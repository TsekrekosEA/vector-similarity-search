/**
 * @file ivfpq.hpp
 * @brief Inverted File with Product Quantization (IVF-PQ) for memory-efficient ANN search
 * 
 * Algorithm Overview:
 * IVF-PQ combines coarse quantization (IVF-Flat) with fine quantization (Product Quantization)
 * to achieve massive memory compression while maintaining reasonable search accuracy. The
 * algorithm divides the vector space into coarse Voronoi cells, then compresses vectors within
 * each cell using product quantization. This gives memory reduction of 32x-256x compared to
 * storing original vectors.
 * 
 * Theoretical Foundation:
 * Product Quantization decomposes a d-dimensional vector into m sub-vectors of d/m dimensions
 * each, then quantizes each sub-vector independently. The key insight is that joint quantization
 * of all dimensions requires 2^k codewords (exponential growth), but decomposing into m groups
 * requires only m·2^k codewords (linear growth). For example, quantizing a 128-D vector as 8
 * sub-vectors of 16-D each with 8-bit codes requires 8·256 = 2048 centroids, while joint
 * quantization would require 256^8 centroids (intractable). The compressed representation stores
 * m codes (one per sub-vector) instead of d floats.
 * 
 * Two-Level Quantization:
 * IVF-PQ uses coarse quantization followed by fine quantization. First, coarse quantization
 * partitions the dataset into k Voronoi cells using k-means clustering, storing the coarse
 * centroid. Second, fine quantization compresses residuals (difference between vector and its
 * coarse centroid) using product quantization. Each residual is split into m sub-vectors and
 * each sub-vector is replaced with its nearest quantization centroid's code. This residual
 * quantization is more accurate than quantizing the original vector directly because residuals
 * have smaller magnitude and tighter distribution.
 * 
 * Asymmetric Distance Computation (ADC):
 * During query processing, we use asymmetric distance which computes distance from uncompressed
 * query to compressed vectors. The query residual (query minus coarse centroid) is kept uncompressed
 * for precision. For each candidate vector, we decompose the query residual into m sub-vectors,
 * compute distance from each query sub-vector to all quantization centroids in that sub-space
 * (precomputed lookup table of m·2^k distances), then sum distances to the quantization centroids
 * indicated by the candidate's codes. This gives approximate distance ≈ ||q_res - x_res||² where
 * q_res is query residual (exact) and x_res is reconstructed from codes.
 * 
 * Memory Savings:
 * Original storage uses d·4 bytes per vector (d floats). Compressed storage uses m·k/8 bytes
 * per vector (m codes of k bits each). For SIFT (d=128), using m=8 sub-vectors with k=8 bits
 * gives 8·1 = 8 bytes per vector (64x compression from 512 bytes). For MNIST (d=784), using
 * m=49 sub-vectors with k=8 bits gives 49 bytes per vector (6.4x compression from 314 bytes
 * if uint8_t). Memory for quantization centroids is m·2^k·(d/m) = d·2^k which is 128·256 = 32KB
 * for SIFT, negligible for million-point datasets.
 * 
 * Algorithm Details:
 * Index construction runs coarse clustering on dataset to get k coarse centroids, then for each
 * coarse cluster: computes residuals (vector minus coarse centroid) for all vectors in cluster,
 * splits residuals into m sub-vectors, runs k-means with 2^nbits centroids on each sub-vector
 * group independently (m·k independent k-means problems), assigns each sub-vector to nearest
 * quantization centroid and stores its code (0 to 2^nbits-1). Query processing computes query
 * residual for nprobe nearest coarse centroids, precomputes distances from query sub-vectors to
 * all quantization centroids (lookup table), for each candidate in probed clusters sums distances
 * using candidate's codes (table lookups), sorts candidates by approximate distance, then refines
 * top candidates by computing exact distance from original query to full vectors.
 * 
 * Parameter Tuning:
 * Number of coarse clusters k typically follows sqrt(n) to 4·sqrt(n) like IVF-Flat. Larger k
 * gives better coarse quantization but more overhead. Number of sub-vectors m is constrained by
 * m divides d evenly. Common choices: for SIFT (d=128) use m=8 (16-D sub-vectors) or m=16 (8-D
 * sub-vectors). Larger m gives more compression but more table lookups. Number of bits nbits
 * determines codebook size 2^nbits. Common choices: nbits=8 (256 centroids per sub-space, good
 * accuracy) or nbits=4 (16 centroids, higher compression, lower accuracy). Memory for codebooks
 * is m·2^nbits·(d/m) = d·2^nbits, which is manageable for nbits ≤ 8. Number of probes nprobe
 * trades accuracy for speed like IVF-Flat. Larger nprobe searches more clusters but computes
 * more distance table lookups.
 * 
 * Complexity Analysis:
 * Let n = dataset size, k = coarse clusters, m = sub-vectors, s = sub-vector dimensionality (d/m),
 * c_coarse = 2^nbits centroids per sub-vector. Index construction takes O(iterations·n·k·d) for
 * coarse k-means, then O(m·iterations·(n/k)·c_coarse·s) = O(m·iterations·n·2^nbits·s) for fine
 * k-means. Space is O(n·m·nbits/8 + k·d + m·k·c_coarse·s) = O(n·m + k·d + k·m·2^nbits·d/m).
 * Query takes O(k·d) for coarse search, O(nprobe·m·c_coarse·s) = O(nprobe·m·2^nbits·d/m) for
 * distance tables, O(candidates·m) for asymmetric distance via table lookups, O(top·d) for exact
 * refinement. Distance computation is dominated by table lookups (O(m) per candidate) which is
 * much faster than full distance (O(d)).
 * 
 * Comparison to IVF-Flat:
 * IVF-Flat stores original vectors with O(n·d) memory and exact distances during fine search.
 * IVF-PQ stores compressed vectors with O(n·m·nbits/8) memory and approximate distances. For
 * SIFT with m=8 and nbits=8, IVF-PQ uses 8 bytes per vector vs 512 bytes for IVF-Flat (64x
 * compression). IVF-PQ has slightly lower recall (approximate distances) but much faster queries
 * (table lookups vs full distance) and much less memory (critical for billion-scale datasets).
 * IVF-PQ is preferred when memory is constrained or when datasets are very large (100M+ vectors).
 * 
 * Implementation Details:
 * For each coarse cluster, we store m independent Ivfflat instances, one per sub-vector group.
 * Each Ivfflat clusters the residual sub-vectors into 2^nbits clusters. Codes are stored in
 * code_per_piece_per_image_per_cluster[coarse_id].get_row(local_img_id)[sub_vector_id]. Residual
 * computation handles uint8_t carefully by casting to int32_t to avoid underflow (since residuals
 * can be negative). The type TWithNegatives is int8_t for uint8_t data and float for float data.
 * CodebookIndex is uint8_t for nbits ≤ 8 or uint32_t for larger nbits. Template parameters provide
 * type safety and avoid runtime overhead.
 * 
 * Extensions:
 * Optimized Product Quantization (OPQ) applies rotation before PQ to decorrelate dimensions,
 * improving accuracy. Polysemous codes reuse PQ codes as binary hashes for fast pre-filtering.
 * GPU implementations parallelize distance table computation and asymmetric distance lookups.
 * 
 * @authors Τσεκρέκος Έγκορ-Ανδριανός, Δημακόπουλος Θεόδωρος
 */

#pragma once

#include <data_types.hpp>
#include <ivfflat.hpp>
#include <random>

// IVF-PQ Index: Inverted File with Product Quantization
//
// Architecture: The index has three levels of data structures. First level is coarse_clusters
// (an Ivfflat instance) partitioning dataset into k Voronoi cells. Second level is
// clusters_per_dimension_group_per_coarse_cluster storing m·k Ivfflat instances for quantizing
// residual sub-vectors (for each of k coarse clusters, m sub-vector quantizers). Third level is
// code_per_piece_per_image_per_cluster storing compressed codes (for each coarse cluster, each
// image in that cluster, m codes).
//
// Template Parameters:
// T is the dataset element type (float for SIFT, uint8_t for MNIST). TWithNegatives must support
// negative values for residual computation (float for float data, int8_t for uint8_t data since
// residuals can be negative when subtracting centroids). CodebookIndex is the code storage type
// (uint8_t for nbits ≤ 8, uint32_t for larger nbits). These provide type safety without runtime
// overhead.
//
// Design Rationale:
// Storing residuals (vector minus coarse centroid) instead of raw vectors for product quantization
// improves accuracy because residuals have smaller magnitude and tighter distribution. Independent
// quantization of each sub-vector group (m separate k-means problems) avoids the curse of
// dimensionality that joint quantization would face. Precomputing distance tables (query sub-vector
// to all quantization centroids) enables fast approximate distance via table lookups (O(m) instead
// of O(d) per candidate).
template <typename T, typename TWithNegatives /* one bit shorter is okay */, typename CodebookIndex>
// TWithNegatives would be replaced with constexpr if it wasn't for the private members below.
class Ivfpq {
  public:
    // Construct an IVF-PQ index with default printing enabled. Initializes data structures but
    // does not build the index. Call build() to run coarse clustering and product quantization.
    // Parameters: dataset is the reference to vectors (not copied), k_clusters is the number of
    // coarse Voronoi cells (typically sqrt(n) to 4·sqrt(n)), dimensions_of_sub_vectors is d/m
    // where m is the number of sub-vectors (d must be divisible by m), nbits is the number of
    // bits per code (codebook size = 2^nbits, typically 4 or 8), seed is the random seed for
    // k-means initialization.
    Ivfpq(const Matrix<T>& dataset, int k_clusters, int dimensions_of_sub_vectors, int nbits,
          int seed);
    
    // Construct an IVF-PQ index with configurable printing. Use should_print=false to disable
    // progress output when building multiple indices or during benchmarking.
    Ivfpq(const Matrix<T>& dataset, int k_clusters, int dimensions_of_sub_vectors, int nbits,
          int seed, bool should_print);
    
    // Build the index: run coarse k-means and product quantization. Algorithm: first, build coarse
    // clusters via k-means to get k Voronoi cells; second, for each coarse cluster: extract vectors
    // in that cluster, compute residuals (vector minus coarse centroid), split each residual into
    // m sub-vectors of dimension d/m, for each sub-vector group run k-means with 2^nbits centroids
    // to build sub-vector quantizer, assign each sub-vector to nearest quantizer centroid and store
    // code. This builds m·k independent k-means problems. The codes form a compressed representation:
    // each vector is represented by m codes of nbits each instead of d floats.
    // Time: O(iterations·n·k·d + m·iterations·n·2^nbits·d/m), Space: O(n·m + k·m·2^nbits·d/m)
    void build();
    
    // Query the index to find candidate nearest neighbors using asymmetric distance computation.
    // Three-phase search: first, coarse search finds num_nearest_centroids closest coarse centroids
    // to query; second, for each probed cluster: compute query residual, precompute distance table
    // from query sub-vectors to all quantization centroids (m·2^nbits distances), for each candidate
    // in cluster compute approximate distance via table lookups (sum of distances indexed by
    // candidate's codes), sort cluster candidates; third, merge and refine top candidates by
    // computing exact distances to original vectors. Returns candidates sorted by exact distance.
    // Asymmetric distance approximates ||query - vector||² ≈ ||query_residual - reconstructed_residual||².
    // Time: O(k·d + nprobe·m·2^nbits·d/m + candidates·m + top·d) where table lookups dominate
    std::vector<std::tuple<double, int>>
    get_candidates(const T* pixels_of_query, int num_nearest_centroids, int num_results) const;
    
    // Calculate silhouette score for coarse clustering quality. Measures how well the coarse
    // k-means separated the dataset into clusters. Good coarse clustering (high silhouette)
    // improves IVF-PQ accuracy because residuals within each cluster have smaller magnitude and
    // tighter distribution, making them easier to quantize accurately.
    // Range: [-1, 1], higher is better
    double get_silhouette() const {
        return coarse_clusters.get_silhouette();
    }
    
    // Print index statistics for debugging. Outputs coarse cluster statistics (number of clusters,
    // vectors per cluster) and fine quantizer statistics for each sub-vector group in each coarse
    // cluster. Useful for diagnosing clustering quality and understanding memory usage.
    void print() const;

  private:
    const int dimensions_of_sub_vectors; // Sub-vector dimensionality (d/m)
    int pieces_of_a_vector;              // Number of sub-vectors (m = d / dimensions_of_sub_vectors)
    const int nbits;                     // Bits per code (codebook size = 2^nbits)
    const int seed;                      // Random seed for k-means
    const bool should_print = true;      // Enable progress output
    const Matrix<T>& dataset;            // Reference to original dataset (not owned)
    
    // Coarse quantizer: partitions dataset into k Voronoi cells. Used to find nprobe nearest
    // coarse centroids during queries.
    Ivfflat<T> coarse_clusters;

    // Residual storage for sub-vector groups. residual_dimension_groups_per_offset[i] contains
    // residuals (vector minus coarse centroid) for sub-vector group i%m in coarse cluster i/m.
    // These matrices are inputs to the fine k-means clusterers. Size: k·m matrices of size
    // (n/k)×(d/m) on average.
    std::vector<Matrix<TWithNegatives>> residual_dimension_groups_per_offset;

    // Fine quantizers: one Ivfflat per sub-vector group per coarse cluster. Total of k·m
    // quantizers. clusters_per_dimension_group_per_coarse_cluster[coarse_id * m + sub_vec_id]
    // contains the Ivfflat that quantizes the sub_vec_id-th sub-vector group for vectors in
    // coarse cluster coarse_id. Each Ivfflat has 2^nbits centroids representing quantization
    // codewords for that sub-space.
    std::vector<Ivfflat<TWithNegatives>> clusters_per_dimension_group_per_coarse_cluster;

    // Compressed codes: for each coarse cluster, stores m codes per vector.
    // code_per_piece_per_image_per_cluster[coarse_id].get_row(local_img_id)[sub_vec_id] is the
    // code (0 to 2^nbits-1) for the sub_vec_id-th sub-vector of the local_img_id-th vector in
    // coarse cluster coarse_id. This is the final compressed representation, using m·nbits/8
    // bytes per vector instead of d·sizeof(T) bytes.
    std::vector<Matrix<CodebookIndex>> code_per_piece_per_image_per_cluster;

    // Compute approximate distances to candidates in a single coarse cluster using asymmetric
    // distance computation. Precomputes distance table from query sub-vectors to all quantization
    // centroids, then for each candidate sums distances indexed by candidate's codes. Sorts
    // candidates by approximate distance and inserts top candidates_per_cluster into the global
    // candidate list at insert_position. This is the core IVF-PQ distance computation using table
    // lookups for speed.
    void add_local_candidates(std::vector<std::tuple<double, int>>& candidates,
                              size_t insert_position, const T* pixels_of_query,
                              size_t candidates_per_cluster, size_t i_centroid) const;
};
