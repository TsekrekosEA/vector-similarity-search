#include <algorithm> // partial_sort
#include <cstdint>
#include <iostream>
#include <ivfpq.hpp>
#include <limits> // std::numeric_limits<double>::infinity()
#include <sstream>
#include <utils.hpp>

template <typename T, typename TWithNegatives, typename CodebookIndex>
Ivfpq<T, TWithNegatives, CodebookIndex>::Ivfpq(const Matrix<T>& dataset, int k_clusters,
                                               int dimensions_of_sub_vectors, int nbits, int seed)
    : dimensions_of_sub_vectors(dimensions_of_sub_vectors), nbits(nbits), seed(seed),
      dataset(dataset), coarse_clusters(dataset, k_clusters, seed) {
    // Calculate number of sub-vectors (m). If dimensions don't divide evenly, excess dimensions
    // are discarded. For example, if d=130 and dimensions_of_sub_vectors=16, we get m=8 sub-vectors
    // covering 128 dimensions, ignoring the last 2 dimensions. This is acceptable in practice since
    // losing a few dimensions has minimal impact on search quality.
    pieces_of_a_vector = dataset.get_cols() / dimensions_of_sub_vectors;
}
template <typename T, typename TWithNegatives, typename CodebookIndex>
Ivfpq<T, TWithNegatives, CodebookIndex>::Ivfpq(const Matrix<T>& dataset, int k_clusters,
                                               int dimensions_of_sub_vectors, int nbits, int seed,
                                               bool should_print)
    : dimensions_of_sub_vectors(dimensions_of_sub_vectors), nbits(nbits), seed(seed),
      should_print(should_print), dataset(dataset), coarse_clusters(dataset, k_clusters, seed) {

    // Calculate number of sub-vectors, discarding excess dimensions if d is not divisible by
    // dimensions_of_sub_vectors.
    pieces_of_a_vector = dataset.get_cols() / dimensions_of_sub_vectors;
    if (!should_print) {
        coarse_clusters.disable_printing();
    }
}

template <typename T, typename TWithNegatives, typename CodebookIndex>
void Ivfpq<T, TWithNegatives, CodebookIndex>::build() {

    if (should_print) {
        std::cout << "Building IVFPQ index for " << dataset.get_rows() << " points..." << std::endl;
    }
    
    // Build coarse clusters first (IVF part). Use 20 iterations instead of default 10 for better
    // coarse clustering since residual quality depends on good coarse centroids.
    coarse_clusters.set_max_build_iterations(20);
    coarse_clusters.build();

    const size_t num_coarse = coarse_clusters.get_centroids().get_rows();

    // Preallocate space for compressed codes. Reserve prevents reallocation which would invalidate
    // pointers to elements.
    code_per_piece_per_image_per_cluster.reserve(num_coarse);
    residual_dimension_groups_per_offset.reserve(num_coarse * pieces_of_a_vector);

    // For each coarse cluster, build m sub-vector quantizers (Product Quantization part)
    for (size_t centroid_id = 0; centroid_id < num_coarse; centroid_id++) {

        const std::vector<int>& image_ids =
            coarse_clusters.get_image_ids_per_cluster()[centroid_id];
        auto& cpic = code_per_piece_per_image_per_cluster;
        cpic.emplace_back(image_ids.size(), pieces_of_a_vector);
        Matrix<CodebookIndex>& code_per_piece_per_image = cpic[cpic.size() - 1];

        // For each sub-vector group, build a quantizer
        for (int i_piece = 0; i_piece < pieces_of_a_vector; i_piece++) {
            int first_pixel = i_piece * dimensions_of_sub_vectors;

            // Compute residuals (vector minus coarse centroid) for this sub-vector group.
            // Residual quantization is more accurate than raw vector quantization because residuals
            // have smaller magnitude and tighter distribution after removing the coarse structure.
            std::vector<Matrix<TWithNegatives>>& rdg = residual_dimension_groups_per_offset;
            rdg.emplace_back(image_ids.size(), dimensions_of_sub_vectors);
            Matrix<TWithNegatives>& residual_i_th_pieces = rdg[rdg.size() - 1];
            
            for (size_t i_image = 0; i_image < image_ids.size(); i_image++) {
                TWithNegatives* residual_piece = residual_i_th_pieces.get_row(i_image);
                const T* of_image = &dataset.get_row(image_ids[i_image])[first_pixel];
                const T* of_centroid =
                    &coarse_clusters.get_centroids().get_row(centroid_id)[first_pixel];
                
                // Compute residual with careful type handling. For uint8_t, cast to int32_t to
                // avoid underflow (residuals can be negative). Result fits in int8_t since
                // |residual| < 256.
                for (int i = 0; i < dimensions_of_sub_vectors; i++) {
                    if constexpr (std::is_same_v<T, uint8_t>) {
                        residual_piece[i] =
                            static_cast<TWithNegatives>(static_cast<int32_t>(of_image[i]) -
                                                        static_cast<int32_t>(of_centroid[i]));
                    } else {
                        residual_piece[i] = of_image[i] - of_centroid[i];
                    }
                }
            }

            // Run k-means on residual sub-vectors to build sub-vector quantizer. Each quantizer
            // has 2^nbits centroids (codebook). The centroids represent quantization codewords
            // for this sub-space.
            std::vector<Ivfflat<TWithNegatives>>& cpd =
                clusters_per_dimension_group_per_coarse_cluster;
            cpd.emplace_back(residual_i_th_pieces, 1u << nbits, seed);
            
            assert(cpd.size() - 1 == centroid_id * pieces_of_a_vector + i_piece);
            assert(cpd.size() == residual_dimension_groups_per_offset.size());
            
            Ivfflat<TWithNegatives>& inner_clusters = cpd[cpd.size() - 1];
            if (should_print) {
                std::stringstream ss;
                ss << "Inner cluster " << cpd.size() << "/" << (num_coarse * pieces_of_a_vector);
                inner_clusters.set_progress_bar_label(ss.str());
                inner_clusters.stop_flickering();
            } else {
                inner_clusters.disable_printing();
            }
            inner_clusters.build();

            // Assign codes: for each residual sub-vector, find its nearest quantization centroid
            // and store the centroid's index as the code. This compresses the sub-vector from
            // (d/m) floats to a single code of nbits.
            const size_t n = inner_clusters.get_centroids().get_rows(); // may be less than 2^nbits if cluster too small
            for (size_t inner_centroid_index = 0; inner_centroid_index < n;
                 inner_centroid_index++) {

                // Assign code inner_centroid_index to all sub-vectors in this quantization cluster
                const std::vector<int>& image_id =
                    inner_clusters.get_image_ids_per_cluster()[inner_centroid_index];
                for (size_t i = 0; i < image_id.size(); i++) {
                    code_per_piece_per_image.get_row(image_id[i])[i_piece] = inner_centroid_index;
                }
            }
        }
    }
    if (should_print) {
        std::cout << "IVFPQ index build complete.                      "
                  << std::endl; // Spaces overwrite progress bar
    }
}

// Helper function to partially sort a vector and return the effective size. If num_nearest_centroids
// is less than vector size, uses partial_sort for O(n·log(k)) instead of full sort's O(n·log(n)).
// If num_nearest_centroids exceeds size, falls back to full sort. Returns the actual number of
// sorted elements (min of requested and available).
static size_t sort_n(std::vector<std::tuple<double, int>>& distances,
                     size_t num_nearest_centroids) {
    if (num_nearest_centroids <= distances.size()) {
        std::partial_sort(distances.begin(), distances.begin() + num_nearest_centroids,
                          distances.end());

        return num_nearest_centroids;
    } else {
        std::sort(distances.begin(), distances.end());
        return distances.size();
    }
}

/*  Before:

  Built index: k=16 m=28 nbits=8 (build time: 181.15s)
    nprobe=1 | Recall=0.647 | Speedup=48.5x
    nprobe=4 | Recall=0.749 | Speedup=12.4x
Building IVFPQ index for 60000 points...
IVFPQ index build complete.                      ids, 0/409 centroid pixels modified
  Built index: k=16 m=49 nbits=8 (build time: 196.41s)
    nprobe=1 | Recall=0.650 | Speedup=39.8x
    nprobe=4 | Recall=0.754 | Speedup=10.0x
Building IVFPQ index for 60000 points...
IVFPQ index build complete.                      ids, 0/358 centroid pixels modified
  Built index: k=16 m=56 nbits=8 (build time: 215.54s)
    nprobe=1 | Recall=0.651 | Speedup=36.9x
    nprobe=4 | Recall=0.753 | Speedup=9.3x
Building IVFPQ index for 60000 points...
IVFPQ index build complete.                      roids, 0/179 centroid pixels modified
  Built index: k=16 m=112 nbits=8 (build time: 250.96s)
    nprobe=1 | Recall=0.650 | Speedup=24.5x
*/

// Compute approximate distances to candidates in a single coarse cluster using asymmetric distance
// computation (ADC). This is the core of IVF-PQ query processing, using precomputed distance tables
// for fast approximate distance via table lookups instead of full distance computations.
template <typename T, typename TWithNegatives, typename CodebookIndex>
void Ivfpq<T, TWithNegatives, CodebookIndex>::add_local_candidates(
    std::vector<std::tuple<double, int>>& candidates, size_t insert_position,
    const T* pixels_of_query, size_t candidates_per_cluster, size_t i_centroid) const {

    // Step 1: Compute query residual (query minus coarse centroid). This residual will be
    // decomposed into sub-vectors and compared against quantization centroids. Residual space
    // is where product quantization operates.
    const size_t pixels_per_image = dataset.get_cols();
    std::vector<TWithNegatives> residual(pixels_per_image);
    const T* coarse_centroid_pixels = coarse_clusters.get_centroids().get_row(i_centroid);
    for (size_t i = 0; i < pixels_per_image; i++) {
        if constexpr (std::is_same_v<T, uint8_t>) {
            residual[i] =
                static_cast<TWithNegatives>(static_cast<int32_t>(pixels_of_query[i]) -
                                            static_cast<int32_t>(coarse_centroid_pixels[i]));
        } else {
            residual[i] = pixels_of_query[i] - coarse_centroid_pixels[i];
        }
    }

    // Step 2: Precompute distance table (asymmetric distance lookup table). For each sub-vector
    // group, compute distances from query sub-vector to all 2^nbits quantization centroids. This
    // gives m·2^nbits distances. These distances are reused for all candidates in this cluster,
    // amortizing the O(m·2^nbits·d/m) cost across all candidates. This is the key optimization
    // of asymmetric distance computation.
    Matrix<double> distance_to_centroids_of_pieces(pieces_of_a_vector, 1 << nbits);
    for (int i_piece = 0; i_piece < pieces_of_a_vector; i_piece++) {
        const Ivfflat<TWithNegatives>& inner_clusters =
            clusters_per_dimension_group_per_coarse_cluster[i_centroid * pieces_of_a_vector +
                                                            i_piece];
        const Matrix<TWithNegatives>& inner_centroids = inner_clusters.get_centroids();
        double* distance_to_centroids = distance_to_centroids_of_pieces.get_row(i_piece);
        TWithNegatives* residual_piece = &residual[i_piece * dimensions_of_sub_vectors];
        
        // Compute squared Euclidean distance from query sub-vector to each quantization centroid
        for (size_t i = 0; i < inner_centroids.get_rows() /* may not be 1 << nbits */; i++) {
            distance_to_centroids[i] = euclidean_distance_squared<TWithNegatives>(
                residual_piece, inner_centroids.get_row(i), dimensions_of_sub_vectors);
        }
    }

    // Step 3: Compute approximate distance to each candidate via table lookups. For each candidate,
    // retrieve its m codes, sum distances indexed by those codes from the precomputed table. This
    // is O(m) per candidate instead of O(d) for full distance. The approximate distance is
    // ||q_res - x_res||² ≈ sum_{i=1}^m ||q_res_i - c_{code_i}||² where c_{code_i} is the
    // quantization centroid indicated by the candidate's code for sub-vector i.
    const std::vector<int>& image_ids_in_cluster =
        coarse_clusters.get_image_ids_per_cluster()[i_centroid];
    size_t images_in_the_cluster = image_ids_in_cluster.size();
    std::vector<std::tuple<double, int>> ds(images_in_the_cluster);

    for (size_t i_image = 0; i_image < images_in_the_cluster; i_image++) {

        double asymmetric_distance = 0;

        // Sum distances indexed by codes (table lookups)
        const CodebookIndex* code_per_piece =
            code_per_piece_per_image_per_cluster[i_centroid].get_row(i_image);
        for (int i_piece = 0; i_piece < pieces_of_a_vector; i_piece++) {
            int code_of_x = code_per_piece[i_piece];
            asymmetric_distance += distance_to_centroids_of_pieces.unchecked(i_piece, code_of_x);
        }

        // Store local index (within cluster), will map to global later
        ds[i_image] = std::make_tuple(asymmetric_distance, i_image);
    }

    // Step 4: Select top candidates from this cluster based on approximate distance. We over-sample
    // (candidates_per_cluster > num_results) to compensate for PQ approximation errors. Some
    // candidates with good approximate distance may have poor exact distance, so taking more
    // candidates increases chance of capturing true nearest neighbors.
    size_t n = sort_n(ds, candidates_per_cluster);

    // Step 5: Refine selected candidates by computing exact distance to original vectors. Map local
    // cluster indices to global dataset indices and compute full Euclidean distance. This refinement
    // corrects for PQ approximation errors. Insert into global candidate list at specified position.
    for (size_t i = 0; i < n; i++) {
        // Map local cluster index to global dataset index
        int local_image_index = std::get<1>(ds[i]);
        int global_image_id = image_ids_in_cluster[local_image_index];

        // Calculate exact distance using full original vector
        double real_distance_squared = euclidean_distance_squared(
            pixels_of_query, dataset.get_row(global_image_id), pixels_per_image);

        // Store exact distance and global image ID in candidate list
        if (insert_position + i < candidates.size()) {
            candidates[insert_position + i] =
                std::make_tuple(real_distance_squared, global_image_id);
        }
    }
}

// Query the IVF-PQ index to find approximate nearest neighbors. Three-phase algorithm: coarse
// search to find nprobe nearest coarse centroids, asymmetric distance computation within probed
// clusters using PQ codes and lookup tables, refinement with exact distances. Returns num_results
// candidates sorted by exact distance.
template <typename T, typename TWithNegatives, typename CodebookIndex>
std::vector<std::tuple<double, int>> Ivfpq<T, TWithNegatives, CodebookIndex>::get_candidates(
    const T* pixels_of_query, int num_nearest_centroids, int num_results) const {

    // Allocate candidate buffer. We over-sample by taking 20x num_results candidates per probed
    // cluster to compensate for PQ approximation errors. Some candidates with good approximate
    // distance (computed via PQ codes) have poor exact distance, so over-sampling increases the
    // chance of capturing true nearest neighbors. This is a critical parameter: too small causes
    // recall loss, too large wastes computation on refinement.
    size_t candidates_per_cluster = num_results * 20;
    size_t max_candidates = candidates_per_cluster * num_nearest_centroids + num_results;

    std::vector<std::tuple<double, int>> candidates(
        max_candidates,
        std::make_tuple(std::numeric_limits<double>::infinity(), -1)); // Initialize with sentinels
    const size_t pixels_per_image = dataset.get_cols();

    // Phase 1: Coarse search to find nprobe = num_nearest_centroids nearest coarse centroids.
    // Compute exact distance from query to all k coarse centroids and select closest nprobe.
    // This determines which Voronoi cells to search, same as IVF-Flat.
    const Matrix<T>& centroids = coarse_clusters.get_centroids();
    std::vector<std::tuple<double, int>> distances(centroids.get_rows());
    for (size_t i = 0; i < centroids.get_rows(); i++) {
        double d =
            euclidean_distance_squared(pixels_of_query, centroids.get_row(i), pixels_per_image);
        distances[i] = std::make_tuple(d, i);
    }
    const size_t min_nearest_centroids = sort_n(distances, num_nearest_centroids);

    // Phase 2 & 3: For each probed cluster, compute asymmetric distances (ADC) to all candidates
    // in that cluster using precomputed distance tables, then refine by computing exact distances.
    // After each cluster, re-sort the global candidate list to keep best num_results at front.
    for (size_t c = 0; c < min_nearest_centroids; c++) {
        size_t i_centroid = std::get<1>(distances[c]);
        size_t insert_position = num_results + c * candidates_per_cluster;

        // Process this cluster: compute distance table, compute approximate distances via table
        // lookups, select top candidates_per_cluster, refine with exact distances
        add_local_candidates(candidates, insert_position, pixels_of_query, candidates_per_cluster,
                             i_centroid);

        // Re-sort after each cluster to maintain best num_results at front. This ensures we
        // always keep the global best candidates as we incrementally add candidates from each
        // probed cluster.
        sort_n(candidates, num_results);
    }

    // Return top num_results with exact distances. Some entries may have infinite distance and -1
    // index if fewer than num_results candidates exist in probed clusters (happens when clusters
    // are small or nprobe is low).
    candidates.resize(num_results);
    return candidates;
}

template <typename T, typename TWithNegatives, typename CodebookIndex>
void Ivfpq<T, TWithNegatives, CodebookIndex>::print() const {

    std::cout << "=== coarse clusters ===" << std::endl;
    coarse_clusters.print();
    std::cout << std::endl;
    size_t total_coarse = coarse_clusters.get_centroids().get_rows();
    for (size_t coarse_cluster = 0; coarse_cluster < total_coarse; coarse_cluster++) {
        for (int dimension_group = 0; dimension_group < pieces_of_a_vector; dimension_group++) {
            std::cout << "=== inner cluster, coarse = " << coarse_cluster
                      << ", dimension group = " << dimension_group << " ===" << std::endl;
            size_t idx = coarse_cluster * pieces_of_a_vector + dimension_group;
            clusters_per_dimension_group_per_coarse_cluster[idx].print();
            std::cout << std::endl;
        }
    }
}

// Explicit template instantiations
template class Ivfpq<float, float, uint8_t>;
template class Ivfpq<float, float, uint32_t>;
template class Ivfpq<uint8_t, int8_t, uint8_t>;
template class Ivfpq<uint8_t, int8_t, uint32_t>;
