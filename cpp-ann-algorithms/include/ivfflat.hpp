/**
 * @file ivfflat.hpp
 * @brief Inverted File with Flat (IVF-Flat) index for approximate nearest neighbor search
 * 
 * Algorithm Overview:
 * IVF-Flat is a partition-based ANN algorithm that divides the dataset into clusters using
 * k-means, then searches only the most relevant clusters during queries. This dramatically
 * reduces the search space from n points to a small fraction of n.
 * 
 * Theoretical Foundation:
 * The key insight is that if we partition the vector space into regions (Voronoi cells), a
 * query's nearest neighbors are likely to be in the same region or nearby regions. The index
 * consists of a coarse quantizer (k centroids defining k Voronoi cells) and inverted lists
 * (for each centroid, a list of vectors in that cell). This is called "inverted file" by
 * analogy to text search inverted indices: in text search we have word to list of documents
 * containing the word, in IVF we have centroid to list of vectors closest to that centroid.
 * 
 * Two-Level Search:
 * The search has two phases. First, coarse search finds the nprobe nearest centroids to the
 * query by computing distance from query to all k centroids and selecting nprobe closest
 * centroids. Second, fine search exhaustively searches vectors in those nprobe clusters by
 * taking the union of all inverted lists for selected centroids. If the true nearest neighbor
 * is in cluster C_i and C_i is not among the nprobe selected clusters, we'll miss it. Thus
 * larger nprobe gives higher recall (more clusters searched) while smaller nprobe gives faster
 * search (fewer candidates).
 * 
 * Algorithm Details:
 * Index construction runs k-means clustering on the dataset by initializing k centroids as
 * random data points, then iterating: assign points to nearest centroid, recompute centroids
 * as means of assigned points, converge when centroids stabilize. Then build inverted lists by
 * storing each point's ID in its cluster's list. Query processing finds nprobe nearest centroids
 * (coarse search), collects candidates from those nprobe clusters (fine search), computes exact
 * distances to all candidates, and returns top-N nearest neighbors.
 * 
 * K-Means Clustering:
 * We use Lloyd's Algorithm: initialize by selecting k random data points as initial centroids,
 * then repeat until convergence by assigning each point to its nearest centroid and recomputing
 * each centroid as the mean of its assigned points. Convergence stops when centroids change by
 * less than threshold. Our implementation stops if fewer than 10% of centroid pixels change or
 * after max_centroid_revisions iterations (default: 10). Complexity is O(iterations · n · k · d)
 * time where iterations is approximately 10-50 (typically converges quickly), n is dataset size,
 * k is number of clusters, d is dimensionality, and O(k · d + n) space for centroids and assignments.
 * 
 * Parameter Tuning:
 * Number of clusters k typically follows rule of thumb k ≈ sqrt(n) to k ≈ 4·sqrt(n). For n=1M,
 * use k ≈ 1000 to 4000. Larger k gives smaller clusters and faster queries but slower build,
 * while smaller k gives larger clusters, slower queries but faster build. Number of probes nprobe
 * typically ranges from 1 to k/10. nprobe=1 gives very fast search but low recall (60-70%),
 * nprobe=10 gives moderate speed and good recall (85-95%), nprobe=100 gives slow search but high
 * recall (98-99%). Recommended configurations: fast low accuracy uses k=1000 nprobe=1, balanced
 * uses k=1000 nprobe=10, high accuracy uses k=4000 nprobe=50.
 * 
 * Complexity Analysis:
 * Let n = dataset size, k = clusters, d = dimensions, c = candidates per query. Index construction
 * takes O(iterations · n · k · d) time approximately O(10 · n · k · d) and O(n + k · d) space for
 * inverted lists and centroids. Query takes O(k · d + nprobe · c/k · d) time for finding nprobe
 * nearest centroids plus computing distances to candidates. If c ≈ nprobe · n/k, this is O(k · d + c · d).
 * Space is O(c) for candidate set. With nprobe=10 and k=1000, we search 10 clusters out of 1000,
 * so expected candidates ≈ 10/1000 · n = n/100, giving speedup ≈ 100x over brute-force (minus overhead).
 * 
 * Comparison to LSH:
 * IVF-Flat uses data-driven partitioning (k-means) with k Voronoi cells and searches nearest cells,
 * while LSH uses random partitioning (hash functions) with L hash tables and takes union of hash buckets.
 * IVF-Flat has slower build time (k-means) but often better accuracy, while LSH has faster index
 * construction and theoretically guaranteed properties. IVF-Flat advantages include better accuracy
 * (data-aware partitioning) and tunable via nprobe (smooth accuracy-speed tradeoff). LSH advantages
 * include faster index construction and theoretically guaranteed properties.
 * 
 * Extensions:
 * IVF-Flat is often combined with compression techniques: IVF-PQ adds Product Quantization for
 * memory compression, IVF-OPQ uses optimized product quantization, IVF-HNSW uses HNSW for coarse
 * quantizer (faster centroid search).
 * 
 * @authors Δημακόπουλος Θεόδωρος
 */

#pragma once

#include <data_types.hpp>
#include <string>

// Ivfflat: Inverted File index with flat storage of original vectors
//
// Architecture: The index has centroids (k cluster centers as d-dimensional vectors), inverted
// lists (k lists mapping centroid_id to vector_ids in that cluster), and references to the
// original dataset (not copied) for distance computations.
//
// The index is "flat" because it stores references to original vectors, not compressed versions.
// This preserves exact distance computations during the fine search phase.
//
// Design Choices:
// We use Lloyd's Algorithm with Early Stopping because standard k-means can take 50-100 iterations
// on large datasets, but for ANN we don't need perfect clusters since "good enough" suffices. Early
// stopping at 10 iterations balances quality vs build time. Empirically, clusters after 10 iterations
// are approximately 95% as good as converged. For convergence detection, we track number of changed
// centroid pixels and stop if less than 10% of pixels change (configurable via
// set_how_many_centroid_pixels_end_the_build). This is faster than checking centroid movement since
// no sqrt is needed. For degenerate cases: if dataset size < k_clusters we create single cluster
// (no benefit from clustering), if dataset size = 0 we create empty inverted list (for IVF-PQ sub-clusters).
template <typename T> class Ivfflat {
  public:
    // Construct an IVF-Flat index. Initializes data structures but does not build the index.
    // Call build() to run k-means and populate inverted lists. The dataset is stored by reference
    // to avoid copying millions of vectors. The caller must ensure dataset lifetime exceeds the
    // index lifetime.
    Ivfflat(const Matrix<T>& dataset, int k_clusters, int seed);
    
    // Configure convergence threshold for k-means. Sets the number of changed centroid pixels
    // that triggers early stopping. Lower values give more iterations (better clusters, slower
    // build), higher values give fewer iterations (faster build, slightly worse clusters).
    // Default: 10% of total centroid pixels
    void set_how_many_centroid_pixels_end_the_build(int n) {
        changed_centroid_pixels_that_would_end_build = n;
    }
    
    // Set maximum k-means iterations. Hard limit on clustering iterations to prevent excessive
    // build time. Default: 10. Rationale: For ANN applications, 10 iterations usually provide
    // sufficiently good clusters. Perfect convergence is unnecessary since we're doing approximate
    // search anyway.
    void set_max_build_iterations(int n) {
        max_centroid_revisions = n;
    }
    
    // Disable progress output. Useful for benchmarking or when using the index as a component
    // (e.g., in IVF-PQ where multiple Ivfflat instances are created).
    void disable_printing() {
        should_print = false;
    }
    
    // Disable terminal screen clearing during progress updates. By default, progress bars clear
    // the screen for cleaner output. Disable for logging or when multiple indices are built
    // concurrently.
    void stop_flickering() {
        should_clear_progress = false;
    }
    
    // Set custom label for progress bar. Useful when building multiple indices to distinguish
    // which is building. Example labels: "IVF-PQ Coarse", "IVF-PQ Sub-cluster 5"
    void set_progress_bar_label(std::string s) {
        label = s;
    }
    
    // Build the index: run k-means and populate inverted lists. Algorithm: initialize centroids
    // as k random data points, then Lloyd's iterations repeating until convergence or max iterations:
    // assignment phase assigns each vector to its nearest centroid, update phase recomputes each
    // centroid as the mean of its cluster, convergence check counts changed centroid pixels and
    // stops if less than threshold. Finally inverted lists are ready for queries. Optimization:
    // During assignment, we use eucl_d_sq_if_smaller_else_inf() which aborts distance computation
    // as soon as it exceeds the current best, providing approximately 30-40% speedup in
    // high-dimensional spaces.
    // Time: O(iterations · n · k · d), Space: O(n + k · d)
    void build();
    
    // Query the index to find candidate nearest neighbors. Two-phase search: coarse search finds
    // num_nearest_clusters closest centroids to query, fine search collects all vectors in those
    // clusters as candidates. Returns candidates sorted by distance (closest first), up to num_results.
    // Algorithm: compute distance from query to all k centroids, select nprobe = num_nearest_clusters
    // closest centroids, for each selected centroid retrieve its inverted list (vector IDs), compute
    // exact distance from query to each vector, add (distance, id) tuples to candidate set, sort
    // candidates by distance, return top num_results. Returns squared distances to avoid sqrt overhead;
    // callers must take sqrt if absolute distances are needed.
    // Time: O(k·d + c·d + c·log(c)) where c = candidates
    std::vector<std::tuple<double, int>>
    get_candidates(const T* pixels_of_query, int num_nearest_clusters, int num_results) const;
    
    // Calculate silhouette score (clustering quality metric). Silhouette score measures how
    // well-separated clusters are: score near +1 means points are close to their cluster and far
    // from others (good), score near 0 means points are on cluster boundaries (mediocre), score
    // near -1 means points may be in wrong clusters (bad). Use case: diagnosing cluster quality
    // or selecting optimal k value. Note: expensive to compute O(n²·d), so only use for analysis,
    // not in production query pipelines.
    // Range: [-1, 1]
    double get_silhouette() const;

    // Print index statistics for debugging. Outputs number of clusters, vectors per cluster
    // (min, max, mean, stddev), and memory usage.
    void print() const;
    
    // Get the centroid matrix for analysis or IVF-PQ. Use case: IVF-PQ needs centroids to compute
    // residuals.
    const Matrix<T>& get_centroids() const {
        return centroids;
    }
    
    // Get inverted lists for analysis. Returns const reference to vector of inverted lists where
    // image_ids_per_cluster[i] contains IDs of vectors in cluster i.
    const std::vector<std::vector<int>>& get_image_ids_per_cluster() const {
        return image_ids_per_cluster;
    }

  private:
    // Initialize k centroids as random dataset points. K-means++ initialization (selecting distant
    // points) could provide better initial clusters, but random initialization is simpler and
    // converges fast enough for ANN purposes.
    void initialize_centroids_as_random_images(int seed);
    
    // Assign each vector to its nearest centroid (k-means assignment phase). Clears and repopulates
    // image_ids_per_cluster. Time: O(n · k · d)
    void group_by_nearest_centroid();
    
    // Compute average distance from a vector to a set of vectors. Used in silhouette score calculation.
    double average_distance(const T* image, const std::vector<int>& image_ids) const;

    const int seed;                    // Random seed for centroid initialization
    const int pixels_per_image;        // Vector dimensionality (d)
    const bool worth_clustering;       // False if dataset too small for k clusters
    
    // K-means convergence parameters
    int changed_centroid_pixels_that_would_end_build = 1; // Convergence threshold
    int max_centroid_revisions = 10;                      // Max iterations
    
    // Output control
    bool should_print = true;          // Enable progress output
    bool should_clear_progress = true; // Clear screen during progress
    std::string label = "IVF-Flat";    // Progress bar label
    
    // Core data structures
    const Matrix<T>& dataset;                      // Reference to original dataset (not owned)
    Matrix<T> centroids;                           // k×d matrix of cluster centers
    std::vector<std::vector<int>> image_ids_per_cluster; // Inverted lists: cluster_id to vector_ids
};