/**
 * @file brute_force.hpp
 * @brief Exhaustive linear scan for ground truth nearest neighbor computation
 * 
 * Algorithm Overview:
 * Brute-force search computes exact nearest neighbors via exhaustive linear scan: for each query,
 * compute exact Euclidean distance to all n dataset vectors, sort by distance, return top-N. This
 * is guaranteed to find the true nearest neighbors (no approximation) but has O(n·d) time per
 * query where n is dataset size and d is dimensionality. For large datasets (SIFT: 1M vectors,
 * MNIST: 60K vectors) this is expensive, which is why ANN algorithms exist. However, brute-force
 * is essential for benchmarking: it provides ground truth to measure ANN algorithm accuracy.
 * 
 * Role in Benchmarking:
 * In the benchmarking pipeline, brute-force runs after the ANN algorithm to fill ground truth
 * fields (id_true and distance_true) in the ANearNeighbor structures. The ANN algorithm already
 * filled approximate results (id and distance_approximate). By comparing approximate to true
 * results, we can calculate accuracy metrics: Recall@N (fraction of queries where true NN is in
 * approximate top-N) and Average Approximation Factor (mean of distance_approximate / distance_true).
 * This two-phase approach separates approximate search timing from validation timing.
 * 
 * Implementation Strategy:
 * The function computes full distance matrix (query_count × n distances) but avoids materializing
 * it in memory. For each query: iterate through all n dataset vectors computing distance, maintain
 * min-heap of top-N, extract sorted results. This uses O(N) space per query instead of O(n). For
 * efficiency, distances are computed as squared Euclidean (avoiding sqrt) since comparison order
 * is preserved. Final results take sqrt for output.
 * 
 * Integration with ANN Results:
 * If existing_output is provided (non-null), the function operates in validation mode: it assumes
 * id and distance_approximate fields are already populated by an ANN algorithm, and ONLY fills
 * distance_true and id_true fields. This allows using a single Output structure throughout the
 * pipeline. If existing_output is null, the function operates standalone: it creates a new Output
 * and fills all fields (id, distance_approximate, id_true, distance_true) with brute-force results
 * (approximate and true are identical since brute-force is exact).
 * 
 * Complexity Analysis:
 * Time is O(queries · n · d) for distance computations plus O(queries · n · log(N)) for sorting.
 * For SIFT (queries=10K, n=1M, d=128, N=10): approximately 10K · 1M · 128 = 1.28 trillion flops
 * for distances, plus 10K · 1M · log(10) ≈ 33 million comparisons. On modern CPU (10 GFLOPS),
 * this takes ~128 seconds. Space is O(queries · N) for results plus O(n) for distance buffer per
 * query. This is why ANN algorithms are necessary: brute-force is too slow for real-time applications.
 * 
 * Speedup Metric:
 * The benchmarking pipeline reports "speedup vs brute-force" as t_brute / t_approx. For example,
 * if brute-force takes 100 seconds and LSH takes 2 seconds, speedup is 50x. Typical ANN algorithms
 * achieve 10x-100x speedup while maintaining 90%+ recall, making them practical for large-scale
 * applications like image search or recommendation systems.
 * 
 * Optimization Opportunities:
 * The current implementation is single-threaded. Parallelizing across queries or using SIMD
 * intrinsics for distance computation could provide 4x-8x speedup. However, since brute-force is
 * only used for offline validation (not production queries), optimization is lower priority. The
 * template design already provides type-specific optimization (uint8_t arithmetic vs float arithmetic).
 * 
 * @authors Τσεκρέκος Έγκορ-Ανδριανός, Δημακόπουλος Θεόδωρος
 */

#pragma once

#include <arg_parse.hpp>
#include <cstdint>
#include <data_types.hpp>
#include <memory>

// Brute-force nearest neighbor search via exhaustive linear scan. Computes exact distances from
// each query to all dataset vectors and returns the top-N nearest neighbors. This is the ground
// truth algorithm for benchmarking ANN methods.
//
// Template parameter T is the vector element type (float for SIFT, uint8_t for MNIST). The function
// supports two modes:
//
// 1. Standalone mode (existing_output == nullptr): Creates new Output structure and fills all fields
//    (id, distance_approximate, id_true, distance_true) with brute-force results. Approximate and
//    true results are identical since brute-force is exact. Use this mode to run brute-force as
//    the primary search algorithm.
//
// 2. Validation mode (existing_output != nullptr): Assumes id and distance_approximate fields are
//    already filled by an ANN algorithm. ONLY fills distance_true and id_true fields for ground
//    truth comparison. This enables the benchmarking pipeline: run ANN first (fast, approximate),
//    then run brute-force (slow, exact) to measure accuracy. The function does NOT modify id or
//    distance_approximate in this mode.
//
// Algorithm: For each query, compute exact Euclidean distance to all n dataset vectors using
// squared distance (avoiding sqrt for efficiency), maintain top-N in priority queue, extract sorted
// results and take sqrt for final distances. Time: O(queries · n · d), Space: O(queries · N).
//
// Returns: unique_ptr<Output> containing nearest neighbor results and ground truth for accuracy
// metrics (recall, average approximation factor).
template <typename T>
std::unique_ptr<Output>
brute_force_querying(const Matrix<T>& input_images, const Matrix<T>& query_images,
                     BruteforceArguments& args, std::unique_ptr<Output> existing_output = nullptr);
