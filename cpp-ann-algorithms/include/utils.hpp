/**
 * @file utils.hpp
 * @brief Core utility functions for vector similarity search
 * 
 * This file provides essential building blocks used across all ANN algorithms. The distance
 * calculation functions offer optimized Euclidean distance computations with early termination.
 * LSH utilities generate random projections and hash function components. Candidate filtering
 * functions perform k-NN and range search on candidate sets. Metrics calculation provides
 * comprehensive performance evaluation.
 * 
 * Design Principles:
 * Template-based genericity ensures type safety without runtime overhead. Raw pointer interfaces
 * enable zero-copy, cache-friendly operations. Optimized inner loops maximize performance in
 * distance-dominated workloads.
 * 
 * @authors Τσεκρέκος Έγκορ-Ανδριανός, Δημακόπουλος Θεόδωρος
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <data_types.hpp>
#include <random>
#include <vector>

// Distance Calculation Utilities

// Compute Euclidean distance between two vectors using the standard L2 norm formula:
// d(v1, v2) = sqrt(sum((v1[i] - v2[i])^2)). This is the most common distance metric for
// high-dimensional vector spaces, satisfying metric axioms and providing intuitive geometric
// meaning. Uses raw pointers for maximum cache efficiency. The sqrt operation is relatively
// expensive (10-20 cycles on modern CPUs), so for distance comparisons consider using
// euclidean_distance_squared() to avoid unnecessary sqrt calculations.
// Time: O(dimension), Space: O(1)
template <typename T> double euclidean_distance(const T* vec1, const T* vec2, size_t dimension);

// Compute squared Euclidean distance without the square root: sum((v1[i] - v2[i])^2).
// Use this instead of euclidean_distance() when only distance comparisons are needed, not
// absolute distance values. Since sqrt is monotonic, distance ordering is preserved:
// d1 < d2 if and only if d1² < d2². Avoiding sqrt() provides approximately 20-30% speedup
// in distance-dominated workloads, which is crucial for algorithms like k-NN that perform
// millions of comparisons. Typical use cases include finding nearest neighbors (only ordering
// matters), sorting by distance, range queries (after squaring the threshold), and internal
// distance computations in clustering algorithms.
// Time: O(dimension), Space: O(1)
template <typename T>
double euclidean_distance_squared(const T* vec1, const T* vec2, size_t dimension);

// Compute squared Euclidean distance with early termination. Calculates the distance but
// stops early if it exceeds a threshold, returning infinity. This optimization is valuable
// when searching for k nearest neighbors, where we maintain the k-th best distance as a
// threshold. If a candidate's partial distance sum already exceeds this threshold, we can
// abort the calculation early since this candidate can never be in the top-k. In high-dimensional
// spaces (d > 100), early termination can skip 50-80% of dimension computations for distant
// points, providing significant speedup.
// Time: O(dimension) worst-case, O(1) to O(dimension) average, Space: O(1)
template <typename T>
double eucl_d_sq_if_smaller_else_inf(const T* vec1, const T* vec2, size_t dimension,
                                     double other_distance);

// LSH Random Projection Utilities (shared by LSH and Hypercube)

// Generate a random projection vector with components drawn from a standard normal distribution
// N(0,1). These vectors define random hyperplanes used in LSH hash functions. Random projection
// is a dimensionality reduction technique based on the Johnson-Lindenstrauss lemma. Projecting
// high-dimensional vectors onto random directions approximately preserves pairwise distances with
// high probability. The LSH hash function h_v(p) = floor((p·v + t) / w) uses this vector v along
// with a random offset t and bucket width w. Normal distribution is rotationally invariant,
// ensuring no directional bias in the hash function, which is essential for the probabilistic
// guarantees of LSH.
// Time: O(vector_dimension), Space: O(vector_dimension)
std::vector<float> generate_random_projection_vector(size_t vector_dimension, std::mt19937& gen);

// Generate random offset uniformly distributed in [0, w) for LSH hash functions. The offset t
// in the hash function h(p) = floor((p·v + t) / w) provides randomization of bucket boundaries.
// Different offsets create different bucket alignments, reducing correlation between hash functions.
// Without offset (t=0), all hash functions would have bucket boundaries at the same locations
// (multiples of w). With random offset, bucket boundaries are staggered, creating more diverse
// hash functions and improving LSH collision properties.
// Time: O(1), Space: O(1)
float generate_random_offset(float w, std::mt19937& gen);

// Candidate Filtering Utilities (shared by LSH and Hypercube)

// Find k nearest neighbors from a candidate set. Given a set of candidate point IDs (from
// LSH/Hypercube buckets), compute exact distances to the query and return the k nearest neighbors.
// ANN algorithms work in two phases: first, candidate generation uses hash functions to identify
// promising candidates (fast but approximate), then candidate refinement computes exact distances
// and selects top-k (slower but accurate). This function performs the refinement phase.
//
// Algorithm: Compute exact distance from query to each candidate, sort candidates by distance
// ascending, return top-N as ANearNeighbor objects. Complexity is O(c·d + c·log(c)) where c is
// number of candidates and d is dimensionality. For LSH/Hypercube to be effective, c should be
// much smaller than dataset size n (typically c = O(log n) to O(sqrt(n))).
// Time: O(c·d + c·log(c)), Space: O(c)
template <typename T>
std::vector<ANearNeighbor> find_k_nearest_from_candidates(const std::vector<int>& candidate_ids,
                                                          const T* query_point,
                                                          const Matrix<T>& dataset, int N);

// Find all neighbors within radius R from candidates. Range search variant that returns ALL
// candidates within distance R (not just top-k). For each candidate, compute exact distance to
// query and add to result if distance <= R. Differs from k-NN in that k-NN returns fixed-size
// results (exactly k neighbors) while range returns variable-size results (0 to all candidates).
// k-NN always returns k results (or fewer if dataset is small) while range may return 0 results
// if no points are within R. k-NN requires sorting while range just needs filtering. Use cases
// include finding all near-duplicates within threshold, density estimation (count neighbors
// within R), and anomaly detection (points with few neighbors).
// Time: O(c·d), Space: O(c) worst-case
template <typename T>
std::vector<ImageId> find_in_range_from_candidates(const std::vector<int>& candidate_ids,
                                                   const T* query_point, const Matrix<T>& dataset,
                                                   double R);

// Metrics Calculation Utilities

// Calculate comprehensive performance metrics for ANN search. Computes aggregate statistics
// from per-query results to evaluate algorithm performance. This function is central to the
// benchmarking pipeline.
//
// Metrics Computed:
//
// Average Approximation Factor (AF): mean(distance_approximate / distance_true) across all
// queries. Measures distance accuracy with ideal value 1.0 (perfect) and typical range 1.0-1.5
// depending on algorithm and parameters. Quantifies the quality of approximate distances.
//
// Recall@N: fraction of queries where id_true appears in top-N results. Measures retrieval
// accuracy with range [0.0, 1.0]. For example, 0.95 means 95% of queries found the true nearest
// neighbor. This is the primary metric for evaluating ANN effectiveness.
//
// Queries Per Second (QPS): total_queries / t_approximate_total. Measures throughput with
// higher being better (faster search). Quantifies search speed.
//
// Average Query Times: t_approximate_average is mean query time for ANN search, t_true_average
// is mean query time for brute-force search. Speedup = t_true_average / t_approximate_average.
//
// Usage: After running both approximate and brute-force search, fill output.queries with id and
// distance_approximate (from ANN) and id_true and distance_true (from brute-force), then call
// calculate_metrics(output, t_approx_total, t_true_total) to populate all metric fields.
// Time: O(q·n) where q = queries, n = neighbors per query, Space: O(1)
// Note: If timing parameters are 0.0, timing metrics are set to 0.0
void calculate_metrics(Output& output, double t_approximate_total = 0.0, double t_true_total = 0.0);
