#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <random>
#include <data_types.hpp>

// ============================================================================
// DISTANCE CALCULATION UTILITIES
// ============================================================================

// Compute the Euclidean distance between two vectors
template <typename T>
double euclidean_distance(const T* vec1, const T* vec2, size_t dimension);

// Similar to euclidean_distance but skip the computation of the square root
template <typename T>
double euclidean_distance_squared(const T* vec1, const T* vec2, size_t dimension);

// Often faster than the above, when seeking the smallest distance in a list.
// Returns infinity if the distance is bigger than the other distance.
template <typename T>
double eucl_d_sq_if_smaller_else_inf(const T* vec1, const T* vec2, size_t dimension, double other_distance);

// ============================================================================
// LSH RANDOM PROJECTION UTILITIES (shared by LSH and Hypercube)
// ============================================================================

// Generate a random projection vector from a normal distribution N(0,1)
// Used to create the random hyperplanes for LSH and Hypercube hash functions
std::vector<float> generate_random_projection_vector(size_t vector_dimension, std::mt19937& gen);

// Generate a random offset from a uniform distribution U(0, w)
// Used to shift the hash function bins
float generate_random_offset(float w, std::mt19937& gen);

// ============================================================================
// CANDIDATE FILTERING UTILITIES (shared by LSH and Hypercube)
// ============================================================================

// Find the k nearest neighbors from a set of candidate point IDs
template<typename T>
std::vector<ANearNeighbor> find_k_nearest_from_candidates(
    const std::vector<int>& candidate_ids,
    const T* query_point,
    const Matrix<T>& dataset,
    int N
);

// Find all neighbors within a fixed radius R from a set of candidate point IDs
template<typename T>
std::vector<ImageId> find_in_range_from_candidates(
    const std::vector<int>& candidate_ids,
    const T* query_point,
    const Matrix<T>& dataset,
    double R
);

// ============================================================================
// METRICS CALCULATION UTILITIES
// ============================================================================

// Calculate performance metrics for the output
// - average_af: Average Approximation Factor (distance_approximate / distance_true)
// - recall_at_n: Fraction of queries where true NN appears in approximate results
// - queries_per_second: QPS based on approximate search time
// - t_approximate_average: Average time per query for approximate search
// - t_true_average: Average time per query for brute force search
// 
// If timing parameters are not provided (0.0), timing metrics will be set to 0.0
void calculate_metrics(Output& output, double t_approximate_total = 0.0, double t_true_total = 0.0);
