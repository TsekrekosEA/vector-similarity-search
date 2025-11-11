#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <utils.hpp>

template <typename T> double euclidean_distance(const T* vec1, const T* vec2, size_t dimension) {
    return sqrt(euclidean_distance_squared(vec1, vec2, dimension));
}

template <typename T>
double euclidean_distance_squared(const T* vec1, const T* vec2, size_t dimension) {
    double sum_of_squares = 0.0;
    for (size_t i = 0; i < dimension; i++) {
        double diff = static_cast<double>(vec1[i]) - static_cast<double>(vec2[i]);
        sum_of_squares += diff * diff;
    }
    return sum_of_squares;
}

// With this function: Built index: k=16 m=28 nbits=8 (build time: 179.57s)   nprobe=1 |
// Recall=0.470 | Speedup=62.6x With just square d: Built index: k=16 m=28 nbits=8 (build time:
// 236.75s)   nprobe=1 | Recall=0.470 | Speedup=63.8x
template <typename T>
double eucl_d_sq_if_smaller_else_inf(const T* vec1, const T* vec2, size_t dimension,
                                     double other_value) {
    double sum_of_squares = 0.0;
    for (size_t i = 0; i < dimension; i++) {
        double diff = static_cast<double>(vec1[i]) - static_cast<double>(vec2[i]);
        sum_of_squares += diff * diff;
        if (sum_of_squares > other_value) {
            return std::numeric_limits<double>::infinity();
        }
    }
    return sum_of_squares;
}

std::vector<float> generate_random_projection_vector(size_t vector_dimension, std::mt19937& gen) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> random_line(vector_dimension);
    for (size_t i = 0; i < vector_dimension; ++i) {
        random_line[i] = dist(gen);
    }
    return random_line;
}

float generate_random_offset(float w, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(0.0f, w);
    return dist(gen);
}

template <typename T>
std::vector<ANearNeighbor> find_k_nearest_from_candidates(const std::vector<int>& candidate_ids,
                                                          const T* query_point,
                                                          const Matrix<T>& dataset, int N) {
    // Only true distance used here, no need for ANearNeighbor struct yet
    using TempNeighbor = std::pair<double, int>;
    std::priority_queue<TempNeighbor> top_candidates;

    // Find top closest points based on true Euclidean distance
    for (int id : candidate_ids) {
        const T* candidate_ptr = dataset.get_row(id);
        // Calculate distance using utility function (takes raw pointers)
        double dist = euclidean_distance(query_point, candidate_ptr, dataset.get_cols());

        if (static_cast<int>(top_candidates.size()) < N) {
            top_candidates.push({dist, id});
        } else if (dist < top_candidates.top().first) {
            top_candidates.pop();
            top_candidates.push({dist, id});
        }
    }

    std::vector<ANearNeighbor> final_neighbors;
    while (!top_candidates.empty()) {
        TempNeighbor temp_neighbor = top_candidates.top();
        top_candidates.pop();

        ANearNeighbor final_neighbor;
        final_neighbor.id = temp_neighbor.second;
        // For LSH and Hypercube, we only set distance_approximate here
        final_neighbor.distance_approximate = temp_neighbor.first;
        final_neighbors.push_back(final_neighbor);
    }
    std::reverse(final_neighbors.begin(), final_neighbors.end());

    return final_neighbors;
}

template <typename T>
std::vector<ImageId> find_in_range_from_candidates(
    // Get all unique candidate IDs
    const std::vector<int>& candidate_ids, const T* query_point, const Matrix<T>& dataset,
    double R) {
    std::vector<ImageId> result;
    // Check if they are within the fixed radius R
    for (int id : candidate_ids) {
        const T* candidate_ptr = dataset.get_row(id);
        double dist = euclidean_distance(query_point, candidate_ptr, dataset.get_cols());

        if (dist <= R) {
            result.push_back(static_cast<ImageId>(id));
        }
    }

    return result;
}

template double euclidean_distance<float>(const float*, const float*, size_t);
template double euclidean_distance_squared<float>(const float*, const float*, size_t);
template double eucl_d_sq_if_smaller_else_inf<float>(const float*, const float*, size_t, double);
template std::vector<ANearNeighbor> find_k_nearest_from_candidates<float>(const std::vector<int>&,
                                                                          const float*,
                                                                          const Matrix<float>&,
                                                                          int);
template std::vector<ImageId> find_in_range_from_candidates<float>(const std::vector<int>&,
                                                                   const float*,
                                                                   const Matrix<float>&, double);

template double euclidean_distance<uint8_t>(const uint8_t*, const uint8_t*, size_t);
template double euclidean_distance_squared<uint8_t>(const uint8_t*, const uint8_t*, size_t);
template double eucl_d_sq_if_smaller_else_inf<uint8_t>(const uint8_t*, const uint8_t*, size_t,
                                                       double);
template std::vector<ANearNeighbor> find_k_nearest_from_candidates<uint8_t>(const std::vector<int>&,
                                                                            const uint8_t*,
                                                                            const Matrix<uint8_t>&,
                                                                            int);
template std::vector<ImageId> find_in_range_from_candidates<uint8_t>(const std::vector<int>&,
                                                                     const uint8_t*,
                                                                     const Matrix<uint8_t>&,
                                                                     double);

template double euclidean_distance<int8_t>(const int8_t*, const int8_t*, size_t);
template double euclidean_distance_squared<int8_t>(const int8_t*, const int8_t*, size_t);
template double eucl_d_sq_if_smaller_else_inf<int8_t>(const int8_t*, const int8_t*, size_t, double);
template std::vector<ANearNeighbor> find_k_nearest_from_candidates<int8_t>(const std::vector<int>&,
                                                                           const int8_t*,
                                                                           const Matrix<int8_t>&,
                                                                           int);
template std::vector<ImageId> find_in_range_from_candidates<int8_t>(const std::vector<int>&,
                                                                    const int8_t*,
                                                                    const Matrix<int8_t>&, double);

// ============================================================================
// METRICS CALCULATION
// ============================================================================

void calculate_metrics(Output& output, double t_approximate_total, double t_true_total) {
    if (output.queries.empty()) {
        output.average_af = 0.0;
        output.recall_at_n = 0.0;
        output.queries_per_second = 0.0;
        output.t_approximate_average = 0.0;
        output.t_true_average = 0.0;
        return;
    }

    double total_af = 0.0;
    size_t queries_with_true_nn = 0;
    size_t valid_queries = 0;
    size_t num_queries = output.queries.size();

    for (const auto& query_result : output.queries) {
        if (query_result.nearest_neighbors.empty()) {
            continue;
        }

        const auto& first_neighbor = query_result.nearest_neighbors[0];

        // Skip queries where approximate algorithm found no candidates
        // (distance_approximate = 0 and id = 0 indicates no neighbors found)
        if (first_neighbor.distance_approximate == 0.0 && first_neighbor.id == 0) {
            continue; // LSH/Hypercube found nothing - skip this query
        }

        valid_queries++;

        // Calculate Approximation Factor for first neighbor
        // Only calculate AF if distance_true > 0
        if (first_neighbor.distance_true > 0.0) {
            double af = first_neighbor.distance_approximate / first_neighbor.distance_true;
            total_af += af;
        } else {
            // If distance_true is 0, the point is identical (perfect match)
            // Treat as AF = 1.0 (approximate found the exact same point)
            total_af += 1.0;
        }

        // Calculate Recall@N: Check if true nearest neighbor appears in approximate results
        // The true NN is id_true of the first neighbor (the actual closest point)
        ImageId true_nearest_id = first_neighbor.id_true;

        bool found_true_nn = false;
        for (const auto& neighbor : query_result.nearest_neighbors) {
            if (neighbor.id == true_nearest_id) {
                found_true_nn = true;
                break;
            }
        }

        if (found_true_nn) {
            queries_with_true_nn++;
        }
    }

    // Calculate final metrics
    if (valid_queries > 0) {
        output.average_af = total_af / valid_queries;
        output.recall_at_n = static_cast<double>(queries_with_true_nn) / valid_queries;
    } else {
        output.average_af = 0.0;
        output.recall_at_n = 0.0;
    }

    // Calculate timing metrics
    if (t_approximate_total > 0.0 && num_queries > 0) {
        output.t_approximate_average = t_approximate_total / num_queries;
        output.queries_per_second = num_queries / t_approximate_total;
    } else {
        output.t_approximate_average = 0.0;
        output.queries_per_second = 0.0;
    }

    if (t_true_total > 0.0 && num_queries > 0) {
        output.t_true_average = t_true_total / num_queries;
    } else {
        output.t_true_average = 0.0;
    }
}
