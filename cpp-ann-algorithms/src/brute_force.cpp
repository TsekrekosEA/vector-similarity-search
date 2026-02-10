#include <algorithm>
#include <brute_force.hpp>
#include <tuple>
#include <utils.hpp>

// Brute force nearest neighbor search algorithm
// Computes exact distances to all points and returns the k nearest neighbors
// Template parameter T can be float (for SIFT) or uint8_t (for MNIST)
// If existing_output is provided, ONLY fills distance_true fields without modifying id or
// distance_approximate
template <typename T>
std::unique_ptr<Output>
brute_force_querying(const Matrix<T>& input_images, const Matrix<T>& query_images,
                     BruteforceArguments& args, std::unique_ptr<Output> existing_output) {
    // Check if we're filling existing output BEFORE moving it
    bool filling_existing = (existing_output != nullptr);

    // Use existing output or create new one
    auto output = existing_output ? std::move(existing_output) : std::make_unique<Output>();

    // Only set algorithm name and resize if creating new output
    if (!filling_existing) {
        output->algorithm = "brute force";
        output->queries.resize(query_images.get_rows());
    }

    // i is necessary,  for (auto query : queries)  wouldn't be enough
    for (size_t i_query = 0; i_query < query_images.get_rows(); i_query++) {
        const T* query_pixels = query_images.get_row(i_query);
        std::vector<std::tuple<double, ImageId>> euclidean_distances;

        // Optimization: Reserve capacity upfront to avoid reallocations
        euclidean_distances.reserve(input_images.get_rows());

        for (ImageId i_input = 0; i_input < input_images.get_rows(); i_input++) {
            const T* input_pixels = input_images.get_row(i_input);
            // Optimization: Use squared distance (avoids expensive sqrt)
            // sqrt will be taken only for the final N neighbors returned
            double distance_squared =
                euclidean_distance_squared(query_pixels, input_pixels, query_images.get_cols());
            euclidean_distances.push_back(std::make_tuple(distance_squared, i_input));
        }

        // Optimization: Use partial_sort instead of full sort
        // Only sorts the first N elements, much faster for large datasets when N << dataset_size
        // Time complexity: O(n log k) instead of O(n log n) where k = N
        size_t N = (args.common.number_of_nearest > 0)
                       ? static_cast<size_t>(args.common.number_of_nearest)
                       : 1;
        size_t sort_count = std::min(N, euclidean_distances.size());

        if (args.common.search_for_range && !filling_existing) {
            // For range search, we need full sort to find all points within radius efficiently
            std::sort(euclidean_distances.begin(), euclidean_distances.end());
        } else {
            // For nearest neighbor only, partial sort is much faster
            std::partial_sort(euclidean_distances.begin(), euclidean_distances.begin() + sort_count,
                              euclidean_distances.end());
        }

        // ALWAYS fill the N true nearest neighbors (id_true and distance_true)
        // This is needed for proper metrics calculation (Average AF, Recall@N)
        if (filling_existing) {
            // Fill distance_true and id_true for all N neighbors
            // Resize if needed to ensure we have N slots
            if (output->queries[i_query].nearest_neighbors.size() < N) {
                output->queries[i_query].nearest_neighbors.resize(N);
            }

            for (size_t i = 0; i < N && i < euclidean_distances.size(); i++) {
                // Convert squared distance back to actual distance for output
                output->queries[i_query].nearest_neighbors[i].distance_true =
                    std::sqrt(std::get<0>(euclidean_distances[i]));
                output->queries[i_query].nearest_neighbors[i].id_true =
                    std::get<1>(euclidean_distances[i]);
            }
        } else {
            // Creating new output - fill everything
            for (size_t i = 0; i < N && i < euclidean_distances.size(); i++) {
                ANearNeighbor nn;
                nn.id_true = std::get<1>(euclidean_distances[i]);
                // Convert squared distance back to actual distance for output
                nn.distance_true = std::sqrt(std::get<0>(euclidean_distances[i]));
                output->queries[i_query].nearest_neighbors.push_back(nn);
            }
        }

        if (args.common.search_for_range && !filling_existing) {
            // Optimization: Compare with squared radius to avoid sqrt
            double radius_squared = args.common.radius * args.common.radius;

            // Optimization: Pre-count points within radius, then reserve capacity
            size_t count_in_range = 0;
            for (const auto& dist_pair : euclidean_distances) {
                if (std::get<0>(dist_pair) <= radius_squared) {
                    count_in_range++;
                } else {
                    break; // Already sorted, can stop early
                }
            }

            output->queries[i_query].r_near_neighbors.reserve(count_in_range);
            for (size_t i_near = 0; i_near < count_in_range; i_near++) {
                output->queries[i_query].r_near_neighbors.push_back(
                    std::get<1>(euclidean_distances[i_near]));
            }
        }
        // TODO what should r_near_neighbors be if !search_for_range?
    }
    return output;
}

template std::unique_ptr<Output> brute_force_querying<float>(const Matrix<float>&,
                                                             const Matrix<float>&,
                                                             BruteforceArguments&,
                                                             std::unique_ptr<Output>);
template std::unique_ptr<Output> brute_force_querying<uint8_t>(const Matrix<uint8_t>&,
                                                               const Matrix<uint8_t>&,
                                                               BruteforceArguments&,
                                                               std::unique_ptr<Output>);
