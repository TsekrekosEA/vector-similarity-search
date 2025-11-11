#include "hypercube.hpp"
#include "utils.hpp"       // For generate_random_projection_vector and candidate finders
#include "arg_parse.hpp"   // For HypercubeArguments
#include <iostream>
#include <chrono>
#include <unordered_set>
#include <queue>

// ============================================================================
// HypercubeIndex Class Implementation
// ============================================================================

template<typename T>
HypercubeIndex<T>::HypercubeIndex(int kproj_input, int dimensions, std::uint32_t seed)
    : kproj(kproj_input) {
    
    if (kproj_input > 64) { // !!!! Might need to change this, uint64_t is more efficient but 
                            // 64 is an arbitrary constraint for kproj
        throw std::invalid_argument("Hypercube kproj cannot exceed 64 for uint64_t keys.");
    }

    std::cout << "Initializing Hypercube Index with kproj=" << kproj_input << std::endl;
    std::mt19937 random_generator{seed};
    
    random_lines.reserve(kproj);
    for (int i = 0; i < kproj; ++i) {
        random_lines.push_back(generate_random_projection_vector(dimensions, random_generator));
    }
}

template<typename T>
uint64_t HypercubeIndex<T>::get_hash_key(const T* data_point, size_t dim) const {
    uint64_t key = 0;
    for (int i = 0; i < kproj; ++i) {
        double dot_product = 0.0;
        for (size_t j = 0; j < dim; ++j) {
            // check if our point is on the "positive" or "negative" side of the plane
            dot_product += random_lines[i][j] * static_cast<double>(data_point[j]);
        }
        if (dot_product >= 0) {
            // using OR set the bit of the i-th bit to 1 if the point is on the positive side of plane
            key |= (uint64_t(1) << i);
        }
    }
    // final compact hash key of 64 binary bits of 1 and 0
    return key;
}

template<typename T>
void HypercubeIndex<T>::build(const Matrix<T>& dataset) {
    std::cout << "Building Hypercube index for " << dataset.get_rows() << " points..." << std::endl;
    // build our map by taking all points in our dataset, hashing them and putting them in their respective buckets
    for (size_t i = 0; i < dataset.get_rows(); ++i) {
        const T* row_ptr = dataset.get_row(i);
        uint64_t key = get_hash_key(row_ptr, dataset.get_cols());
        table[key].push_back(i);
    }
    std::cout << "Hypercube index build complete." << std::endl;
}

template<typename T>
std::vector<int> HypercubeIndex<T>::get_candidates(const T* query_point, size_t dim, int probes, int M) const {
    //
    std::unordered_set<int> candidates_set;
    std::unordered_set<uint64_t> visited_keys;
    // since BFS is used for search, queue works great here
    uint64_t query_key = get_hash_key(query_point, dim);
    // hypercube graph has cycles, need to keep track of visited keys
    std::queue<uint64_t> q;
    q.push(query_key);
    visited_keys.insert(query_key);

    int probes_checked = 0;

    while (!q.empty() && probes_checked < probes) {
        uint64_t current_key = q.front();
        q.pop();
        probes_checked++;
        // if the current bucket has other keys, process them first
        auto it = table.find(current_key);
        if (it != table.end()) {
            candidates_set.insert(it->second.begin(), it->second.end());
            if (static_cast<int>(candidates_set.size()) >= M) {
                break; // Stop if we've collected enough candidates
            }
        }
        
        // Generate neighbors at Hamming distance 1 from the current key
        for (int i = 0; i < kproj; ++i) {
            uint64_t neighbor_key = current_key ^ (uint64_t(1) << i);
            if (visited_keys.find(neighbor_key) == visited_keys.end()) {
                q.push(neighbor_key);
                visited_keys.insert(neighbor_key);
            }
        }
    }
    return std::vector<int>(candidates_set.begin(), candidates_set.end());
}


template<typename T>
std::vector<ANearNeighbor> HypercubeIndex<T>::find_k_nearest(const T* query_point, const Matrix<T>& dataset, int N, int probes, int M) const {
    std::vector<int> candidate_ids = get_candidates(query_point, dataset.get_cols(), probes, M);
    // use the same utility function as LSH for the final refinement step
    return find_k_nearest_from_candidates(candidate_ids, query_point, dataset, N);
}

template<typename T>
std::vector<ImageId> HypercubeIndex<T>::find_in_range(const T* query_point, const Matrix<T>& dataset, double R, int probes, int M) const {
    std::vector<int> candidate_ids = get_candidates(query_point, dataset.get_cols(), probes, M);
    // use the same utility function as LSH
    return find_in_range_from_candidates(candidate_ids, query_point, dataset, R);
}

// HIGH-LEVEL QUERY FUNCTION IMPLEMENTATION

template <typename T>
std::unique_ptr<Output> hypercube_querying(
    const Matrix<T>& input_images,
    const Matrix<T>& query_images,
    HypercubeArguments& args
) {
    // 1. Initialize the main output object.
    auto output_data = std::make_unique<Output>();
    output_data->algorithm = "Hypercube";
    output_data->queries.resize(query_images.get_rows()); // Pre-size the vector

    // 2. Build the Hypercube index.
    std::cout << "Building Hypercube index..." << std::endl;
    auto start_build = std::chrono::high_resolution_clock::now();
    HypercubeIndex<T> index(args.kproj, input_images.get_cols(), args.seed);
    index.build(input_images);
    auto end_build = std::chrono::high_resolution_clock::now();
    std::cout << "Hypercube index built in " 
              << std::chrono::duration<double>(end_build - start_build).count() << "s." << std::endl;

    // 3. Loop through all queries to find their neighbors.
    std::cout << "Processing " << query_images.get_rows() << " queries with Hypercube..." << std::endl;
    for (size_t i = 0; i < query_images.get_rows(); ++i) {
        const T* query_ptr = query_images.get_row(i);
        
        // --- Perform the approximate k-NN search ---
        std::vector<ANearNeighbor> neighbors = index.find_k_nearest(
            query_ptr, 
            input_images, 
            args.common.number_of_nearest, 
            args.probes, 
            args.M
        );
        
        output_data->queries[i].nearest_neighbors = neighbors;

        // --- Perform range search if the flag is set ---
        if (args.common.search_for_range) {
             std::vector<ImageId> range_neighbors = index.find_in_range(
                query_ptr, 
                input_images, 
                args.common.radius, 
                args.probes, 
                args.M
             );
            output_data->queries[i].r_near_neighbors = range_neighbors;
        }
    }
    
    std::cout << "Hypercube querying complete." << std::endl;
    // The final summary statistics are INTENTIONALLY left blank.
    // They will be calculated in main() after this function returns.
    return output_data;
}


// EXPLICIT TEMPLATE INSTANTIATIONS
template class HypercubeIndex<float>;
template class HypercubeIndex<uint8_t>;

template std::unique_ptr<Output> hypercube_querying<float>(const Matrix<float>&, const Matrix<float>&, HypercubeArguments&);
template std::unique_ptr<Output> hypercube_querying<uint8_t>(const Matrix<uint8_t>&, const Matrix<uint8_t>&, HypercubeArguments&);