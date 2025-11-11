#include <cmath>
#include <iostream>
#include <lsh.hpp>
#include <unordered_set>
#include <utils.hpp>

template <typename T>
int lsh_base_hash(const std::vector<float>& random_line, const T* data_point, size_t dim, float t,
                  float w) {
    double dot_product = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dot_product += static_cast<double>(random_line[i]) * static_cast<double>(data_point[i]);
    }
    return static_cast<int>(
        std::floor((dot_product + static_cast<double>(t)) / static_cast<double>(w)));
}

template <typename T>
SuperHash<T>::SuperHash(int k, size_t vector_dimension, float w_input, std::mt19937& gen)
    : w(w_input) {
    // Reserve space for efficiency
    random_lines.reserve(k);
    offsets.reserve(k);

    // Create k independent base hash functions
    for (int i = 0; i < k; ++i) {
        random_lines.push_back(generate_random_projection_vector(vector_dimension, gen));
        offsets.push_back(generate_random_offset(w, gen));
    }

    // Initialize r_values with k random integers
    r_values.reserve(k);
    std::uniform_int_distribution<int> distrib; // Use default range
    for (int i = 0; i < k; ++i) {
        r_values.push_back(distrib(gen));
    }
}

// This function takes a data point and applies the k stored hash functions to it.
template <typename T> long long SuperHash<T>::get_hash_key(const T* data_point, size_t dim) const {
    long long combined_hash = 0;
    for (size_t i = 0; i < random_lines.size(); ++i) {
        int h_value = lsh_base_hash(random_lines[i], data_point, dim, offsets[i], w);

        // Combine using the random linear combination. Cast to long long to prevent overflow.
        combined_hash += static_cast<long long>(r_values[i]) * h_value;
    }

    // The double modulo handles negative results correctly in C++.
    // (a % M + M) % M ensures the result is always in [0, M-1].
    return (combined_hash % M + M) % M;
}

// LSH_Index Implementation
template <typename T>
LSH_Index<T>::LSH_Index(int L_input, int k_input, float w_input, int dimensions, std::uint32_t seed)
    : L(L_input) {
    std::cout << "Initializing LSH Index with L=" << L_input << ", k=" << k_input << std::endl;
    std::mt19937 random_generator{seed};

    hash_functions.reserve(L);
    for (int i = 0; i < L; ++i) {
        // Each call to the SuperHash constructor creates a new, unique set of k random projections.
        // Emplace back works better here because push_back would make redundant copying calls
        hash_functions.emplace_back(k_input, dimensions, w_input, random_generator);
    }
    // Initialize L empty hash tables
    tables.resize(L);
}

template <typename T> void LSH_Index<T>::build(const Matrix<T>& dataset) {
    std::cout << "Building LSH index for " << dataset.get_rows() << " points..." << std::endl;

    // Outer loop: Iterate through each of the L hash table systems
    for (int i = 0; i < L; ++i) {
        std::cout << "  Populating table " << i + 1 << "/" << L << "..." << std::endl;

        // Inner loop: Iterate through every point in the dataset
        for (size_t j = 0; j < dataset.get_rows(); ++j) {
            const T* row_ptr = dataset.get_row(j);
            // Directly pass the pointer
            long long key = hash_functions[i].get_hash_key(row_ptr, dataset.get_cols());
            tables[i][key].push_back(j);
        }
    }
    std::cout << "LSH index build complete." << std::endl;
}

// Returns a vector of id's of points in the dataset
template <typename T>
std::vector<int> LSH_Index<T>::get_candidates(const T* query_point, size_t dim) const {
    // Use an unordered_set to collect unique candidate point IDs from all L tables, avoid
    // duplicates from different superhashes
    std::unordered_set<int> candidates_set;
    // Query each of the L hash tables
    for (int i = 0; i < L; ++i) {
        // Compute the hash key for the query point using the i-th hash function
        long long key = hash_functions[i].get_hash_key(query_point, dim);
        // Look up this key in the i-th hash table
        auto it = tables[i].find(key);

        // If the key exists in the table, add all point IDs in that bucket to our candidates
        if (it != tables[i].end()) {
            const std::vector<int>& bucket = it->second;
            candidates_set.insert(bucket.begin(), bucket.end());
        }
    }
    // Convert the set to a vector and return
    std::vector<int> candidates(candidates_set.begin(), candidates_set.end());
    return candidates;
}

template <typename T>
std::vector<ANearNeighbor> LSH_Index<T>::find_k_nearest(const T* query_point,
                                                        const Matrix<T>& dataset, int N) const {
    std::vector<int> candidate_ids = get_candidates(query_point, dataset.get_cols());
    return find_k_nearest_from_candidates(candidate_ids, query_point, dataset, N);
}

template <typename T>
std::vector<ImageId> LSH_Index<T>::find_in_range(const T* query_point, const Matrix<T>& dataset,
                                                 double R) const {
    std::vector<int> candidate_ids = get_candidates(query_point, dataset.get_cols());
    return find_in_range_from_candidates(candidate_ids, query_point, dataset, R);
}

template <typename T>
std::unique_ptr<Output> lsh_querying(const Matrix<T>& input_images, const Matrix<T>& query_images,
                                     LshArguments& args) {
    auto output = std::make_unique<Output>();
    output->algorithm = "LSH";
    output->queries.resize(query_images.get_rows());

    LSH_Index<T> index(args.L, args.k, static_cast<float>(args.w), input_images.get_cols(),
                       static_cast<std::uint32_t>(args.seed));
    index.build(input_images);

    std::cout << "Processing " << query_images.get_rows() << " queries..." << std::endl;

    for (size_t i_query = 0; i_query < query_images.get_rows(); i_query++) {
        const T* query_pixels = query_images.get_row(i_query);

        size_t N = (args.common.number_of_nearest > 0)
                       ? static_cast<size_t>(args.common.number_of_nearest)
                       : 1;

        std::vector<ANearNeighbor> neighbors = index.find_k_nearest(query_pixels, input_images, N);

        output->queries[i_query].nearest_neighbors = neighbors;

        if (args.common.search_for_range) {
            std::vector<ImageId> range_neighbors =
                index.find_in_range(query_pixels, input_images, args.common.radius);
            output->queries[i_query].r_near_neighbors = range_neighbors;
        }
    }

    std::cout << "LSH querying complete." << std::endl;
    return output;
}

template class SuperHash<float>;
template class SuperHash<uint8_t>;

template class LSH_Index<float>;
template class LSH_Index<uint8_t>;

template std::unique_ptr<Output> lsh_querying<float>(const Matrix<float>&, const Matrix<float>&,
                                                     LshArguments&);
template std::unique_ptr<Output> lsh_querying<uint8_t>(const Matrix<uint8_t>&,
                                                       const Matrix<uint8_t>&, LshArguments&);

template int lsh_base_hash<float>(const std::vector<float>&, const float*, size_t, float, float);
template int lsh_base_hash<uint8_t>(const std::vector<float>&, const uint8_t*, size_t, float,
                                    float);
