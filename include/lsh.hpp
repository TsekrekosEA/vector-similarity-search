#pragma once

#include <data_types.hpp>
#include <arg_parse.hpp>
#include <vector>
#include <random>
#include <unordered_map>
#include <memory>
#include <cstdint>

// Compute the base LSH hash value using the formula: h(p) = ⌊(p·v + t) / w⌋
template<typename T>
int lsh_base_hash(const std::vector<float>& random_line, const T* data_point, size_t dim, float t, float w);

// SuperHash: Represents a composite hash function made of k independent base hash functions
// Template parameter T allows it to work with different data types (float, uint8_t, etc.)
template<typename T>
class SuperHash {
public:
    SuperHash(int k, size_t vector_dimension, float w, std::mt19937& gen);
    
    // Return type is now long long
    long long get_hash_key(const T* data_point, size_t dim) const;

private:
    std::vector<std::vector<float>> random_lines; // Stores k random vectors v
    std::vector<float> offsets;                   // Stores k random offsets t
    float w;                                      // The window size w

    // New members for composite hashing
    std::vector<int> r_values; // Stores k random integers for the combination
    const long long M = 4294967291; // A large prime (2^32 - 5)
};

using LshHashTable = std::unordered_map<long long, std::vector<int>>;

// LSH_Index: Main class for Locality Sensitive Hashing index
// Template parameter T allows it to work with different data types (float or uint8_t)
template<typename T>
class LSH_Index {
public:
    LSH_Index(int L_input, int k_input, float w_input, int dimensions, std::uint32_t seed);
    void build(const Matrix<T>& dataset);
    
    // Query the LSH index to find candidate nearest neighbors
    // Returns a vector of point IDs that are candidates for being near the query point
    // All query methods take raw pointers for efficiency (avoids vector copies)
    std::vector<int> get_candidates(const T* query_point, size_t dim) const;
    std::vector<ANearNeighbor> find_k_nearest(const T* query_point, const Matrix<T>& dataset, int N) const;
    std::vector<ImageId> find_in_range(const T* query_point, const Matrix<T>& dataset, double R) const;
    
private:
    int L; // Number of hash tables
    // The LSH index is composed of L hash functions and L tied hash tables
    std::vector<SuperHash<T>> hash_functions;
    std::vector<LshHashTable> tables;
};

// HIGH-LEVEL QUERY FUNCTION
template <typename T>
std::unique_ptr<Output> lsh_querying(
    const Matrix<T>& input_images,
    const Matrix<T>& query_images,
    LshArguments& args
);
