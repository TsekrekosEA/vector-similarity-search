#pragma once

#include <cstdint>
#include <data_types.hpp>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

// Forward declaration of argument struct
struct HypercubeArguments;

// The Hypercube hash table uses a fast 64-bit integer as its key.
using HypercubeHashTable = std::unordered_map<uint64_t, std::vector<int>>;

// Main class for the Hypercube LSH index, templated on data type 'T'.
template <typename T> class HypercubeIndex {
  public:
    HypercubeIndex(int kproj_input, int dimensions, std::uint32_t seed);

    void build(const Matrix<T>& dataset);

    // Public query methods that mirror the LSH_Index interface
    std::vector<int> get_candidates(const T* query_point, size_t dim, int probes, int M) const;
    std::vector<ANearNeighbor> find_k_nearest(const T* query_point, const Matrix<T>& dataset, int N,
                                              int probes, int M) const;
    // Add find_in_range for consistency
    std::vector<ImageId> find_in_range(const T* query_point, const Matrix<T>& dataset, double R,
                                       int probes, int M) const;

  private:
    uint64_t get_hash_key(const T* data_point, size_t dim) const;

    int kproj;
    std::vector<std::vector<float>> random_lines;
    HypercubeHashTable table;
};

// HIGH-LEVEL QUERY FUNCTION DECLARATION
template <typename T>
std::unique_ptr<Output> hypercube_querying(const Matrix<T>& input_images,
                                           const Matrix<T>& query_images, HypercubeArguments& args);