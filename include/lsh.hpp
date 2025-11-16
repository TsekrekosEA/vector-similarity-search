/**
 * @file lsh.hpp
 * @brief Locality-Sensitive Hashing (LSH) implementation for approximate nearest neighbor search
 * 
 * Algorithm Overview:
 * LSH is a probabilistic technique that hashes similar vectors to the same buckets with high
 * probability. Unlike cryptographic hashes which aim for randomness, LSH preserves similarity:
 * nearby points in the original space hash to the same bucket, while distant points hash to
 * different buckets.
 * 
 * Theoretical Foundation:
 * For Euclidean space, we use random projection hash functions: h(p) = floor((p·v + t) / w)
 * where p is the input vector, v is a random unit vector from N(0,1), t is a random offset
 * from U(0, w), and w is the bucket width parameter. For two points p and q at distance d,
 * the collision probability is approximately P(h(p) = h(q)) ≈ 1 - d/w for small d. Nearby
 * points have high collision probability while distant points have low collision probability.
 * 
 * Amplification via Composite Hashing:
 * A single hash function has limited discrimination. We combine k hash functions into a "super
 * hash" function g(p) = (h₁(p), h₂(p), ..., hₖ(p)). The collision probability becomes
 * P(g(p) = g(q)) = P(h(p) = h(q))^k, which for nearby points is (1 - d/w)^k (still high if k
 * is moderate and d is small), and for distant points it's very low (exponential decay).
 * 
 * Amplification via Multiple Tables:
 * To increase recall and avoid missing nearby points due to bad hash luck, we create L independent
 * hash tables with L different super hash functions. The probability that p and q collide in at
 * least one table is 1 - (1 - (1 - d/w)^k)^L. Parameters k and L control the accuracy-speed
 * tradeoff: larger k gives fewer false positives (better precision, but may miss neighbors),
 * while larger L gives higher recall (less likely to miss neighbors, but more candidates to check).
 * 
 * Implementation Strategy:
 * The index uses L hash tables (std::unordered_map), where each table maps hash keys to lists
 * of point IDs. We maintain L independent super hash functions, each combining k base hashes.
 * The query process computes L super hash keys for the query point, looks up each key in its
 * corresponding table, collects all candidate point IDs (union across tables), computes exact
 * distances to candidates, and returns the top-k nearest neighbors.
 * 
 * Key Optimization - Composite Hash Key:
 * Instead of storing k individual hash values per table, we combine them into a single integer
 * key using a random linear combination: key = (r₁·h₁ + r₂·h₂ + ... + rₖ·hₖ) mod M, where
 * rᵢ are random integers and M is a large prime. This reduces storage overhead and enables
 * fast unordered_map lookups.
 * 
 * Parameter Tuning Guidelines:
 * Typical values for k are 4-10 (higher k gives better precision, lower recall). Typical values
 * for L are 5-20 (higher L gives higher recall, more candidates). Bucket width w should be set
 * to around 4 for normalized float data, or 10-20 for uint8_t data. Example tradeoffs: k=4, L=5
 * gives fast search with moderate accuracy (70-80% recall), while k=8, L=10 gives slower search
 * with high accuracy (90-95% recall).
 * 
 * Complexity Analysis:
 * Let n = dataset size, d = dimensionality, c = candidates per query. Index construction takes
 * O(L · n · k · d) time to hash all points into all tables, and O(L · n) space to store point
 * IDs in tables. Query time is O(L · k · d + c · d) for computing L hashes plus refining
 * candidates, with O(c) space for the candidate set. For LSH to be efficient, c should be
 * much smaller than n, typically c = O(n^p) where p < 1.
 * 
 * @authors Τσεκρέκος Έγκορ-Ανδριανός
 */

#pragma once

#include <arg_parse.hpp>
#include <cstdint>
#include <data_types.hpp>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

// Compute a single LSH base hash value using the formula: h(p) = floor((p·v + t) / w).
// Geometric interpretation: p·v projects point p onto the random direction v, (p·v + t)
// shifts the projection by offset t, division by w discretizes the real line into buckets
// of width w, and floor assigns an integer bucket ID. Points that project close together
// on direction v get the same hash value.
// Time: O(dim)
template <typename T>
int lsh_base_hash(const std::vector<float>& random_line, const T* data_point, size_t dim, float t,
                  float w);

// SuperHash: Composite hash function combining k independent base hashes
//
// A single hash function h(p) has limited ability to distinguish similar from dissimilar points.
// By combining k independent hash functions, we create a more discriminative hash function
// g(p) = (h₁(p), ..., hₖ(p)). Instead of storing a k-dimensional tuple, we compute a single
// integer key: key(p) = (r₁·h₁(p) + r₂·h₂(p) + ... + rₖ·hₖ(p)) mod M, where hᵢ(p) are base
// hash functions, rᵢ are random integers for universal hashing, and M is a large prime
// (4294967291 = 2^32 - 5). This random linear combination preserves the collision properties
// of the k-tuple hash while enabling efficient integer-based hash table operations. Each base
// hash hᵢ uses independent random vector vᵢ ~ N(0,1)^d and independent random offset tᵢ ~ U(0, w).
template <typename T> class SuperHash {
  public:
    // Construct a super hash with k base hash functions. Initializes k independent random
    // projections and offsets, plus k random integers for the composite key calculation.
    SuperHash(int k, size_t vector_dimension, float w, std::mt19937& gen);

    // Compute the composite hash key for a data point. Applies all k base hash functions and
    // combines them using random linear combination modulo M. Uses long long arithmetic to
    // prevent integer overflow during the linear combination. The double modulo ((x % M + M) % M)
    // correctly handles negative values in C++.
    // Time: O(k · dim)
    long long get_hash_key(const T* data_point, size_t dim) const;

  private:
    std::vector<std::vector<float>> random_lines; // k random projection vectors (vᵢ)
    std::vector<float> offsets;                   // k random offsets (tᵢ)
    float w;                                      // Bucket width parameter
    std::vector<int> r_values;                    // k random integers for linear combination
    const long long M = 4294967291;               // Large prime (2^32 - 5) for modular arithmetic
};

// LshHashTable: Hash table mapping composite keys to lists of point IDs. Each bucket (hash key)
// contains a vector of dataset point IDs that hash to that key. During queries, we look up the
// query's hash key and examine all points in that bucket as candidates.
using LshHashTable = std::unordered_map<long long, std::vector<int>>;

// LSH_Index: Main LSH index structure with L hash tables
//
// Architecture: The index maintains L independent hash table systems with L super hash functions
// (each combining k base hashes) and L hash tables (each mapping keys to point ID lists).
//
// Amplification Strategy: Multiple tables increase recall. If a nearby point is missed in one
// table due to hash randomness, it's likely to be found in another table.
//
// Query Strategy: Hash query point using all L super hash functions, look up each hash key in
// its corresponding table, collect all candidate IDs (union across tables, removing duplicates),
// refine candidates by computing exact distances, and return top-k nearest neighbors.
//
// Memory Footprint: Storage is O(L · n) where n = dataset size. Each point appears once per table
// (L times total). Hash tables add overhead (approximately 2-3x for unordered_map). Typical sizes:
// for n=1M points with L=10, approximately 40-80 MB for just point IDs. For n=10M with L=20,
// approximately 800 MB to 1.6 GB.
template <typename T> class LSH_Index {
  public:
    // Construct an LSH index with specified parameters. Creates L independent super hash functions
    // but does not build the tables. Call build() after construction to populate tables with
    // dataset points.
    LSH_Index(int L_input, int k_input, float w_input, int dimensions, std::uint32_t seed);
    
    // Build the index by hashing all dataset points. Populates all L hash tables by iterating
    // through each table and each point, computing the hash key, and inserting the point ID into
    // the appropriate bucket. After build(), the index is ready for queries.
    // Time: O(L · n · k · d) where d = dimensions, Space: O(L · n)
    void build(const Matrix<T>& dataset);

    // Query the index to find candidate nearest neighbors. Returns a list of candidate point IDs
    // (not distances). These candidates need to be refined by computing exact distances. For each
    // table, computes the query's hash key and retrieves all points in that bucket, then returns
    // the union of candidates with duplicates removed.
    // Time: O(L · k · d + c) where c = total candidates
    std::vector<int> get_candidates(const T* query_point, size_t dim) const;
    
    // Find k nearest neighbors (high-level interface). Combines candidate generation and refinement:
    // gets candidates via get_candidates(), computes exact distances to candidates, sorts, and
    // returns top-k.
    std::vector<ANearNeighbor> find_k_nearest(const T* query_point, const Matrix<T>& dataset,
                                              int N) const;
    
    // Find all neighbors within radius R. Range search variant that returns all candidates within
    // distance R.
    std::vector<ImageId> find_in_range(const T* query_point, const Matrix<T>& dataset,
                                       double R) const;

  private:
    int L; // Number of hash tables (amplification factor)
    std::vector<SuperHash<T>> hash_functions; // L independent super hash functions
    std::vector<LshHashTable> tables;         // L hash tables mapping keys to point ID lists
};

// High-level LSH query function. Complete pipeline: index construction, querying, and metric
// calculation. Builds LSH index from input_images, queries index for each query vector, and
// populates output structure with results. This function is called by main() and handles all
// LSH-specific logic.
template <typename T>
std::unique_ptr<Output> lsh_querying(const Matrix<T>& input_images, const Matrix<T>& query_images,
                                     LshArguments& args);
