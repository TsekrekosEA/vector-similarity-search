/**
 * @file hypercube.hpp
 * @brief Hypercube LSH: A variant of LSH using binary hash codes and Hamming distance
 * 
 * Algorithm Overview:
 * Hypercube LSH is an elegant variant of standard LSH that maps high-dimensional vectors to
 * vertices of a k-dimensional binary hypercube {0,1}^k. Each vertex represents a bucket, and
 * similar vectors are mapped to the same or nearby vertices.
 * 
 * Theoretical Foundation:
 * For each of k dimensions, we define a binary hash function: h_i(p) = sign(p · v_i) where
 * the result is 0 if p·v_i < 0 and 1 if p·v_i >= 0, with v_i being a random unit vector
 * from N(0,1)^d. Each hash function determines which side of a random hyperplane the point
 * lies on. The k binary hashes form a k-bit binary code, mapping each point to a vertex of
 * the k-dimensional hypercube: h(p) = (h_1(p), h_2(p), ..., h_k(p)) in {0,1}^k. For example,
 * with k=3, point p_1 might hash to (1,0,1) which is vertex 101 in binary or 5 in decimal,
 * while point p_2 hashes to (1,0,0) which is vertex 100 or 4 in decimal.
 * 
 * Similarity Preservation:
 * Points that are close in the original space tend to have similar hash codes with low Hamming
 * distance. If two points are on the same side of most hyperplanes, their hash codes differ
 * in few bits.
 * 
 * Differences from Standard LSH:
 * Standard LSH uses L hash tables with real-valued hash keys, exact key matching, O(L·n) memory,
 * and query strategy based on union of L buckets. Hypercube LSH uses 1 table with k-bit binary
 * keys, multi-probe search (Hamming neighbors), O(n) memory, and BFS on the hypercube graph.
 * 
 * Key Advantages:
 * Memory efficiency comes from using a single table instead of L tables (providing L times
 * reduction). Fast hash computation uses binary operations which are faster than float arithmetic.
 * The elegant hypercube geometry enables principled multi-probe search. The method scales well
 * to very high-dimensional data (k up to 64 with uint64_t).
 * 
 * Multi-Probe Search Strategy:
 * Instead of using multiple tables like standard LSH, Hypercube LSH uses multi-probe search by
 * examining multiple buckets per query through exploring nearby vertices in the hypercube.
 * Starting from the query's hash vertex, we perform BFS visiting vertices at increasing Hamming
 * distances: distance 0 is the exact hash (same bucket), distance 1 flips 1 bit (k neighbors),
 * distance 2 flips 2 bits (k·(k-1)/2 neighbors), and distance h flips h bits (C(k,h) neighbors).
 * Hamming distance correlates with geometric distance: Hamming dist = 0 has high probability of
 * being near query, dist = 1 has moderate probability (differ on 1 hyperplane), dist = 2 has
 * lower probability (differ on 2 hyperplanes), and so on.
 * 
 * Implementation Optimizations:
 * We pack the k-bit hash into a single uint64_t integer where bit i stores h_i(p), allowing
 * keys up to k=64 dimensions and enabling fast bitwise operations (XOR for neighbor generation).
 * We use a single hash table (unordered_map<uint64_t, vector<int>>) where keys are uint64_t
 * hash codes (vertices of hypercube) and values are lists of point IDs hashing to that vertex.
 * To find Hamming distance 1 neighbors, we flip each bit: neighbor_key = current_key XOR (1 << i)
 * which is extremely fast (single CPU cycle) compared to re-hashing. BFS stops when we've
 * examined 'probes' vertices or collected M candidates, whichever comes first.
 * 
 * Parameter Tuning:
 * kproj (number of hash functions / hypercube dimension): Typical values 10-16. Higher k gives
 * more buckets (2^k total), more precise but sparser. Lower k gives fewer buckets, denser but
 * less discriminative. probes (number of vertices to examine): Typical values 10-100. Higher
 * probes gives higher recall but slower. Rule of thumb: probes ≈ kproj^2 gives good recall.
 * M (maximum candidates to collect): Typical values 100-1000. Higher M gives more candidates
 * to refine, higher recall but slower.
 * 
 * Complexity Analysis:
 * Let n = dataset size, d = dimensions, k = kproj, c = candidates. Index construction takes
 * O(n · k · d) time to hash all points and O(n) space for single table. Query time is
 * O(probes · k · d + c · d) for multi-probe search (BFS + hash computations) plus refining
 * candidates with exact distances, using O(c) space for the candidate set. For efficiency,
 * probes should be much less than 2^k and c should be much less than n.
 * 
 * Comparison to Standard LSH:
 * Hypercube is often preferred when memory is constrained (single table vs L tables), dataset
 * is not too large (< 10M points), or query time is more important than index size. Standard
 * LSH is preferred when dataset is very large (> 10M points), higher recall is required, or
 * construction time is critical (no BFS during queries).
 * 
 * @authors Τσεκρέκος Έγκορ-Ανδριανός
 */

#pragma once

#include <cstdint>
#include <data_types.hpp>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

// Forward declaration of argument struct
struct HypercubeArguments;

// HypercubeHashTable: Hash table mapping binary hash codes to point ID lists. Keys are uint64_t
// (k-bit binary codes representing hypercube vertices) and values are vector<int> (IDs of points
// hashing to that vertex). The hash table has at most 2^k buckets (one per hypercube vertex),
// but in practice many buckets are empty for large k.
using HypercubeHashTable = std::unordered_map<uint64_t, std::vector<int>>;

// HypercubeIndex: Hypercube LSH index using binary hash codes and multi-probe search
//
// Architecture: The index uses k random projection vectors (each defines a binary hash) and a
// single hash table mapping uint64_t keys to point ID lists, with BFS-based multi-probe query
// strategy.
//
// Memory Efficiency: Uses approximately 5-10x less memory than standard LSH with L tables.
// Standard LSH uses O(L · n) where L ≈ 10-20, while Hypercube uses O(n). For n=1M points,
// standard LSH uses approximately 40-160 MB while Hypercube uses approximately 8-16 MB. For
// n=10M points, the difference is even more dramatic.
template <typename T> class HypercubeIndex {
  public:
    // Construct a Hypercube index. Initializes k random projection vectors but does not build
    // the table. Call build() to populate the table with dataset points. Constraint: kproj <= 64
    // due to uint64_t key representation. Throws std::invalid_argument if kproj > 64.
    HypercubeIndex(int kproj_input, int dimensions, std::uint32_t seed);

    // Build the index by hashing all dataset points. Maps each point to its hypercube vertex:
    // compute k-bit hash code for each point, then insert point ID into table[hash_code]. After
    // build(), the index is ready for queries.
    // Time: O(n · k · d), Space: O(n)
    void build(const Matrix<T>& dataset);

    // Query the index using multi-probe BFS. Performs breadth-first search on the hypercube graph,
    // starting from the query's hash vertex and exploring nearby vertices. Algorithm: compute
    // query_key = hash(query), initialize BFS queue with query_key, then while (probes_checked < probes)
    // and (candidates < M): pop vertex from queue, add all points in that bucket to candidates,
    // generate Hamming distance 1 neighbors by flipping each bit, add unvisited neighbors to queue.
    // Returns unique candidate IDs. Stopping conditions: examined 'probes' vertices OR collected
    // M candidates, whichever comes first.
    // Time: O(probes · k · d + c) where c = candidates
    std::vector<int> get_candidates(const T* query_point, size_t dim, int probes, int M) const;
    
    // Find k nearest neighbors (high-level interface). Combines multi-probe search with candidate
    // refinement: get candidates via BFS, compute exact distances, return top-N.
    std::vector<ANearNeighbor> find_k_nearest(const T* query_point, const Matrix<T>& dataset, int N,
                                              int probes, int M) const;
    
    // Find all neighbors within radius R. Range search using multi-probe strategy.
    std::vector<ImageId> find_in_range(const T* query_point, const Matrix<T>& dataset, double R,
                                       int probes, int M) const;

  private:
    // Compute k-bit hash code for a data point. Algorithm: key = 0 (64-bit integer), then for
    // i = 0 to k-1: if (data_point · random_lines[i]) >= 0 then set bit i of key to 1, else
    // bit i remains 0. Returns key. Geometric interpretation: each bit encodes which side of a
    // hyperplane the point is on. Bit i = 0 means negative side of hyperplane v_i, bit i = 1
    // means positive side. We use bitwise OR to set bits: key |= (1ULL << i) which is much
    // faster than array operations or string concatenation.
    // Time: O(k · d)
    uint64_t get_hash_key(const T* data_point, size_t dim) const;

    int kproj;  // Number of hash functions (hypercube dimension k)
    std::vector<std::vector<float>> random_lines; // k random projection vectors
    HypercubeHashTable table;  // Single hash table mapping hash codes to point ID lists
};

// High-level Hypercube query function. Complete pipeline: index construction, querying, and
// result formatting. Builds Hypercube index from input_images, queries index for each query
// vector (with multi-probe search), and populates output structure with results. This function
// is called by main() and handles all Hypercube-specific logic.
template <typename T>
std::unique_ptr<Output> hypercube_querying(const Matrix<T>& input_images,
                                           const Matrix<T>& query_images, HypercubeArguments& args);