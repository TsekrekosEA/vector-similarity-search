/**
 * @file data_types.hpp
 * @brief Core data structures for high-performance vector similarity search
 * 
 * This file defines the fundamental data structures used throughout the ANN search engine.
 * The Matrix<T> class provides a cache-friendly, contiguous 2D array container. The result
 * structures (ANearNeighbor, OutputForOneQuery, Output) store query results and performance metrics.
 * 
 * Design Philosophy:
 * The design prioritizes cache locality and memory efficiency for high-dimensional vector operations.
 * All structures use contiguous memory layouts to maximize CPU cache utilization during intensive
 * distance computations. This is critical for performance when working with millions of vectors.
 * 
 * @authors Τσεκρέκος Έγκορ-Ανδριανός
 */

#pragma once
#include <algorithm> // sort
#include <cassert>
#include <cmath> // sqrt
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Matrix: High-performance 2D container with cache-optimized memory layout
//
// Unlike std::vector<std::vector<T>>, this Matrix stores all elements in a single contiguous
// block of memory. This provides several critical advantages for our use case. First, sequential
// access patterns maximize CPU cache hits. When iterating through a row, the data is already
// in L1/L2 cache. Second, hardware prefetchers can anticipate future memory accesses, reducing
// latency during distance calculations. Third, we eliminate pointer chasing overhead compared to
// vector-of-vectors which requires pointer dereferencing per row. Finally, contiguous data enables
// compiler auto-vectorization and manual SIMD optimizations for distance computations.
//
// Memory Layout:
// Data is stored in row-major order: [row0][row1][row2]...
// For a 3x4 matrix, memory layout is: [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23]
// Element (i,j) is located at index: i * cols + j
//
// Template Parameter:
// The element type T is typically float for SIFT or uint8_t for MNIST. Template design allows
// type-safe code reuse across different datasets without runtime polymorphism overhead.
//
// Typical Usage:
//   Matrix<float> vectors(1000000, 128);  // 1M 128-dimensional SIFT vectors
//   Matrix<uint8_t> images(60000, 784);   // 60K 784-dimensional MNIST images
//   const float* vec = vectors.get_row(i);
//   for (int j = 0; j < 128; j++) {
//       process(vec[j]);  // Sequential memory access is cache-friendly
//   }

template <typename T> class Matrix {
  private:
    size_t rows;  // Number of rows (vectors) in the matrix
    size_t cols;  // Number of columns (dimensions per vector)
    std::vector<T> data;  // Contiguous storage: size = rows × cols

  public:
    // Construct a matrix with the specified dimensions. Allocates a single contiguous block
    // of memory sized rows × cols. All elements are value-initialized (0 for numeric types).
    // Time: O(rows × cols), Space: O(rows × cols)
    Matrix(size_t input_rows, size_t input_cols)
        : rows(input_rows), cols(input_cols), data(rows * cols) {
    }

    // Bounds-checked element access (read/write). Provides safe access to individual matrix
    // elements with runtime bounds checking using assertions in debug builds. Slightly slower
    // than unchecked() due to assertion overhead. Use for correctness-critical code or debugging.
    T& at(size_t input_row, size_t input_col) {
        assert(input_row < rows && input_col < cols);
        return data[input_row * cols + input_col];
    }

    // Bounds-checked element access (read-only). Const version of at() for read-only access.
    const T& at(size_t input_row, size_t input_col) const {
        assert(input_row < rows && input_col < cols);
        return data[input_row * cols + input_col];
    }

    // Unchecked element access (read/write). Direct access without bounds checking for maximum
    // performance. Caller MUST ensure indices are valid to avoid undefined behavior. Use for
    // performance-critical inner loops where bounds are guaranteed. No safety net in release builds.
    T& unchecked(size_t input_row, size_t input_col) {
        return data[input_row * cols + input_col];
    }
    
    // Unchecked element access (read-only). Const version of unchecked().
    const T& unchecked(size_t input_row, size_t input_col) const {
        return data[input_row * cols + input_col];
    }

    // Get raw pointer to a row for high-performance iteration. Returns a pointer to the first
    // element of the specified row, enabling efficient sequential access patterns that maximize
    // CPU cache utilization. This is the PREFERRED method for iterating over vector elements
    // during distance calculations, as it allows sequential memory access (optimal cache behavior),
    // compiler auto-vectorization (SIMD optimization), and minimal pointer arithmetic overhead.
    //
    // Example usage in distance calculation:
    //   const float* v1 = matrix.get_row(i);
    //   const float* v2 = matrix.get_row(j);
    //   for (size_t d = 0; d < dim; d++) {
    //       dist += (v1[d] - v2[d]) * (v1[d] - v2[d]);
    //   }
    const T* get_row(size_t input_row) const {
        assert(input_row < rows);
        return &data[input_row * cols];
    }

    // Get mutable pointer to a row (non-const version). Allows modification of row elements via
    // pointer. Use cautiously; prefer const version when read-only access suffices.
    T* get_row(size_t input_row) {
        assert(input_row < rows);
        return &data[input_row * cols];
    }

    // Get the number of rows (vectors) in the matrix
    size_t get_rows() const {
        return rows;
    }
    
    // Get the number of columns (vector dimensionality)
    size_t get_cols() const {
        return cols;
    }

    // Access the underlying contiguous data vector. Exposes the internal std::vector for
    // specialized operations like bulk I/O operations (loading/saving entire datasets), direct
    // memory manipulation, or integration with external libraries. Use only when necessary
    // for performance or I/O operations, as direct manipulation bypasses matrix abstraction.
    std::vector<T>& get_raw_data() {
        return data;
    }
};

// ImageId: Type alias for vector/image identifiers
// Represents the unique ID of a vector in the dataset (typically its index). Using size_t
// allows indexing datasets with billions of vectors.
using ImageId = size_t;

// ANearNeighbor: Represents a single approximate nearest neighbor result
//
// Stores both the approximate search result and the ground truth for validation. This dual
// representation enables calculation of accuracy metrics. The Approximation Factor (AF) is
// computed as distance_approximate / distance_true, measuring how close the approximate
// distance is to the true distance. Recall@N measures whether the true nearest neighbor
// appears in the approximate results.
//
// Field Descriptions:
// id - The ID returned by the approximate algorithm
// id_true - The actual nearest neighbor ID (from brute-force search)
// distance_approximate - Distance computed by the approximate algorithm
// distance_true - Actual distance to the true nearest neighbor
//
// Note: The approximate algorithm initially populates only 'id' and 'distance_approximate'.
// The brute-force phase later fills 'id_true' and 'distance_true' for metric calculation.
class ANearNeighbor {
  public:
    ImageId id = 0;                    // ID returned by approximate algorithm
    ImageId id_true = 0;               // True nearest neighbor ID (ground truth)
    double distance_approximate = 0.0; // Distance computed by ANN algorithm
    double distance_true = 0.0;        // Actual distance to true nearest neighbor
};

// OutputForOneQuery: Contains results for a single query vector
//
// Stores both k-NN results (top-N nearest neighbors) and range search results (all neighbors
// within radius R). Separating k-NN and range results allows efficient handling of different
// query types without wasting memory.
//
// nearest_neighbors - Top-N approximate nearest neighbors (ordered by distance)
// r_near_neighbors - All neighbors within radius R (unordered)
class OutputForOneQuery {
  public:
    std::vector<ANearNeighbor> nearest_neighbors;
    std::vector<ImageId> r_near_neighbors;
};

// Output: Complete output structure for a batch of queries
//
// Contains results for all queries plus aggregate performance metrics. This structure is used
// for writing results to output files, computing benchmark statistics, and comparing algorithms.
//
// Performance Metrics Explained:
//
// average_af (Average Approximation Factor):
// Definition: mean(distance_approximate / distance_true) across all queries. Measures how close
// approximate distances are to true distances. Ideal value is 1.0 (perfect accuracy). Typical
// range is 1.0 to 1.5, with lower being better.
//
// recall_at_n (Recall@N):
// Definition: (number of queries with true NN in results) / (total queries). Measures the
// probability of finding the actual nearest neighbor. Range is 0.0 to 1.0, with higher being
// better. For example, 0.95 means 95% of queries return the true nearest neighbor.
//
// queries_per_second (QPS):
// Definition: total_queries / t_approximate_total. Measures query throughput. Higher values
// indicate faster search capability.
//
// t_approximate_average:
// Average time per query for approximate search, in seconds, excluding index construction.
//
// t_true_average:
// Average time per query for brute-force search. Used to calculate speedup ratio as
// t_true_average / t_approximate_average.
//
// Design Rationale:
// Storing per-query results alongside aggregate metrics enables detailed analysis of query-specific
// performance, detection of performance outliers, verification of metric calculations, and export
// of complete results for external analysis.
class Output {
  public:
    std::string algorithm;                    // Algorithm name ("LSH", "Hypercube", "IVF-Flat", etc.)
    std::vector<OutputForOneQuery> queries;   // Per-query results
    
    // Aggregate accuracy metrics
    double average_af;      // Average Approximation Factor (mean distance ratio)
    double recall_at_n;     // Fraction of queries where true NN is in top-N
    
    // Performance metrics
    double queries_per_second;     // Query throughput (QPS)
    double t_approximate_average;  // Average query time for approximate search (seconds)
    double t_true_average;         // Average query time for brute-force search (seconds)
};;

// ============================================================================
// RESULT STRUCTURES
// ============================================================================

/**
 * @typedef ImageId
 * @brief Type alias for vector/image identifiers
 * 
 * Represents the unique ID of a vector in the dataset (typically its index).
 * Using size_t allows indexing datasets with billions of vectors.
 */

/**
 * @typedef ImageId
 * @brief Type alias for vector/image identifiers
 * 
 * Represents the unique ID of a vector in the dataset (typically its index).
 * Using size_t allows indexing datasets with billions of vectors.
 */
using ImageId = size_t;

/**
 * @class ANearNeighbor
 * @brief Represents a single approximate nearest neighbor result
 * 
 * Stores both the approximate search result and the ground truth for validation.
 * This dual representation enables calculation of accuracy metrics like:
 * - Approximation Factor (AF): distance_approximate / distance_true
 * - Recall@N: Whether the true nearest neighbor appears in approximate results
 * 
 * FIELD DESCRIPTIONS:
 * ===================
 * id: The ID returned by the approximate algorithm
 * id_true: The actual nearest neighbor ID (from brute-force search)
 * distance_approximate: Distance computed by the approximate algorithm
 * distance_true: Actual distance to the true nearest neighbor
 * 
 * NOTE: The approximate algorithm initially populates only 'id' and 'distance_approximate'.
 * The brute-force phase later fills 'id_true' and 'distance_true' for metric calculation.
 */
class ANearNeighbor {
  public:
    ImageId id = 0;                    ///< ID returned by approximate algorithm
    ImageId id_true = 0;               ///< True nearest neighbor ID (ground truth)
    double distance_approximate = 0.0; ///< Distance computed by ANN algorithm
    double distance_true = 0.0;        ///< Actual distance to true nearest neighbor
};

/**
 * @class OutputForOneQuery
 * @brief Contains results for a single query vector
 * 
 * Stores both k-NN results (top-N nearest neighbors) and range search results
 * (all neighbors within radius R).
 * 
 * FIELD DESCRIPTIONS:
 * ===================
 * nearest_neighbors: Top-N approximate nearest neighbors (ordered by distance)
 * r_near_neighbors: All neighbors within radius R (unordered)
 * 
 * Design Note: Separating k-NN and range results allows efficient handling of
 * different query types without wasting memory.
 */
class OutputForOneQuery {
  public:
    std::vector<ANearNeighbor> nearest_neighbors; ///< Top-N nearest neighbors
    std::vector<ImageId> r_near_neighbors;        ///< All neighbors within radius R
};

/**
 * @class Output
 * @brief Complete output structure for a batch of queries
 * 
 * Contains results for all queries plus aggregate performance metrics.
 * This structure is used for:
 * 1. Writing results to output files
 * 2. Computing benchmark statistics
 * 3. Comparing algorithms scientifically
 * 
 * PERFORMANCE METRICS:
 * ====================
 * average_af: Average Approximation Factor across all queries
 *   - Definition: mean(distance_approximate / distance_true)
 *   - Interpretation: How close approximate distances are to true distances
 *   - Ideal value: 1.0 (perfect accuracy)
 *   - Typical range: 1.0 - 1.5 (lower is better)
 * 
 * recall_at_n: Fraction of queries where true NN is found
 *   - Definition: (# queries with true NN in results) / (total queries)
 *   - Interpretation: Probability of finding the actual nearest neighbor
 *   - Range: [0.0, 1.0] (higher is better)
 *   - Example: 0.95 = 95% of queries return the true nearest neighbor
 * 
 * queries_per_second: Query throughput
 *   - Definition: total_queries / t_approximate_total
 *   - Interpretation: Sustainable query rate
 *   - Higher is better (indicates faster search)
 * 
 * t_approximate_average: Average time per query (approximate search)
 *   - In seconds, excluding index construction
 * 
 * t_true_average: Average time per query (brute-force search)
 *   - Used to calculate speedup: t_true_average / t_approximate_average
 * 
 * DESIGN RATIONALE:
 * =================
 * Storing per-query results (queries vector) alongside aggregate metrics enables:
 * - Detailed analysis of query-specific performance
 * - Detection of performance outliers
 * - Verification of metric calculations
 * - Export of complete results for external analysis
 */
class Output {
  public:
    std::string algorithm;                    ///< Algorithm name ("LSH", "Hypercube", etc.)
    std::vector<OutputForOneQuery> queries;   ///< Per-query results
    
    // Aggregate accuracy metrics
    double average_af;      ///< Average Approximation Factor (mean distance ratio)
    double recall_at_n;     ///< Fraction of queries where true NN is in top-N
    
    // Performance metrics
    double queries_per_second;     ///< Query throughput (QPS)
    double t_approximate_average;  ///< Average query time - approximate search (seconds)
    double t_true_average;         ///< Average query time - brute-force search (seconds)
};