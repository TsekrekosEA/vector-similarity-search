/**
 * @file file_io.hpp
 * @brief Binary file I/O for SIFT and MNIST datasets
 * 
 * File Format Support:
 * This module handles binary file formats for two popular ANN benchmarking datasets: SIFT1M
 * (1 million 128-dimensional float vectors) using .fvecs/.bvecs format, and MNIST (60,000
 * 784-dimensional uint8 vectors) using IDX format. Both formats store vectors in row-major
 * layout with metadata headers. SIFT uses little-endian 4-byte integers for dimensions and
 * 4-byte floats for elements. MNIST uses big-endian 4-byte integers with magic numbers for
 * type validation.
 * 
 * SIFT .fvecs Format:
 * The .fvecs format is a simple binary format storing float vectors: for each vector, first
 * 4 bytes are dimension d as int32_t (should be 128), then d * 4 bytes are the float elements.
 * This repeats for n vectors. Total file size is n * (4 + d * 4) bytes. No global header exists,
 * so the loader must infer n by dividing file size by bytes per vector. The .bvecs format is
 * identical but uses uint8_t elements (d bytes instead of d * 4 bytes per vector).
 * 
 * MNIST IDX Format:
 * The IDX format has a global header followed by data: first 4 bytes are magic number (0x00000803
 * for unsigned byte images), next 4 bytes are number of images n, next 4 bytes are number of rows
 * (28 for MNIST), next 4 bytes are number of columns (28 for MNIST). Then follows n * rows * cols
 * bytes of pixel data. All integers are big-endian (network byte order). We flatten 28×28 images
 * into 784-D vectors for ANN search.
 * 
 * Memory Efficiency:
 * All load functions return std::unique_ptr<Matrix<T>> to manage large allocations on the heap.
 * SIFT datasets can be 512MB (1M vectors × 128 floats × 4 bytes), MNIST datasets are 47MB (60K
 * vectors × 784 bytes). Heap allocation avoids stack overflow and enables move semantics for
 * efficient transfer. The truncated variants allow loading only the first n vectors, useful for
 * testing with smaller datasets to save memory and reduce build time.
 * 
 * Output Format:
 * The write_output function writes results in human-readable text format: one line per query with
 * nearest neighbor IDs and distances, followed by aggregate metrics (recall, average approximation
 * factor, queries per second, speedup). This format is parsable for further analysis or visualization.
 * 
 * @authors Τσεκρέκος Έγκορ-Ανδριανός
 */

#pragma once
#include "data_types.hpp"
#include <fstream> // For std::ofstream
#include <memory>  // For std::unique_ptr

// Load MNIST dataset from IDX binary format. Reads header with magic number validation (expects
// 0x00000803 for unsigned byte images), dimensions (n × 28 × 28), then pixel data. Flattens 28×28
// images into 784-D vectors. Returns n × 784 matrix of uint8_t. File format uses big-endian
// integers so byte swapping may be needed on little-endian systems. Throws on I/O error or invalid
// magic number.
std::unique_ptr<Matrix<unsigned char>> load_mnist_data(const std::string& filepath);

// Load first n vectors from MNIST dataset. Useful for testing with smaller datasets to reduce
// memory usage (60K vectors = 47MB, truncating to 10K = 7.8MB). Reads full header but stops after
// first_n images. Throws if first_n exceeds dataset size.
std::unique_ptr<Matrix<unsigned char>> load_mnist_data_truncated(const std::string& filepath,
                                                                 uint32_t first_n);

// Load SIFT dataset from .fvecs binary format. Each vector is stored as 4-byte dimension d
// (should be 128) followed by d floats (4 bytes each). Infers number of vectors n from file size
// divided by bytes per vector (4 + 128*4 = 516 bytes). Returns n × 128 matrix of floats. Throws
// on I/O error or size mismatch (file size not multiple of 516).
std::unique_ptr<Matrix<float>> load_sift_data(const std::string& filepath);

// Load first n vectors from SIFT dataset. Useful for testing (1M vectors = 512MB, truncating to
// 100K = 51.2MB). Reads vectors sequentially and stops after first_n. No dimension validation on
// truncated read. Throws if first_n exceeds dataset size.
std::unique_ptr<Matrix<float>> load_sift_data_truncated(const std::string& filepath,
                                                        uint32_t first_n);

// Open output file for writing results. Returns unique_ptr<ofstream> for automatic resource
// management. Throws on file creation failure (e.g., invalid path, permissions). The caller
// passes the stream to write_output.
std::unique_ptr<std::ofstream> initialize_output_file(const std::string& filepath);

// Write benchmark results to output file in text format. Writes per-query results: nearest
// neighbor IDs, approximate distances, true distances (from brute-force), and radius search
// results if applicable. Then writes aggregate metrics: algorithm name, recall@N (fraction of
// queries where true NN is in top-N approximate), average approximation factor (mean of
// distance_approximate / distance_true), queries per second, speedup vs brute-force. Format is
// human-readable for inspection and parsable for automated analysis.
void write_output(std::ofstream& out, const Output& output_data);