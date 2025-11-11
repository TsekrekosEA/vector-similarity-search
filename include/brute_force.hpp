#pragma once

#include <memory>
#include <cstdint>
#include <data_types.hpp>
#include <arg_parse.hpp>

// Brute force nearest neighbor search algorithm
// Computes exact distances to all points and returns the k nearest neighbors
// Template parameter T can be float (for SIFT) or uint8_t (for MNIST)
// If existing_output is provided, ONLY fills distance_true fields (doesn't touch id or distance_approximate)
template <typename T>
std::unique_ptr<Output> brute_force_querying(
    const Matrix<T>& input_images,
    const Matrix<T>& query_images,
    BruteforceArguments& args,
    std::unique_ptr<Output> existing_output = nullptr
);
