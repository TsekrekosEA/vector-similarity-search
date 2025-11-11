#pragma once
#include <memory> // For std::unique_ptr
#include <fstream> // For std::ofstream
#include "data_types.hpp"

std::unique_ptr<Matrix<unsigned char>> load_mnist_data(const std::string& filepath);
std::unique_ptr<Matrix<unsigned char>> load_mnist_data_truncated(const std::string& filepath, uint32_t first_n);
std::unique_ptr<Matrix<float>> load_sift_data(const std::string& filepath);
std::unique_ptr<Matrix<float>> load_sift_data_truncated(const std::string& filepath, uint32_t first_n);

std::unique_ptr<std::ofstream> initialize_output_file(const std::string& filepath);
void write_output(std::ofstream& out, const Output& output_data);