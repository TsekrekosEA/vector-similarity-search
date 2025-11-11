#pragma once
#include <vector>
#include <cassert>
#include <memory>
#include <string>
#include <cstdint>
#include <cmath>  // sqrt
#include <algorithm>  // sort

template <typename T>
class Matrix {
    private:
        size_t rows;
        size_t cols;
        std::vector<T> data;

    public:
        Matrix(size_t input_rows, size_t input_cols) : rows(input_rows), cols(input_cols), data(rows * cols) {}

        // Access to one element Read/Write permission
        T& at(size_t input_row, size_t input_col) {
            assert(input_row < rows && input_col < cols);
            return data[input_row * cols + input_col];
        }

        // Access to one element Read Only
        const T& at(size_t input_row, size_t input_col) const {
            assert(input_row < rows && input_col < cols);
            return data[input_row * cols + input_col];
        }

        T& unchecked(size_t input_row, size_t input_col) {
            return data[input_row * cols + input_col];
        }
        const T& unchecked(size_t input_row, size_t input_col) const {
            return data[input_row * cols + input_col];
        }

        // Direct pointer access to a row for faster itteration
        const T* get_row(size_t input_row) const {
            assert(input_row < rows);
            return &data[input_row * cols];
        }
        
        T* get_row(size_t input_row) {
            assert(input_row < rows);
            return &data[input_row * cols];
        }

        // For getting matrix size
        size_t get_rows() const { return rows; }
        size_t get_cols() const { return cols; }

        // access to the contigent vector structure
        std::vector<T>& get_raw_data() { return data; }

    };

using ImageId = size_t;

class ANearNeighbor {
    public:
        ImageId id = 0;
        ImageId id_true = 0;
        double distance_approximate = 0.0;
        double distance_true = 0.0;
};

class OutputForOneQuery {
    public:
        std::vector<ANearNeighbor> nearest_neighbors;
        std::vector<ImageId> r_near_neighbors;
};

class Output {
    public:
        std::string algorithm;
        std::vector<OutputForOneQuery> queries;
        double average_af;
        double recall_at_n;  // fraction of correct queries, ie those where the actual nearest neighbor is within the returned N approximate nearest.
        double queries_per_second;
        double t_approximate_average;
        double t_true_average;
};