#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include "file_io.hpp"

// standard function to swap endianess, assumes Little-Endian System Execution
inline uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0x00FF00FF);
    return (val << 16) | (val >> 16);
}

std::string filepath = "data/query.dat";  // note: it's overriden in the functions

static std::unique_ptr<Matrix<unsigned char>> load_mnist_data_backend(const std::string& filepath, uint32_t first_n_images, bool should_truncate){
    // open file in binary mode
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open MNIST file: " << filepath << std::endl;
        return nullptr;
    }
    // define containers for characteristic data in dataset
    uint32_t magic_number, num_images, num_rows, num_cols;

    // file reads as char, needs casting to uint32_t for proper format
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&num_images), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&num_rows), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&num_cols), sizeof(uint32_t));

    // assuming program is run on a little-endian machine which most modern computers are, 
    // swap Big-endian format of mnist file type to little-endian
    magic_number = swap_endian(magic_number);
    num_images = swap_endian(num_images);
    num_rows = swap_endian(num_rows);
    num_cols = swap_endian(num_cols);

    // perhaps truncate the images
    if (should_truncate) {
        if (first_n_images > num_images) first_n_images = num_images;
    }
    else {
        first_n_images = num_images;
    }

    // Check to make sure file passed really is mnist
    if (magic_number != 2051) {
        std::cerr << "Error: Invalid MNIST image file. Incorrect magic number: " << magic_number << std::endl;
        return nullptr;
    }

    // Define the dimension our data matrix needs to be
    size_t dimensions = num_rows * num_cols;

    // allocate a single, flat block of memory for the entire image dataset
    auto dataset_matrix = std::make_unique<Matrix<unsigned char>>(first_n_images, dimensions);

    // using the metadata found in beginning of mnist file, calculate total required amount of bytes to read
    size_t total_bytes = first_n_images * dimensions * sizeof(unsigned char);
    // returning a raw pointer to the beginning of our Matrix's vector, write all the data bytes
    file.read(reinterpret_cast<char*>(dataset_matrix->get_raw_data().data()), total_bytes);

    if (!file || file.gcount() != static_cast<std::streamsize>(total_bytes)) {
        std::cerr << "Error: Failed to read the full pixel data from MNIST file." << std::endl;
        return nullptr;
    }

    std::cout << "Successfully loaded " << dataset_matrix->get_rows() << " MNIST images (" 
              << dataset_matrix->get_cols() << " dimensions)." << std::endl;

    // return the unique pointer to our raw data, preventing needless copying
    // the file is closed automatically by ifstream's destructor
    return dataset_matrix;
}

std::unique_ptr<Matrix<unsigned char>> load_mnist_data(const std::string& filepath){
    return load_mnist_data_backend(filepath, 0, false);
}
std::unique_ptr<Matrix<unsigned char>> load_mnist_data_truncated(const std::string& filepath, uint32_t first_n){
    return load_mnist_data_backend(filepath, first_n, true);
}

static std::unique_ptr<Matrix<float>> load_sift_data_backend(const std::string& filepath, uint32_t first_n_images, bool should_truncate){
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open SIFT file: " << filepath << std::endl;
        return nullptr;
    }

    // the .fvecs file does not have a header with total number of vectors,
    // so we read vector by vector because each vector is prefixed by it's dimension
    // we read into a temporaray vector, because we do not know the size of the file, until we finsh reading
    std::vector<std::vector<float>> temp_data;
    int dimension = -1;

    while (file) {
        int current_dimension;
        // Read the 4-byte integer dimension for the next vector
        file.read(reinterpret_cast<char*>(&current_dimension), sizeof(int32_t));

        if (file.eof()) {
            break;
        }

        if (dimension == -1) {
            dimension = current_dimension; // Set the vector dimensions from first vector
        } else if (dimension != current_dimension) {
            std::cerr << "Error: Inconsistent vector dimensions in SIFT file." << std::endl;
            return nullptr;
        }

        std::vector<float> vec(dimension);
        file.read(reinterpret_cast<char*>(vec.data()), dimension * sizeof(float));

        if (!file) { // Check for incomplete read
            std::cerr << "Error: Incomplete vector read from SIFT file." << std::endl;
            return nullptr;
        }
        
        temp_data.push_back(vec);
        if (should_truncate && --first_n_images == 0) break;
    }

    if (temp_data.empty()) {
        std::cerr << "Warning: No data loaded from SIFT file: " << filepath << std::endl;
        return nullptr;
    }

    // Now that all the vector data is loaded into our temp_data struct
    // copy it into our Matrix
    size_t num_vectors = temp_data.size();
    auto dataset_matrix = std::make_unique<Matrix<float>>(num_vectors, dimension);

    for (size_t i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < dimension; ++j) {
            dataset_matrix->at(i, j) = temp_data[i][j];
        }
    }

    std::cout << "Successfully loaded " << dataset_matrix->get_rows() << " SIFT vectors (" 
              << dataset_matrix->get_cols() << " dimensions)." << std::endl;

    return dataset_matrix;
}
std::unique_ptr<Matrix<float>> load_sift_data(const std::string& filepath){
    return load_sift_data_backend(filepath, 0, false);
}
std::unique_ptr<Matrix<float>> load_sift_data_truncated(const std::string& filepath, uint32_t first_n){
    return load_sift_data_backend(filepath, first_n, true);
}




std::unique_ptr<std::ofstream> initialize_output_file(const std::string& filepath) {
    auto out_file = std::make_unique<std::ofstream>(filepath);
    if (!out_file->is_open()) {
        std::cerr << "Error: Could not open output file: " << filepath << std::endl;
    }
    // Set precision for floating-point numbers for consistent output, 6 is standard in C++
    *out_file << std::fixed << std::setprecision(6);
    return out_file;
}

void write_output(std::ofstream& out, const Output& output_data) {
    if (!out.is_open()) {
        std::cerr << "Error: Output file stream is not open." << std::endl;
        return;
    }

    // Write Header
    out << output_data.algorithm << std::endl;
    out << std::endl;

    // write query results
    for (size_t query_id = 0; query_id < output_data.queries.size(); query_id++) {
        auto query_result = output_data.queries[query_id];
        out << "Query: " << query_id << std::endl;

        // Loop through the N nearest neighbors (PDF format: no separate "True Nearest neighbor" line)
        for (size_t i = 0; i < query_result.nearest_neighbors.size(); ++i) {
            const auto& neighbor = query_result.nearest_neighbors[i];
            out << "Nearest neighbor-" << i + 1 << ": " << neighbor.id << std::endl;
            out << "distanceApproximate: " << neighbor.distance_approximate << std::endl;
            out << "distanceTrue: " << neighbor.distance_true << std::endl;
        }

        // Write the (optional) R-near neighbors
        if (!query_result.r_near_neighbors.empty()) {
            out << "R-near neighbors:" << std::endl;
            for (const auto& r_neighbor_id : query_result.r_near_neighbors) {
                out << r_neighbor_id << std::endl;
            }
        }
        
        out << std::endl;
    }

    // final summary block
    out << "Average AF: " << output_data.average_af << std::endl;
    out << "Recall@N: " << output_data.recall_at_n << std::endl;
    out << "QPS: " << output_data.queries_per_second << std::endl;
    out << "tApproximateAverage: " << output_data.t_approximate_average << std::endl;
    out << "tTrueAverage: " << output_data.t_true_average << std::endl;
}
