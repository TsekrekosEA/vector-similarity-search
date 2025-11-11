#include <stdio.h>
#include <string.h>
#include <chrono>

#include <arg_parse.hpp>
#include <file_io.hpp>
#include <brute_force.hpp>
#include <lsh.hpp>
#include <hypercube.hpp>
#include <ivfflat.hpp>
#include <ivfpq.hpp>
#include <utils.hpp>

int main(int argc, char **argv) {

    char *algorithm_argument = get_algorithm_argument(argc, argv);  // can call exit(-1)
    if (strcmp(algorithm_argument, "-lsh") == 0) {
        LshArguments a{};
        parse_lsh_arguments(&a, argc, argv);
        
        if (a.common.image_type == IMG_SIFT) {
            std::unique_ptr<Matrix<float>> input_images = load_sift_data(a.common.input_file);
            std::unique_ptr<Matrix<float>> query_images = load_sift_data(a.common.query_file);
            
            // Run LSH FIRST to get approximate neighbors (fills id and distance_approximate)
            auto t_approx_start = std::chrono::high_resolution_clock::now();
            std::unique_ptr<Output> output_data = lsh_querying<float>(*input_images, *query_images, a);
            auto t_approx_end = std::chrono::high_resolution_clock::now();
            
            // Then run brute force to fill distance_true fields
            BruteforceArguments brute_args{};
            brute_args.common = a.common;
            auto t_true_start = std::chrono::high_resolution_clock::now();
            output_data = brute_force_querying<float>(*input_images, *query_images, brute_args, std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();
            
            // Calculate timing metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;
            
            // Calculate performance metrics (Average AF, Recall@N, QPS, timing)
            calculate_metrics(*output_data, t_approx.count(), t_true.count());
        
            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        }
        else if (a.common.image_type == IMG_MNIST) {
            std::unique_ptr<Matrix<uint8_t>> input_images = load_mnist_data(a.common.input_file);
            std::unique_ptr<Matrix<uint8_t>> query_images = load_mnist_data(a.common.query_file);
            
            // Run LSH FIRST to get approximate neighbors (fills id and distance_approximate)
            auto t_approx_start = std::chrono::high_resolution_clock::now();
            std::unique_ptr<Output> output_data = lsh_querying<uint8_t>(*input_images, *query_images, a);
            auto t_approx_end = std::chrono::high_resolution_clock::now();
            
            // Then run brute force to fill distance_true fields
            BruteforceArguments brute_args{};
            brute_args.common = a.common;
            auto t_true_start = std::chrono::high_resolution_clock::now();
            output_data = brute_force_querying<uint8_t>(*input_images, *query_images, brute_args, std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();
            
            // Calculate timing metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;
            
            // Calculate performance metrics (Average AF, Recall@N, QPS, timing)
            calculate_metrics(*output_data, t_approx.count(), t_true.count());

            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        }
        else {
            fprintf(stderr, "%s:%d: unsupported file format provided.\n", __FILE__, __LINE__);
        }
    }
    else if (strcmp(algorithm_argument, "-hypercube") == 0) {
        HypercubeArguments a{};
        parse_hypercube_arguments(&a, argc, argv);
        
        if (a.common.image_type == IMG_SIFT) {
            std::unique_ptr<Matrix<float>> input_images = load_sift_data(a.common.input_file);
            std::unique_ptr<Matrix<float>> query_images = load_sift_data(a.common.query_file);
            
            // Run Hypercube FIRST to get approximate neighbors (fills id and distance_approximate)
            auto t_approx_start = std::chrono::high_resolution_clock::now();
            std::unique_ptr<Output> output_data = hypercube_querying<float>(*input_images, *query_images, a);
            auto t_approx_end = std::chrono::high_resolution_clock::now();
            
            // Then run brute force to fill distance_true fields
            BruteforceArguments brute_args{};
            brute_args.common = a.common;
            auto t_true_start = std::chrono::high_resolution_clock::now();
            output_data = brute_force_querying<float>(*input_images, *query_images, brute_args, std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();
            
            // Calculate timing metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;
            
            // Calculate performance metrics (Average AF, Recall@N, QPS, timing)
            calculate_metrics(*output_data, t_approx.count(), t_true.count());
            
            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        }
        else if (a.common.image_type == IMG_MNIST) {
            std::unique_ptr<Matrix<uint8_t>> input_images = load_mnist_data(a.common.input_file);
            std::unique_ptr<Matrix<uint8_t>> query_images = load_mnist_data(a.common.query_file);
            
            // Run Hypercube FIRST to get approximate neighbors (fills id and distance_approximate)
            auto t_approx_start = std::chrono::high_resolution_clock::now();
            std::unique_ptr<Output> output_data = hypercube_querying<uint8_t>(*input_images, *query_images, a);
            auto t_approx_end = std::chrono::high_resolution_clock::now();
            
            // Then run brute force to fill distance_true fields
            BruteforceArguments brute_args{};
            brute_args.common = a.common;
            auto t_true_start = std::chrono::high_resolution_clock::now();
            output_data = brute_force_querying<uint8_t>(*input_images, *query_images, brute_args, std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();
            
            // Calculate timing metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;
            
            // Calculate performance metrics (Average AF, Recall@N, QPS, timing)
            calculate_metrics(*output_data, t_approx.count(), t_true.count());
            
            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        }
        else {
            fprintf(stderr, "%s:%d: unsupported file format provided.\n", __FILE__, __LINE__);
        }
    }
    else if (strcmp(algorithm_argument, "-ivfflat") == 0) {
        IvfflatArguments a{};
        parse_ivfflat_arguments(&a, argc, argv);
        
        if (a.common.image_type == IMG_SIFT) {
            std::unique_ptr<Matrix<float>> input_images = load_sift_data(a.common.input_file);
            std::unique_ptr<Matrix<float>> query_images = load_sift_data(a.common.query_file);
            
            // Build IVFFlat index
            auto t_approx_start = std::chrono::high_resolution_clock::now();
            Ivfflat<float> index(*input_images, a.kclusters, a.seed);
            index.build();
            
            // Query using IVFFlat
            std::unique_ptr<Output> output_data = std::make_unique<Output>();
            output_data->algorithm = "IVFFlat";
            output_data->queries.resize(query_images->get_rows());
            
            for (size_t query_id = 0; query_id < query_images->get_rows(); query_id++) {
                const float* query_point = query_images->get_row(query_id);
                auto candidates = index.get_candidates(query_point, a.nprobe, a.common.number_of_nearest);
                
                output_data->queries[query_id].nearest_neighbors.resize(a.common.number_of_nearest);
                for (int i = 0; i < a.common.number_of_nearest && i < static_cast<int>(candidates.size()); i++) {
                    auto [dist_squared, image_id] = candidates[i];
                    output_data->queries[query_id].nearest_neighbors[i].id = image_id;
                    output_data->queries[query_id].nearest_neighbors[i].distance_approximate = std::sqrt(dist_squared);
                }
                
                // Range search if requested
                if (a.common.search_for_range) {
                    auto all_candidates = index.get_candidates(query_point, a.nprobe, input_images->get_rows());
                    for (auto [dist_squared, image_id] : all_candidates) {
                        double dist = std::sqrt(dist_squared);
                        if (dist <= a.common.radius && image_id >= 0) {
                            output_data->queries[query_id].r_near_neighbors.push_back(image_id);
                        }
                    }
                }
            }
            auto t_approx_end = std::chrono::high_resolution_clock::now();
            
            // Run brute force to fill distance_true fields
            BruteforceArguments brute_args{};
            brute_args.common = a.common;
            auto t_true_start = std::chrono::high_resolution_clock::now();
            output_data = brute_force_querying<float>(*input_images, *query_images, brute_args, std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();
            
            // Calculate metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;
            calculate_metrics(*output_data, t_approx.count(), t_true.count());
            
            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        }
        else if (a.common.image_type == IMG_MNIST) {
            std::unique_ptr<Matrix<uint8_t>> input_images = load_mnist_data(a.common.input_file);
            std::unique_ptr<Matrix<uint8_t>> query_images = load_mnist_data(a.common.query_file);
            
            // Build IVFFlat index
            auto t_approx_start = std::chrono::high_resolution_clock::now();
            Ivfflat<uint8_t> index(*input_images, a.kclusters, a.seed);
            index.build();
            
            // Query using IVFFlat
            std::unique_ptr<Output> output_data = std::make_unique<Output>();
            output_data->algorithm = "IVFFlat";
            output_data->queries.resize(query_images->get_rows());
            
            for (size_t query_id = 0; query_id < query_images->get_rows(); query_id++) {
                const uint8_t* query_point = query_images->get_row(query_id);
                auto candidates = index.get_candidates(query_point, a.nprobe, a.common.number_of_nearest);
                
                output_data->queries[query_id].nearest_neighbors.resize(a.common.number_of_nearest);
                for (int i = 0; i < a.common.number_of_nearest && i < static_cast<int>(candidates.size()); i++) {
                    auto [dist_squared, image_id] = candidates[i];
                    output_data->queries[query_id].nearest_neighbors[i].id = image_id;
                    output_data->queries[query_id].nearest_neighbors[i].distance_approximate = std::sqrt(dist_squared);
                }
                
                // Range search if requested
                if (a.common.search_for_range) {
                    auto all_candidates = index.get_candidates(query_point, a.nprobe, input_images->get_rows());
                    for (auto [dist_squared, image_id] : all_candidates) {
                        double dist = std::sqrt(dist_squared);
                        if (dist <= a.common.radius && image_id >= 0) {
                            output_data->queries[query_id].r_near_neighbors.push_back(image_id);
                        }
                    }
                }
            }
            auto t_approx_end = std::chrono::high_resolution_clock::now();
            
            // Run brute force to fill distance_true fields
            BruteforceArguments brute_args{};
            brute_args.common = a.common;
            auto t_true_start = std::chrono::high_resolution_clock::now();
            output_data = brute_force_querying<uint8_t>(*input_images, *query_images, brute_args, std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();
            
            // Calculate metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;
            calculate_metrics(*output_data, t_approx.count(), t_true.count());
            
            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        }
        else {
            fprintf(stderr, "%s:%d: unsupported file format provided.\n", __FILE__, __LINE__);
        }
    }
    else if (strcmp(algorithm_argument, "-ivfpq") == 0) {
        IvfpqArguments a{};
        parse_ivfpq_arguments(&a, argc, argv);
        
        if (a.common.image_type == IMG_SIFT) {
            std::unique_ptr<Matrix<float>> input_images = load_sift_data(a.common.input_file);
            std::unique_ptr<Matrix<float>> query_images = load_sift_data(a.common.query_file);
            
            // Calculate dimensions per subvector
            int dimensions_per_subvector = input_images->get_cols() / a.M;
            
            // Build IVFPQ index
            auto t_approx_start = std::chrono::high_resolution_clock::now();
            Ivfpq<float, float, uint8_t> index(*input_images, a.kclusters, dimensions_per_subvector, a.nbits, a.seed);
            index.build();
            
            // Query using IVFPQ
            std::unique_ptr<Output> output_data = std::make_unique<Output>();
            output_data->algorithm = "IVFPQ";
            output_data->queries.resize(query_images->get_rows());
            
            for (size_t query_id = 0; query_id < query_images->get_rows(); query_id++) {
                const float* query_point = query_images->get_row(query_id);
                auto candidates = index.get_candidates(query_point, a.nprobe, a.common.number_of_nearest);
                
                output_data->queries[query_id].nearest_neighbors.resize(a.common.number_of_nearest);
                for (int i = 0; i < a.common.number_of_nearest && i < static_cast<int>(candidates.size()); i++) {
                    auto [dist_squared, image_id] = candidates[i];
                    output_data->queries[query_id].nearest_neighbors[i].id = image_id;
                    output_data->queries[query_id].nearest_neighbors[i].distance_approximate = std::sqrt(dist_squared);
                }
                
                // Range search if requested
                if (a.common.search_for_range) {
                    auto all_candidates = index.get_candidates(query_point, a.nprobe, input_images->get_rows());
                    for (auto [dist_squared, image_id] : all_candidates) {
                        double dist = std::sqrt(dist_squared);
                        if (dist <= a.common.radius && image_id >= 0) {
                            output_data->queries[query_id].r_near_neighbors.push_back(image_id);
                        }
                    }
                }
            }
            auto t_approx_end = std::chrono::high_resolution_clock::now();
            
            // Run brute force to fill distance_true fields
            BruteforceArguments brute_args{};
            brute_args.common = a.common;
            auto t_true_start = std::chrono::high_resolution_clock::now();
            output_data = brute_force_querying<float>(*input_images, *query_images, brute_args, std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();
            
            // Calculate metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;
            calculate_metrics(*output_data, t_approx.count(), t_true.count());
            
            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        }
        else if (a.common.image_type == IMG_MNIST) {
            std::unique_ptr<Matrix<uint8_t>> input_images = load_mnist_data(a.common.input_file);
            std::unique_ptr<Matrix<uint8_t>> query_images = load_mnist_data(a.common.query_file);
            
            // Calculate dimensions per subvector
            int dimensions_per_subvector = input_images->get_cols() / a.M;
            
            // Build IVFPQ index
            auto t_approx_start = std::chrono::high_resolution_clock::now();
            Ivfpq<uint8_t, int8_t, uint8_t> index(*input_images, a.kclusters, dimensions_per_subvector, a.nbits, a.seed);
            index.build();
            
            // Query using IVFPQ
            std::unique_ptr<Output> output_data = std::make_unique<Output>();
            output_data->algorithm = "IVFPQ";
            output_data->queries.resize(query_images->get_rows());
            
            for (size_t query_id = 0; query_id < query_images->get_rows(); query_id++) {
                const uint8_t* query_point = query_images->get_row(query_id);
                auto candidates = index.get_candidates(query_point, a.nprobe, a.common.number_of_nearest);
                
                output_data->queries[query_id].nearest_neighbors.resize(a.common.number_of_nearest);
                for (int i = 0; i < a.common.number_of_nearest && i < static_cast<int>(candidates.size()); i++) {
                    auto [dist_squared, image_id] = candidates[i];
                    output_data->queries[query_id].nearest_neighbors[i].id = image_id;
                    output_data->queries[query_id].nearest_neighbors[i].distance_approximate = std::sqrt(dist_squared);
                }
                
                // Range search if requested
                if (a.common.search_for_range) {
                    auto all_candidates = index.get_candidates(query_point, a.nprobe, input_images->get_rows());
                    for (auto [dist_squared, image_id] : all_candidates) {
                        double dist = std::sqrt(dist_squared);
                        if (dist <= a.common.radius && image_id >= 0) {
                            output_data->queries[query_id].r_near_neighbors.push_back(image_id);
                        }
                    }
                }
            }
            auto t_approx_end = std::chrono::high_resolution_clock::now();
            
            // Run brute force to fill distance_true fields
            BruteforceArguments brute_args{};
            brute_args.common = a.common;
            auto t_true_start = std::chrono::high_resolution_clock::now();
            output_data = brute_force_querying<uint8_t>(*input_images, *query_images, brute_args, std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();
            
            // Calculate metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;
            calculate_metrics(*output_data, t_approx.count(), t_true.count());
            
            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        }
        else {
            fprintf(stderr, "%s:%d: unsupported file format provided.\n", __FILE__, __LINE__);
        }
    }
    else if (strcmp(algorithm_argument, "-bruteforce") == 0) {
        BruteforceArguments a{};
        parse_bruteforce_arguments(&a, argc, argv);
        if (a.common.image_type == IMG_SIFT) {
            std::unique_ptr<Matrix<float>> input_images = load_sift_data(a.common.input_file);
            std::unique_ptr<Matrix<float>> query_images = load_sift_data(a.common.query_file);
            
            auto t_start = std::chrono::high_resolution_clock::now();
            std::unique_ptr<Output> output_data = brute_force_querying<float>(*input_images, *query_images, a);
            auto t_end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double> t_elapsed = t_end - t_start;
            output_data->t_true_average = t_elapsed.count() / query_images->get_rows();
            output_data->queries_per_second = query_images->get_rows() / t_elapsed.count();
            
            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        }
        else if (a.common.image_type == IMG_MNIST) {
            std::unique_ptr<Matrix<uint8_t>> input_images = load_mnist_data(a.common.input_file);
            std::unique_ptr<Matrix<uint8_t>> query_images = load_mnist_data(a.common.query_file);
            
            auto t_start = std::chrono::high_resolution_clock::now();
            std::unique_ptr<Output> output_data = brute_force_querying<uint8_t>(*input_images, *query_images, a);
            auto t_end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double> t_elapsed = t_end - t_start;
            output_data->t_true_average = t_elapsed.count() / query_images->get_rows();
            output_data->queries_per_second = query_images->get_rows() / t_elapsed.count();
            
            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        }
        else {
            fprintf(stderr, "%s:%d: unsupported algorithm specified.\n", __FILE__, __LINE__);
        }
    }
    else {
        fprintf(stderr, "Internal error when finding the argument "
            "that denotes the algorithm, returned '%s'\n", algorithm_argument);
        return -1;
    }

    return 0;
}