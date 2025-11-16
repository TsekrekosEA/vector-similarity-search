/**
 * @file main.cpp
 * @brief Main entry point for the ANN search engine benchmarking suite
 * 
 * Program Architecture:
 * This program implements a command-line interface for benchmarking approximate nearest neighbor
 * (ANN) algorithms on high-dimensional vector datasets. The architecture follows a template-based
 * design supporting both float (SIFT) and uint8_t (MNIST) data types through compile-time
 * polymorphism. The program processes commands in this pipeline: parse command-line arguments to
 * determine algorithm and parameters, load dataset and query files from disk in binary format,
 * instantiate and build the selected algorithm's index structure, execute queries to find
 * approximate nearest neighbors (timed separately), run brute-force linear scan to find true
 * nearest neighbors for validation (timed separately), calculate performance metrics comparing
 * approximate to true results, write results to output file.
 * 
 * Two-Phase Benchmarking:
 * The benchmarking methodology separates approximate search from ground truth computation to
 * measure each independently. Phase 1 runs the ANN algorithm (LSH, Hypercube, IVF-Flat, or IVF-PQ)
 * to populate the approximate results (id and distance_approximate fields in ANearNeighbor
 * structures). Phase 2 runs brute-force exhaustive search to populate ground truth (id_true and
 * distance_true fields). This separation allows measuring ANN speedup (t_true / t_approx) and
 * accuracy metrics (recall, average approximation factor) independently. The Output structure
 * holds both approximate and true results per query, enabling direct comparison.
 * 
 * Template Instantiation:
 * The code uses C++ templates to support multiple data types without code duplication. Each
 * algorithm is templated on element type T (float for SIFT, uint8_t for MNIST). The main
 * function acts as a dispatcher: after parsing arguments and determining dataset type, it
 * instantiates the appropriate template specialization (e.g., lsh_querying<float> or
 * lsh_querying<uint8_t>). This provides type safety and optimal performance (no virtual function
 * overhead, full compiler optimization) while keeping a single codebase. Template explicit
 * instantiations in each algorithm's .cpp file ensure compilation.
 * 
 * Timing Methodology:
 * All timing uses std::chrono::high_resolution_clock for nanosecond precision. Timing is split
 * into two phases: t_approx measures only the approximate search time (index build + queries),
 * t_true measures only the brute-force validation time (linear scan for all queries). This
 * separation is critical because index build time is a one-time cost amortized across many
 * queries in production, while query time matters for each request. The metrics calculation
 * reports queries per second (QPS) as query_count / t_approx and speedup as t_true / t_approx.
 * 
 * Algorithm Selection:
 * The program supports four ANN algorithms selected via command-line flag: -lsh for Locality
 * Sensitive Hashing with L hash tables and composite hashing, -hypercube for Hypercube LSH with
 * single table and multi-probe BFS, -ivfflat for Inverted File with exact vectors and two-level
 * search, -ivfpq for Inverted File with Product Quantization and asymmetric distance. Each has
 * algorithm-specific parameters (L and k for LSH, kproj/probes/M for Hypercube, kclusters/nprobe
 * for IVF-Flat, kclusters/nprobe/nbits/subvector_dims for IVF-PQ) plus common parameters (N for
 * top-N, R for radius search).
 * 
 * Output Format:
 * Results are written to a text file with per-query nearest neighbors (IDs and distances) plus
 * aggregate metrics: average_approximation_factor measures distance inflation (avg of
 * distance_approximate / distance_true), recall_at_N measures accuracy (fraction of queries where
 * true NN is in top-N approximate results), queries_per_second measures throughput (queries /
 * t_approx), speedup_vs_brute_force measures efficiency (t_true / t_approx). These metrics enable
 * direct comparison across algorithms and parameter settings.
 * 
 * Memory Management:
 * The program uses smart pointers (std::unique_ptr) for automatic memory management. Dataset
 * matrices are heap-allocated via unique_ptr to handle large data (SIFT: 1M × 128 floats = 512MB,
 * MNIST: 60K × 784 uint8 = 47MB). Ownership transfer via std::move avoids copies when passing
 * Output structures between phases. Index structures are stack-allocated within their scope for
 * RAII cleanup.
 * 
 * @authors Τσεκρέκος Έγκορ-Ανδριανός, Δημακόπουλος Θεόδωρος
 */

#include <chrono>
#include <stdio.h>
#include <string.h>

#include <arg_parse.hpp>
#include <brute_force.hpp>
#include <file_io.hpp>
#include <hypercube.hpp>
#include <ivfflat.hpp>
#include <ivfpq.hpp>
#include <lsh.hpp>
#include <utils.hpp>

int main(int argc, char** argv) {

    // Determine which algorithm to run based on command-line flag. This call can exit(-1) if the
    // algorithm flag is missing or invalid, providing early validation before expensive argument
    // parsing.
    char* algorithm_argument = get_algorithm_argument(argc, argv);
    if (strcmp(algorithm_argument, "-lsh") == 0) {
        LshArguments a{};
        parse_lsh_arguments(&a, argc, argv);

        if (a.common.image_type == IMG_SIFT) {
            std::unique_ptr<Matrix<float>> input_images = load_sift_data(a.common.input_file);
            std::unique_ptr<Matrix<float>> query_images = load_sift_data(a.common.query_file);

            // Run LSH FIRST to get approximate neighbors (fills id and distance_approximate)
            auto t_approx_start = std::chrono::high_resolution_clock::now();
            std::unique_ptr<Output> output_data =
                lsh_querying<float>(*input_images, *query_images, a);
            auto t_approx_end = std::chrono::high_resolution_clock::now();

            // Then run brute force to fill distance_true fields
            BruteforceArguments brute_args{};
            brute_args.common = a.common;
            auto t_true_start = std::chrono::high_resolution_clock::now();
            output_data = brute_force_querying<float>(*input_images, *query_images, brute_args,
                                                      std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();

            // Calculate timing metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;

            // Calculate performance metrics (Average AF, Recall@N, QPS, timing)
            calculate_metrics(*output_data, t_approx.count(), t_true.count());

            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        } else if (a.common.image_type == IMG_MNIST) {
            std::unique_ptr<Matrix<uint8_t>> input_images = load_mnist_data(a.common.input_file);
            std::unique_ptr<Matrix<uint8_t>> query_images = load_mnist_data(a.common.query_file);

            // Run LSH FIRST to get approximate neighbors (fills id and distance_approximate)
            auto t_approx_start = std::chrono::high_resolution_clock::now();
            std::unique_ptr<Output> output_data =
                lsh_querying<uint8_t>(*input_images, *query_images, a);
            auto t_approx_end = std::chrono::high_resolution_clock::now();

            // Then run brute force to fill distance_true fields
            BruteforceArguments brute_args{};
            brute_args.common = a.common;
            auto t_true_start = std::chrono::high_resolution_clock::now();
            output_data = brute_force_querying<uint8_t>(*input_images, *query_images, brute_args,
                                                        std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();

            // Calculate timing metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;

            // Calculate performance metrics (Average AF, Recall@N, QPS, timing)
            calculate_metrics(*output_data, t_approx.count(), t_true.count());

            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        } else {
            fprintf(stderr, "%s:%d: unsupported file format provided.\n", __FILE__, __LINE__);
        }
    } else if (strcmp(algorithm_argument, "-hypercube") == 0) {
        HypercubeArguments a{};
        parse_hypercube_arguments(&a, argc, argv);

        if (a.common.image_type == IMG_SIFT) {
            std::unique_ptr<Matrix<float>> input_images = load_sift_data(a.common.input_file);
            std::unique_ptr<Matrix<float>> query_images = load_sift_data(a.common.query_file);

            // Run Hypercube FIRST to get approximate neighbors (fills id and distance_approximate)
            auto t_approx_start = std::chrono::high_resolution_clock::now();
            std::unique_ptr<Output> output_data =
                hypercube_querying<float>(*input_images, *query_images, a);
            auto t_approx_end = std::chrono::high_resolution_clock::now();

            // Then run brute force to fill distance_true fields
            BruteforceArguments brute_args{};
            brute_args.common = a.common;
            auto t_true_start = std::chrono::high_resolution_clock::now();
            output_data = brute_force_querying<float>(*input_images, *query_images, brute_args,
                                                      std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();

            // Calculate timing metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;

            // Calculate performance metrics (Average AF, Recall@N, QPS, timing)
            calculate_metrics(*output_data, t_approx.count(), t_true.count());

            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        } else if (a.common.image_type == IMG_MNIST) {
            std::unique_ptr<Matrix<uint8_t>> input_images = load_mnist_data(a.common.input_file);
            std::unique_ptr<Matrix<uint8_t>> query_images = load_mnist_data(a.common.query_file);

            // Run Hypercube FIRST to get approximate neighbors (fills id and distance_approximate)
            auto t_approx_start = std::chrono::high_resolution_clock::now();
            std::unique_ptr<Output> output_data =
                hypercube_querying<uint8_t>(*input_images, *query_images, a);
            auto t_approx_end = std::chrono::high_resolution_clock::now();

            // Then run brute force to fill distance_true fields
            BruteforceArguments brute_args{};
            brute_args.common = a.common;
            auto t_true_start = std::chrono::high_resolution_clock::now();
            output_data = brute_force_querying<uint8_t>(*input_images, *query_images, brute_args,
                                                        std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();

            // Calculate timing metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;

            // Calculate performance metrics (Average AF, Recall@N, QPS, timing)
            calculate_metrics(*output_data, t_approx.count(), t_true.count());

            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        } else {
            fprintf(stderr, "%s:%d: unsupported file format provided.\n", __FILE__, __LINE__);
        }
    } else if (strcmp(algorithm_argument, "-ivfflat") == 0) {
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
                auto candidates =
                    index.get_candidates(query_point, a.nprobe, a.common.number_of_nearest);

                output_data->queries[query_id].nearest_neighbors.resize(a.common.number_of_nearest);
                for (int i = 0;
                     i < a.common.number_of_nearest && i < static_cast<int>(candidates.size());
                     i++) {
                    auto [dist_squared, image_id] = candidates[i];
                    output_data->queries[query_id].nearest_neighbors[i].id = image_id;
                    output_data->queries[query_id].nearest_neighbors[i].distance_approximate =
                        std::sqrt(dist_squared);
                }

                // Range search if requested
                if (a.common.search_for_range) {
                    auto all_candidates =
                        index.get_candidates(query_point, a.nprobe, input_images->get_rows());
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
            output_data = brute_force_querying<float>(*input_images, *query_images, brute_args,
                                                      std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();

            // Calculate metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;
            calculate_metrics(*output_data, t_approx.count(), t_true.count());

            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        } else if (a.common.image_type == IMG_MNIST) {
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
                auto candidates =
                    index.get_candidates(query_point, a.nprobe, a.common.number_of_nearest);

                output_data->queries[query_id].nearest_neighbors.resize(a.common.number_of_nearest);
                for (int i = 0;
                     i < a.common.number_of_nearest && i < static_cast<int>(candidates.size());
                     i++) {
                    auto [dist_squared, image_id] = candidates[i];
                    output_data->queries[query_id].nearest_neighbors[i].id = image_id;
                    output_data->queries[query_id].nearest_neighbors[i].distance_approximate =
                        std::sqrt(dist_squared);
                }

                // Range search if requested
                if (a.common.search_for_range) {
                    auto all_candidates =
                        index.get_candidates(query_point, a.nprobe, input_images->get_rows());
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
            output_data = brute_force_querying<uint8_t>(*input_images, *query_images, brute_args,
                                                        std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();

            // Calculate metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;
            calculate_metrics(*output_data, t_approx.count(), t_true.count());

            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        } else {
            fprintf(stderr, "%s:%d: unsupported file format provided.\n", __FILE__, __LINE__);
        }
    } else if (strcmp(algorithm_argument, "-ivfpq") == 0) {
        IvfpqArguments a{};
        parse_ivfpq_arguments(&a, argc, argv);

        if (a.common.image_type == IMG_SIFT) {
            std::unique_ptr<Matrix<float>> input_images = load_sift_data(a.common.input_file);
            std::unique_ptr<Matrix<float>> query_images = load_sift_data(a.common.query_file);

            // Calculate dimensions per subvector
            int dimensions_per_subvector = input_images->get_cols() / a.M;

            // Build IVFPQ index
            auto t_approx_start = std::chrono::high_resolution_clock::now();
            Ivfpq<float, float, uint8_t> index(*input_images, a.kclusters, dimensions_per_subvector,
                                               a.nbits, a.seed);
            index.build();

            // Query using IVFPQ
            std::unique_ptr<Output> output_data = std::make_unique<Output>();
            output_data->algorithm = "IVFPQ";
            output_data->queries.resize(query_images->get_rows());

            for (size_t query_id = 0; query_id < query_images->get_rows(); query_id++) {
                const float* query_point = query_images->get_row(query_id);
                auto candidates =
                    index.get_candidates(query_point, a.nprobe, a.common.number_of_nearest);

                output_data->queries[query_id].nearest_neighbors.resize(a.common.number_of_nearest);
                for (int i = 0;
                     i < a.common.number_of_nearest && i < static_cast<int>(candidates.size());
                     i++) {
                    auto [dist_squared, image_id] = candidates[i];
                    output_data->queries[query_id].nearest_neighbors[i].id = image_id;
                    output_data->queries[query_id].nearest_neighbors[i].distance_approximate =
                        std::sqrt(dist_squared);
                }

                // Range search if requested
                if (a.common.search_for_range) {
                    auto all_candidates =
                        index.get_candidates(query_point, a.nprobe, input_images->get_rows());
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
            output_data = brute_force_querying<float>(*input_images, *query_images, brute_args,
                                                      std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();

            // Calculate metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;
            calculate_metrics(*output_data, t_approx.count(), t_true.count());

            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        } else if (a.common.image_type == IMG_MNIST) {
            std::unique_ptr<Matrix<uint8_t>> input_images = load_mnist_data(a.common.input_file);
            std::unique_ptr<Matrix<uint8_t>> query_images = load_mnist_data(a.common.query_file);

            // Calculate dimensions per subvector
            int dimensions_per_subvector = input_images->get_cols() / a.M;

            // Build IVFPQ index
            auto t_approx_start = std::chrono::high_resolution_clock::now();
            Ivfpq<uint8_t, int8_t, uint8_t> index(*input_images, a.kclusters,
                                                  dimensions_per_subvector, a.nbits, a.seed);
            index.build();

            // Query using IVFPQ
            std::unique_ptr<Output> output_data = std::make_unique<Output>();
            output_data->algorithm = "IVFPQ";
            output_data->queries.resize(query_images->get_rows());

            for (size_t query_id = 0; query_id < query_images->get_rows(); query_id++) {
                const uint8_t* query_point = query_images->get_row(query_id);
                auto candidates =
                    index.get_candidates(query_point, a.nprobe, a.common.number_of_nearest);

                output_data->queries[query_id].nearest_neighbors.resize(a.common.number_of_nearest);
                for (int i = 0;
                     i < a.common.number_of_nearest && i < static_cast<int>(candidates.size());
                     i++) {
                    auto [dist_squared, image_id] = candidates[i];
                    output_data->queries[query_id].nearest_neighbors[i].id = image_id;
                    output_data->queries[query_id].nearest_neighbors[i].distance_approximate =
                        std::sqrt(dist_squared);
                }

                // Range search if requested
                if (a.common.search_for_range) {
                    auto all_candidates =
                        index.get_candidates(query_point, a.nprobe, input_images->get_rows());
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
            output_data = brute_force_querying<uint8_t>(*input_images, *query_images, brute_args,
                                                        std::move(output_data));
            auto t_true_end = std::chrono::high_resolution_clock::now();

            // Calculate metrics
            std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
            std::chrono::duration<double> t_true = t_true_end - t_true_start;
            calculate_metrics(*output_data, t_approx.count(), t_true.count());

            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        } else {
            fprintf(stderr, "%s:%d: unsupported file format provided.\n", __FILE__, __LINE__);
        }
    } else if (strcmp(algorithm_argument, "-bruteforce") == 0) {
        BruteforceArguments a{};
        parse_bruteforce_arguments(&a, argc, argv);
        if (a.common.image_type == IMG_SIFT) {
            std::unique_ptr<Matrix<float>> input_images = load_sift_data(a.common.input_file);
            std::unique_ptr<Matrix<float>> query_images = load_sift_data(a.common.query_file);

            auto t_start = std::chrono::high_resolution_clock::now();
            std::unique_ptr<Output> output_data =
                brute_force_querying<float>(*input_images, *query_images, a);
            auto t_end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> t_elapsed = t_end - t_start;
            output_data->t_true_average = t_elapsed.count() / query_images->get_rows();
            output_data->queries_per_second = query_images->get_rows() / t_elapsed.count();

            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        } else if (a.common.image_type == IMG_MNIST) {
            std::unique_ptr<Matrix<uint8_t>> input_images = load_mnist_data(a.common.input_file);
            std::unique_ptr<Matrix<uint8_t>> query_images = load_mnist_data(a.common.query_file);

            auto t_start = std::chrono::high_resolution_clock::now();
            std::unique_ptr<Output> output_data =
                brute_force_querying<uint8_t>(*input_images, *query_images, a);
            auto t_end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> t_elapsed = t_end - t_start;
            output_data->t_true_average = t_elapsed.count() / query_images->get_rows();
            output_data->queries_per_second = query_images->get_rows() / t_elapsed.count();

            std::unique_ptr<std::ofstream> s = initialize_output_file(a.common.output_file);
            write_output(*s, *output_data);
        } else {
            fprintf(stderr, "%s:%d: unsupported algorithm specified.\n", __FILE__, __LINE__);
        }
    } else {
        fprintf(stderr,
                "Internal error when finding the argument "
                "that denotes the algorithm, returned '%s'\n",
                algorithm_argument);
        return -1;
    }

    return 0;
}