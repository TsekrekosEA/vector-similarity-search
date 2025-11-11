// benchmark.cpp - Efficient benchmark system that pre-computes ground truth
// This avoids redundant brute force computation for algorithm performance analysis

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include <brute_force.hpp>
#include <data_types.hpp>
#include <file_io.hpp>
#include <hypercube.hpp>
#include <ivfflat.hpp>
#include <ivfpq.hpp>
#include <lsh.hpp>
#include <utils.hpp>

// ============================================================================
// SMART PARALLELIZATION CONTROL
// ============================================================================

// Determine optimal number of parallel tasks based on system resources
inline size_t get_optimal_parallelism() {
    unsigned int hw_cores = std::thread::hardware_concurrency();

    // If hardware_concurrency fails, default to 4
    if (hw_cores == 0) {
        return 4;
    }

    // Use 75% of available cores to avoid oversubscription
    // Leave some cores for OS and other processes
    // Minimum of 1, maximum of actual cores
    size_t optimal = std::max(1u, static_cast<unsigned int>(hw_cores * 0.75));

    return std::min(optimal, static_cast<size_t>(hw_cores));
}

// Batch configurations to run in controlled parallel chunks
template <typename ConfigType, typename TaskFunc>
void run_batched_parallel(const std::vector<ConfigType>& configs, TaskFunc task_func) {
    const size_t max_parallel = get_optimal_parallelism();
    const size_t total_configs = configs.size();

    std::cout << "  [Parallelization: " << max_parallel << " concurrent tasks, " << total_configs
              << " total configurations]" << std::endl;

    for (size_t batch_start = 0; batch_start < total_configs; batch_start += max_parallel) {
        size_t batch_end = std::min(batch_start + max_parallel, total_configs);
        size_t batch_size = batch_end - batch_start;

        // Launch batch
        std::vector<std::future<void>> batch_futures;
        batch_futures.reserve(batch_size);

        for (size_t i = batch_start; i < batch_end; ++i) {
            batch_futures.push_back(
                std::async(std::launch::async, [&, i]() { task_func(configs[i]); }));
        }

        // Wait for batch to complete before starting next batch
        for (auto& future : batch_futures) {
            future.get();
        }
    }
}

// ============================================================================
// THREAD-SAFE OUTPUT HANDLING
// ============================================================================

std::mutex cout_mutex;
std::mutex csv_mutex;

void thread_safe_print(const std::string& message) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << message << std::flush;
}

void thread_safe_csv_write(std::ofstream& csv_file, const std::string& line) {
    std::lock_guard<std::mutex> lock(csv_mutex);
    csv_file << line << std::flush;
}

// ============================================================================
// GROUND TRUTH CACHE
// ============================================================================

// Save ground truth results to a binary file for reuse
template <typename T> void save_ground_truth(const std::string& filename, const Output& output) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to save ground truth to: " << filename << std::endl;
        return;
    }

    size_t num_queries = output.queries.size();
    file.write(reinterpret_cast<const char*>(&num_queries), sizeof(num_queries));

    for (const auto& query : output.queries) {
        size_t num_neighbors = query.nearest_neighbors.size();
        file.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(num_neighbors));

        for (const auto& nn : query.nearest_neighbors) {
            file.write(reinterpret_cast<const char*>(&nn.id_true), sizeof(nn.id_true));
            file.write(reinterpret_cast<const char*>(&nn.distance_true), sizeof(nn.distance_true));
        }
    }

    std::cout << "Ground truth saved: " << filename << std::endl;
}

// Load pre-computed ground truth from binary file
std::unique_ptr<Output> load_ground_truth(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return nullptr;
    }

    auto output = std::make_unique<Output>();
    output->algorithm = "ground_truth";

    size_t num_queries;
    file.read(reinterpret_cast<char*>(&num_queries), sizeof(num_queries));
    output->queries.resize(num_queries);

    for (auto& query : output->queries) {
        size_t num_neighbors;
        file.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));
        query.nearest_neighbors.resize(num_neighbors);

        for (auto& nn : query.nearest_neighbors) {
            file.read(reinterpret_cast<char*>(&nn.id_true), sizeof(nn.id_true));
            file.read(reinterpret_cast<char*>(&nn.distance_true), sizeof(nn.distance_true));
        }
    }

    std::cout << "Ground truth loaded: " << filename << " (" << num_queries << " queries)"
              << std::endl;
    return output;
}

// ============================================================================
// BENCHMARK UTILITIES
// ============================================================================

// Fill approximate results with ground truth data for metrics calculation
void fill_with_ground_truth(Output& approx_output, const Output& ground_truth) {
    for (size_t i = 0; i < approx_output.queries.size() && i < ground_truth.queries.size(); i++) {
        auto& approx_query = approx_output.queries[i];
        const auto& gt_query = ground_truth.queries[i];

        // Ensure we have enough slots
        size_t num_to_copy =
            std::min(approx_query.nearest_neighbors.size(), gt_query.nearest_neighbors.size());

        for (size_t j = 0; j < num_to_copy; j++) {
            approx_query.nearest_neighbors[j].id_true = gt_query.nearest_neighbors[j].id_true;
            approx_query.nearest_neighbors[j].distance_true =
                gt_query.nearest_neighbors[j].distance_true;
        }
    }
}

// Print benchmark results in a formatted table
void print_benchmark_result(const std::string& algorithm, const std::string& params, double avg_af,
                            double recall, double qps, double t_approx, double speedup_vs_brute) {
    std::ostringstream oss;
    oss << std::left << std::setw(15) << algorithm << std::setw(30) << params << std::right
        << " AF=" << std::fixed << std::setprecision(4) << std::setw(7) << avg_af
        << " Recall=" << std::setw(6) << std::setprecision(3) << recall << " QPS=" << std::setw(8)
        << std::setprecision(1) << qps << " Time=" << std::setw(8) << std::setprecision(6)
        << t_approx << "s" << " Speedup=" << std::setw(6) << std::setprecision(1)
        << speedup_vs_brute << "x" << std::endl;
    thread_safe_print(oss.str());
}

// Write benchmark result to CSV file (for Python plotting)
void write_csv_result(std::ofstream& csv_file, const std::string& dataset,
                      const std::string& algorithm, const std::string& params, double avg_af,
                      double recall, double qps, double t_approx, double speedup_vs_brute) {
    std::ostringstream oss;
    oss << dataset << "," << algorithm << "," << params << "," << std::fixed << std::setprecision(6)
        << avg_af << "," << recall << "," << qps << "," << t_approx << "," << speedup_vs_brute
        << std::endl;
    thread_safe_csv_write(csv_file, oss.str());
}

// ============================================================================
// BENCHMARK EXECUTION
// ============================================================================

template <typename T>
void run_lsh_benchmark(const Matrix<T>& input_images, const Matrix<T>& query_images,
                       const Output& ground_truth, ImageType img_type, int N,
                       double brute_time_per_query, std::ofstream& csv_file,
                       const std::string& dataset_name, bool enable_range_search = false,
                       double R = 0.0) {
    std::cout << "\n--- LSH Parameter Sweep ---" << std::endl;

    // Cross join of all parameter combinations
    std::vector<std::tuple<int, int, double>> configs;

    if (img_type == IMG_MNIST) {
        // MNIST: k × L × w cross join
        std::vector<int> k_values = {2, 4, 8};
        std::vector<int> L_values = {4, 6, 10, 14};
        std::vector<double> w_values = {1000.0, 2000.0, 3000.0};

        for (int k : k_values) {
            for (int L : L_values) {
                for (double w : w_values) {
                    configs.emplace_back(k, L, w);
                }
            }
        }
    } else { // SIFT
        // SIFT: k × L × w cross join
        std::vector<int> k_values = {4, 6, 10};
        std::vector<int> L_values = {4, 6, 10, 14};
        std::vector<double> w_values = {175.0, 250.0, 400.0};

        for (int k : k_values) {
            for (int L : L_values) {
                for (double w : w_values) {
                    configs.emplace_back(k, L, w);
                }
            }
        }
    }

    // Run configurations in batched parallel execution
    auto task = [&](const std::tuple<int, int, double>& config) {
        auto [k, L, w] = config;

        LshArguments args{};
        args.common.number_of_nearest = N;
        args.common.image_type = img_type;
        args.common.search_for_range = enable_range_search;
        args.common.radius = static_cast<int>(R);
        args.k = k;
        args.L = L;
        args.w = w;
        // TODO: Allow overriding the benchmark seed from CLI if experiments need variability.
        args.seed = 1;

        auto start = std::chrono::high_resolution_clock::now();
        auto output = lsh_querying(input_images, query_images, args);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        double t_per_query = elapsed / query_images.get_rows();

        // Fill with ground truth
        fill_with_ground_truth(*output, ground_truth);

        // Calculate metrics (modifies output in-place)
        calculate_metrics(*output, elapsed,
                          0.0); // Pass approximate time, no brute force time in benchmark

        double avg_af = output->average_af;
        double recall = output->recall_at_n;
        double qps = output->queries_per_second;

        double speedup = brute_time_per_query / t_per_query;

        std::string params =
            "k=" + std::to_string(k) + " L=" + std::to_string(L) + " w=" + std::to_string(w);
        if (enable_range_search) {
            params += " R=" + std::to_string(R);
        }
        print_benchmark_result("LSH", params, avg_af, recall, qps, t_per_query, speedup);
        write_csv_result(csv_file, dataset_name, "LSH", params, avg_af, recall, qps, t_per_query,
                         speedup);
    };

    run_batched_parallel(configs, task);
}

template <typename T>
void run_hypercube_benchmark(const Matrix<T>& input_images, const Matrix<T>& query_images,
                             const Output& ground_truth, ImageType img_type, int N,
                             double brute_time_per_query, std::ofstream& csv_file,
                             const std::string& dataset_name, bool enable_range_search = false,
                             double R = 0.0) {
    std::cout << "\n--- Hypercube Parameter Sweep ---" << std::endl;

    // Cross join of all parameter combinations (kproj × probes, M derived from probes)
    std::vector<std::tuple<int, int, int>> configs;

    if (img_type == IMG_MNIST) {
        // MNIST: tuned cross join (kproj × probes) with M derived from probes
        const std::vector<int> kproj_values = {8, 10, 12, 14, 16};
        const std::vector<int> probes_values = {8, 16, 32, 64, 100, 124, 256, 512, 1024, 2048};
        const int dataset_rows = static_cast<int>(input_images.get_rows());

        for (int kproj : kproj_values) {
            for (int probes : probes_values) {
                const int derived_M = std::min(dataset_rows, probes * 40);
                configs.emplace_back(kproj, derived_M, probes);
            }
        }
    } else { // SIFT
        // SIFT: tuned cross join (kproj × probes) with M derived from probes
        const std::vector<int> kproj_values = {8, 10, 12, 14, 16};
        const std::vector<int> probes_values = {8, 16, 32, 64, 100, 124, 256, 512, 1024, 2048};
        const int dataset_rows = static_cast<int>(input_images.get_rows());

        for (int kproj : kproj_values) {
            for (int probes : probes_values) {
                const int derived_M = std::min(dataset_rows, probes * 15);
                configs.emplace_back(kproj, derived_M, probes);
            }
        }
    }

    // Run configurations in batched parallel execution
    auto task = [&](const std::tuple<int, int, int>& config) {
        auto [kproj, M, probes] = config;

        HypercubeArguments args{};
        args.common.number_of_nearest = N;
        args.common.image_type = img_type;
        args.common.search_for_range = enable_range_search;
        args.common.radius = static_cast<int>(R);
        args.kproj = kproj;
        args.M = M;
        args.probes = probes;
        args.seed = 1;

        auto start = std::chrono::high_resolution_clock::now();
        auto output = hypercube_querying(input_images, query_images, args);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        double t_per_query = elapsed / query_images.get_rows();

        // Fill with ground truth
        fill_with_ground_truth(*output, ground_truth);

        // Calculate metrics (modifies output in-place)
        calculate_metrics(*output, elapsed,
                          0.0); // Pass approximate time, no brute force time in benchmark

        double avg_af = output->average_af;
        double recall = output->recall_at_n;
        double qps = output->queries_per_second;

        double speedup = brute_time_per_query / t_per_query;

        std::string params = "kproj=" + std::to_string(kproj) + " M=" + std::to_string(M) +
                             " probes=" + std::to_string(probes);
        if (enable_range_search) {
            params += " R=" + std::to_string(R);
        }
        print_benchmark_result("Hypercube", params, avg_af, recall, qps, t_per_query, speedup);
        write_csv_result(csv_file, dataset_name, "Hypercube", params, avg_af, recall, qps,
                         t_per_query, speedup);
    };

    run_batched_parallel(configs, task);
}

template <typename T>
void run_ivfflat_benchmark(const Matrix<T>& input_images, const Matrix<T>& query_images,
                           const Output& ground_truth, ImageType img_type, int N,
                           double brute_time_per_query, std::ofstream& csv_file,
                           const std::string& dataset_name, bool enable_range_search = false,
                           double R = 0.0) {
    std::cout << "\n--- IVF-Flat Parameter Sweep ---" << std::endl;

    // Cross join of all parameter combinations
    std::vector<std::tuple<int, int>> configs;

    if (img_type == IMG_MNIST) {
        // MNIST: k_clusters × nprobe cross join
        const std::vector<int> k_clusters_values = {16, 32, 64, 128, 256};
        const std::vector<int> nprobe_values = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32};

        for (int k_clusters : k_clusters_values) {
            for (int nprobe : nprobe_values) {
                configs.emplace_back(k_clusters, nprobe);
            }
        }
    } else { // SIFT
        // SIFT: k_clusters × nprobe cross join (heavily reduced for 1M dataset)
        const std::vector<int> k_clusters_values = {16, 32, 64, 128, 256};
        const std::vector<int> nprobe_values = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32};

        for (int k_clusters : k_clusters_values) {
            for (int nprobe : nprobe_values) {
                configs.emplace_back(k_clusters, nprobe);
            }
        }
    }

    // OPTIMIZATION: Group by k_clusters and build index once per k_clusters value
    // Then test all nprobe values on the same index (nprobe only affects querying, not building)
    std::cout << "  [Optimized: building index once per k_clusters value]" << std::endl;

    // Group configs by k_clusters
    std::map<int, std::vector<int>> build_groups;
    for (const auto& [k_clusters, nprobe] : configs) {
        build_groups[k_clusters].push_back(nprobe);
    }

    std::cout << "  [Index builds needed: " << build_groups.size() << " (down from "
              << configs.size() << ")]" << std::endl;

    for (const auto& [k_clusters, nprobe_values] : build_groups) {
        // Build IVF-Flat index once for this k_clusters value
        auto start_build = std::chrono::high_resolution_clock::now();
        Ivfflat<T> index(input_images, k_clusters, 1);
        index.build();
        auto end_build = std::chrono::high_resolution_clock::now();
        double build_time = std::chrono::duration<double>(end_build - start_build).count();

        std::cout << "  Built index: k_clusters=" << k_clusters << " (build time: " << std::fixed
                  << std::setprecision(2) << build_time << "s)" << std::endl;

        // Test all nprobe values on this index
        for (int nprobe : nprobe_values) {
            // Query phase
            auto start = std::chrono::high_resolution_clock::now();

            Output output;
            output.algorithm = "IVF-Flat";
            output.queries.resize(query_images.get_rows());

            for (size_t i = 0; i < query_images.get_rows(); i++) {
                const T* query_point = query_images.get_row(i);
                auto candidates = index.get_candidates(query_point, nprobe, N);

                output.queries[i].nearest_neighbors.resize(N);
                for (size_t j = 0; j < std::min(candidates.size(), static_cast<size_t>(N)); j++) {
                    auto [dist, id] = candidates[j];
                    output.queries[i].nearest_neighbors[j].id = id;
                    output.queries[i].nearest_neighbors[j].distance_approximate = std::sqrt(dist);
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            double t_per_query = elapsed / query_images.get_rows();

            // Fill with ground truth
            fill_with_ground_truth(output, ground_truth);

            // Calculate metrics
            calculate_metrics(output, elapsed, 0.0);

            double avg_af = output.average_af;
            double recall = output.recall_at_n;
            double qps = output.queries_per_second;

            double speedup = brute_time_per_query / t_per_query;

            std::string params =
                "k_clusters=" + std::to_string(k_clusters) + " nprobe=" + std::to_string(nprobe);
            if (enable_range_search) {
                params += " R=" + std::to_string(R);
            }

            std::cout << "    nprobe=" << nprobe << " | Recall=" << std::setprecision(3) << recall
                      << " | Speedup=" << std::setprecision(1) << speedup << "x" << std::endl;

            write_csv_result(csv_file, dataset_name, "IVF-Flat", params, avg_af, recall, qps,
                             t_per_query, speedup);
        }
    }
}

template <typename T, typename TWithNegatives>
void run_ivfpq_benchmark(const Matrix<T>& input_images, const Matrix<T>& query_images,
                         const Output& ground_truth, ImageType img_type, int N,
                         double brute_time_per_query, std::ofstream& csv_file,
                         const std::string& dataset_name, bool enable_range_search = false,
                         double R = 0.0) {
    std::cout << "\n--- IVF-PQ Parameter Sweep ---" << std::endl;

    // Cross join of all parameter combinations to get exactly 50 configs
    // Parameters: k_clusters × nprobe × m_subvectors × nbits
    std::vector<std::tuple<int, int, int, int>> configs;

    if (img_type == IMG_MNIST) {
        // MNIST: 784 dimensions
        // m_subvectors must divide 784: factors are 1,2,4,7,8,14,16,28,49,56,98,112,196,392,784
        // Use: 4,7,8,14,16,28,49,56,98 (reasonable sub-vector sizes)

        // k_clusters × nprobe × m_subvectors × nbits = 50 configs
        const std::vector<int> k_clusters_values = {16, 32, 64, 128, 256}; // 5 values
        const std::vector<int> nprobe_values = {1, 2, 4, 8};               // 4 values
        const std::vector<int> m_subvectors_values = {
            7, 8, 14, 16, 28, 49};                       // 6 values (784/7=112, 784/8=98, etc.)
        const std::vector<int> nbits_values = {4, 6, 8}; // 3 values

        // To get exactly 50 configs, we'll use a strategic subset
        // 5 k_clusters × 2 nprobe × 5 m_subvectors = 50
        for (int k_clusters : k_clusters_values) {                // 5
            for (int nprobe : {1, 4}) {                           // 2
                for (int m_subvectors : {28, 49, 56, 112, 196}) { // 5
                    int nbits = 8;                                // Fixed at 8 bits for consistency
                    configs.emplace_back(k_clusters, nprobe, m_subvectors, nbits);
                }
            }
        }
    } else { // SIFT
        // SIFT: 128 dimensions
        // m_subvectors must divide 128: factors are 1,2,4,8,16,32,64,128
        // Use: 4,8,16,32,64 (reasonable sub-vector sizes)

        // k_clusters × nprobe × m_subvectors × nbits = 50 configs
        const std::vector<int> k_clusters_values = {8, 16, 32, 64, 128}; // 5 values
        const std::vector<int> nprobe_values = {1, 2, 4, 8};             // 4 values
        const std::vector<int> m_subvectors_values = {4, 8, 16, 32, 64}; // 5 values
        const std::vector<int> nbits_values = {4, 6, 8};                 // 3 values

        // To get exactly 50 configs: 5 k_clusters × 2 nprobe × 5 m_subvectors = 50
        for (int k_clusters : k_clusters_values) {             // 5
            for (int nprobe : {1, 4}) {                        // 2
                for (int m_subvectors : m_subvectors_values) { // 5
                    int nbits = 8;                             // Fixed at 8 bits for consistency
                    configs.emplace_back(k_clusters, nprobe, m_subvectors, nbits);
                }
            }
        }
    }

    std::cout << "  [Total configurations: " << configs.size() << "]" << std::endl;

    // OPTIMIZATION: Group by (k_clusters, m_subvectors, nbits) and build index once per group
    // Then test all nprobe values on the same index (nprobe only affects querying, not building)
    std::cout << "  [Optimized: building index once per (k_clusters, m_subvectors, nbits) group]"
              << std::endl;

    // Group configs by build parameters
    std::map<std::tuple<int, int, int>, std::vector<int>> build_groups;
    for (const auto& [k_clusters, nprobe, m_subvectors, nbits] : configs) {
        build_groups[{k_clusters, m_subvectors, nbits}].push_back(nprobe);
    }

    std::cout << "  [Index builds needed: " << build_groups.size() << " (down from "
              << configs.size() << ")]" << std::endl;

    for (const auto& [build_params, nprobe_values] : build_groups) {
        auto [k_clusters, m_subvectors, nbits] = build_params;
        int dims_per_subvector = input_images.get_cols() / m_subvectors;

        // Build IVF-PQ index once for this group
        auto start_build = std::chrono::high_resolution_clock::now();
        Ivfpq<T, TWithNegatives, uint8_t> index(input_images, k_clusters, dims_per_subvector, nbits,
                                                1);
        index.build();
        auto end_build = std::chrono::high_resolution_clock::now();
        double build_time = std::chrono::duration<double>(end_build - start_build).count();

        std::cout << "  Built index: k=" << k_clusters << " m=" << m_subvectors
                  << " nbits=" << nbits << " (build time: " << std::fixed << std::setprecision(2)
                  << build_time << "s)" << std::endl;

        // Test all nprobe values on this index
        for (int nprobe : nprobe_values) {
            // Query phase
            auto start = std::chrono::high_resolution_clock::now();

            Output output;
            output.algorithm = "IVF-PQ";
            output.queries.resize(query_images.get_rows());

            for (size_t i = 0; i < query_images.get_rows(); i++) {
                const T* query_point = query_images.get_row(i);
                auto candidates = index.get_candidates(query_point, nprobe, N);

                output.queries[i].nearest_neighbors.resize(N);
                for (size_t j = 0; j < std::min(candidates.size(), static_cast<size_t>(N)); j++) {
                    auto [dist, id] = candidates[j];
                    output.queries[i].nearest_neighbors[j].id = id;
                    output.queries[i].nearest_neighbors[j].distance_approximate = std::sqrt(dist);
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            double t_per_query = elapsed / query_images.get_rows();

            // Fill with ground truth
            fill_with_ground_truth(output, ground_truth);

            // Calculate metrics
            calculate_metrics(output, elapsed, 0.0);

            double avg_af = output.average_af;
            double recall = output.recall_at_n;
            double qps = output.queries_per_second;

            double speedup = brute_time_per_query / t_per_query;

            std::string params =
                "k=" + std::to_string(k_clusters) + " nprobe=" + std::to_string(nprobe) +
                " m=" + std::to_string(m_subvectors) + " nbits=" + std::to_string(nbits);
            if (enable_range_search) {
                params += " R=" + std::to_string(R);
            }

            std::cout << "    nprobe=" << nprobe << " | Recall=" << std::setprecision(3) << recall
                      << " | Speedup=" << std::setprecision(1) << speedup << "x" << std::endl;

            write_csv_result(csv_file, dataset_name, "IVF-PQ", params, avg_af, recall, qps,
                             t_per_query, speedup);
        }
    }
}

// ============================================================================
// MAIN BENCHMARK RUNNER
// ============================================================================

template <typename T>
void run_dataset_benchmark(const std::string& dataset_name, const std::string& input_path,
                           const std::string& query_path, const std::string& gt_cache_path,
                           ImageType img_type, int N = 10, bool enable_range_search = false,
                           bool run_lsh = true, bool run_hypercube = true, bool run_ivfflat = true,
                           bool run_ivfpq = true) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "BENCHMARK: " << dataset_name << std::endl;
    std::cout << "Input: " << input_path << std::endl;
    std::cout << "Query: " << query_path << std::endl;
    std::cout << "N=" << N << " nearest neighbors" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Load datasets
    std::cout << "\nLoading datasets..." << std::endl;
    std::unique_ptr<Matrix<T>> input_images;
    std::unique_ptr<Matrix<T>> query_images;

    if constexpr (std::is_same_v<T, uint8_t>) {
        input_images = load_mnist_data(input_path);
        query_images = load_mnist_data(query_path);
    } else if constexpr (std::is_same_v<T, float>) {
        input_images = load_sift_data(input_path);
        query_images = load_sift_data(query_path);
    }

    // Try to load cached ground truth
    std::cout << "\nChecking for cached ground truth..." << std::endl;
    auto ground_truth = load_ground_truth(gt_cache_path);
    double brute_time_per_query = 0;

    if (!ground_truth) {
        std::cout << "No cached ground truth found. Computing (this will take a while)..."
                  << std::endl;

        BruteforceArguments brute_args{};
        brute_args.common.number_of_nearest = N;
        brute_args.common.image_type = img_type;

        auto start = std::chrono::high_resolution_clock::now();
        ground_truth = brute_force_querying(*input_images, *query_images, brute_args);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        brute_time_per_query = elapsed / query_images->get_rows();

        std::cout << "Brute force completed in " << std::fixed << std::setprecision(2) << elapsed
                  << "s (" << std::setprecision(6) << brute_time_per_query << "s per query)"
                  << std::endl;

        // Save for future use
        save_ground_truth<T>(gt_cache_path, *ground_truth);
    } else {
        // FAST MODE: Use hardcoded baseline times from previous runs
        // Uncomment the section below to re-measure if needed
        /*
        std::cout << "Using cached baseline times (comment out to re-measure)..." << std::endl;

        if (img_type == IMG_MNIST) {
            brute_time_per_query = 0.125492;  // From previous MNIST run (60K dataset)
        } else {
            brute_time_per_query = 0.110198;  // From previous SIFT run (1M dataset)
        }

        std::cout << "Brute force baseline: " << std::setprecision(6)
                  << brute_time_per_query << "s per query (cached)" << std::endl;
        */
        /* UNCOMMENT THIS BLOCK TO RE-MEASURE BRUTE FORCE BASELINE:      */
        std::cout << "Measuring brute force baseline time..." << std::endl;
        BruteforceArguments brute_args{};
        brute_args.common.number_of_nearest = N;
        brute_args.common.image_type = img_type;

        auto start = std::chrono::high_resolution_clock::now();
        auto dummy = brute_force_querying(*input_images, *query_images, brute_args);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        brute_time_per_query = elapsed / query_images->get_rows();
        std::cout << "Brute force baseline: " << std::setprecision(6) << brute_time_per_query
                  << "s per query" << std::endl;
    }

    // Open CSV file for this dataset
    std::string csv_filename = "results/benchmark_" + dataset_name + ".csv";
    std::ofstream csv_file(csv_filename);

    // Write CSV header
    csv_file << "Dataset,Algorithm,Parameters,AvgAF,Recall,QPS,TimePerQuery,Speedup" << std::endl;

    // Determine appropriate radius for range search based on dataset
    double R = (img_type == IMG_MNIST) ? 2000.0 : 2.0;

    // Run k-nearest neighbor benchmarks
    std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         K-NEAREST NEIGHBOR BENCHMARKS (N=" << N << ")              ║"
              << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

    if (run_lsh) {
        run_lsh_benchmark(*input_images, *query_images, *ground_truth, img_type, N,
                          brute_time_per_query, csv_file, dataset_name, false, 0.0);
    }
    if (run_hypercube) {
        run_hypercube_benchmark(*input_images, *query_images, *ground_truth, img_type, N,
                                brute_time_per_query, csv_file, dataset_name, false, 0.0);
    }
    if (run_ivfflat) {
        run_ivfflat_benchmark(*input_images, *query_images, *ground_truth, img_type, N,
                              brute_time_per_query, csv_file, dataset_name, false, 0.0);
    }
    if (run_ivfpq) {
        if constexpr (std::is_same_v<T, uint8_t>) {
            run_ivfpq_benchmark<uint8_t, int8_t>(*input_images, *query_images, *ground_truth,
                                                 img_type, N, brute_time_per_query, csv_file,
                                                 dataset_name, false, 0.0);
        } else {
            run_ivfpq_benchmark<T, T>(*input_images, *query_images, *ground_truth, img_type, N,
                                      brute_time_per_query, csv_file, dataset_name, false, 0.0);
        }
    }

    // Run range search benchmarks (only if enabled)
    if (enable_range_search) {
        std::cout << "\n╔════════════════════════════════════════════════════════════╗"
                  << std::endl;
        std::cout << "║         RANGE SEARCH BENCHMARKS (R=" << R << ")                ║"
                  << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

        if (run_lsh) {
            run_lsh_benchmark(*input_images, *query_images, *ground_truth, img_type, N,
                              brute_time_per_query, csv_file, dataset_name, true, R);
        }
        if (run_hypercube) {
            run_hypercube_benchmark(*input_images, *query_images, *ground_truth, img_type, N,
                                    brute_time_per_query, csv_file, dataset_name, true, R);
        }
        if (run_ivfflat) {
            run_ivfflat_benchmark(*input_images, *query_images, *ground_truth, img_type, N,
                                  brute_time_per_query, csv_file, dataset_name, true, R);
        }
        if (run_ivfpq) {
            if constexpr (std::is_same_v<T, uint8_t>) {
                run_ivfpq_benchmark<uint8_t, int8_t>(*input_images, *query_images, *ground_truth,
                                                     img_type, N, brute_time_per_query, csv_file,
                                                     dataset_name, true, R);
            } else {
                run_ivfpq_benchmark<T, T>(*input_images, *query_images, *ground_truth, img_type, N,
                                          brute_time_per_query, csv_file, dataset_name, true, R);
            }
        }
    }

    csv_file.close();
    std::cout << "\n CSV results saved to: " << csv_filename << std::endl;

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Benchmark complete for " << dataset_name << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << R"(
╔════════════════════════════════════════════════════════════╗
║     NEAREST NEIGHBOR ALGORITHM BENCHMARK SUITE             ║
║  Pre-computed Ground Truth for Efficient Analysis          ║
╚════════════════════════════════════════════════════════════╝
)" << std::endl;

    // Default paths (can be overridden by command line)
    std::string mnist_input = "data/mnist/train-images.idx3-ubyte";
    std::string mnist_query = "data/mnist/t10k-images.idx3-ubyte";
    std::string sift_input = "data/sift/sift_base.fvecs";
    std::string sift_query = "data/sift/sift_query.fvecs";

    std::string gt_dir = "output/ground_truth";
    mkdir(gt_dir.c_str(), 0777);

    bool run_mnist = true;
    bool run_sift = true;
    bool enable_range_search = false;
    bool run_lsh = true;
    bool run_hypercube = true;
    bool run_ivfflat = true;
    bool run_ivfpq = true;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--mnist-only") {
            run_sift = false;
        } else if (arg == "--sift-only") {
            run_mnist = false;
        } else if (arg == "--range") {
            enable_range_search = true;
        } else if (arg == "--lsh-only") {
            run_hypercube = false;
            run_ivfflat = false;
            run_ivfpq = false;
        } else if (arg == "--hypercube-only") {
            run_lsh = false;
            run_ivfflat = false;
            run_ivfpq = false;
        } else if (arg == "--ivfflat-only") {
            run_lsh = false;
            run_hypercube = false;
            run_ivfpq = false;
        } else if (arg == "--ivfpq-only") {
            run_lsh = false;
            run_hypercube = false;
            run_ivfflat = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n"
                      << "Options:\n"
                      << "  --mnist-only      Run only MNIST benchmarks\n"
                      << "  --sift-only       Run only SIFT benchmarks\n"
                      << "  --lsh-only        Run only LSH benchmarks\n"
                      << "  --hypercube-only  Run only Hypercube benchmarks\n"
                      << "  --ivfflat-only    Run only IVF-Flat benchmarks\n"
                      << "  --ivfpq-only      Run only IVFPQ benchmarks\n"
                      << "  --range           Enable range search benchmarks\n"
                      << "  --help, -h        Show this help message\n"
                      << std::endl;
            return 0;
        }
    }

    // Run benchmarks
    if (run_mnist) {
        run_dataset_benchmark<uint8_t>(
            "MNIST", mnist_input, mnist_query, gt_dir + "/mnist_ground_truth_N10.bin", IMG_MNIST,
            10, enable_range_search, run_lsh, run_hypercube, run_ivfflat, run_ivfpq);
    }

    if (run_sift) {
        run_dataset_benchmark<float>(
            "SIFT", sift_input, sift_query, gt_dir + "/sift_ground_truth_N10.bin", IMG_SIFT, 10,
            enable_range_search, run_lsh, run_hypercube, run_ivfflat, run_ivfpq);
    }

    std::cout << "\nAll benchmarks completed successfully!" << std::endl;
    return 0;
}
