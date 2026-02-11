/**
 * @file benchmark.cpp
 * @brief Comprehensive ANN algorithm benchmarking suite with intelligent caching and parallelization
 * 
 * System Overview:
 * This benchmarking suite evaluates four approximate nearest neighbor algorithms (LSH, Hypercube,
 * IVF-Flat, IVF-PQ) across multiple parameter configurations on SIFT and MNIST datasets. The system
 * implements several key optimizations: ground truth caching to avoid redundant brute-force
 * computation, index reuse to minimize expensive build operations, batched parallelization for
 * concurrent testing while respecting system resources, and CSV output generation for automated
 * analysis and visualization. The goal is efficient parameter sweeps covering thousands of
 * configurations to identify optimal accuracy-speed tradeoffs.
 * 
 * Ground Truth Caching Strategy:
 * Computing ground truth via brute-force is the bottleneck in ANN benchmarking. For SIFT (1M
 * vectors × 128D) with 10K queries, brute-force takes approximately 100+ seconds. For MNIST (60K
 * vectors × 784D) with 10K queries, it takes approximately 12+ seconds. Since ground truth is
 * dataset-dependent but algorithm-independent, we compute it once and cache to binary files in
 * output/ground_truth/ directory. Subsequent benchmark runs load cached results in under 1 second,
 * enabling rapid iteration on algorithm parameters without waiting for validation. Cache files are
 * keyed by dataset name and N value (e.g., sift_ground_truth_N10.bin).
 * 
 * Index Reuse Optimization:
 * Some ANN algorithm parameters affect only querying, not index construction. For IVF-Flat, nprobe
 * (number of clusters to search) affects query time but index is determined by k_clusters alone.
 * For IVF-PQ, nprobe is query-time while k_clusters/m/nbits determine index structure. The system
 * groups configurations by index-determining parameters, builds each unique index once, then tests
 * all query-time parameter variations on that index. This reduces index builds from hundreds to
 * dozens, saving significant time. For example, testing 5 k_clusters values with 10 nprobe values
 * each requires only 5 index builds instead of 50.
 * 
 * Batched Parallelization:
 * Testing hundreds of configurations sequentially is slow. The system implements batched parallel
 * execution using std::async: determines optimal parallelism as 75% of CPU cores (leaving headroom
 * for OS and other processes), splits configurations into batches of optimal size, launches tasks
 * within each batch concurrently, waits for batch completion before starting next batch (prevents
 * memory explosion from too many concurrent index builds). This provides 4x-8x speedup on typical
 * multi-core systems while maintaining stability. Mutex-protected output ensures clean console and
 * CSV logging without interleaving.
 * 
 * Parameter Sweep Design:
 * Each algorithm undergoes comprehensive parameter sweeps exploring the accuracy-speed tradeoff
 * space. LSH sweeps k (hash functions per table), L (number of tables), w (bucket width) in
 * Cartesian product (3×4×3 = 36 configs for MNIST, similar for SIFT). Hypercube sweeps kproj
 * (hypercube dimensions), probes (vertices to search), M (candidate limit derived from probes)
 * yielding 5×10 = 50 configs per dataset. IVF-Flat sweeps k_clusters and nprobe yielding 5×10 = 50
 * configs. IVF-PQ sweeps k_clusters, nprobe, m (sub-vectors), nbits (bits per code) with strategic
 * subsets totaling 50 configs. Parameters are tuned differently for MNIST (high-dimensional,
 * integer) vs SIFT (moderate-dimensional, float) to explore relevant ranges.
 * 
 * Metrics Collection:
 * For each configuration, the system measures: recall@N (fraction of queries where true nearest
 * neighbor appears in approximate top-N, higher is better, target ≥ 0.90), average approximation
 * factor (mean ratio of approximate distance to true distance, lower is better, ideal = 1.0),
 * queries per second (throughput, higher is better, measures scalability), time per query (latency,
 * lower is better, measures responsiveness), speedup vs brute-force (ratio of brute-force time to
 * approximate time, higher is better, typical range 10x-100x). These metrics enable multi-objective
 * optimization: balance recall (accuracy) against QPS (speed).
 * 
 * CSV Output Format:
 * Results are written to results/benchmark_DATASET.csv with columns: Dataset, Algorithm, Parameters,
 * AvgAF, Recall, QPS, TimePerQuery, Speedup. Each row represents one parameter configuration's
 * performance. This format enables automated analysis via Python/pandas for generating plots
 * (recall vs QPS curves, parameter sensitivity analysis) and statistical summaries (best
 * configurations, Pareto frontiers). The accompanying plot_results.py script visualizes these
 * tradeoffs.
 * 
 * Threading and Safety:
 * Concurrent execution requires careful synchronization. The system uses mutex-protected console
 * output (cout_mutex prevents interleaved prints), mutex-protected CSV writing (csv_mutex ensures
 * atomic line writes), thread-safe progress reporting (each thread builds independent status
 * strings before acquiring lock), and thread-local result storage (each task computes metrics
 * independently before writing). This maintains correctness while maximizing parallelism.
 * 
 * Memory Management:
 * Benchmark runs can be memory-intensive when building large indices concurrently. The system
 * manages memory by batching parallel tasks (prevents too many concurrent index builds), reusing
 * indices across query parameter variations (amortizes memory cost), loading datasets once and
 * passing by reference (avoids copies), using smart pointers for automatic cleanup (prevents leaks).
 * Peak memory usage is dominated by dataset size (SIFT: 512MB, MNIST: 47MB) plus largest index
 * structure (IVF-PQ with many clusters and sub-vectors can reach hundreds of MB).
 * 
 * Command-Line Interface:
 * The program accepts flags for flexible testing: --mnist-only or --sift-only runs single dataset,
 * --lsh-only/--hypercube-only/--ivfflat-only/--ivfpq-only tests single algorithm, --range enables
 * radius search benchmarks (finds all neighbors within distance R, more expensive than top-N),
 * --help displays usage information. Default behavior runs all algorithms on both datasets with
 * top-N search only.
 * 
 * Benchmark Workflow:
 * Typical workflow: first run with ground truth computation (cache miss) taking 100+ seconds for
 * full validation, subsequent runs load cached ground truth in <1 second and proceed directly to
 * algorithm testing, each algorithm completes parameter sweep in minutes (depending on parallelism),
 * CSV results accumulate incrementally during execution, final visualization via plot_results.py
 * generates recall-QPS curves and parameter analysis. This workflow enables rapid experimentation
 * with algorithm tuning.
 * 
 * Integration with Main Application:
 * This benchmark suite is separate from the main.cpp query interface. Main.cpp is for single-shot
 * queries with specific parameters (production use case), benchmark.cpp is for systematic evaluation
 * across parameter ranges (research/tuning use case). They share underlying algorithm implementations
 * and file I/O but have different purposes and interfaces.
 * 
 * @authors Τσεκρέκος Έγκορ-Ανδριανός, Δημακόπουλος Θεόδωρος
 */

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
// This section implements intelligent parallelization that adapts to system resources. The goal is
// to maximize throughput without oversubscribing CPU cores (which causes context switching overhead)
// or exhausting memory (from too many concurrent index builds). The system queries hardware
// concurrency, applies a 75% heuristic (leaving headroom for OS and background tasks), and batches
// configurations to control memory usage.

// Determine optimal number of parallel tasks based on system resources. Returns 75% of hardware
// concurrency with minimum 1 and maximum equal to actual core count. The 75% heuristic prevents
// CPU oversubscription: on 8-core system, uses 6 threads leaving 2 cores for OS, I/O, and other
// processes. This typically provides better throughput than 100% utilization due to reduced
// context switching and thermal throttling. Falls back to 4 threads if hardware_concurrency fails.
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

// Execute configurations in controlled parallel batches to manage system resources. This function
// implements batched parallelization: splits total configurations into batches of size max_parallel,
// launches all tasks in current batch concurrently using std::async, waits for batch completion
// before starting next batch. This prevents memory explosion from too many concurrent index builds
// (each IVF-PQ index can use hundreds of MB) while still achieving good parallelism. Sequential
// batching also simplifies progress tracking and error handling compared to fully asynchronous
// execution. Template parameters: ConfigType is the configuration struct type (tuple of parameters),
// TaskFunc is the callable that processes one configuration (typically a lambda capturing benchmark
// context).
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
// Concurrent tasks writing to shared output streams (stdout, CSV file) require synchronization to
// prevent interleaved writes. These mutexes and wrapper functions ensure atomic operations: each
// thread formats its complete message/line before acquiring the lock, writes atomically, releases
// lock. This maintains output integrity without forcing all computation to be sequential.

std::mutex cout_mutex; // Protects std::cout for console output
std::mutex csv_mutex;  // Protects CSV file writes

// Print a message to console with thread-safety. Formats the message completely before acquiring
// the mutex, ensuring minimal lock duration. The flush ensures output appears immediately rather
// than buffering indefinitely (important for progress monitoring).
void thread_safe_print(const std::string& message) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << message << std::flush;
}

// Write a line to CSV file with thread-safety. Similar to thread_safe_print but for file I/O.
// Each thread constructs its complete CSV line before acquiring the lock, ensuring atomic line
// writes (no partial lines or interleaving). The flush ensures data is written to disk immediately
// for crash resilience (if benchmark crashes mid-run, completed results are preserved).
void thread_safe_csv_write(std::ofstream& csv_file, const std::string& line) {
    std::lock_guard<std::mutex> lock(csv_mutex);
    csv_file << line << std::flush;
}

// ============================================================================
// GROUND TRUTH CACHE
// ============================================================================
// Ground truth computation (brute-force nearest neighbor search) is the major bottleneck in ANN
// benchmarking. For SIFT with 1M base vectors and 10K queries, brute-force takes 100+ seconds. For
// MNIST with 60K vectors and 10K queries, it takes 12+ seconds. Since ground truth depends only on
// dataset and N (not algorithm or parameters), we compute once and cache to binary files. Subsequent
// runs load cached results in under 1 second, enabling rapid parameter exploration. Cache format is
// simple binary serialization: number of queries, then for each query (number of neighbors, then
// id_true and distance_true pairs). Template parameter T is unused but maintained for consistency
// with dataset type.

// Save ground truth results to binary file for future reuse. Serializes the nearest neighbor results
// (id_true and distance_true fields) in binary format. Does NOT save approximate results (id and
// distance_approximate) since those are algorithm-specific. File format: uint64 num_queries, then
// for each query (uint64 num_neighbors, then num_neighbors pairs of (int id_true, double distance_true)).
// Uses binary for compactness and speed (text format would be 10x larger and slower to parse).
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

// Load pre-computed ground truth from binary file. Returns nullptr if file doesn't exist or read
// fails (cache miss). On success, returns Output structure with id_true and distance_true fields
// populated, id and distance_approximate left empty. The caller will fill approximate results by
// running ANN algorithms. Binary deserialization is fast (loading SIFT ground truth with 10K queries
// × 10 neighbors takes <1 second) compared to recomputing which takes 100+ seconds.
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
// These helper functions handle common benchmarking tasks: filling approximate results with ground
// truth for metrics calculation, formatting benchmark results for console and CSV output. They
// abstract repetitive operations and ensure consistent formatting across all algorithms.

// Fill approximate results with ground truth data to enable metrics calculation. After running an
// ANN algorithm, the Output structure has id and distance_approximate filled but id_true and
// distance_true are empty. This function copies id_true and distance_true from the cached ground
// truth into the approximate output, enabling calculate_metrics() to compute recall and approximation
// factor by comparing approximate to true results. Handles mismatched sizes gracefully (if approximate
// returned fewer than N neighbors due to small candidate pool).
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

// Format and print benchmark result to console in aligned table format. Displays algorithm name,
// parameter string, and key metrics (average approximation factor, recall, QPS, time, speedup)
// in fixed-width columns for readability. Uses thread-safe printing to prevent interleaving when
// multiple configurations complete concurrently. The formatted output enables quick visual assessment
// of results during benchmark execution.
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

// Write benchmark result to CSV file for automated analysis and plotting. One line per configuration
// with comma-separated values matching the CSV header. Uses thread-safe writing to prevent line
// interleaving or corruption when multiple threads write concurrently. The CSV format enables
// pandas/matplotlib analysis via plot_results.py for generating recall-QPS curves and parameter
// sensitivity plots. Fixed precision ensures consistent numeric formatting across runs.
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
// These functions implement parameter sweeps for each ANN algorithm. Each function explores the
// accuracy-speed tradeoff space by testing Cartesian products of parameter ranges. Parameter values
// are tuned per-dataset (MNIST vs SIFT have different characteristics requiring different ranges).
// All functions follow the same pattern: generate configuration list, define task lambda capturing
// context, run batched parallel execution, collect and output metrics.

// Run LSH parameter sweep testing combinations of k (hash functions per table), L (number of tables),
// and w (bucket width). The Cartesian product explores how these parameters affect the fundamental
// LSH tradeoff: larger k gives better filtering per table but slower hashing, larger L gives higher
// recall but more memory and slower queries, larger w gives wider buckets (more collisions, higher
// recall but more candidates). MNIST uses different ranges than SIFT due to dimensionality and
// distance distribution differences. Template parameter T is the vector element type (uint8_t for
// MNIST, float for SIFT).
template <typename T>
void run_lsh_benchmark(const Matrix<T>& input_images, const Matrix<T>& query_images,
                       const Output& ground_truth, ImageType img_type, int N,
                       double brute_time_per_query, std::ofstream& csv_file,
                       const std::string& dataset_name, bool enable_range_search = false,
                       double R = 0.0) {
    std::cout << "\n--- LSH Parameter Sweep ---" << std::endl;

    // Generate all parameter combinations via Cartesian product. LSH has three key parameters that
    // interact: k controls per-table selectivity (higher k = fewer candidates per table but requires
    // closer match), L controls multi-table redundancy (higher L = more chances to find neighbors),
    // w controls bucket granularity (higher w = coarser buckets, more collisions). The cross join
    // explores how these interact to affect recall and speed. Parameter ranges are tuned based on
    // dataset characteristics and empirical testing.
    std::vector<std::tuple<int, int, double>> configs;

    if (img_type == IMG_MNIST) {
        // MNIST parameter ranges: k × L × w = 3 × 4 × 3 = 36 configurations. MNIST is high-dimensional
        // (784D) with integer pixel values (0-255), so bucket widths are large (1000-3000) relative
        // to typical Euclidean distances (100-500). Small k values (2-8) work because dimensionality
        // provides natural selectivity. L values (4-14) balance recall improvement against memory cost.
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
        // SIFT parameter ranges: k × L × w = 3 × 4 × 3 = 36 configurations. SIFT is moderate-dimensional
        // (128D) with float values (typically 0-500), so bucket widths are smaller (175-400) relative
        // to typical distances (10-100). Higher k values (4-10) are beneficial due to lower dimensionality.
        // L values similar to MNIST. The different ranges reflect the different distance distributions
        // and computational costs between uint8_t and float arithmetic.
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

    // Execute all configurations in batched parallel mode. Each task is a lambda that captures the
    // benchmark context (datasets, ground truth, file handles) and processes one parameter configuration:
    // builds LSH index, runs queries, fills ground truth, calculates metrics, outputs results. The
    // batched execution provides parallelism (4-8x speedup on multi-core systems) while controlling
    // resource usage (prevents memory exhaustion from too many concurrent index builds).
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
        // Seed is fixed at 1 for reproducibility. This ensures LSH builds identical hash functions
        // across runs, enabling fair comparison. Could be exposed as CLI parameter for testing
        // variance across random seeds, but fixed seed is standard for benchmarking.
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

// Run Hypercube parameter sweep testing combinations of kproj (hypercube dimensions), probes (vertices
// to search), and M (maximum candidates, derived from probes). The hypercube algorithm embeds vectors
// into {0,1}^kproj vertices, then searches multiple vertices via BFS multi-probe. Key tradeoffs:
// larger kproj gives more vertices (2^kproj) so higher recall but exponential growth limits practical
// range (typically 8-16), larger probes searches more vertices (higher recall, slower), M derived as
// probes × multiplier (40 for MNIST, 15 for SIFT) to limit candidates per vertex. The cross join
// explores kproj × probes = 5 × 10 = 50 configurations per dataset.
template <typename T>
void run_hypercube_benchmark(const Matrix<T>& input_images, const Matrix<T>& query_images,
                             const Output& ground_truth, ImageType img_type, int N,
                             double brute_time_per_query, std::ofstream& csv_file,
                             const std::string& dataset_name, bool enable_range_search = false,
                             double R = 0.0) {
    std::cout << "\n--- Hypercube Parameter Sweep ---" << std::endl;

    // Generate configurations with M derived from probes. Hypercube has three parameters but M is
    // derived rather than independent: M = min(dataset_size, probes × multiplier) where multiplier
    // depends on dataset. The multiplier controls how many candidates to collect per probed vertex.
    // Higher multiplier gives more candidates (better recall, slower), lower multiplier gives faster
    // queries. Different multipliers for MNIST (40) vs SIFT (15) reflect different cluster densities.
    std::vector<std::tuple<int, int, int>> configs;

    if (img_type == IMG_MNIST) {
        // MNIST Hypercube ranges: kproj × probes = 5 × 10 = 50 configurations. kproj values (8-16)
        // span practical hypercube dimensions (2^8=256 to 2^16=65536 vertices). probes values (8-2048)
        // explore multi-probe aggressiveness from minimal (8 vertices) to extensive (2048 vertices).
        // M derived as min(60000, probes × 40) balances candidate collection against memory.
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
        // SIFT Hypercube ranges: same kproj and probes as MNIST (5 × 10 = 50 configurations) but
        // different M multiplier (15 instead of 40). SIFT has sparser hypercube population due to
        // float vectors and different distance distribution, so lower M multiplier suffices. Also
        // limits memory usage for the larger SIFT dataset (1M vs 60K vectors).
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

// Run IVF-Flat parameter sweep with intelligent index reuse. IVF-Flat has two parameters: k_clusters
// (determines index structure via k-means) and nprobe (query-time parameter selecting how many
// clusters to search). Key optimization: nprobe doesn't affect index structure, so we group
// configurations by k_clusters, build each unique index once, then test all nprobe values on that
// index. This reduces index builds from k_clusters × nprobe (e.g., 5 × 10 = 50) to just k_clusters
// (e.g., 5), saving massive time since k-means clustering is expensive. The cross join explores
// k_clusters × nprobe = 5 × 10 = 50 configurations per dataset.
template <typename T>
void run_ivfflat_benchmark(const Matrix<T>& input_images, const Matrix<T>& query_images,
                           const Output& ground_truth, ImageType img_type, int N,
                           double brute_time_per_query, std::ofstream& csv_file,
                           const std::string& dataset_name, bool enable_range_search = false,
                           double R = 0.0) {
    std::cout << "\n--- IVF-Flat Parameter Sweep ---" << std::endl;

    // Generate configurations for k_clusters × nprobe cross join. IVF-Flat partitions the dataset
    // into k Voronoi cells via k-means, then searches nprobe nearest cells during queries. Key
    // tradeoffs: larger k_clusters gives finer partitioning (smaller cells, faster queries per cell)
    // but slower build and more overhead, larger nprobe searches more cells (higher recall, slower
    // queries). The ranges explore practical values from coarse (16 clusters) to fine (256 clusters)
    // partitioning, and from aggressive (1 probe) to exhaustive (32 probes) search.
    std::vector<std::tuple<int, int>> configs;

    if (img_type == IMG_MNIST) {
        // MNIST IVF-Flat ranges: k_clusters × nprobe = 5 × 10 = 50 configurations. k_clusters
        // (16-256) spans from very coarse (16 clusters, ~3750 vectors per cluster) to fine (256
        // clusters, ~234 vectors per cluster) partitioning of the 60K dataset. nprobe (1-32) spans
        // from very aggressive single-probe search to extensive multi-probe. These ranges balance
        // exploration of the accuracy-speed tradeoff space against benchmark runtime.
        const std::vector<int> k_clusters_values = {16, 32, 64, 128, 256};
        const std::vector<int> nprobe_values = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32};

        for (int k_clusters : k_clusters_values) {
            for (int nprobe : nprobe_values) {
                configs.emplace_back(k_clusters, nprobe);
            }
        }
    } else { // SIFT
        // SIFT IVF-Flat ranges: same k_clusters and nprobe as MNIST (5 × 10 = 50 configurations).
        // For the 1M SIFT dataset, k_clusters (16-256) gives partitions from very coarse (16 clusters,
        // ~62500 vectors per cluster) to fine (256 clusters, ~3900 vectors per cluster). Same nprobe
        // values as MNIST. The ranges are intentionally reduced from theoretical optimal (k ≈ sqrt(n)
        // ≈ 1000 for 1M) to keep benchmark runtime manageable while still covering key tradeoffs.
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
// This function orchestrates the complete benchmark workflow for a single dataset. It handles ground
// truth caching (load if exists, compute and save if not), brute-force baseline measurement (for
// speedup calculation), CSV file initialization, and sequential execution of all enabled algorithms.
// The function is templated on vector element type T to support both SIFT (float) and MNIST (uint8_t)
// through compile-time polymorphism, avoiding code duplication and runtime overhead.

// Run comprehensive benchmark suite on a single dataset. Loads dataset, manages ground truth caching,
// measures brute-force baseline, then runs all enabled algorithms with their parameter sweeps. Results
// are written to CSV file incrementally (one line per configuration as it completes) for crash
// resilience and progress monitoring. Template parameter T is the vector element type (float for
// SIFT, uint8_t for MNIST), enabling type-safe operations without virtual function overhead.
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
