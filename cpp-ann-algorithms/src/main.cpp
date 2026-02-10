/**
 * @file main.cpp
 * @brief Main entry point for the ANN search engine
 *
 * This program implements a command-line interface for Approximate Nearest Neighbor (ANN) search
 * on high-dimensional vector datasets. It supports four algorithms — LSH, Hypercube, IVF-Flat,
 * and IVF-PQ — each templated on element type (float for SIFT, uint8_t for MNIST) for type-safe,
 * zero-overhead polymorphism.
 *
 * Architecture:
 *   1. Parse CLI arguments to determine algorithm and parameters
 *   2. Load dataset/query files from binary format (SIFT .fvecs or MNIST .idx)
 *   3. Run the selected ANN algorithm (timed) to populate approximate results
 *   4. Run brute-force exhaustive search (timed) to fill ground truth
 *   5. Compute metrics: Recall@N, Average AF, QPS, speedup
 *   6. Write results to output file
 *
 * The two-phase benchmarking methodology (steps 3-4) enables measuring ANN speedup and accuracy
 * independently. Template dispatch via run_algorithm<T>() eliminates the need for per-type
 * code duplication.
 *
 * @authors Egor-Andrianos Tsekrekos, Theodoros Dimakopoulos
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>

#include <arg_parse.hpp>
#include <brute_force.hpp>
#include <file_io.hpp>
#include <hypercube.hpp>
#include <ivfflat.hpp>
#include <ivfpq.hpp>
#include <lsh.hpp>
#include <utils.hpp>

// ---------------------------------------------------------------------------
// Core benchmark pipeline: run ANN → brute-force → metrics → output
// ---------------------------------------------------------------------------
// Accepts a callable that performs the approximate search and returns Output.
// This factored design eliminates per-algorithm, per-type duplication.
template <typename T>
static void run_algorithm(const Matrix<T>& input, const Matrix<T>& queries,
                          AlgorithmIndependentArguments& common,
                          std::function<std::unique_ptr<Output>()> search_fn) {
    // Phase 1: approximate search (timed)
    auto t_approx_start = std::chrono::high_resolution_clock::now();
    std::unique_ptr<Output> output = search_fn();
    auto t_approx_end = std::chrono::high_resolution_clock::now();

    // Phase 2: brute-force ground truth (timed)
    BruteforceArguments brute_args{};
    brute_args.common = common;
    auto t_true_start = std::chrono::high_resolution_clock::now();
    output = brute_force_querying<T>(input, queries, brute_args, std::move(output));
    auto t_true_end = std::chrono::high_resolution_clock::now();

    // Compute metrics and write results
    std::chrono::duration<double> t_approx = t_approx_end - t_approx_start;
    std::chrono::duration<double> t_true = t_true_end - t_true_start;
    calculate_metrics(*output, t_approx.count(), t_true.count());

    auto s = initialize_output_file(common.output_file);
    write_output(*s, *output);
}

// ---------------------------------------------------------------------------
// IVF-style query loop (shared by IVF-Flat and IVF-PQ)
// ---------------------------------------------------------------------------
// Runs get_candidates on each query and populates output with nearest neighbors
// and optional range results. Factored from the two IVF branches.
template <typename T, typename Index>
static std::unique_ptr<Output> ivf_query_loop(Index& index, const Matrix<T>& queries,
                                              const Matrix<T>& input,
                                              const AlgorithmIndependentArguments& common,
                                              const char* algorithm_name, int nprobe) {
    auto output = std::make_unique<Output>();
    output->algorithm = algorithm_name;
    output->queries.resize(queries.get_rows());

    for (size_t qid = 0; qid < queries.get_rows(); qid++) {
        const T* qvec = queries.get_row(qid);
        auto candidates = index.get_candidates(qvec, nprobe, common.number_of_nearest);

        auto& nn = output->queries[qid].nearest_neighbors;
        nn.resize(common.number_of_nearest);
        for (int i = 0; i < common.number_of_nearest && i < static_cast<int>(candidates.size()); i++) {
            auto [dist_sq, id] = candidates[i];
            nn[i].id = id;
            nn[i].distance_approximate = std::sqrt(dist_sq);
        }

        if (common.search_for_range) {
            auto all = index.get_candidates(qvec, nprobe, input.get_rows());
            for (auto [dist_sq, id] : all) {
                if (std::sqrt(dist_sq) <= common.radius && id >= 0) {
                    output->queries[qid].r_near_neighbors.push_back(id);
                }
            }
        }
    }
    return output;
}

// ---------------------------------------------------------------------------
// Dataset type dispatch: loads data and delegates to the algorithm runner
// ---------------------------------------------------------------------------
// Handles the SIFT/MNIST branch so each algorithm block needs only one call.
template <typename ArgsT, typename Fn>
static void dispatch_by_type(ArgsT& a, Fn run_for_type) {
    if (a.common.image_type == IMG_SIFT) {
        auto input = load_sift_data(a.common.input_file);
        auto queries = load_sift_data(a.common.query_file);
        run_for_type(*input, *queries, a);
    } else if (a.common.image_type == IMG_MNIST) {
        auto input = load_mnist_data(a.common.input_file);
        auto queries = load_mnist_data(a.common.query_file);
        run_for_type(*input, *queries, a);
    } else {
        std::fprintf(stderr, "%s:%d: unsupported dataset format.\n", __FILE__, __LINE__);
    }
}

// ===========================================================================
int main(int argc, char** argv) {
    char* alg = get_algorithm_argument(argc, argv);

    if (std::strcmp(alg, "-lsh") == 0) {
        LshArguments a{};
        parse_lsh_arguments(&a, argc, argv);
        dispatch_by_type(a, [](auto& input, auto& queries, auto& args) {
            using T = std::remove_reference_t<decltype(input.unchecked(0, 0))>;
            run_algorithm<T>(input, queries, args.common, [&]() {
                return lsh_querying<T>(input, queries, args);
            });
        });

    } else if (std::strcmp(alg, "-hypercube") == 0) {
        HypercubeArguments a{};
        parse_hypercube_arguments(&a, argc, argv);
        dispatch_by_type(a, [](auto& input, auto& queries, auto& args) {
            using T = std::remove_reference_t<decltype(input.unchecked(0, 0))>;
            run_algorithm<T>(input, queries, args.common, [&]() {
                return hypercube_querying<T>(input, queries, args);
            });
        });

    } else if (std::strcmp(alg, "-ivfflat") == 0) {
        IvfflatArguments a{};
        parse_ivfflat_arguments(&a, argc, argv);
        dispatch_by_type(a, [](auto& input, auto& queries, auto& args) {
            using T = std::remove_reference_t<decltype(input.unchecked(0, 0))>;
            run_algorithm<T>(input, queries, args.common, [&]() {
                Ivfflat<T> index(input, args.kclusters, args.seed);
                index.build();
                return ivf_query_loop<T>(index, queries, input, args.common,
                                         "IVFFlat", args.nprobe);
            });
        });

    } else if (std::strcmp(alg, "-ivfpq") == 0) {
        IvfpqArguments a{};
        parse_ivfpq_arguments(&a, argc, argv);
        dispatch_by_type(a, [](auto& input, auto& queries, auto& args) {
            using T = std::remove_reference_t<decltype(input.unchecked(0, 0))>;
            int dims_per_sub = input.get_cols() / args.M;
            run_algorithm<T>(input, queries, args.common, [&]() {
                // uint8_t residuals need int8_t to avoid underflow; float stays float
                if constexpr (std::is_same_v<T, uint8_t>) {
                    Ivfpq<uint8_t, int8_t, uint8_t> index(input, args.kclusters,
                                                          dims_per_sub, args.nbits, args.seed);
                    index.build();
                    return ivf_query_loop<T>(index, queries, input, args.common,
                                             "IVFPQ", args.nprobe);
                } else {
                    Ivfpq<float, float, uint8_t> index(input, args.kclusters,
                                                       dims_per_sub, args.nbits, args.seed);
                    index.build();
                    return ivf_query_loop<T>(index, queries, input, args.common,
                                             "IVFPQ", args.nprobe);
                }
            });
        });

    } else if (std::strcmp(alg, "-bruteforce") == 0) {
        BruteforceArguments a{};
        parse_bruteforce_arguments(&a, argc, argv);
        dispatch_by_type(a, [](auto& input, auto& queries, auto& args) {
            using T = std::remove_reference_t<decltype(input.unchecked(0, 0))>;
            auto t_start = std::chrono::high_resolution_clock::now();
            auto output = brute_force_querying<T>(input, queries, args);
            auto t_end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = t_end - t_start;
            output->t_true_average = elapsed.count() / queries.get_rows();
            output->queries_per_second = queries.get_rows() / elapsed.count();

            auto s = initialize_output_file(args.common.output_file);
            write_output(*s, *output);
        });

    } else {
        std::fprintf(stderr, "Unknown algorithm flag: '%s'\n", alg);
        return 1;
    }

    return 0;
}