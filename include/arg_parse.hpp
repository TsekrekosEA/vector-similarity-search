/**
 * @file arg_parse.hpp
 * @brief Command-line argument parsing for ANN algorithm configurations
 * 
 * Argument Structure Design:
 * The argument parsing system uses a hierarchy of structs to organize parameters. Common parameters
 * shared by all algorithms (input file, query file, output file, N for top-N search, R for radius
 * search) are grouped in AlgorithmIndependentArguments. Each algorithm has its own struct
 * (LshArguments, HypercubeArguments, IvfflatArguments, IvfpqArguments) containing a common field
 * plus algorithm-specific parameters. This design avoids parameter bloat in main() and provides
 * type safety through distinct structures.
 * 
 * Parameter Categories:
 * Common parameters apply to all algorithms: input_file is the dataset binary file path,
 * query_file is the query vectors binary file path, output_file is the results text file path,
 * number_of_nearest is N for top-N search, radius is R for range search, search_for_range enables
 * range search mode, image_type determines dataset format (SIFT or MNIST). Algorithm-specific
 * parameters tune each method: LSH uses k (hash functions per table), L (number of tables), w
 * (bucket width); Hypercube uses kproj (number of projections), M (max candidates), probes (number
 * of vertices to probe); IVF-Flat uses kclusters (number of Voronoi cells), nprobe (clusters to
 * search); IVF-PQ adds M (sub-vectors), nbits (bits per code).
 * 
 * Image Type Detection:
 * The image_type field is automatically inferred from file extensions: .fvecs or .bvecs files are
 * SIFT (float/byte vectors), .idx-ubyte or .idx3-ubyte files are MNIST (IDX format). This allows
 * the main function to instantiate the correct template specialization (float vs uint8_t) without
 * requiring explicit type flags. The detection happens during argument parsing and is stored in
 * the common struct.
 * 
 * Seed Parameter:
 * All randomized algorithms (LSH, Hypercube, IVF-Flat, IVF-PQ) accept a seed parameter for
 * reproducibility. This controls random projection vector generation (LSH, Hypercube) and k-means
 * centroid initialization (IVF-Flat, IVF-PQ). Setting the same seed ensures identical index
 * construction across runs, critical for debugging and fair benchmarking.
 * 
 * Range Search Mode:
 * The search_for_range flag enables R-near neighbor search in addition to top-N search. When
 * enabled, the algorithm returns all points within distance R of the query (stored in
 * r_near_neighbors field). This is more expensive than top-N (must examine more candidates) but
 * useful for applications requiring all neighbors within a threshold. The radius parameter R
 * is specified via the -R flag.
 * 
 * Error Handling:
 * The parse functions validate required arguments and can call exit(-1) on missing or invalid
 * parameters. The get_algorithm_argument function ensures an algorithm flag (-lsh, -hypercube,
 * -ivfflat, -ivfpq) is present before detailed parsing begins. Zeroing the structs (memset to 0)
 * provides default values and allows detecting unset fields.
 * 
 * Future Improvements:
 * The comment block notes desired refactorings: placing help messages near struct members for
 * maintainability, parsing image type before algorithm arguments to enable better validation,
 * allowing partial argument parsing (parsing unrelated flags without errors). These would improve
 * code organization and user experience but don't affect current functionality.
 * 
 * @authors Δημακόπουλος Θεόδωρος
 */

#pragma once

/*
Argument parsing should be re-written.

- Ideally, the help messages should be somewhere near the struct/class members.
- Algorithm functions shouldn't take file names or image types as arguments.
- Arguments that don't match the algorithm should be parsed anyway.
- The image type should be parsed before the algorithm's arguments are prepared.
*/

// Detect which algorithm the user wants to run based on command-line flag. Returns pointer to the
// algorithm flag string (-lsh, -hypercube, -ivfflat, or -ivfpq). This function performs early
// validation before expensive argument parsing, exiting with error message if no valid algorithm
// flag is found. Can call exit(-1) on missing or invalid algorithm.
char* get_algorithm_argument(int argc, char** argv);

// Initialization note: setting these structs to zero (memset) marks all fields as unset, providing
// default values and enabling detection of missing required parameters.

// Image dataset type enumeration. Determines binary format parser and template instantiation type.
// IMG_SIFT uses .fvecs format with float elements, IMG_MNIST uses .idx format with uint8_t elements.
// IMG_UNSET is an internal marker for uninitialized state.
enum ImageType {
    IMG_UNSET, // Internal state, not set by user
    IMG_SIFT,  // SIFT dataset (.fvecs), float vectors
    IMG_MNIST  // MNIST dataset (.idx), uint8_t vectors
};

// Parameters common to all algorithms. Stores dataset paths, search configuration, and detected
// image type. Embedded in all algorithm-specific structs to avoid parameter duplication.
struct AlgorithmIndependentArguments {
    char *input_file;   // Dataset binary file path (.fvecs or .idx)
    char *query_file;   // Query vectors binary file path (same format as input)
    char *output_file;  // Results text file path
    int number_of_nearest; // N for top-N nearest neighbor search
    int radius;         // R for range search (find all within distance R)
    ImageType image_type; // Dataset format (SIFT or MNIST), auto-detected from file extension
    bool search_for_range; // Enable R-near neighbor search in addition to top-N
    bool range_bool_has_been_set; // Internal flag to track if range parameters were provided
};

// LSH algorithm parameters. Configures L hash tables with k hash functions each. Larger L gives
// better recall but more memory, larger k gives better candidate filtering but slower hashing.
struct LshArguments {
    AlgorithmIndependentArguments common; // Shared parameters
    int k;      // Number of hash functions per table (amplification within table)
    int L;      // Number of hash tables (amplification across tables)
    double w;   // Bucket width for LSH hash functions (controls quantization)
    int seed;   // Random seed for projection vector generation
};

// Hypercube algorithm parameters. Uses kproj binary hash functions to embed vectors into hypercube
// vertices, then searches multiple vertices via BFS-like probing. M limits candidates per vertex.
struct HypercubeArguments {
    AlgorithmIndependentArguments common; // Shared parameters
    int kproj;  // Number of projections (dimension of hypercube, typically 10-16)
    int M;      // Maximum candidates to examine (early stopping for speed)
    int probes; // Number of hypercube vertices to probe (multi-probe search)
    double w;   // Bucket width (accepted per spec but not used in current implementation)
    int seed;   // Random seed for projection vector generation
};

// IVF-Flat algorithm parameters. Partitions dataset into kclusters Voronoi cells via k-means, then
// searches nprobe nearest cells. Larger kclusters gives finer partitioning (faster queries, slower
// build), larger nprobe gives higher recall (more cells searched).
struct IvfflatArguments {
    AlgorithmIndependentArguments common; // Shared parameters
    int kclusters; // Number of Voronoi cells (coarse quantization clusters)
    int nprobe;    // Number of nearest clusters to search during queries
    int seed;      // Random seed for k-means centroid initialization
};

// IVF-PQ algorithm parameters. Combines IVF-Flat's coarse quantization with Product Quantization's
// fine quantization. M is number of sub-vectors (must divide dimensionality evenly), nbits is bits
// per code (codebook size = 2^nbits, typically 4 or 8). Larger M gives more compression, larger
// nbits gives better accuracy.
struct IvfpqArguments {
    AlgorithmIndependentArguments common; // Shared parameters
    int kclusters; // Number of coarse Voronoi cells
    int nprobe;    // Number of coarse clusters to search
    int seed;      // Random seed for k-means initialization
    int M;         // Number of sub-vectors (product quantization granularity)
    int nbits;     // Bits per code (determines codebook size 2^nbits)
};

// Brute-force algorithm parameters. Used for ground truth computation via exhaustive linear scan.
// The fraction_of_pixels parameter allows early termination after examining a fraction of dimensions
// (useful for approximate brute-force in very high dimensions, though rarely used).
struct BruteforceArguments {
    AlgorithmIndependentArguments common; // Shared parameters
    double fraction_of_pixels; // Fraction of dimensions to use (1.0 = full distance, rarely changed)
};

// Parse LSH-specific arguments from command line. Expects flags: -d input_file, -q query_file,
// -o output_file, -N number_of_nearest, -R radius, -k hash_functions, -L tables, -w bucket_width,
// -seed random_seed. Validates presence of required arguments and exits on error.
void parse_lsh_arguments(LshArguments* a, int argc, char** argv);

// Parse Hypercube-specific arguments. Expects flags: -d, -q, -o, -N, -R, -k kproj, -M max_candidates,
// -probes num_vertices, -w bucket_width, -seed random_seed. Note: -k here means kproj (projections),
// different from LSH's k (hash functions per table).
void parse_hypercube_arguments(HypercubeArguments* a, int argc, char** argv);

// Parse IVF-Flat-specific arguments. Expects flags: -d, -q, -o, -N, -R, -k kclusters, -nprobe
// num_clusters, -seed random_seed. Note: -k here means kclusters (number of Voronoi cells).
void parse_ivfflat_arguments(IvfflatArguments* a, int argc, char** argv);

// Parse IVF-PQ-specific arguments. Expects flags: -d, -q, -o, -N, -R, -k kclusters, -nprobe
// num_clusters, -M subvectors, -nbits bits_per_code, -seed random_seed. Combines IVF and PQ parameters.
void parse_ivfpq_arguments(IvfpqArguments* a, int argc, char** argv);

// Parse brute-force arguments. Expects flags: -d, -q, -o, -N, -R. The fraction_of_pixels parameter
// is rarely used (defaults to 1.0 for full distance computation).
void parse_bruteforce_arguments(BruteforceArguments* a, int argc, char** argv);
