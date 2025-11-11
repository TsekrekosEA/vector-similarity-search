#pragma once

/*
Argument parsing should be re-written.

- Ideally, the help messages should be somewhere near the struct/class members.
- Algorithm functions shouldn't take file names or image types as arguments.
- Arguments that don't match the algorithm should be parsed anyway.
- The image type should be parsed before the algorithm's arguments are prepared.
*/

// can exit(-1)
char* get_algorithm_argument(int argc, char** argv);

// setting the structs below to zero marks all the fields as unset

enum ImageType {
    IMG_UNSET, // ignore, as a user
    IMG_SIFT,
    IMG_MNIST
};

struct AlgorithmIndependentArguments {
    char *input_file, *query_file, *output_file;
    int number_of_nearest;
    int radius;
    ImageType image_type;
    bool search_for_range;
    bool range_bool_has_been_set; // ignore, as a user
};

struct LshArguments {
    AlgorithmIndependentArguments common;
    int k, L;
    double w;
    int seed;
};

struct HypercubeArguments {
    AlgorithmIndependentArguments common;
    int kproj, M, probes;
    double w; // accepted per PDF spec but not used in implementation
    int seed;
};

struct IvfflatArguments {
    AlgorithmIndependentArguments common;
    int kclusters, nprobe, seed;
};

struct IvfpqArguments {
    AlgorithmIndependentArguments common;
    int kclusters, nprobe, seed, M, nbits;
};

struct BruteforceArguments {
    AlgorithmIndependentArguments common;
    double fraction_of_pixels;
};

void parse_lsh_arguments(LshArguments* a, int argc, char** argv);
void parse_hypercube_arguments(HypercubeArguments* a, int argc, char** argv);
void parse_ivfflat_arguments(IvfflatArguments* a, int argc, char** argv);
void parse_ivfpq_arguments(IvfpqArguments* a, int argc, char** argv);
void parse_bruteforce_arguments(BruteforceArguments* a, int argc, char** argv);
