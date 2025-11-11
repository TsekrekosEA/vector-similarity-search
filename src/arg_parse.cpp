#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <arg_parse.hpp>

static char usage_message[] =
    "Usage: %s <args> <algorithm>\n"
    "\n"
    "Algorithm: -lsh | -hypercube | -ivfflat | -ivfpq | -bruteforce\n"
    "\n"
    "Common arguments:\n"
    "    -d <input file>\n"
    "    -q <query file>\n"
    "    -o <output file>\n"
    "    -N <number of nearest>\n"
    "    -type <sift|mnist>\n"
    "    -R <radius>\n"
    "    -range <true|false>\n"
    "\n"
    "LSH arguments:\n"
    "    -k <int>\n"
    "    -L <int>\n"
    "    -w <double>\n"
    "    -seed <int>\n"
    "\n"
    "Hypercube arguments:\n"
    "    -kproj <int>\n"
    "    -w <double>\n"
    "    -M <int>\n"
    "    -probes <int>\n"
    "    -seed <int>\n"
    "\n"
    "IVFFlat arguments:\n"
    "    -kclusters <int>\n"
    "    -nprobe <int>\n"
    "    -seed <int>\n"
    "\n"
    "IVFPQ arguments:\n"
    "    -kclusters <int>\n"
    "    -nprobe <int>\n"
    "    -M <int>\n"
    "    -nbits <int>\n"
    "    -seed <int>\n"
    "\n"
    "Brute force arguments:\n"
    "    -frac <double>  fraction of pixels to consider\n"
    ;

static bool is_algorithm_argument(char *arg) {
    if (
        strcmp(arg, "-lsh") == 0 ||
        strcmp(arg, "-hypercube") == 0 ||
        strcmp(arg, "-ivfflat") == 0 ||
        strcmp(arg, "-ivfpq") == 0 ||
        strcmp(arg, "-bruteforce") == 0
    ) {
        return true;
    }
    return false;
}

// No argv[i] can be -lsh etc. except the equivalent flag.
// -lsh would be invalid as a number value
// and it's not a wanted file name.
char *get_algorithm_argument(int argc, char **argv) {
    for (int i = argc - 1; i >= 1; i--) {  // end to start
        if (is_algorithm_argument(argv[i])) return argv[i];
    }
    fprintf(stderr, usage_message, argv[0]);
    exit(-1);
}

// can exit(-1)
static bool parse_if_algorithm_independent(AlgorithmIndependentArguments *a, int i, char **argv) {

    if (0);
#define IF(X) else if (strcmp(argv[i], X) == 0)

    IF ("-d") a->input_file        =      argv[++i];
    IF ("-q") a->query_file        =      argv[++i];
    IF ("-o") a->output_file       =      argv[++i];
    IF ("-N") a->number_of_nearest = atoi(argv[++i]);
    IF ("-R") a->radius            = atoi(argv[++i]);
    IF ("-type") {
        char *given = argv[++i];
        if (strcmp(given, "sift") == 0) a->image_type = IMG_SIFT;
        else if (strcmp(given, "mnist") == 0) a->image_type = IMG_MNIST;
        else {
            fprintf(stderr, "wrong -type: %s, expected sift or mnist\n\n", given);
            exit(-1);
        }
    }
    IF ("-range") {
        char *given = argv[++i];
        if (strcmp(given, "true") == 0) a->search_for_range = true;
        else if (strcmp(given, "false") == 0) a->search_for_range = false;
        else {
            fprintf(stderr, "invalid argument for -range (%s) expected true or false\n\n", given);
            exit(-1);
        }
        a->range_bool_has_been_set = true;
    }

#undef IF

    else return false;
    return true;
}

static void assert_no_unset(AlgorithmIndependentArguments *a) {

    if (!a->input_file || !a->query_file || !a->output_file) {
        fprintf(stderr, "missing the input, query or output files (-d -q -o)\n");
        exit(-1);
    }
    if (a->image_type == IMG_UNSET) {
        fprintf(stderr, "the image type hasn't been set (-type sift|mnist)\n");
        exit(-1);
    }
    if (!a->range_bool_has_been_set) {
        fprintf(stderr, "no -range true|false provided\n");
        exit(-1);
    }
    if (a->search_for_range && !a->radius) {
        fprintf(stderr, "provided '-range true' but no '-R <radius>'\n");  // TODO is this right?
        exit(-1);
    }
}

void parse_lsh_arguments(LshArguments *a, int argc, char **argv) {

    // Set defaults per PDF spec
    a->seed = 1;
    a->k = 4;
    a->L = 5;
    a->w = 4.0;
    a->common.number_of_nearest = 1;

    for (int i = 1; i < argc; i++) {
        if (is_algorithm_argument(argv[i])) {
            continue;
        }
        if (parse_if_algorithm_independent(&(a->common), i, argv)) {
            i++;
            continue;
        }

#define IF(X) else if (strcmp(argv[i], X) == 0)

        IF ("-k") a->k = atoi(argv[++i]);
        IF ("-L") a->L = atoi(argv[++i]);
        IF ("-w") a->w = atof(argv[++i]);
        IF ("-seed") a->seed = atoi(argv[++i]);

#undef IF

        else {
            fprintf(stderr, "unknown flag '%s'\n\n", argv[i]);
            fprintf(stderr, usage_message, argv[0]);
            exit(-1);
        }
    }
    assert_no_unset(&(a->common));
}

void parse_hypercube_arguments(HypercubeArguments *a, int argc, char **argv) {

    a->seed = 1;
    a->kproj = 14;
    a->w = 4.0;  // default per PDF spec (accepted but not used in implementation)
    a->M = 10;
    a->probes = 2;
    a->common.number_of_nearest = 1;

    for (int i = 1; i < argc; i++) {
        if (is_algorithm_argument(argv[i])) {
            continue;
        }
        if (parse_if_algorithm_independent(&(a->common), i, argv)) {
            i++;
            continue;
        }

#define IF(X) else if (strcmp(argv[i], X) == 0)

        IF ("-kproj")  a->kproj  = atoi(argv[++i]);
        IF ("-w")      a->w      = atof(argv[++i]);  // accepted per PDF spec but not used
        IF ("-M")      a->M      = atoi(argv[++i]);
        IF ("-probes") a->probes = atoi(argv[++i]);
        IF ("-seed")   a->seed   = atoi(argv[++i]);

#undef IF

        else {
            fprintf(stderr, "unknown flag '%s'\n\n", argv[i]);
            fprintf(stderr, usage_message, argv[0]);
            exit(-1);
        }
    }
    assert_no_unset(&(a->common));

    if (a->common.number_of_nearest <= 0) {
        a->common.number_of_nearest = 1;
    }
}

void parse_ivfflat_arguments(IvfflatArguments *a, int argc, char **argv) {

    // Set defaults per PDF spec
    a->seed = 1;
    a->kclusters = 50;
    a->nprobe = 5;
    a->common.number_of_nearest = 1;

    for (int i = 1; i < argc; i++) {
        if (is_algorithm_argument(argv[i])) {
            continue;
        }
        if (parse_if_algorithm_independent(&(a->common), i, argv)) {
            i++;
            continue;
        }

#define IF(X) else if (strcmp(argv[i], X) == 0)

        IF ("-kclusters") a->kclusters = atoi(argv[++i]);
        IF ("-nprobe")    a->nprobe    = atoi(argv[++i]);
        IF ("-seed")      a->seed      = atoi(argv[++i]);

#undef IF

        else {
            fprintf(stderr, "unknown flag '%s'\n\n", argv[i]);
            fprintf(stderr, usage_message, argv[0]);
            exit(-1);
        }
    }
    assert_no_unset(&(a->common));
}

void parse_ivfpq_arguments(IvfpqArguments *a, int argc, char **argv) {

    // Set defaults per PDF spec
    a->seed = 1;
    a->kclusters = 50;
    a->nprobe = 5;
    a->nbits = 8;  // for 2^8 = 256 clusters
    a->M = 16;
    a->common.number_of_nearest = 1;

    for (int i = 1; i < argc; i++) {
        if (is_algorithm_argument(argv[i])) {
            continue;
        }
        if (parse_if_algorithm_independent(&(a->common), i, argv)) {
            i++;
            continue;
        }

#define IF(X) else if (strcmp(argv[i], X) == 0)

        IF ("-kclusters") a->kclusters = atoi(argv[++i]);
        IF ("-nprobe")    a->nprobe    = atoi(argv[++i]);
        IF ("-seed")      a->seed      = atoi(argv[++i]);
        IF ("-M")         a->M         = atoi(argv[++i]);
        IF ("-nbits")     a->nbits     = atoi(argv[++i]);
#undef IF

        else {
            fprintf(stderr, "unknown flag '%s'\n\n", argv[i]);
            fprintf(stderr, usage_message, argv[0]);
            exit(-1);
        }
    }
    assert_no_unset(&(a->common));
}

void parse_bruteforce_arguments(BruteforceArguments *a, int argc, char **argv) {

    for (int i = 1; i < argc; i++) {
        if (is_algorithm_argument(argv[i])) {
            continue;
        }
        if (parse_if_algorithm_independent(&(a->common), i, argv)) {
            i++;
            continue;
        }

#define IF(X) else if (strcmp(argv[i], X) == 0)

        IF ("-frac") a->fraction_of_pixels = atof(argv[++i]);

#undef IF

        else {
            fprintf(stderr, "unknown flag '%s'\n\n", argv[i]);
            fprintf(stderr, usage_message, argv[0]);
            exit(-1);
        }
    }
    assert_no_unset(&(a->common));
}
