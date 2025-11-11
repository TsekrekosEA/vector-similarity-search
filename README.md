# Approximate Nearest Neighbor Search

Implementation and benchmarking of several approximate nearest neighbor (ANN) search algorithms in C++. This project evaluates the performance of LSH, Hypercube, IVF-Flat, and 
IVF-PQ against a brute-force exact search on the MNIST and SIFT datasets.

## Creators
Τσεκρέκος ΄Εγκορ-Ανδριανός, sdi2300203
Δημακόπουλος Θεόδωρος, sdi1900048

## Algorithms Implemented

-   **Brute Force:** Exact nearest neighbor search for ground truth generation.
-   **LSH (Locality-Sensitive Hashing):** Hash-based approximate search using random projections.
-   **Hypercube:** Binary hypercube projection with BFS-based vertex exploration.
-   **IVF-Flat:** Inverted file index with k-means clustering, storing full vectors.
-   **IVF-PQ:** Inverted file with product quantization for memory-efficient compressed search.

## Project Structure

```
.
├── Makefile
├── README.md
├── src/
│   ├── main.cpp
│   ├── benchmark.cpp
│   ├── lsh.cpp, hypercube.cpp, ivfflat.cpp, ivfpq.cpp, brute_force.cpp
│   └── ... (utility files)
├── include/
│   ├── lsh.hpp, hypercube.hpp, ivfflat.hpp, ivfpq.hpp, brute_force.hpp
│   └── ... (utility headers)
├── data/         # Datasets (git-ignored)
├── output/       # Algorithm output files (git-ignored)
└── results/      # Benchmark CSVs and plots (git-ignored)
```

---

## Setup and Building

#### Prerequisites
-   C++ compiler with C++17 support (e.g., `g++` or `clang++`)
-   `make`
-   `wget` and `tar`

#### 1. Download Datasets
The datasets are not included in the repository.

**SIFT1M:**
```bash
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz -O data/sift.tar.gz
cd data && tar -zxvf sift.tar.gz && cd ..
```

**MNIST:**
Download the four `.idx-ubyte` files from [this repository](https://github.com/mrgloom/MNIST-dataset-in-different-formats/tree/master/data/Original%20dataset) and place them in the `data/mnist/` directory.

#### 2. Build
```bash
# Build the main search program (bin/search)
make

# Build the benchmarking harness (bin/benchmark)
make bin/benchmark

# Clean all compiled files
make clean
```
---

## Usage

### Basic Search
The main program `bin/search` is used for running a single algorithm configuration.

**Syntax:**
```bash
./bin/search -<algorithm> -d <input> -q <query> -o <output> [parameters...]
```
-   **Common Parameters:** `-N <int>`, `-R <float>`, `-type <mnist|sift>`
-   **Algorithm Flags:** `-lsh`, `-hypercube`, `-ivfflat`, `-ivfpq`

**Example (Hypercube on MNIST):**
```bash
./bin/search -hypercube -d data/mnist/train-images.idx3-ubyte \
    -q data/mnist/t10k-images.idx3-ubyte -o output/hypercube.txt \
    -type mnist -kproj 14 -M 1000 -probes 50 -N 10 -range false
```

### Benchmarking
The `bin/benchmark` program runs a full parameter sweep for all algorithms and generates CSV files in the `results/` directory.

**Run Benchmarks:**
```bash
# Run sweeps for all algorithms (this may take a long time)
./bin/benchmark

# Run for a specific algorithm
./bin/benchmark --lsh-only
```

---

## Performance Metrics

-   **Recall@N:** Accuracy metric. Fraction of queries where the true #1 neighbor is found. (Higher is better).
-   **Average AF:** Quality metric. Average ratio of approximate distance to true distance. (Lower is better, 1.0 is perfect).
-   **Speedup:** Performance metric. How many times faster the algorithm is than brute-force. (Higher is better).
-   **QPS (Queries Per Second):** Throughput metric. (Higher is better).
```