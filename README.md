# High-Performance Approximate Nearest Neighbor Search Engine

This project is a C++17 implementation of a high-performance vector similarity search engine, built entirely from scratch. It provides a framework for indexing and searching through millions of high-dimensional vectors using several state-of-the-art Approximate Nearest Neighbor (ANN) algorithms.

The engine is benchmarked on standard industry datasets, **SIFT1M** (1 million 128-D float vectors) and **MNIST** (60,000 784-D unsigned char vectors), to scientifically evaluate the performance and accuracy trade-offs of each implemented technique.

![alt text](results/MNIST_recall_vs_speedup.png)

## Project Highlights

-   **Algorithms Implemented from Scratch:**:
    -   **LSH (Locality-Sensitive Hashing):** Using random projections and an efficient single-integer key scheme.
    -   **Hypercube LSH:** A high-performance implementation using `uint64_t` keys and a multi-probe BFS search strategy.
    -   **IVF-Flat (Inverted File):** A partition-based index built on a custom k-means implementation.
    -   **IVF-PQ (Inverted File with Product Quantization):** An advanced index combining IVF with vector compression for massive memory savings and accelerated search via Asymmetric Distance Computation (ADC).

-   **High-Performance C++ Design:** The codebase emphasizes modern C++ best practices for speed and safety.
    -   **Cache-Friendly Data Structures:** Utilizes a custom `Matrix` class with a contiguous memory layout to maximize cache locality and processing speed.
    -   **Template-Based Genericity:** The entire system is generic, supporting both `float` and `uint8_t` data types without code duplication.
    -   **RAII and Smart Pointers:** Ensures robust, leak-free memory management.

-   **Scientific Benchmarking & Analysis:** The project includes a comprehensive C++ benchmarking harness and Python visualization scripts to produce detailed performance reports.
    -   **Key Metrics:** Gathers crucial metrics like Recall@N, Queries Per Second (QPS), and Average Approximation Factor (AF).
    -   **Data-Driven Insights:** The generated plots provide a clear, empirical analysis of the speed vs. accuracy trade-offs for each algorithm on different types of data.

## Creators

- Τσεκρέκος ΄Εγκορ-Ανδριανός,
- Δημακόπουλος Θεόδωρος

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
