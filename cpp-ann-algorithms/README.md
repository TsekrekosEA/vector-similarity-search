# High-Performance Approximate Nearest Neighbor Search Engine

A C++17 implementation of a high-performance vector similarity search engine, built entirely from scratch. This project provides a framework for indexing and searching through millions of high-dimensional vectors using several state-of-the-art Approximate Nearest Neighbor (ANN) algorithms.

The engine is benchmarked on standard industry datasets, **SIFT1M** (1 million 128-D float vectors) and **MNIST** (60,000 784-D `uint8_t` vectors), to scientifically evaluate the performance and accuracy trade-offs of each implemented technique.

![MNIST Recall vs Speedup](images/MNIST_recall_vs_speedup.png)
*Figure 1: Recall vs. Speedup comparison for all implemented algorithms on the MNIST dataset.*

![SIFT Recall vs Speedup](images/SIFT_recall_vs_speedup.png)
*Figure 2: Recall vs. Speedup comparison for all implemented algorithms on the SIFT1M dataset.*

---

## Table of Contents
- [Project Highlights](#project-highlights)
- [Project Structure](#project-structure)
- [Setup and Building](#setup-and-building)
- [Usage](#usage)
- [Benchmarking & Visualization](#benchmarking--visualization)
- [Performance Metrics](#performance-metrics)
- [Authors](#authors)

---

## Project Highlights

-   **Algorithms Implemented from Scratch:** 
    -   **LSH (Locality-Sensitive Hashing):** Using random projections and an efficient single-integer key scheme.
    -   **Hypercube LSH:** A high-performance implementation using `uint64_t` keys and a multi-probe BFS search strategy.
    -   **IVF-Flat (Inverted File):** A partition-based index built on a custom, optimized k-means implementation.
    -   **IVF-PQ (Inverted File with Product Quantization):** An advanced index combining IVF with vector compression for massive memory savings and accelerated search via Asymmetric Distance Computation (ADC).

-   **High-Performance C++ Design:** The codebase emphasizes modern C++17 best practices for speed and safety.
    -   **Cache-Friendly Data Structures:** Utilizes a custom `Matrix` class with a contiguous memory layout to maximize cache locality and processing speed.
    -   **Template-Based Genericity:** The entire system is generic, supporting both `float` and `uint8_t` data types without code duplication.
    -   **RAII and Smart Pointers:** Ensures robust, leak-free memory management.

-   **Scientific Benchmarking & Analysis:** Includes a comprehensive C++ benchmarking harness and Python visualization scripts to produce detailed performance reports.
    -   **Key Metrics:** Gathers crucial metrics like Recall@N, Queries Per Second (QPS), and Average Approximation Factor (AF).
    -   **Data-Driven Insights:** The generated plots provide a clear, empirical analysis of the speed vs. accuracy trade-offs.

---

## Project Structure
```.
├── Makefile          # Build system for main program and benchmarks
├── README.md         # This file
├── scripts/          # Helper scripts for downloading data
├── src/              # Source files (.cpp) for all algorithms and utilities
├── include/          # Header files (.hpp) for all modules
├── data/             # Datasets (git-ignored)
├── output/           # Individual algorithm output files (git-ignored)
└── results/          # Benchmark CSVs and generated plots (git-ignored)
```
---

## Setup and Building

#### Prerequisites
-   C++ compiler with C++17 support (e.g., `g++` or `clang++`)
-   `make`
-   `wget` and `tar`
-   Python 3 with `pandas` and `matplotlib` (for visualization)

#### 1. Download Datasets
The algorithms are generic and can work with any dataset in the appropriate binary format. The scripts below download the specific **SIFT1M** and **MNIST** datasets used for the benchmarks presented in this project.

**SIFT1M Dataset:**
```bash
./scripts/download_sift.sh
```

**MNIST Dataset:**
```bash
./scripts/download_mnist.sh
```

#### 2. Build
```bash
# Build the main search program (bin/search)
make

# Build the benchmarking harness (bin/benchmark)
make benchmark

# Clean all compiled files
make clean
```
---

## Usage

### Basic Search
The `bin/search` program can run any implemented algorithm on a compatible dataset.

**Syntax:**
```bash
./bin/search -<alg> -d <input_file> -q <query_file> -o <output_file> [parameters...]
```
-   **Common:** `-N <int>`, `-R <float>`, `-type <mnist|sift>`
-   **Algorithm Flags:** `-lsh`, `-hypercube`, `-ivfflat`, `-ivfpq`

**Example (IVF-Flat on the SIFT1M benchmark dataset):**
```bash
./bin/search -ivfflat -d data/sift/sift_base.fvecs \
    -q data/sift/sift_query.fvecs -o output/ivfflat_sift.txt \
    -type sift -k_clusters 256 -nprobe 4 -N 10
```
---

## Benchmarking & Visualization

The `bin/benchmark` program runs a full parameter sweep on the included SIFT1M and MNIST datasets to generate the performance data. The `plot_results.py` script visualizes this data.

**1. Run Benchmarks:**
```bash
# Run sweeps for all algorithms on the default datasets
./bin/benchmark

# Or run for a specific algorithm
./bin/benchmark --ivfflat-only
```

**2. Generate Plots:**
```bash
python3 plot_results.py
```
This script reads the benchmark CSVs from `results/` and saves comprehensive comparison plots to the same directory.

---

## Performance Metrics

-   **Recall@N:** Accuracy metric. Fraction of queries where the true #1 nearest neighbor is found. (Higher is better).
-   **Average AF:** Quality metric. Average ratio of approximate distance to true distance. (Lower is better, 1.0 is perfect).
-   **Speedup:** Performance metric. How many times faster the algorithm is than brute-force. (Higher is better).
-   **QPS (Queries Per Second):** Throughput metric. (Higher is better).

---

## Authors

-   Egor-Andrianos Tsekrekos
-   Theodoros Dimakopoulos

## License

MIT License — see [LICENSE](../LICENSE) for details.

