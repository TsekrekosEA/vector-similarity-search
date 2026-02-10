# Approximate Nearest Neighbor Search: From Theory to Protein Discovery

A comprehensive study and implementation of Approximate Nearest Neighbor (ANN) algorithms — from foundational data structures in C++ to learned hash functions in Python, culminating in a real-world application: discovering similar proteins that traditional sequence alignment (BLAST) might miss.

<p align="center">
  <img src="cpp-ann-algorithms/images/SIFT_recall_vs_speedup.png" width="48%" alt="SIFT Recall vs Speedup"/>
  <img src="cpp-ann-algorithms/images/MNIST_recall_vs_speedup.png" width="48%" alt="MNIST Recall vs Speedup"/>
</p>
<p align="center"><em>Recall vs. Speedup trade-offs across all implemented algorithms on SIFT1M and MNIST.</em></p>

---

## Project Overview

This project implements and benchmarks **six ANN methods** end-to-end, progressing through three levels of sophistication:

| Component | Description | Language | Key Techniques |
|-----------|-------------|----------|----------------|
| [**cpp-ann-algorithms/**](cpp-ann-algorithms/) | Four ANN algorithms implemented from scratch | C++17 | LSH, Hypercube, IVF-Flat, IVF-PQ |
| [**neural-lsh/**](neural-lsh/) | Learned hash functions via graph partitioning | Python / PyTorch | KaHIP, MLP classifier, multi-probe search |
| [**protein-similarity-search/**](protein-similarity-search/) | Real-world protein similarity pipeline | Python / C++ | ESM-2 embeddings, BLAST comparison |

### Key Results

| Method | SIFT1M Recall@10 | MNIST Recall@10 | Protein Recall@50 vs BLAST |
|--------|:-:|:-:|:-:|
| LSH | ~0.65 | ~0.80 | 0.38 |
| Hypercube | ~0.45 | ~0.70 | 0.60 |
| IVF-Flat | ~0.95 | ~0.95 | 0.67 |
| IVF-PQ | ~0.85 | ~0.90 | 0.67 |
| Neural LSH | ~0.90 | ~0.95 | **0.67** (9.5 QPS — 4x faster than BLAST) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        cpp-ann-algorithms/ (C++17)                         │
│                                                                             │
│   ┌─────────┐  ┌───────────┐  ┌──────────┐  ┌────────────────────────┐     │
│   │   LSH   │  │ Hypercube │  │ IVF-Flat │  │       IVF-PQ           │     │
│   │L tables │  │ BFS probe │  │ k-means  │  │ Product Quantization   │     │
│   │k hashes │  │ XOR flip  │  │ inv.list │  │ Asymmetric Distance    │     │
│   └────┬────┘  └─────┬─────┘  └────┬─────┘  └──────────┬────────────┘     │
│        └──────────────┼─────────────┼───────────────────┘                   │
│                       ▼             ▼                                       │
│              ┌──────────────────────────────┐                               │
│              │  Matrix<T> · distance funcs  │                               │
│              │  metrics · file I/O · bench  │                               │
│              └──────────────────────────────┘                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                         neural-lsh/ (Python)                                │
│                                                                             │
│   Dataset ──► k-NN Graph ──► KaHIP Partition ──► Train MLP ──► Multi-probe │
│              (scikit-learn)   (graph cutting)    (PyTorch)     search       │
├─────────────────────────────────────────────────────────────────────────────┤
│                  protein-similarity-search/ (Python + C++)                  │
│                                                                             │
│   Swiss-Prot ──► ESM-2 Embed ──► ANN Search ──► Compare vs BLAST          │
│   (573K proteins)  (320-dim)     (all methods)   (Recall@N evaluation)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- C++17 compiler (GCC 7+ or Clang 5+)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- BLAST+ (for protein search only)

### 1. C++ Algorithms

```bash
cd cpp-ann-algorithms
./scripts/download_sift.sh   # Download SIFT1M dataset (~400MB)
./scripts/download_mnist.sh  # Download MNIST dataset (~50MB)
make                         # Build with -O2 -std=c++17

# Run LSH on SIFT1M
./bin/search -lsh -d data/sift/sift_base.fvecs \
    -q data/sift/sift_query.fvecs -o output/lsh_sift.txt \
    -type sift -k 4 -L 5 -N 10

# Run full benchmark suite
make build-all
./bin/benchmark
python3 plot_results.py
```

### 2. Neural LSH

```bash
cd neural-lsh
uv sync                     # Install dependencies

# Build index (k-NN graph → KaHIP partition → MLP training)
uv run python nlsh_build.py -d ../data/mnist/train-images.idx3-ubyte \
    -i output/nlsh_index.pth -type mnist -m 100 --epochs 10

# Search
uv run python nlsh_search.py -d ../data/mnist/train-images.idx3-ubyte \
    -q ../data/mnist/t10k-images.idx3-ubyte \
    -i output/nlsh_index.pth -o output/results.txt -type mnist -N 10 -T 5
```

### 3. Protein Similarity Search

```bash
cd protein-similarity-search
./setup.sh                   # Download Swiss-Prot, build BLAST DB, generate embeddings
./run_pipeline.sh            # Run all methods and compare against BLAST
```

---

## Algorithms Implemented

### Locality-Sensitive Hashing (LSH)
Random projection hash functions with **L independent hash tables**, each using a composite SuperHash of **k base hashes**. Provides probabilistic guarantees on finding near neighbors with sub-linear query time.

### Hypercube LSH
Maps vectors to vertices of a **k-dimensional binary hypercube** via sign of random projections. Uses **BFS-based multi-probe search** at increasing Hamming distances. Single hash table design uses ~5-10x less memory than standard LSH.

### IVF-Flat (Inverted File Index)
Partitions the dataset into **Voronoi cells** via k-means clustering. Queries search the **nprobe nearest cells** for exact distance computation. Custom k-means implementation with early-termination distance optimization.

### IVF-PQ (Product Quantization)
Extends IVF-Flat with **vector compression**: splits each vector into M sub-vectors, quantizes each independently. Uses **Asymmetric Distance Computation (ADC)** — precomputing distance tables for O(M) lookups per candidate instead of O(D).

### Neural LSH
A **learned hashing** approach: builds a k-NN graph over the dataset, partitions it with KaHIP (balanced graph cutting), then trains an MLP to predict partition membership. At query time, the network routes queries to the most likely partitions in a single forward pass.

---

## Benchmarking

The C++ component includes a comprehensive benchmarking harness with:
- **Ground truth caching** — brute-force results serialized to disk, loaded in <1s on subsequent runs
- **Index reuse** — for IVF methods, builds index once and tests multiple `nprobe` values
- **Batched parallelism** — uses `std::async` with 75% of hardware cores
- **Parameter sweeps** — ~200 configurations across all algorithms
- **CSV export + visualization** — 22 plot types via matplotlib/seaborn

```bash
cd cpp-ann-algorithms
./bin/benchmark --sift-only --ivfflat-only   # Targeted sweep
python3 plot_results.py                       # Generate all visualizations
```

---

## Project Structure

```
.
├── LICENSE
├── README.md                       # This file
├── .clang-format                   # C++ formatting rules
├── .gitignore
│
├── cpp-ann-algorithms/             # Part 1: C++17 ANN implementations
│   ├── Makefile
│   ├── README.md
│   ├── include/                    # Header files with algorithm docs
│   ├── src/                        # Implementation files
│   ├── scripts/                    # Dataset download scripts
│   ├── images/                     # Benchmark result plots
│   └── plot_results.py             # Visualisation (22 plot types)
│
├── neural-lsh/                     # Part 2: Learned hashing (Python/PyTorch)
│   ├── pyproject.toml
│   ├── README.md
│   ├── nlsh_build.py               # Index construction pipeline
│   ├── nlsh_search.py              # Multi-probe query pipeline
│   ├── model.py                    # MLPClassifier (PyTorch module)
│   ├── parsers.py                  # MNIST / SIFT binary readers
│   ├── run_experiments.sh          # Parameterised experiment runner
│   └── plot_results.py             # Neural LSH visualisation
│
├── protein-similarity-search/      # Part 3: Bioinformatics application
│   ├── pyproject.toml
│   ├── README.md
│   ├── setup.sh                    # End-to-end environment setup
│   ├── run_pipeline.sh             # Orchestration script
│   ├── protein_embed.py            # ESM-2 embedding generation
│   ├── protein_embed_backend.py    # Embedding implementation
│   ├── protein_search.py           # Multi-method search & BLAST comparison
│   └── results.txt                 # Sample benchmark results
│
└── docs/                           # GitHub Pages site (project write-up)
    └── index.md
```

---

## Technical Highlights

- **From-scratch implementations** — no external ANN libraries (FAISS, Annoy, etc.); every algorithm built from first principles
- **Modern C++17** — templates with `if constexpr`, structured bindings, smart pointers, RAII
- **Cache-optimized data structures** — custom contiguous `Matrix<T>` class for cache-friendly iteration
- **Template-based zero-cost abstraction** — single codebase supports `float` and `uint8_t` without virtual dispatch
- **Asymmetric Distance Computation** — IVF-PQ uses precomputed lookup tables for O(M) distance instead of O(D)
- **Reproducible experiments** — seeded RNG, ground truth caching, automated parameter sweeps
- **Real-world application** — ESM-2 protein language model embeddings enable detection of remote homologs missed by BLAST

---

## Authors

- **Egor-Andrianos Tsekrekos**

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
