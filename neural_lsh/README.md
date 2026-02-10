# Neural LSH — Learned Hash Functions for ANN Search

A **learned hashing** approach to Approximate Nearest Neighbour (ANN) search.
Instead of hand-crafting hash functions (as in classical LSH), we use a neural
network to *learn* a partitioning of the dataset that groups nearby points
together — enabling fast, single-forward-pass query routing.

```
Dataset ──► k-NN Graph ──► KaHIP Partition ──► Train MLP ──► Multi-probe Search
           (scikit-learn)   (graph cutting)    (PyTorch)
```

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Running Experiments](#running-experiments)
- [Performance Metrics](#performance-metrics)
- [Authors](#authors)

---

## How It Works

1. **k-NN graph** — Compute each point's *k* nearest neighbours (scikit-learn)
   and build a symmetrised, weighted adjacency graph.
2. **Graph partitioning** — Feed the graph to [KaHIP](https://github.com/KaHIP/KaHIP)
   to produce *m* balanced partitions (blocks), minimising edge-cut across
   partition boundaries so that similar points land in the same block.
3. **MLP training** — Train a small MLP classifier (PyTorch) to predict the
   KaHIP block label from the raw feature vector.  The network learns to
   imitate the graph partitioner.
4. **Multi-probe search** — At query time, run a single forward pass through
   the MLP to obtain the top-*T* predicted partitions, then exhaustively
   search those partitions for nearest neighbours.

The key insight: traditional LSH uses random projections that are data-agnostic.
Neural LSH replaces them with a *data-dependent* partitioning learned by
knowledge distillation — KaHIP is the teacher, the MLP is the student.

---

## Project Structure

```
neural-lsh/
├── pyproject.toml          # Dependencies (uv / pip)
├── README.md               # This file
│
├── nlsh_build.py           # Index construction pipeline (CLI)
├── nlsh_search.py          # Multi-probe query pipeline  (CLI)
├── model.py                # MLPClassifier (PyTorch module)
├── parsers.py              # MNIST (IDX3) & SIFT (.fvecs) readers
│
├── brute_force.py          # Exact search for ground-truth generation
├── metrics.py              # AF, Recall@N, QPS computation
├── results.py              # Dataclasses for search output
├── output_writer.py        # Detailed output file writer
├── ground_truth_cache.py   # Caches brute-force results to disk
│
├── plot_results.py         # Visualisation (Recall vs Speedup, etc.)
├── run_experiments.sh      # Parameterised experiment runner
└── data/                   # Datasets (git-ignored)
```

---

## Setup

**Prerequisites:** Python 3.10+, a C++ compiler + CMake (required by the
`kahip` package).

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install .
```

---

## Usage

### 1. Build an Index

```bash
python nlsh_build.py \
    -d data/mnist/train-images.idx3-ubyte \
    -i output/index.pth \
    -type mnist \
    --knn 10 -m 100 --epochs 10
```

| Flag | Description | Default |
|------|-------------|---------|
| `-d` | Input dataset path | *required* |
| `-i` | Output index path (.pth) | *required* |
| `-type` | `mnist` or `sift` | *required* |
| `--knn` | Neighbours for the k-NN graph | 10 |
| `-m` | Number of KaHIP partitions | 100 |
| `--kahip_mode` | `fast` / `eco` / `strong` | `eco` |
| `--imbalance` | KaHIP imbalance factor | 0.03 |
| `--layers` | Hidden layers in the MLP | 3 |
| `--nodes` | Nodes per hidden layer | 64 |
| `--epochs` | Training epochs | 10 |
| `--batch_size` | Mini-batch size | 128 |
| `--lr` | Learning rate | 0.001 |
| `--seed` | Random seed | 1 |

### 2. Search

```bash
python nlsh_search.py \
    -d data/mnist/train-images.idx3-ubyte \
    -q data/mnist/t10k-images.idx3-ubyte \
    -i output/index.pth \
    -o output/results.txt \
    -type mnist -N 10 -T 5
```

| Flag | Description | Default |
|------|-------------|---------|
| `-d` | Dataset file | *required* |
| `-q` | Query file | *required* |
| `-i` | Index file from build step | *required* |
| `-o` | Output file (skip with `--minimal-output`) | — |
| `-type` | `mnist` or `sift` | *required* |
| `-N` | Nearest neighbours to return | 1 |
| `-T` | Partitions to probe (multi-probe) | 5 |
| `-R` | Radius for range search | 2000/2800 |
| `--range` | Enable range search (`true`/`false`) | `true` |
| `--csv-output-file` | Append one-line metrics to CSV | — |
| `--minimal-output` | Skip detailed output file | off |

---

## Running Experiments

A single parameterised script replaces the previous per-dataset experiment
scripts.  Pass hyperparameter ranges as space-separated strings:

```bash
# MNIST sweep (default)
./run_experiments.sh

# SIFT sweep
./run_experiments.sh --dataset sift --k "10 20" --m "128 256" --probes "5 10 20"

# Custom
./run_experiments.sh --dataset mnist --k "2 4 8" --m "8 32 128" --epochs 100 --probes "1 5 10"
```

Results are written to `output/experiments_{dataset}.csv`.  Visualise with:

```bash
python plot_results.py                          # auto-detects MNIST/SIFT CSVs
python plot_results.py output/my_results.csv    # explicit path
```

---

## Performance Metrics

| Metric | Description | Goal |
|--------|-------------|------|
| **Recall@N** | Fraction of queries where the true NN is in the result set | Higher |
| **Average AF** | Mean ratio of approximate to true distance (1.0 = perfect) | Lower |
| **QPS** | Queries per second | Higher |
| **Speedup** | Wall-time ratio vs brute force | Higher |

---

## Authors

- Egor-Andrianos Tsekrekos

## License

MIT License — see [LICENSE](../LICENSE) for details.
