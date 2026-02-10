---
layout: default
---

# Approximate Nearest Neighbour Search: From Theory to Protein Discovery

> **TL;DR** — We implement six ANN algorithms from scratch (C++ and Python),
> benchmark them on SIFT1M and MNIST, then apply them to a real problem:
> finding similar proteins in Swiss-Prot (573 K entries) using ESM-2  
> embeddings — 4× faster than BLAST with 67 % recall.

---

## 1. The Problem

Given a database of *N* vectors in *D* dimensions, find the *k* closest
vectors to a query point.  Brute-force search is $O(ND)$ per query — too
slow when *N* reaches millions and latency budgets are milliseconds.

**Approximate** Nearest Neighbour (ANN) algorithms trade a small accuracy
loss for orders-of-magnitude speedups by using clever data structures that
narrow the search to a small candidate set.

This project walks through three levels of sophistication:

| Level | What | Where |
|-------|------|-------|
| **Classical algorithms** | LSH, Hypercube, IVF-Flat, IVF-PQ — built from scratch in C++17 | `cpp-ann-algorithms/` |
| **Learned hashing** | Neural LSH — an MLP trained on KaHIP graph partitions replaces hand-crafted hash functions | `neural-lsh/` |
| **Real-world application** | Protein similarity search — ESM-2 embeddings + all ANN methods vs. BLAST | `protein-similarity-search/` |

---

## 2. Classical ANN Algorithms (C++17)

All four algorithms share the same core abstraction: a `Matrix<T>` class
that stores vectors contiguously in memory for cache-friendly iteration,
and a common `Output` structure that records ids, distances, and timing.

### 2.1 Locality-Sensitive Hashing (LSH)

**Idea:** Project each vector through *k* random hyperplanes, concatenate
the sign bits into a single hash key, and repeat with *L* independent tables.
Because nearby vectors are likely to collide in at least one table, we can
restrict the distance computation to the collision set.

**Complexity:** $O(L \cdot |\text{bucket}|)$ per query, with high-probability
guarantees for $(1+\epsilon)$-approximate neighbours.

### 2.2 Hypercube LSH

**Idea:** Map each *D*-dimensional vector to a single vertex of a
*k*-dimensional binary hypercube (one bit per random projection).
Multi-probe search at increasing Hamming distances explores nearby
vertices via BFS.

**Trade-off:** Uses a single hash table (5–10× less memory than multi-table
LSH) at the cost of slightly more probing.

### 2.3 IVF-Flat (Inverted File Index)

**Idea:** Partition the dataset into Voronoi cells via k-means clustering.
At query time, find the nearest *nprobe* centroids and exhaustively search
those cells.  Custom k-means with early-termination distance checks.

### 2.4 IVF-PQ (Product Quantization)

**Idea:** Extend IVF with vector compression.  Each stored vector is split
into *M* sub-vectors, each quantized to one of 256 centroids.  A distance
table is precomputed once per query, enabling O(*M*) approximate distance
lookups instead of O(*D*).

**Asymmetric Distance Computation (ADC):** The query is *not* quantized —
only the database vectors are — preserving query-side precision while
compressing storage to *M* bytes per vector.

### Results on Standard Benchmarks

<p align="center">
  <img src="../cpp-ann-algorithms/images/SIFT_recall_vs_speedup.png" width="48%"
       alt="SIFT Recall vs Speedup"/>
  <img src="../cpp-ann-algorithms/images/MNIST_recall_vs_speedup.png" width="48%"
       alt="MNIST Recall vs Speedup"/>
</p>
<p align="center"><em>Recall@10 vs. Speedup on SIFT1M (128-D, 1 M vectors) and MNIST (784-D, 60 K vectors).</em></p>

Key observations:
- **IVF-Flat** achieves the highest recall (~0.95) but at moderate speedup.
- **IVF-PQ** trades ~5 % recall for ~2× more speedup via compressed storage.
- **LSH** offers consistent sub-linear performance with tuneable L and k.
- **Hypercube** is the most memory-efficient but needs careful probe tuning.

---

## 3. Neural LSH — Learned Hash Functions (Python / PyTorch)

Classical LSH uses *random* projections that know nothing about the data
distribution.  Neural LSH replaces them with a **data-dependent partitioning
learned by knowledge distillation**.

### Pipeline

```
Dataset ──► k-NN Graph ──► KaHIP Partition ──► Train MLP ──► Multi-probe Search
           (scikit-learn)   (graph cutting)     (PyTorch)
```

1. Build a symmetrised k-NN graph over the training set.
2. Feed the graph to [KaHIP](https://github.com/KaHIP/KaHIP), which cuts it
   into *m* balanced partitions while minimising inter-partition edges.
   Nearby points end up in the same partition.
3. Train a small MLP to predict the partition label from the raw feature
   vector — the MLP *imitates* the graph partitioner.
4. At query time, one forward pass produces the top-*T* predicted partitions;
   exhaustively search those partitions.

### Why It Works

KaHIP solves a global optimisation problem (balanced min-cut) that captures
the data manifold.  The MLP distils that knowledge into a fast, fixed-cost
forward pass.  Increasing *T* (the number of probes) smoothly trades latency
for recall — exactly the knob an engineer wants.

### Hyperparameter Sensitivity

| Parameter | Effect | Typical range |
|-----------|--------|:---:|
| *k* (k-NN graph) | Denser graph → better partitions, slower build | 2–20 |
| *m* (partitions) | More partitions → finer buckets, must increase *T* | 16–1024 |
| *T* (probes) | More probes → higher recall, lower speedup | 1–m/4 |
| epochs | More training → better partition prediction | 1–100 |

---

## 4. Protein Similarity Search — A Real-World Application

As a final capstone, we apply all five ANN methods to **biological sequence
search**: given a query protein, find the most similar proteins in
[Swiss-Prot](https://www.uniprot.org/) (573 K entries).

### Challenge

Proteins are variable-length amino acid sequences (20–35 K residues).
Traditional tools like **BLAST** align sequences character-by-character —
effective but slow and unable to detect very remote homologs.

### Our Approach

1. **Embed** every protein into a fixed 320-dimensional vector using
   [ESM-2](https://github.com/facebookresearch/esm), a protein language model
   pre-trained on 250 M sequences.
2. **Index** the embeddings with each ANN method.
3. **Search** with the same query proteins and measure Recall@50 against
   BLAST's top-50 hits.

### Results

| Method | QPS | Recall@50 vs BLAST | Notes |
|--------|:---:|:---:|---|
| **Neural LSH** | **9.5** | **0.67** | Best speed–accuracy trade-off |
| Hypercube | 1.1 | 0.60 | Single-table, low memory |
| BLAST | 2.3 | 1.00 | Reference (sequence alignment) |
| Euclidean LSH | 0.4 | 0.38 | Lower accuracy with default params |
| IVF-Flat | <0.1 | 0.67 | High index build time (~1 600 s) |
| IVF-PQ | <0.1 | 0.67 | High index build time (~3 800 s) |

**Key finding:** Neural LSH is **4× faster than BLAST** while retrieving
two-thirds of the same hits — *and* can surface structural homologs that
BLAST misses entirely because they share no detectable sequence similarity.

---

## 5. Technical Highlights

| Aspect | Details |
|--------|---------|
| **From-scratch implementations** | No FAISS, Annoy, or ScaNN — every algorithm built from first principles |
| **Modern C++17** | Templates with `if constexpr`, structured bindings, RAII, `std::async` parallelism |
| **Cache-optimised** | Custom contiguous `Matrix<T>` for cache-friendly iteration |
| **Zero-cost generics** | Single codebase supports `float` (SIFT) and `uint8_t` (MNIST) with no virtual dispatch |
| **ADC distance tables** | IVF-PQ precomputes lookup tables for $O(M)$ distance instead of $O(D)$ |
| **Reproducible experiments** | Seeded RNG, ground-truth caching, automated parameter sweeps |
| **Knowledge distillation** | KaHIP → MLP pipeline turns graph partitioning into a fast forward pass |
| **Protein language model** | ESM-2 (650 M params) embeds proteins into 320-D vectors capturing evolutionary information |

---

## 6. Getting Started

```bash
# Clone
git clone https://github.com/<user>/ann-search.git && cd ann-search

# ── Part 1: C++ algorithms ──
cd cpp-ann-algorithms
./scripts/download_sift.sh && ./scripts/download_mnist.sh
make && ./bin/benchmark
python3 plot_results.py

# ── Part 2: Neural LSH ──
cd ../neural-lsh
uv sync
./run_experiments.sh                          # MNIST default sweep
./run_experiments.sh --dataset sift           # SIFT sweep
python3 plot_results.py

# ── Part 3: Protein similarity ──
cd ../protein-similarity-search
./setup.sh            # download Swiss-Prot, build BLAST DB, generate embeddings
./run_pipeline.sh     # run all methods and compare
```

---

## Authors

- **Egor-Andrianos Tsekrekos**
- **Theodoros Dimakopoulos**

[MIT License](../LICENSE)
