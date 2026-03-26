# Algorithm Parameters

This document explains the parameters for each ANN algorithm, their effects on performance, and tuning guidelines.

## Table of Contents
- [LSH (Locality Sensitive Hashing)](#lsh-locality-sensitive-hashing)
- [Hypercube](#hypercube)
- [IVF-Flat](#ivf-flat)
- [IVF-PQ (Product Quantization)](#ivf-pq-product-quantization)
- [Neural LSH](#neural-lsh)
- [Parameter Tuning Strategies](#parameter-tuning-strategies)

---

## LSH (Locality Sensitive Hashing)

### Parameters

**`k` - Hash functions per table**
- **Range:** 1-20 (typical: 4-10)
- **Effect:** More hash functions = tighter buckets, fewer false positives
- **Trade-off:** Higher k reduces recall but increases precision
- **Default:** 6

**`L` - Number of hash tables**
- **Range:** 1-50 (typical: 5-15)
- **Effect:** More tables = higher recall (more chances to find neighbors)
- **Trade-off:** Higher L increases memory and query time
- **Default:** 8

**`w` - Bucket width**
- **Range:** 0.1-10.0 (dataset-dependent)
- **Effect:** Larger w = wider buckets, more candidates per query
- **Trade-off:** Too large = slow queries, too small = low recall
- **Default:** 1.0 (SIFT), 5.0 (MNIST)

### Usage

```bash
./bin/search \
    -algorithm lsh \
    -k 6 \
    -L 8 \
    -w 1.0 \
    -d dataset.dat \
    -q queries.dat \
    -o results.txt
```

### Tuning Guidelines

**For high-dimensional data (d > 500):**
- Increase L (10-20 tables)
- Moderate k (5-8)
- Tune w based on average distance

**For low-dimensional data (d < 100):**
- Fewer tables needed (L = 5-10)
- Higher k (8-12)
- Smaller w

**General strategy:**
1. Start with defaults
2. If recall is low: increase L or w
3. If queries are slow: decrease L or w
4. If too many candidates: increase k or decrease w

### Performance Characteristics

| k | L | Recall | Query Time | Memory |
|---|---|--------|------------|--------|
| 4 | 5 | Low | Fast | Low |
| 6 | 8 | Medium | Medium | Medium |
| 8 | 15 | High | Slow | High |

---

## Hypercube

### Parameters

**`k` (or `kproj`) - Projection dimensions**
- **Range:** 6-16 (typical: 10-14)
- **Effect:** Creates 2^k hypercube vertices
- **Trade-off:** Higher k = exponentially more vertices, finer partitioning
- **Default:** 12 (4096 vertices)

**`w` - Window width**
- **Range:** 0.5-5.0
- **Effect:** Similar to LSH bucket width
- **Trade-off:** Larger w = more neighbors considered
- **Default:** 1.5

**`M` - Max candidates to check**
- **Range:** 100-10000
- **Effect:** Limits search even if more candidates exist
- **Trade-off:** Safety valve for query time
- **Default:** 5000

**`probes` - Vertices to probe**
- **Range:** 1-10
- **Effect:** Number of nearby vertices to check (Hamming distance)
- **Trade-off:** More probes = higher recall, slower queries
- **Default:** 2

### Usage

```bash
./bin/search \
    -algorithm hypercube \
    -k 12 \
    -w 1.5 \
    -M 5000 \
    -probes 2 \
    -d dataset.dat \
    -q queries.dat \
    -o results.txt
```

### Tuning Guidelines

**Balancing k and probes:**
- Lower k (10) + more probes (4-6) = faster build, flexible search
- Higher k (14) + fewer probes (1-2) = slower build, precise search

**M as safety net:**
- Set to 10-20% of dataset size
- Prevents worst-case query slowdowns

**Recommended combinations:**
- **Fast:** k=10, probes=2, M=1000
- **Balanced:** k=12, probes=3, M=5000
- **Accurate:** k=14, probes=5, M=10000

### Performance Characteristics

Hypercube has **instant build time** and scales well to moderate datasets (up to ~1M points).

---

## IVF-Flat

### Parameters

**`nlist` - Number of clusters**
- **Range:** sqrt(N) to N/100 (typical: 100-10000)
- **Effect:** More clusters = finer partitioning
- **Trade-off:** Too many = long build time, too few = low recall
- **Default:** 1000
- **Rule of thumb:** nlist ≈ √N

**`nprobe` - Clusters to search**
- **Range:** 1 to nlist (typical: nlist/100 to nlist/10)
- **Effect:** More clusters searched = higher recall
- **Trade-off:** Linear increase in query time
- **Default:** 50
- **Rule of thumb:** nprobe ≈ nlist/20

### Usage

```bash
./bin/search \
    -algorithm ivfflat \
    -nlist 1000 \
    -nprobe 50 \
    -d dataset.dat \
    -q queries.dat \
    -o results.txt
```

### Tuning Guidelines

**Cluster count (nlist):**
- **Small datasets (<100K):** nlist = 100-500
- **Medium datasets (100K-1M):** nlist = 500-2000
- **Large datasets (>1M):** nlist = 2000-10000

**Probe count (nprobe):**
- Start with nprobe = nlist/20
- For 90% recall: typically need nprobe ≈ nlist/10
- For 99% recall: may need nprobe ≈ nlist/2

**Build time optimization:**
- k-means iterations: reduce if dataset is well-structured
- Use sampling for very large datasets

### Performance Characteristics

**Strengths:**
- Highest recall among traditional methods
- Exact distances within searched clusters

**Weaknesses:**
- Long build time (k-means clustering)
- Memory-intensive (stores full vectors)

**Build time scaling:**
- O(N × nlist × iterations × dimension)
- Swiss-Prot (573K, 320D): ~27 minutes with nlist=1000

---

## IVF-PQ (Product Quantization)

### Parameters

**`nlist` - Number of clusters**
- Same as IVF-Flat
- **Default:** 1000

**`nprobe` - Clusters to search**
- Same as IVF-Flat
- **Default:** 50

**`m` - Number of subquantizers**
- **Range:** 4-64 (must divide dimension evenly)
- **Effect:** More subquantizers = better approximation
- **Trade-off:** More memory, slower distance computation
- **Default:** 16
- **Constraint:** dimension % m == 0

**`nbits` - Bits per subquantizer**
- **Range:** 4-12 (typical: 8)
- **Effect:** More bits = larger codebooks, better approximation
- **Trade-off:** 2^nbits centroids per subquantizer
- **Default:** 8 (256 centroids)

### Usage

```bash
./bin/search \
    -algorithm ivfpq \
    -nlist 1000 \
    -nprobe 50 \
    -m 16 \
    -nbits 8 \
    -d dataset.dat \
    -q queries.dat \
    -o results.txt
```

### Tuning Guidelines

**Choosing m:**
- Higher dimensional data: use more subquantizers
- **MNIST (784D):** m=16 (49D per subvector)
- **SIFT (128D):** m=8 or 16
- **Proteins (320D):** m=16 or 20

**Choosing nbits:**
- 8 bits (256 centroids) is usually sufficient
- Increase to 10-12 for critical applications
- Codebook size: m × 2^nbits × dimension/m × sizeof(float)

**Memory calculation:**
- **PQ codes:** N × m × (nbits/8) bytes
- **Codebooks:** m × 2^nbits × (D/m) × 4 bytes
- **Example (573K, 320D, m=16, nbits=8):**
  - Codes: 573K × 16 × 1 = 9.2 MB
  - Codebooks: 16 × 256 × 20 × 4 = 328 KB
  - Total: ~10 MB (vs. 733 MB for IVF-Flat)

### Performance Characteristics

**Compression ratio:**
- m=16, nbits=8: ~20x smaller than IVF-Flat
- Slightly lower recall due to quantization error

**Build time:**
- Longer than IVF-Flat (additional PQ training)
- Swiss-Prot: ~63 minutes

---

## Neural LSH

### Parameters

**`m` (or `blocks`) - Number of partitions**
- **Range:** 50-1000 (typical: 100-500)
- **Effect:** More partitions = finer-grained routing
- **Trade-off:** Higher m = longer training, more memory
- **Default:** 400 (proteins), 200 (MNIST/SIFT)
- **Rule of thumb:** m ≈ N/1000

**`T` (or `top_t`) - Partitions to probe**
- **Range:** 1 to m (typical: m/20 to m/5)
- **Effect:** More probes = higher recall
- **Trade-off:** Linear increase in query time
- **Default:** 50
- **Rule of thumb:** T ≈ m/8

**`knn` - Neighbors for graph construction**
- **Range:** 5-50
- **Effect:** k-NN graph density
- **Trade-off:** Higher knn = better graph, longer build
- **Default:** 10

**`epochs` - Training epochs**
- **Range:** 5-50
- **Effect:** More epochs = better-trained MLP
- **Trade-off:** Longer training time
- **Default:** 15

**`hidden` - MLP architecture**
- **Format:** Comma-separated layer sizes (e.g., "128,128")
- **Effect:** Larger network = more expressive
- **Trade-off:** Slower inference, more memory
- **Default:** "128,128" (2 hidden layers)

**`batch_size` - Training batch size**
- **Range:** 32-512
- **Default:** 128

**`lr` - Learning rate**
- **Range:** 0.0001-0.01
- **Default:** 0.001

### Usage

**Build:**
```bash
python nlsh_build.py \
    -d dataset.dat \
    -i index.pth \
    -type sift \
    --knn 10 \
    -m 200 \
    --epochs 15 \
    --layers 2 \
    --nodes 128
```

**Search:**
```bash
python nlsh_search.py \
    -d dataset.dat \
    -q queries.dat \
    -i index.pth \
    -o results.txt \
    -type sift \
    -N 50 \
    -T 20
```

### Tuning Guidelines

**Partition count (m):**
- Too few = poor routing accuracy
- Too many = sparse partitions, long training
- **Small datasets (<100K):** m=50-100
- **Medium datasets (100K-1M):** m=100-300
- **Large datasets (>1M):** m=300-1000

**Probe count (T):**
- Start with T = m/8
- Increase if recall is low
- T=m means searching everything (no speedup)

**Network architecture:**
- Larger networks help for complex distributions
- Diminishing returns beyond 3 layers
- **Simple data:** "64" (1 layer)
- **Moderate:** "128,128" (2 layers)
- **Complex:** "256,128,64" (3 layers)

**Training:**
- More epochs if training loss hasn't plateaued
- Early stopping if validation accuracy saturates

### Performance Characteristics

**Build time breakdown:**
1. k-NN graph: 30-40% of time
2. KaHIP partitioning: 20-30%
3. MLP training: 30-50%

**Query time:**
- O(T × avg_partition_size)
- Dominated by partition size, not MLP inference

**Memory:**
- MLP weights: ~1-10 MB
- Inverted index: ~8N bytes (indices)
- Total: much smaller than IVF methods

---

## Parameter Tuning Strategies

### General Principles

1. **Start with defaults** - They work reasonably well
2. **Measure baseline** - Run with defaults, note recall and QPS
3. **Tune one parameter at a time** - Isolate effects
4. **Consider build-time vs query-time trade-off**

### Recall-Speed Trade-off

All algorithms have parameters that trade recall for speed:

| Algorithm | Increase Recall | Increase Speed |
|-----------|-----------------|----------------|
| LSH | ↑ L, ↑ w | ↓ L, ↓ w, ↑ k |
| Hypercube | ↑ probes, ↓ k | ↓ probes, ↑ k |
| IVF-* | ↑ nprobe | ↓ nprobe |
| Neural LSH | ↑ T, ↑ m | ↓ T, ↓ m |

### Memory-Accuracy Trade-off

| Algorithm | Memory Usage | Accuracy |
|-----------|--------------|----------|
| LSH | L × hash_table_size | Medium |
| Hypercube | 2^k × bucket_size | Medium |
| IVF-Flat | N × D × 4 bytes | High |
| IVF-PQ | N × m bytes | High |
| Neural LSH | Small (MLP + index) | High |

### Empirical Tuning Process

```python
# 1. Baseline
baseline_params = {...}
baseline_recall, baseline_qps = evaluate(baseline_params)

# 2. Grid search key parameters
for param in key_params:
    for value in param_range:
        recall, qps = evaluate({**baseline_params, param: value})
        plot(value, recall, qps)

# 3. Select based on requirements
# - High recall app: maximize recall subject to QPS > threshold
# - High throughput: maximize QPS subject to recall > threshold
```

### Dataset-Specific Recommendations

**MNIST (784D, uint8):**
- LSH: k=6, L=10, w=5.0
- Hypercube: k=12, probes=3
- IVF-*: nlist=500, nprobe=50
- Neural LSH: m=200, T=25

**SIFT (128D, float32):**
- LSH: k=5, L=8, w=1.0
- Hypercube: k=12, probes=2
- IVF-*: nlist=1000, nprobe=50
- Neural LSH: m=200, T=20

**Proteins (320D, float32, 573K):**
- LSH: k=6, L=8, w=1.0
- Hypercube: k=12, probes=2, M=5000
- IVF-*: nlist=1000, nprobe=50
- Neural LSH: m=400, T=50

---

## Advanced Topics

### Adaptive Parameters

Some applications benefit from query-specific parameters:

**Dynamic nprobe (IVF):**
```python
if query_is_outlier:
    nprobe = nlist // 5  # Search more clusters
else:
    nprobe = nlist // 20  # Standard search
```

**Adaptive T (Neural LSH):**
```python
# Based on MLP confidence
if max_probability > 0.8:
    T = 10  # High confidence, fewer probes
else:
    T = 50  # Low confidence, more probes
```

### Multi-Probe Strategies

Instead of fixed probes, use distance-based probing:

**Hypercube:**
- Probe vertices in order of Hamming distance
- Stop when recall threshold met

**IVF:**
- Sort clusters by distance to query
- Probe in order until candidates sufficient

### Parameter Relationships

**LSH:** L and k are related
- Optimal: k ≈ log₂(L)
- Example: L=16 → k≈4, L=128 → k≈7

**IVF:** nlist and nprobe
- For 90% recall: nprobe ≈ 0.1 × nlist
- For 95% recall: nprobe ≈ 0.2 × nlist

**Neural LSH:** m and T
- For balanced search: T ≈ 0.1 × m
- For high recall: T ≈ 0.2 × m

---

## Troubleshooting

**Low recall:**
- Increase: L (LSH), probes (Hypercube), nprobe (IVF), T (Neural LSH)
- Check if parameters match dataset scale

**Slow queries:**
- Decrease: L, nprobe, T
- Consider using PQ compression (IVF-PQ)

**Long build time:**
- Reduce: nlist (IVF), epochs (Neural LSH)
- Use sampling for very large datasets

**High memory usage:**
- Use IVF-PQ instead of IVF-Flat
- Reduce L (LSH) or k (Hypercube)
- Neural LSH is memory-efficient

---

## References

1. Indyk & Motwani (1998) - LSH theory
2. Lv et al. (2007) - Multi-probe LSH
3. Jégou et al. (2011) - Product Quantization
4. Kraska et al. (2018) - Learned index structures

See [RESULTS.md](../RESULTS.md) for empirical parameter evaluation on real datasets.
