# Architecture

## System Overview

```mermaid
graph TB
    subgraph "Data Sources"
        A[FASTA Proteins<br/>Swiss-Prot 573K]
        B[MNIST Images<br/>60K x 784-dim]
        C[SIFT Descriptors<br/>128-dim]
    end
    
    subgraph "Embedding Generation"
        D[ESM-2 Model<br/>facebook/esm2_t6_8M_UR50D]
        E[Binary Parser<br/>parse_mnist.py]
        F[Binary Parser<br/>parse_sift.py]
    end
    
    subgraph "Vector Storage"
        G[Protein Vectors<br/>320-dim float32]
        H[Image Vectors<br/>784-dim uint8]
        I[SIFT Vectors<br/>128-dim float32]
    end
    
    subgraph "Index Building"
        J1[Neural LSH Builder<br/>kNN + KaHIP + MLP]
        J2[C++ Index Builder<br/>LSH/Hypercube/IVF]
    end
    
    subgraph "Search Methods"
        K1[LSH<br/>Random Projections]
        K2[Hypercube<br/>Hamming Space]
        K3[IVF-Flat<br/>Inverted Index]
        K4[IVF-PQ<br/>Product Quantization]
        K5[Neural LSH<br/>Learned Partitions]
        K6[BLAST<br/>Sequence Alignment]
    end
    
    subgraph "Results"
        L[Neighbors + Metrics<br/>Recall, QPS, AF]
        M[Visualizations<br/>Recall vs Speedup]
    end
    
    A -->|protein_embed.py| D
    D -->|.dat + _ids.json| G
    B --> E --> H
    C --> F --> I
    
    G --> J1
    H & I --> J1 & J2
    
    J1 -->|.pth checkpoint| K5
    J2 -->|in-memory structures| K1 & K2 & K3 & K4
    
    G -->|protein_search.py| K1 & K2 & K3 & K4 & K5 & K6
    H & I -->|nlsh_search.py or bin/search| K1 & K2 & K3 & K4 & K5
    
    K1 & K2 & K3 & K4 & K5 & K6 --> L
    L -->|plot_results.py| M
    
    style D fill:#e1f5ff
    style J1 fill:#fff4e6
    style J2 fill:#fff4e6
    style K5 fill:#e8f5e9
    style G fill:#f3e5f5
```

## Component Details

### 1. Data Sources & Embedding Generation

**Protein Data**
- Input: FASTA sequences from Swiss-Prot (~573K proteins)
- Embedding: ESM-2 protein language model (320-dimensional)
- Output: Binary .dat files + JSON ID mappings

**MNIST Data**
- Input: Handwritten digit images (28x28 pixels)
- Format: 784-dimensional uint8 vectors
- Preprocessing: Flatten to vector, no normalization

**SIFT Data**
- Input: SIFT descriptors from images
- Format: 128-dimensional float32 vectors
- Standard computer vision features

### 2. Index Building

**Neural LSH (Python)**
1. Build k-NN graph using scikit-learn
2. Partition graph with KaHIP (balanced graph partitioning)
3. Train PyTorch MLP to predict partitions
4. Save checkpoint: model weights + inverted index

**C++ Algorithms**
- Build in-memory structures during initialization
- LSH: Random projection hash tables
- Hypercube: Hamming space projections
- IVF: k-means clustering + inverted files
- IVF-PQ: k-means + product quantization codebooks

### 3. Search Phase

**Query Processing**
1. Load query vectors
2. For Neural LSH: MLP predicts top-T partitions, probe candidates
3. For C++ methods: Use hash tables/clusters to narrow search space
4. Compute exact distances for candidates
5. Return top-N nearest neighbors

**BLAST (Protein-specific)**
- Traditional sequence alignment algorithm
- Baseline comparison for protein search
- Uses sequence similarity, not embeddings

### 4. Evaluation

**Metrics**
- **Recall@N**: Fraction of true neighbors found
- **QPS**: Queries per second (throughput)
- **Approximation Factor**: Ratio of approximate to true distance
- **Speedup**: vs. brute-force exact search

**Visualization**
- Recall vs. Speedup plots
- Quality analysis charts
- Algorithm comparison tables

## Data Flow Example: Protein Search

```
1. Input: targets.fasta (query proteins)
   ↓
2. ESM-2 generates 320-dim embeddings
   ↓
3. For each algorithm:
   a. Neural LSH: MLP predicts partitions → search top-T
   b. LSH: Hash query → search matching buckets
   c. Hypercube: Project → probe nearby vertices
   d. IVF-*: Assign to clusters → search nprobe clusters
   e. BLAST: Sequence alignment
   ↓
4. Collect top-N neighbors + distances
   ↓
5. Compare all methods:
   - Recall vs BLAST top-N
   - Query time
   - Distance quality
   ↓
6. Generate report: output/results.txt
```

## Performance Characteristics

| Method | Build Time | Query Speed | Accuracy | Memory |
|--------|-----------|-------------|----------|---------|
| **Neural LSH** | ~10 min | **Fastest** (9.5 QPS) | High (0.67 recall) | Medium |
| **Hypercube** | Instant | Fast (1.1 QPS) | Medium (0.60 recall) | Low |
| **IVF-Flat** | ~27 min | Slow (0.1 QPS) | High (0.67 recall) | High |
| **IVF-PQ** | ~63 min | Slow (0.1 QPS) | High (0.67 recall) | Medium |
| **LSH** | Instant | Medium (0.4 QPS) | Lower (0.38 recall) | Medium |
| **BLAST** | N/A | Medium (2.3 QPS) | Reference (1.0) | N/A |

*Benchmarked on Swiss-Prot 573K proteins, 50 nearest neighbors*

## Technology Stack

**Languages & Frameworks**
- C++17 (core algorithms, performance-critical code)
- Python 3.9+ (neural networks, embeddings, orchestration)
- PyTorch (neural network training)

**Key Libraries**
- **KaHIP**: Graph partitioning for Neural LSH
- **scikit-learn**: k-NN graph construction
- **transformers**: ESM-2 protein language model
- **BioPython**: FASTA parsing, BLAST integration

**Build System**
- Make (C++ components)
- pip (Python dependencies)
- Shell scripts (orchestration)
