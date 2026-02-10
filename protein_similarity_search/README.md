# Protein Similarity Search with ANN Algorithms

A comprehensive pipeline for searching similar proteins using Approximate Nearest Neighbor (ANN) algorithms on ESM-2 protein embeddings. This project compares 5 ANN methods against BLAST as a baseline.

## Overview

This project implements protein similarity search using protein language model embeddings (ESM-2) combined with various ANN indexing methods:

| Method | Description | Speed | Accuracy |
|--------|-------------|-------|----------|
| **Euclidean LSH** | Locality Sensitive Hashing | Medium | Lower |
| **Hypercube** | Random projection to hypercube | Fast | Medium |
| **IVF-Flat** | Inverted file with flat storage | Slow* | High |
| **IVF-PQ** | Inverted file with product quantization | Slow* | High |
| **Neural LSH** | Learned hash functions with KaHIP | **Fastest** | **High** |
| BLAST | Sequence alignment (baseline) | Medium | Reference |

*IVF methods have high index build time but fast search after index is built.

## Project Structure

```
protein-similarity-search/
├── run_pipeline.sh          # Main entry point - runs everything
├── setup.sh                 # Environment setup (downloads data, creates venv)
├── protein_embed.py         # Generate ESM-2 embeddings
├── protein_embed_backend.py # Embedding implementation
├── protein_search.py        # ANN search and BLAST comparison
├── pyproject.toml           # Python dependencies (uv/pip)
├── data/
│   ├── swissprot.fasta      # Swiss-Prot database (~573K proteins, downloaded by setup.sh)
│   ├── targets.fasta        # Query proteins (user-provided or default 10 human proteins)
│   ├── protein_vectors.dat  # Pre-computed embeddings (~700MB)
│   ├── query_vectors.dat    # Query embeddings
│   ├── neural_lsh_index.pth # Trained Neural LSH index
│   └── swissprot_db.*       # BLAST database files
└── output/
    ├── results.txt          # Search results
    └── REPORT.md            # Analysis report
```

## Quick Start

### Option 1: Full Setup from Scratch

```bash
# Navigate to project
cd protein-similarity-search

# Run setup script - this will:
#   1. Create Python virtual environment
#   2. Install dependencies
#   3. Download Swiss-Prot database (~90MB compressed, ~250MB extracted)
#   4. Create BLAST database
#   5. Build C++ components
#   6. Generate protein embeddings (takes several hours on CPU)
./setup.sh

# Run the search pipeline
./run_pipeline.sh
```

### Option 2: Using Pre-computed Embeddings

If you already have `protein_vectors.dat`:

```bash
# Setup environment only (skip embedding generation)
./setup.sh --quick

# Run pipeline with existing embeddings
./run_pipeline.sh --skip-embed
```

### Option 3: Step-by-Step Manual Setup

```bash
# 1. Install dependencies (using uv)
uv sync

# Or with pip:
# pip install .

# 2. Download Swiss-Prot (if not using setup.sh)
wget -O data/uniprot_sprot.fasta.gz \
    https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip -c data/uniprot_sprot.fasta.gz > data/swissprot.fasta

# 4. Create BLAST database
makeblastdb -in data/swissprot.fasta -dbtype prot -out data/swissprot_db -parse_seqids

# 5. Build C++ components
cd ../cpp-ann-algorithms && make && cd ../protein-similarity-search

# 6. Generate embeddings (slow - several hours on CPU)
python protein_embed.py -i data/swissprot.fasta -o data/protein_vectors.dat

# 7. Provide query proteins (create data/targets.fasta)

# 8. Run search
python protein_search.py \
    -d data/protein_vectors.dat \
    -q data/targets.fasta \
    -o output/results.txt \
    -method all
```

## Detailed Usage

### setup.sh

The setup script handles all initial configuration:

```bash
./setup.sh              # Full setup (downloads data, generates embeddings)
./setup.sh --quick      # Skip embedding generation
./setup.sh --clean      # Remove all data and start fresh
./setup.sh --check      # Check environment only
./setup.sh --batch=N    # Set batch size for embedding (default: 64 GPU, 8 CPU)
```

What setup.sh does:
1. Checks system requirements (Python 3.9+, pip, wget/curl, BLAST+)
2. Detects GPU/accelerator support
3. Creates Python virtual environment
4. Installs PyTorch and dependencies
5. Downloads Swiss-Prot database from UniProt FTP
6. Creates BLAST database
7. Builds C++ search binary
8. Generates protein embeddings (unless --quick)

### run_pipeline.sh

The main script that orchestrates the search pipeline:

```bash
./run_pipeline.sh                    # Full run
./run_pipeline.sh --skip-embed       # Use existing embeddings
./run_pipeline.sh --quick            # Quick test with 1000 proteins
./run_pipeline.sh --method neural    # Run specific method only
./run_pipeline.sh -N 100             # Change number of neighbors
./run_pipeline.sh --rebuild-index    # Force rebuild Neural LSH index
./run_pipeline.sh --help             # Show all options
```

### protein_embed.py

Generates ESM-2 embeddings for proteins:

```bash
python protein_embed.py \
    -i data/swissprot.fasta \    # Input FASTA file
    -o data/protein_vectors.dat \ # Output embeddings
    -b 8 \                        # Batch size (higher = faster, more RAM)
    -n 1000 \                     # First N proteins only (for testing)
    -v                            # Verbose progress
```

**Output format:** Binary file containing:
- Header: dimension (int32), count (int32)
- Data: float32 vectors (320 dimensions each)
- IDs: JSON file with protein accessions (`*_ids.json`)

### protein_search.py

Runs ANN search and compares with BLAST:

```bash
python protein_search.py \
    -d data/protein_vectors.dat \   # Database embeddings
    -q data/targets.fasta \         # Query proteins FASTA
    -o output/results.txt \         # Output file
    -method all \                   # Method: all|lsh|hypercube|ivfflat|ivfpq|neural|blast
    -N 50 \                         # Top-N neighbors
    --query-embeds data/query.dat \ # Pre-computed query embeddings (optional)
    --blast-db data/swissprot_db    # BLAST database path
```

**Available methods:**
- `all` - Run all methods and compare
- `lsh` - Euclidean LSH only
- `hypercube` - Hypercube projection only
- `ivfflat` - IVF-Flat (FAISS) only
- `ivfpq` - IVF-PQ (FAISS) only
- `neural` - Neural LSH only
- `blast` - BLAST only

## Configuration

Algorithm parameters are defined in `protein_search.py` (Config class):

```python
@dataclass
class Config:
    num_nearest: int = 50       # Top-N neighbors
    
    # LSH parameters
    lsh_k: int = 6              # Hash functions per table
    lsh_L: int = 8              # Number of hash tables
    lsh_w: float = 1.0          # Bucket width
    
    # Hypercube parameters
    hypercube_kproj: int = 12   # Projection dimensions (2^12 vertices)
    hypercube_w: float = 1.5    # Window width
    hypercube_M: int = 5000     # Max candidates
    hypercube_probes: int = 2   # Vertices to probe
    
    # IVF-Flat parameters
    ivfflat_nlist: int = 1000   # Number of clusters
    ivfflat_nprobe: int = 50    # Clusters to search
    
    # IVF-PQ parameters  
    ivfpq_m: int = 16           # Subquantizers
    ivfpq_nbits: int = 8        # Bits per subquantizer
    
    # Neural LSH parameters
    neural_blocks: int = 400    # KaHIP partitions
    neural_T: int = 50          # Partitions to probe
    neural_epochs: int = 15     # Training epochs
    neural_hidden: str = '128,128'  # Network architecture
```

## Output Format

Results are written to `output/results.txt`:

```
================================================================================
Query Protein: P00533
N = 50 (Top-N list size for Recall@N evaluation)

[1] Method Comparison Summary
--------------------------------------------------------------------------------
Method               | Time/query (s)  | QPS        | Recall@N vs BLAST Top-N  
--------------------------------------------------------------------------------
Euclidean LSH        | 3.6585          | 0.4        | 0.11                     
Neural LSH           | 1.0074          | 9.5        | 0.59                     
...

[2] Top-N neighbors per method (showing first 10)
--------------------------------------------------------------------------------
Rank   | Neighbor ID     | L2 Dist    | BLAST Identity | In BLAST Top-N?
--------------------------------------------------------------------------------
1      | P00533          | 0.00       | 100%           | Yes              
2      | P55245          | 0.04       | 99%            | Yes              
...
```

## Installation

### Prerequisites

- Python 3.9+ (3.10+ recommended)
- GCC/G++ compiler (for C++ components)
- BLAST+ (for baseline comparison)
- wget or curl (for downloading Swiss-Prot)
- ~1GB disk space for data files

### Installing BLAST+

```bash
# Ubuntu/Debian
sudo apt-get install ncbi-blast+

# macOS
brew install blast

# Verify installation
blastp -version
```

## Expected Results

With the default configuration on Swiss-Prot (573K proteins):

| Method | QPS | Recall@50 | Notes |
|--------|-----|-----------|-------|
| Neural LSH | **9.5** | **0.67** | Best speed-accuracy trade-off |
| Hypercube | 1.1 | 0.60 | Good balance |
| BLAST | 2.3 | 1.00 | Reference (sequence alignment) |
| Euclidean LSH | 0.4 | 0.38 | Lower accuracy with current params |
| IVF-Flat | <0.1 | 0.67 | High index build time (~1600s) |
| IVF-PQ | <0.1 | 0.67 | High index build time (~3800s) |

**Key findings:**
- **Neural LSH** provides the best speed-accuracy trade-off (4x faster than BLAST, 0.67 recall)
- ESM-2 embeddings can detect **remote homologs** missed by BLAST
- IVF methods have high setup cost but are suitable for static databases

## Query Proteins

The default `data/targets.fasta` includes 10 representative human proteins:

| UniProt ID | Protein | Function |
|------------|---------|----------|
| P00533 | EGFR | Epidermal Growth Factor Receptor (kinase) |
| P04637 | p53 | Tumor suppressor |
| P00915 | CAH1 | Carbonic Anhydrase I |
| P68871 | HBB | Hemoglobin beta subunit |
| P62158 | CALM | Calmodulin |
| P01308 | INS | Insulin |
| P00918 | CAH2 | Carbonic Anhydrase II |
| P69905 | HBA | Hemoglobin alpha subunit |
| P01116 | KRAS | GTPase KRas (oncogene) |
| P02144 | MB | Myoglobin |

You can replace this file with your own query proteins in FASTA format.

## Dependencies

Core dependencies (managed via `pyproject.toml`):
- `torch` — PyTorch for ESM-2 model
- `transformers` — HuggingFace transformers
- `biopython` — FASTA parsing
- `kahip` — Graph partitioning for Neural LSH
- `scikit-learn` — k-NN graph construction
- `numpy`, `tqdm` — Utilities

Install with `uv sync` or `pip install .`.

## Related Components

This project uses components from:
- **Part 1** (`../cpp-ann-algorithms/`): C++ implementation of LSH, Hypercube, IVF-Flat, and IVF-PQ
- **Part 2** (`../neural-lsh/`): Neural LSH implementation with KaHIP graph partitioning

