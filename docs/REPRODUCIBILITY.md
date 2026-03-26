# Reproducibility Guide

This guide provides step-by-step instructions to reproduce the benchmark results reported in [RESULTS.md](../RESULTS.md).

## Quick Overview

**Time required:**
- Setup: 30 minutes - 2 hours (depending on internet speed)
- MNIST/SIFT benchmarks: 1-2 hours
- Protein search (with embedding generation): 4-8 hours (CPU) or 1-2 hours (GPU)

**Prerequisites:**
- Ubuntu 20.04+ (or similar Linux distribution)
- Python 3.9+
- C++ compiler (g++ 9+)
- 8GB RAM minimum (16GB recommended for protein search)
- ~2GB disk space

---

## Environment Setup

### System Requirements

**Verified platforms:**
- Ubuntu 20.04 LTS, 22.04 LTS
- Debian 11+
- macOS should work but not extensively tested

**Dependencies:**
```bash
# Update system
sudo apt-get update

# C++ build tools
sudo apt-get install -y build-essential g++ cmake

# Python development
sudo apt-get install -y python3 python3-pip python3-venv

# BLAST+ (for protein search)
sudo apt-get install -y ncbi-blast+

# Download tools
sudo apt-get install -y wget curl
```

### Python Environment

**Option 1: Virtual environment (recommended)**
```bash
cd vector-similarity-search

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

**Option 2: System-wide (not recommended)**
```bash
pip3 install --user <packages>
```

### Exact Package Versions

For maximum reproducibility, use the exact versions tested:

**Neural LSH:**
```bash
cd algorithms/neural_lsh
pip install -r requirements.txt

# Or manually:
pip install torch==2.9.1
pip install numpy==2.4.1
pip install kahip==3.22
pip install scikit-learn==1.8.0
pip install matplotlib==3.9.0
pip install pandas==2.2.1
pip install seaborn==0.13.2
```

**Protein Search:**
```bash
cd test_framework-protein_folding
pip install -r requirements.txt

# Key packages:
pip install torch==2.9.1
pip install transformers==4.57.3
pip install biopython==1.86
pip install kahip==3.22
pip install scikit-learn==1.8.0
```

**Note:** KaHIP requires CMake and a C++ compiler to build. If installation fails:
```bash
# Install CMake first
sudo apt-get install cmake

# Then retry
pip install kahip
```

---

## Reproducing MNIST/SIFT Benchmarks

### 1. Build C++ Components

```bash
cd algorithms/lsh-hypercube-ivf

# Clean build
make clean
make

# Verify build
ls -lh bin/search
./bin/search --help || echo "Binary built successfully"
```

### 2. Download Datasets

MNIST and SIFT datasets are downloaded automatically by benchmark scripts.

**Manual download (optional):**

**MNIST:**
```bash
# The benchmark script handles this automatically
# If you want to download manually:
mkdir -p data
cd data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
# (parsing scripts included in repository)
```

**SIFT:**
```bash
# Download SIFT1M from TEXMEX
mkdir -p data
cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
```

### 3. Run Benchmarks

**Full benchmark suite:**
```bash
cd algorithms/lsh-hypercube-ivf
make benchmark

# This will:
# - Download datasets if needed
# - Run all algorithms on MNIST and SIFT
# - Generate result files
# - Report performance metrics
```

**Individual dataset benchmarks:**
```bash
# MNIST only
make benchmark-mnist

# SIFT only
make benchmark-sift
```

### 4. Verify Results

Check output in `algorithms/lsh-hypercube-ivf/output/`:
- Performance metrics (Recall@N, QPS)
- Should match values in RESULTS.md within ±5%

**Expected results:**
- **MNIST:** Neural LSH recall ~0.89, QPS ~12
- **SIFT:** Neural LSH recall ~0.85, QPS ~18

Small variations are normal due to:
- Random initialization in Neural LSH
- CPU/memory differences
- Compiler optimizations

---

## Reproducing Protein Search Results

### 1. Setup Environment

```bash
cd test_framework-protein_folding

# Full setup (downloads data, generates embeddings)
./setup.sh

# This will:
# - Create Python virtual environment
# - Install all dependencies
# - Download Swiss-Prot (~90MB compressed)
# - Create BLAST database
# - Build C++ components
# - Generate protein embeddings (LONG - several hours on CPU)
```

**Quick setup (skip embedding generation):**
```bash
./setup.sh --quick

# Then use pre-computed embeddings if available
```

### 2. Generate Embeddings (if not using pre-computed)

This is the most time-consuming step:

```bash
# Activate virtual environment
source venv/bin/activate

# Generate embeddings
python protein_embed.py \
    -i data/swissprot.fasta \
    -o data/protein_vectors.dat \
    -b 64  # batch size (reduce if out of memory)

# Progress: ~2000 proteins/min on GPU, ~100/min on CPU
# Total time: ~30 min (GPU) or 4-8 hours (CPU)
```

**GPU acceleration:**
- Automatically used if CUDA available
- Check with: `python -c "import torch; print(torch.cuda.is_available())"`
- Speeds up embedding ~20x

### 3. Run Protein Search

**All methods:**
```bash
./run_pipeline.sh

# Or if embeddings already generated:
./run_pipeline.sh --skip-embed
```

**Single method (faster):**
```bash
./run_pipeline.sh --method neural --skip-embed
```

**Available methods:**
- `neural` - Neural LSH (recommended)
- `lsh` - Euclidean LSH
- `hypercube` - Hypercube projection
- `ivfflat` - IVF-Flat
- `ivfpq` - IVF-PQ
- `blast` - BLAST baseline
- `all` - Run all methods

### 4. Verify Results

Check `output/results.txt`:
- Neural LSH: QPS ~9.5, Recall@50 ~0.67
- Compare with RESULTS.md tables

**Variations expected:**
- ±10% on QPS (hardware dependent)
- ±0.05 on recall (stochastic algorithms)

---

## Random Seeds and Determinism

### Ensuring Reproducibility

All stochastic components use fixed seeds:

**Neural LSH:**
```python
# In nlsh_build.py
np.random.seed(args.seed)  # Default: 1
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
# KaHIP also uses this seed
```

**To reproduce exact results:**
```bash
# Always use the same seed
python nlsh_build.py -d dataset.dat -i index.pth --seed 42
python nlsh_search.py -i index.pth --seed 42 ...
```

**Sources of non-determinism:**
- CUDA operations (can be forced deterministic with environment variables)
- KaHIP graph partitioning (uses seed but has minor variations)
- OS thread scheduling

**Making CUDA deterministic:**
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python script.py  # Will be slower but deterministic
```

---

## Hardware Specifications

**Benchmarks in RESULTS.md performed on:**
- **CPU:** Intel Xeon / AMD EPYC (typical server CPU)
- **RAM:** 16GB
- **GPU:** NVIDIA GPU with CUDA support (for embeddings)
- **Storage:** SSD

**Performance scaling:**
- **More RAM:** Allows larger batch sizes
- **Better CPU:** Faster k-NN graph construction, KaHIP
- **GPU:** Only affects embedding generation (20x speedup)
- **SSD vs HDD:** Minimal impact (data fits in RAM)

**Minimum specifications:**
- CPU: Dual-core 2GHz+
- RAM: 8GB (12GB for protein search)
- Storage: 5GB free space

---

## Troubleshooting

### Build Failures

**"g++ not found"**
```bash
sudo apt-get install build-essential
```

**"Python.h not found"**
```bash
sudo apt-get install python3-dev
```

**"CMake required for kahip"**
```bash
sudo apt-get install cmake
pip install kahip
```

### Runtime Issues

**"CUDA out of memory"**
```bash
# Reduce batch size
python protein_embed.py -b 8  # instead of 64

# Or use CPU
CUDA_VISIBLE_DEVICES="" python protein_embed.py
```

**"Dataset not found"**
```bash
# Re-run setup
cd test_framework-protein_folding
./setup.sh --clean  # Remove old data
./setup.sh          # Fresh setup
```

**"Different results than reported"**
- Check seed: should be consistent
- Verify package versions: `pip list | grep -E "torch|numpy|kahip"`
- Check dataset version: Swiss-Prot updates regularly

### Result Verification

**Compare your results:**
```bash
# Your results
grep "Recall@50" output/results.txt

# Expected (from RESULTS.md)
# Neural LSH: 0.67 ± 0.05
# Hypercube: 0.60 ± 0.05
```

**If results differ significantly (>20%):**
1. Verify dataset hasn't changed (Swiss-Prot updates quarterly)
2. Check random seeds are set correctly
3. Ensure correct algorithm parameters (see PARAMETERS.md)
4. Report issue with: OS, Python version, package versions

---

## Dataset Versions

### Swiss-Prot
- **Version used in RESULTS.md:** 2025.01 release
- **Protein count:** ~573,000
- **URL:** https://ftp.uniprot.org/pub/databases/uniprot/current_release/

**To use specific version:**
```bash
# Download specific release
wget https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2025_01/knowledgebase/uniprot_sprot-only2025_01.tar.gz
```

### MNIST
- **Version:** Original (unchanging)
- **URL:** http://yann.lecun.com/exdb/mnist/

### SIFT
- **Version:** SIFT1M from TEXMEX
- **URL:** http://corpus-texmex.irisa.fr/

---

## CI/CD Verification

The repository includes GitHub Actions workflows that verify:
- C++ code compiles successfully
- Python imports work correctly
- Basic functionality runs

**To run CI checks locally:**

**C++ build:**
```bash
cd algorithms/lsh-hypercube-ivf
make clean && make
# Should complete without errors
```

**Python imports:**
```bash
cd algorithms/neural_lsh
python -c "import nlsh_build; import nlsh_search"
# Should complete without errors
```

---

## Long-Term Reproducibility

### Archiving Results

**Save your results:**
```bash
# Create archive
tar -czf reproducibility_$(date +%Y%m%d).tar.gz \
    algorithms/lsh-hypercube-ivf/output/ \
    test_framework-protein_folding/output/ \
    *.md

# Save package versions
pip freeze > requirements_frozen.txt
```

### Docker Container (Advanced)

For maximum reproducibility, consider creating a Docker container:

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential python3 python3-pip cmake ncbi-blast+

COPY . /app
WORKDIR /app

RUN cd algorithms/lsh-hypercube-ivf && make
RUN cd algorithms/neural_lsh && pip3 install -r requirements.txt

CMD ["/bin/bash"]
```

---

## Citing Reproduced Results

If you reproduce these results in your research:

```bibtex
@software{Tsekrekos_Vector_Similarity_Search,
  author = {Tsekrekos, Egor-Andrianos},
  title = {Vector Similarity Search: High-Performance ANN Algorithms},
  year = {2025},
  url = {https://github.com/TsekrekosEA/vector-similarity-search},
  note = {Reproduced on [your date] using version [commit hash]}
}
```

---

## Contact

**Issues with reproduction:**
- Open a GitHub issue: https://github.com/TsekrekosEA/vector-similarity-search/issues
- Include: OS, Python version, error messages, steps taken

**Questions:**
- Email: egor.andrianos.tsekrekos@gmail.com
- Provide details about your setup and what you've tried

---

## Appendix: Complete Setup Script

```bash
#!/bin/bash
# Complete setup for reproducing all results

set -e

echo "=== Vector Similarity Search - Reproducibility Setup ==="

# 1. System dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake python3 python3-pip python3-venv ncbi-blast+ wget

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 3. Build C++ components
cd algorithms/lsh-hypercube-ivf
make clean && make
cd ../..

# 4. Install Python dependencies
cd algorithms/neural_lsh
pip install -r requirements.txt
cd ../..

cd test_framework-protein_folding
pip install -r requirements.txt
cd ..

# 5. Run MNIST/SIFT benchmarks
cd algorithms/lsh-hypercube-ivf
make benchmark
cd ../..

# 6. Setup protein search (without embeddings - too long)
cd test_framework-protein_folding
./setup.sh --quick
cd ..

echo ""
echo "✓ Setup complete!"
echo ""
echo "To run protein search with pre-computed embeddings:"
echo "  cd test_framework-protein_folding"
echo "  ./run_pipeline.sh --skip-embed --method neural"
echo ""
echo "To generate embeddings (takes hours):"
echo "  cd test_framework-protein_folding"
echo "  python protein_embed.py -i data/swissprot.fasta -o data/protein_vectors.dat"
```

Save as `reproduce_all.sh` and run with `bash reproduce_all.sh`.
