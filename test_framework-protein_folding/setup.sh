#!/bin/bash
# =============================================================================
# Protein Similarity Search - Setup Script
# =============================================================================
# This script sets up the complete environment for the protein search project:
#   1. Creates Python virtual environment
#   2. Installs Python dependencies (with GPU support if available)
#   3. Downloads Swiss-Prot database
#   4. Creates BLAST database
#   5. Generates protein embeddings
#   6. Builds C++ components
#
# Usage:
#   ./setup.sh              # Full setup
#   ./setup.sh --quick      # Skip embedding (for testing)
#   ./setup.sh --clean      # Clean all data and start fresh
#   ./setup.sh --check      # Check environment only
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Parse arguments
QUICK_MODE=false
CLEAN_MODE=false
CHECK_ONLY=false
EMBED_BATCH_SIZE=64  # Higher for GPU

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            ;;
        --clean)
            CLEAN_MODE=true
            ;;
        --check)
            CHECK_ONLY=true
            ;;
        --batch=*)
            EMBED_BATCH_SIZE="${arg#*=}"
            ;;
        --help|-h)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick     Skip embedding generation (use existing or for testing)"
            echo "  --clean     Remove all data and start fresh"
            echo "  --check     Check environment only, don't install anything"
            echo "  --batch=N   Batch size for embedding (default: 64 for GPU, 8 for CPU)"
            echo "  --help      Show this help message"
            exit 0
            ;;
    esac
done

# =============================================================================
# Clean mode
# =============================================================================
if [ "$CLEAN_MODE" = true ]; then
    print_header "Cleaning all data..."
    
    rm -rf data/*.dat data/*.json data/*.fasta data/*.gz data/swissprot_db.* 2>/dev/null || true
    rm -rf output/* 2>/dev/null || true
    rm -rf venv 2>/dev/null || true
    
    print_success "Cleaned all data and virtual environment"
    echo "Run ./setup.sh again to set up from scratch"
    exit 0
fi

# =============================================================================
# Check system requirements
# =============================================================================
print_header "Checking System Requirements"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_success "Python 3 found: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.9+"
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null || python3 -m pip --version &> /dev/null; then
    print_success "pip found"
else
    print_error "pip not found. Please install pip"
    exit 1
fi

# Check wget or curl
if command -v wget &> /dev/null; then
    DOWNLOADER="wget"
    print_success "wget found"
elif command -v curl &> /dev/null; then
    DOWNLOADER="curl"
    print_success "curl found"
else
    print_error "Neither wget nor curl found. Please install one."
    exit 1
fi

# Check BLAST+
if command -v blastp &> /dev/null; then
    BLAST_VERSION=$(blastp -version 2>&1 | head -1)
    print_success "BLAST+ found: $BLAST_VERSION"
else
    print_warning "BLAST+ not found. Install with: sudo apt install ncbi-blast+"
    print_warning "BLAST is required for ground truth evaluation"
fi

# Check for GPU support
GPU_AVAILABLE=false

# Quick check for GPU using PyTorch (if available)
echo "  Checking for GPU/accelerator support..."
GPU_CHECK=$(python3 -c "
import torch
if torch.cuda.is_available():
    print('cuda:' + torch.cuda.get_device_name(0))
else:
    print('cpu')
" 2>/dev/null || echo "cpu")

if [[ "$GPU_CHECK" == cuda:* ]]; then
    GPU_AVAILABLE=true
    print_success "NVIDIA GPU found: ${GPU_CHECK#cuda:}"
else
    print_warning "No GPU acceleration detected. Will use CPU (embedding will be slower)."
    EMBED_BATCH_SIZE=8
fi

# Check C++ build
CPP_BUILD="../algorithms/lsh-hypercube-ivf/bin/search"
if [ -f "$CPP_BUILD" ]; then
    print_success "C++ search binary found"
else
    print_warning "C++ search binary not found at $CPP_BUILD"
    print_warning "Will attempt to build it"
fi

if [ "$CHECK_ONLY" = true ]; then
    print_header "Environment Check Complete"
    exit 0
fi

# =============================================================================
# Create virtual environment
# =============================================================================
print_header "Setting up Python Virtual Environment"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# =============================================================================
# Install Python dependencies
# =============================================================================
print_header "Installing Python Dependencies"

# Install PyTorch - use standard install (works with CUDA, Intel XPU, or CPU)
# Same approach as Part 2 - just "pip install torch"
echo "Installing PyTorch..."
pip install torch 2>&1 | tail -3
print_success "PyTorch installed"

# Install other dependencies
echo "Installing other dependencies..."
pip install transformers biopython numpy tqdm 2>&1 | tail -3
print_success "All Python dependencies installed"

# Verify PyTorch and check for accelerators
echo ""
echo "Verifying PyTorch installation..."
python3 -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
try:
    import intel_extension_for_pytorch as ipex
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f'  Intel XPU available')
except:
    pass
"

# =============================================================================
# Create data directory
# =============================================================================
print_header "Setting up Data Directory"

mkdir -p data
mkdir -p output

# =============================================================================
# Download Swiss-Prot database
# =============================================================================
print_header "Downloading Swiss-Prot Database"

SWISSPROT_URL="https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
SWISSPROT_GZ="data/uniprot_sprot.fasta.gz"
SWISSPROT_FASTA="data/swissprot.fasta"

if [ -f "$SWISSPROT_FASTA" ]; then
    PROTEIN_COUNT=$(grep -c "^>" "$SWISSPROT_FASTA" 2>/dev/null || echo "0")
    if [ "$PROTEIN_COUNT" -gt 500000 ]; then
        print_success "Swiss-Prot already downloaded ($PROTEIN_COUNT proteins)"
    else
        print_warning "Swiss-Prot file seems incomplete ($PROTEIN_COUNT proteins), re-downloading..."
        rm -f "$SWISSPROT_FASTA" "$SWISSPROT_GZ"
    fi
fi

if [ ! -f "$SWISSPROT_FASTA" ]; then
    echo "Downloading Swiss-Prot database (~90MB compressed)..."
    
    if [ "$DOWNLOADER" = "wget" ]; then
        wget -q --show-progress -O "$SWISSPROT_GZ" "$SWISSPROT_URL"
    else
        curl -L --progress-bar -o "$SWISSPROT_GZ" "$SWISSPROT_URL"
    fi
    
    print_success "Download complete"
    
    echo "Extracting..."
    gunzip -c "$SWISSPROT_GZ" > "$SWISSPROT_FASTA"
    rm -f "$SWISSPROT_GZ"
    
    PROTEIN_COUNT=$(grep -c "^>" "$SWISSPROT_FASTA")
    print_success "Swiss-Prot extracted: $PROTEIN_COUNT proteins"
fi

# =============================================================================
# Create BLAST database
# =============================================================================
print_header "Creating BLAST Database"

BLAST_DB="data/swissprot_db"

if [ -f "${BLAST_DB}.phr" ] && [ -f "${BLAST_DB}.pin" ] && [ -f "${BLAST_DB}.psq" ]; then
    print_success "BLAST database already exists"
else
    if command -v makeblastdb &> /dev/null; then
        echo "Creating BLAST database..."
        makeblastdb -in "$SWISSPROT_FASTA" -dbtype prot -out "$BLAST_DB" -parse_seqids > /dev/null 2>&1
        print_success "BLAST database created"
    else
        print_warning "makeblastdb not found, skipping BLAST database creation"
        print_warning "Install BLAST+ to enable ground truth evaluation"
    fi
fi

# =============================================================================
# Check for query proteins file
# =============================================================================
print_header "Checking Query Proteins File"

QUERY_FASTA="data/targets.fasta"

if [ -f "$QUERY_FASTA" ]; then
    QUERY_COUNT=$(grep -c "^>" "$QUERY_FASTA" 2>/dev/null || echo "0")
    print_success "Query file found: $QUERY_FASTA ($QUERY_COUNT proteins)"
else
    print_warning "Query file not found: $QUERY_FASTA"
    echo ""
    echo "  Please provide your own targets.fasta file with query proteins."
    echo "  This file should contain the proteins you want to search for."
    echo "  Place it at: data/targets.fasta"
    echo ""
    echo "  Example FASTA format:"
    echo "    >sp|P12345|PROT_HUMAN Protein name"
    echo "    MSEQVENCE..."
    echo ""
fi

# =============================================================================
# Build C++ components
# =============================================================================
print_header "Building C++ Components"

CPP_DIR="../algorithms/lsh-hypercube-ivf"

if [ -f "$CPP_DIR/Makefile" ]; then
    echo "Building C++ search binary..."
    cd "$CPP_DIR"
    make clean > /dev/null 2>&1 || true
    if make -j$(nproc) 2>&1 | tail -5; then
        print_success "C++ components built successfully"
    else
        print_warning "C++ build failed (may still work if already built)"
    fi
    cd "$SCRIPT_DIR"
else
    print_warning "C++ Makefile not found at $CPP_DIR/Makefile"
fi

# =============================================================================
# Generate protein embeddings
# =============================================================================
print_header "Generating Protein Embeddings"

EMBEDDINGS_FILE="data/protein_vectors.dat"
EMBEDDINGS_IDS="data/protein_vectors_ids.json"

if [ "$QUICK_MODE" = true ]; then
    print_warning "Quick mode: Skipping embedding generation"
    print_warning "Run without --quick to generate embeddings"
else
    # Check if embeddings already exist and are complete
    if [ -f "$EMBEDDINGS_FILE" ] && [ -f "$EMBEDDINGS_IDS" ]; then
        EMBED_COUNT=$(python3 -c "
import struct, os
with open('$EMBEDDINGS_FILE', 'rb') as f:
    d = struct.unpack('<I', f.read(4))[0]
file_size = os.path.getsize('$EMBEDDINGS_FILE')
n = file_size // (4 + d * 4)
print(n)
" 2>/dev/null || echo "0")
        
        PROTEIN_COUNT=$(grep -c "^>" "$SWISSPROT_FASTA" 2>/dev/null || echo "0")
        
        if [ "$EMBED_COUNT" -eq "$PROTEIN_COUNT" ]; then
            print_success "Embeddings already complete ($EMBED_COUNT vectors)"
        else
            print_warning "Embeddings incomplete ($EMBED_COUNT of $PROTEIN_COUNT)"
            echo "Regenerating embeddings..."
            rm -f "$EMBEDDINGS_FILE" "$EMBEDDINGS_IDS"
        fi
    fi
    
    if [ ! -f "$EMBEDDINGS_FILE" ] || [ ! -f "$EMBEDDINGS_IDS" ]; then
        echo ""
        echo "This will embed all proteins in Swiss-Prot (~570K proteins)."
        if [ "$GPU_AVAILABLE" = true ]; then
            echo "Using GPU with batch size $EMBED_BATCH_SIZE"
            echo "Estimated time: 30-60 minutes"
        else
            echo "Using CPU with batch size $EMBED_BATCH_SIZE"
            echo "Estimated time: Several hours"
        fi
        echo ""
        
        read -p "Start embedding now? [Y/n] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            print_warning "Skipping embedding. Run later with:"
            echo "  source venv/bin/activate"
            echo "  python protein_embed.py -b $EMBED_BATCH_SIZE -v"
        else
            echo "Starting embedding..."
            python3 protein_embed.py -b "$EMBED_BATCH_SIZE" -v
            print_success "Embedding complete!"
        fi
    fi
fi

# =============================================================================
# Generate query embeddings
# =============================================================================
print_header "Generating Query Embeddings"

QUERY_EMBEDDINGS="data/query_vectors.dat"

if [ -f "$QUERY_EMBEDDINGS" ]; then
    print_success "Query embeddings already exist"
elif [ ! -f "$QUERY_FASTA" ]; then
    print_warning "Cannot generate query embeddings - targets.fasta not found"
    echo "  Please provide your query proteins file first, then run:"
    echo "  python3 protein_embed.py -i data/targets.fasta -o data/query_vectors.dat -b 8"
else
    echo "Generating embeddings for query proteins..."
    python3 protein_embed.py -i "$QUERY_FASTA" -o "$QUERY_EMBEDDINGS" -b 8
    print_success "Query embeddings generated"
fi

# =============================================================================
# Final summary
# =============================================================================
print_header "Setup Complete!"

echo ""
echo "Directory structure:"
echo "  data/"
echo "    ├── swissprot.fasta       - Swiss-Prot database ($(grep -c "^>" data/swissprot.fasta 2>/dev/null || echo "?") proteins)"
echo "    ├── swissprot_db.*        - BLAST database"
echo "    ├── protein_vectors.dat   - Protein embeddings"
echo "    ├── protein_vectors_ids.json - Protein ID mapping"
echo "    ├── targets.fasta         - Query proteins (YOU MUST PROVIDE THIS)"
echo "    └── query_vectors.dat     - Query embeddings"
echo ""

# Check what's missing
if [ ! -f "data/targets.fasta" ]; then
    echo -e "${YELLOW}⚠ IMPORTANT: You need to provide data/targets.fasta${NC}"
    echo "  This file should contain your query proteins as specified in the PDF."
    echo ""
fi

if [ ! -f "data/query_vectors.dat" ]; then
    echo -e "${YELLOW}⚠ Query embeddings not yet generated.${NC}"
    echo "  After providing targets.fasta, run:"
    echo "  python protein_embed.py -i data/targets.fasta -o data/query_vectors.dat -b 8"
    echo ""
fi

echo "To run the protein search:"
echo "  source venv/bin/activate"
echo "  python protein_search.py --help"
echo ""
echo "Example commands:"
echo "  python protein_search.py -method lsh         # Run LSH only"
echo "  python protein_search.py -method all         # Run all methods"
echo "  python protein_search.py -method neural      # Run Neural LSH"
echo ""

if [ "$GPU_AVAILABLE" = true ]; then
    echo -e "${GREEN}GPU acceleration is enabled!${NC}"
else
    echo -e "${YELLOW}Running on CPU. For faster embedding, use a machine with NVIDIA GPU.${NC}"
fi

echo ""
print_success "Setup complete!"
