#!/bin/bash
# =============================================================================
# Protein Similarity Search Pipeline
# =============================================================================
# This script runs the complete protein similarity search pipeline:
# 1. Generates protein embeddings (if needed)
# 2. Generates query embeddings (if needed)
# 3. Runs all ANN methods + BLAST comparison
# 4. Outputs results in PDF-required format
#
# Usage:
#   ./run_pipeline.sh                    # Full run
#   ./run_pipeline.sh --skip-embed       # Skip embedding (use existing)
#   ./run_pipeline.sh --quick            # Quick test with 1000 proteins
#   ./run_pipeline.sh --help             # Show help
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
SKIP_EMBED=false
QUICK_MODE=false
NUM_PROTEINS="all"
BATCH_SIZE=8
N_NEIGHBORS=50
METHOD="all"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
ROOT_VENV_DIR="$SCRIPT_DIR/../../.venv"
LOCAL_VENV_DIR="$SCRIPT_DIR/venv"
if [ -d "$ROOT_VENV_DIR" ]; then
    VENV_DIR="$ROOT_VENV_DIR"
else
    VENV_DIR="$LOCAL_VENV_DIR"
fi

# Files
SWISSPROT_FASTA="$DATA_DIR/swissprot.fasta"
TARGETS_FASTA="$DATA_DIR/targets.fasta"
PROTEIN_EMBEDS="$DATA_DIR/protein_vectors.dat"
QUERY_EMBEDS="$DATA_DIR/query_vectors.dat"
NEURAL_INDEX="$DATA_DIR/neural_lsh_index.pth"
OUTPUT_DIR="$SCRIPT_DIR/output"
RESULTS_FILE="$OUTPUT_DIR/results.txt"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
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

show_help() {
    echo "Protein Similarity Search Pipeline"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --skip-embed     Skip embedding generation (use existing embeddings)"
    echo "  --quick          Quick test mode (use first 1000 proteins only)"
    echo "  --method METHOD  Search method: all, lsh, hypercube, ivfflat, ivfpq, neural"
    echo "  --output FILE    Output results file (default: results.txt)"
    echo "  -N NUM           Number of nearest neighbors (default: 50)"
    echo "  --rebuild-index  Force rebuild of Neural LSH index"
    echo "  --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                          # Full pipeline with all proteins"
    echo "  $0 --quick                  # Quick test with 1000 proteins"
    echo "  $0 --skip-embed             # Use existing embeddings"
    echo "  $0 --method lsh -N 100      # Run only LSH with Top-100"
    echo ""
}

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    local missing=false
    
    # Check Python virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        print_error "Virtual environment not found at $VENV_DIR"
        echo "  Run ./setup.sh first to set up the environment"
        missing=true
    else
        print_success "Virtual environment exists"
    fi
    
    # Check Swiss-Prot database
    if [ ! -f "$SWISSPROT_FASTA" ]; then
        print_error "Swiss-Prot database not found at $SWISSPROT_FASTA"
        echo "  Run ./setup.sh first to download the database"
        missing=true
    else
        local protein_count=$(grep -c "^>" "$SWISSPROT_FASTA" 2>/dev/null || echo "0")
        print_success "Swiss-Prot database exists ($protein_count proteins)"
    fi
    
    # Check targets.fasta
    if [ ! -f "$TARGETS_FASTA" ]; then
        print_error "Query proteins file not found at $TARGETS_FASTA"
        echo "  Please provide your targets.fasta file as specified in the PDF"
        missing=true
    else
        local query_count=$(grep -c "^>" "$TARGETS_FASTA" 2>/dev/null || echo "0")
        print_success "Query proteins file exists ($query_count proteins)"
    fi
    
    # Check C++ binary
    local cpp_binary="$SCRIPT_DIR/../algorithms/lsh-hypercube-ivf/bin/search"
    if [ ! -f "$cpp_binary" ]; then
        print_error "C++ search binary not found at $cpp_binary"
        echo "  Run ./setup.sh or build Part 1 manually"
        missing=true
    else
        print_success "C++ search binary exists"
    fi
    
    # Check BLAST
    if ! command -v blastp &> /dev/null; then
        print_warning "BLAST not found - will skip BLAST comparison"
    else
        print_success "BLAST is available"
    fi
    
    if [ "$missing" = true ]; then
        echo ""
        print_error "Missing prerequisites. Please fix the issues above."
        exit 1
    fi
}

estimate_embedding_time() {
    local num_proteins=$1
    local rate=18  # proteins per second on CPU (approximate)
    local seconds=$((num_proteins / rate))
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    
    if [ $hours -gt 0 ]; then
        echo "${hours}h ${minutes}m"
    else
        echo "${minutes}m"
    fi
}

# =============================================================================
# Parse Arguments
# =============================================================================

REBUILD_INDEX=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-embed)
            SKIP_EMBED=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            NUM_PROTEINS=1000
            shift
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --output)
            RESULTS_FILE="$2"
            shift 2
            ;;
        -N)
            N_NEIGHBORS="$2"
            shift 2
            ;;
        --rebuild-index)
            REBUILD_INDEX=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# =============================================================================
# Main Pipeline
# =============================================================================

print_header "Protein Similarity Search Pipeline"

echo "Configuration:"
echo "  Mode: $([ "$QUICK_MODE" = true ] && echo "Quick (1000 proteins)" || echo "Full")"
echo "  Skip embedding: $SKIP_EMBED"
echo "  Method: $METHOD"
echo "  N neighbors: $N_NEIGHBORS"
echo "  Output: $RESULTS_FILE"

# Check prerequisites
check_prerequisites

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# =============================================================================
# Step 1: Generate Protein Embeddings
# =============================================================================

print_header "Step 1: Protein Embeddings"

if [ "$SKIP_EMBED" = true ] && [ -f "$PROTEIN_EMBEDS" ]; then
    print_success "Using existing protein embeddings"
    
    # Check embedding count matches
    embed_count=$(python3 -c "
import struct
count = 0
with open('$PROTEIN_EMBEDS', 'rb') as f:
    while True:
        dim_bytes = f.read(4)
        if not dim_bytes:
            break
        dim = struct.unpack('<I', dim_bytes)[0]
        f.seek(dim * 4, 1)
        count += 1
print(count)
")
    echo "  Existing embeddings: $embed_count proteins"
    
else
    if [ "$QUICK_MODE" = true ]; then
        echo "Generating embeddings for first $NUM_PROTEINS proteins..."
        protein_count=$NUM_PROTEINS
    else
        protein_count=$(grep -c "^>" "$SWISSPROT_FASTA")
        echo "Generating embeddings for all $protein_count proteins..."
        echo "Estimated time: $(estimate_embedding_time $protein_count)"
        echo ""
        read -p "This will take a while. Continue? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    fi
    
    # Remove old embeddings if they exist
    rm -f "$PROTEIN_EMBEDS" "$PROTEIN_EMBEDS.json"
    
    # Generate embeddings
    if [ "$QUICK_MODE" = true ]; then
        python3 protein_embed.py -i "$SWISSPROT_FASTA" -o "$PROTEIN_EMBEDS" -b $BATCH_SIZE -n $NUM_PROTEINS -v
    else
        python3 protein_embed.py -i "$SWISSPROT_FASTA" -o "$PROTEIN_EMBEDS" -b $BATCH_SIZE -v
    fi
    
    print_success "Protein embeddings generated"
fi

# =============================================================================
# Step 2: Generate Query Embeddings
# =============================================================================

print_header "Step 2: Query Embeddings"

# Check if query embeddings exist and match targets
NEED_QUERY_EMBED=false

if [ -f "$QUERY_EMBEDS" ]; then
    # Check if targets.fasta was modified after embeddings
    if [ "$TARGETS_FASTA" -nt "$QUERY_EMBEDS" ]; then
        print_warning "targets.fasta was modified - regenerating query embeddings"
        NEED_QUERY_EMBED=true
    else
        print_success "Using existing query embeddings"
    fi
else
    NEED_QUERY_EMBED=true
fi

if [ "$NEED_QUERY_EMBED" = true ]; then
    echo "Generating query embeddings..."
    python3 protein_embed.py -i "$TARGETS_FASTA" -o "$QUERY_EMBEDS" -b $BATCH_SIZE -v
    print_success "Query embeddings generated"
fi

# =============================================================================
# Step 3: Prepare Neural LSH Index
# =============================================================================

if [[ "$METHOD" == "all" || "$METHOD" == "neural" ]]; then
    print_header "Step 3: Neural LSH Index"
    
    if [ "$REBUILD_INDEX" = true ] && [ -f "$NEURAL_INDEX" ]; then
        echo "Removing old Neural LSH index..."
        rm -f "$NEURAL_INDEX"
    fi
    
    if [ -f "$NEURAL_INDEX" ]; then
        # Check if index was built with different number of proteins
        index_size=$(stat -c%s "$NEURAL_INDEX" 2>/dev/null || echo "0")
        if [ "$index_size" -lt 1000 ]; then
            print_warning "Neural LSH index looks incomplete - will rebuild"
            rm -f "$NEURAL_INDEX"
        else
            print_success "Neural LSH index exists (will reuse)"
        fi
    else
        echo "Neural LSH index will be built during search..."
    fi
fi

# =============================================================================
# Step 4: Run Search
# =============================================================================

print_header "Step 4: Running Protein Search"

echo "Running search with method: $METHOD"
echo "Output: $RESULTS_FILE"
echo ""

python3 protein_search.py \
    -d "$PROTEIN_EMBEDS" \
    -q "$TARGETS_FASTA" \
    -o "$RESULTS_FILE" \
    -method "$METHOD" \
    -N "$N_NEIGHBORS" \
    --swissprot "$SWISSPROT_FASTA" \
    --query-embeds "$QUERY_EMBEDS"

# =============================================================================
# Summary
# =============================================================================

print_header "Pipeline Complete!"

echo ""
echo "Results written to: $RESULTS_FILE"
echo ""

# Show quick summary from results
if [ -f "$RESULTS_FILE" ]; then
    echo "Quick Summary (first query):"
    echo "----------------------------"
    head -20 "$RESULTS_FILE" | grep -E "^(Method|Euclidean|Hypercube|IVF|Neural|BLAST)" | head -7
fi

echo ""
print_success "Done!"
