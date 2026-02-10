#!/usr/bin/env bash
# =============================================================================
# Reproducible Benchmark Runner
# =============================================================================
# Runs all three project parts end-to-end, captures timings and results,
# and writes a Markdown summary to benchmark_report.md.
#
# Prerequisites:
#   Part 1: compiled C++ binaries (run make in classical-ann-search/)
#   Part 2: uv installed, MNIST/SIFT data downloaded
#   Part 3: uv installed, Swiss-Prot FASTA + embeddings generated
#
# Usage:
#   ./benchmark.sh                   Run all parts
#   ./benchmark.sh --part 1          Run only Part 1
#   ./benchmark.sh --part 2          Run only Part 2
#   ./benchmark.sh --part 3          Run only Part 3
#   ./benchmark.sh --dry-run         Print commands without executing
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT="$ROOT_DIR/benchmark_report.md"
DRY_RUN=false
PARTS=(1 2 3)

# ── Argument parsing ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --part)
            PARTS=("$2")
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            head -20 "$0" | tail -16
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

log()  { printf '\033[0;34m[bench]\033[0m %s\n' "$*"; }
ok()   { printf '\033[0;32m[bench]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bench]\033[0m %s\n' "$*"; }
err()  { printf '\033[0;31m[bench]\033[0m %s\n' "$*" >&2; }

run_cmd() {
    if $DRY_RUN; then
        echo "  [dry-run] $*"
    else
        "$@"
    fi
}

elapsed_since() {
    local start=$1
    local now
    now=$(date +%s)
    echo $(( now - start ))
}

# ── Report initialisation ────────────────────────────────────────────────────

{
    echo "# Benchmark Report"
    echo ""
    echo "Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Host: $(uname -n) ($(uname -s) $(uname -m))"
    echo "CPU: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo 'N/A')"
    echo "RAM: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo 'N/A')"
    if command -v nvidia-smi &>/dev/null; then
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
    fi
    echo ""
} > "$REPORT"

# ── Part 1: Classical ANN Search (C++) ────────────────────────────────────────

run_part1() {
    local part_dir="$ROOT_DIR/classical-ann-search"
    log "Part 1: Classical ANN Search"

    if [[ ! -d "$part_dir" ]]; then
        warn "Directory classical-ann-search/ not found, skipping Part 1"
        return 0
    fi

    {
        echo "## Part 1 — Classical ANN Search (C++)"
        echo ""
    } >> "$REPORT"

    # Build
    local t_start
    t_start=$(date +%s)
    log "  Building C++ binaries..."
    run_cmd make -C "$part_dir" -j"$(nproc)" 2>&1 | tail -3
    ok "  Build completed in $(elapsed_since "$t_start")s"

    # Check for test data
    local data_dir="$part_dir/data"
    if [[ ! -d "$data_dir" ]] || [[ -z "$(ls -A "$data_dir" 2>/dev/null)" ]]; then
        warn "  No dataset found in $data_dir — run scripts/download_mnist.sh first"
        echo "No dataset available; skipping benchmark." >> "$REPORT"
        echo "" >> "$REPORT"
        return 0
    fi

    # Run LSH benchmark if binary exists
    local bin="$part_dir/bin/search"
    if [[ -x "$bin" ]]; then
        log "  Running LSH benchmark..."
        t_start=$(date +%s)

        local out_file="$part_dir/output/bench_lsh.txt"
        run_cmd "$bin" \
            -d "$data_dir/train-images.idx3-ubyte" \
            -q "$data_dir/t10k-images.idx3-ubyte" \
            -o "$out_file" \
            -lsh -k 4 -L 5 -N 10 -R 10000 2>&1 | tail -5

        ok "  LSH benchmark completed in $(elapsed_since "$t_start")s"
        {
            echo "LSH benchmark completed in $(elapsed_since "$t_start")s."
            echo ""
        } >> "$REPORT"
    else
        warn "  Binary not found at $bin"
    fi

    echo "" >> "$REPORT"
}

# ── Part 2: Neural LSH (Python) ──────────────────────────────────────────────

run_part2() {
    local part_dir="$ROOT_DIR/neural-lsh"
    log "Part 2: Neural LSH"

    if [[ ! -d "$part_dir" ]]; then
        warn "Directory neural-lsh/ not found, skipping Part 2"
        return 0
    fi

    {
        echo "## Part 2 — Neural LSH (Python)"
        echo ""
    } >> "$REPORT"

    # Sync dependencies
    log "  Syncing Python dependencies..."
    run_cmd uv sync --directory "$part_dir" --quiet 2>&1

    # Check for dataset
    local data_dir="$part_dir/data"
    if [[ ! -d "$data_dir" ]] || [[ -z "$(ls -A "$data_dir" 2>/dev/null)" ]]; then
        warn "  No dataset found in $data_dir"
        echo "No dataset available; skipping benchmark." >> "$REPORT"
        echo "" >> "$REPORT"
        return 0
    fi

    # Build index
    local t_start
    t_start=$(date +%s)
    log "  Building Neural LSH index..."
    run_cmd uv run --directory "$part_dir" python nlsh_build.py \
        --dataset mnist --input_file "$data_dir/train-images.idx3-ubyte" \
        --epochs 20 --num_partitions 16 2>&1 | tail -5

    ok "  Index built in $(elapsed_since "$t_start")s"

    # Search
    t_start=$(date +%s)
    log "  Running Neural LSH search..."
    run_cmd uv run --directory "$part_dir" python nlsh_search.py \
        --dataset mnist \
        --input_file "$data_dir/train-images.idx3-ubyte" \
        --query_file "$data_dir/t10k-images.idx3-ubyte" \
        --num_nearest 10 --num_probes 3 2>&1 | tail -10

    ok "  Search completed in $(elapsed_since "$t_start")s"
    {
        echo "Search completed in $(elapsed_since "$t_start")s."
        echo ""
    } >> "$REPORT"

    echo "" >> "$REPORT"
}

# ── Part 3: Protein Similarity Search ────────────────────────────────────────

run_part3() {
    local part_dir="$ROOT_DIR/protein-similarity-search"
    log "Part 3: Protein Similarity Search"

    if [[ ! -d "$part_dir" ]]; then
        warn "Directory protein-similarity-search/ not found, skipping Part 3"
        return 0
    fi

    {
        echo "## Part 3 — Protein Similarity Search"
        echo ""
    } >> "$REPORT"

    log "  Syncing Python dependencies..."
    run_cmd uv sync --directory "$part_dir" --quiet 2>&1

    # Check for data
    if [[ ! -f "$part_dir/data/vectors.dat" ]]; then
        warn "  Embeddings not found at data/vectors.dat — run protein_embed.py first"
        echo "Embeddings not generated; skipping benchmark." >> "$REPORT"
        echo "" >> "$REPORT"
        return 0
    fi

    local t_start
    t_start=$(date +%s)
    log "  Running protein_search.py..."

    run_cmd uv run --directory "$part_dir" python protein_search.py 2>&1 | tail -15

    ok "  Protein search completed in $(elapsed_since "$t_start")s"
    {
        echo "Protein search completed in $(elapsed_since "$t_start")s."
        echo ""
    } >> "$REPORT"

    # Generate plots if results exist
    if [[ -f "$part_dir/results.txt" ]]; then
        log "  Generating plots..."
        run_cmd uv run --directory "$part_dir" python plot_results.py results.txt -o "$part_dir/images/" 2>&1
    fi

    echo "" >> "$REPORT"
}

# ── Main ──────────────────────────────────────────────────────────────────────

log "Starting benchmark (parts: ${PARTS[*]})"
echo "---" >> "$REPORT"
echo "" >> "$REPORT"

TOTAL_START=$(date +%s)

for part in "${PARTS[@]}"; do
    case "$part" in
        1) run_part1 ;;
        2) run_part2 ;;
        3) run_part3 ;;
        *) err "Unknown part: $part" ;;
    esac
done

TOTAL_ELAPSED=$(elapsed_since "$TOTAL_START")
{
    echo "---"
    echo ""
    echo "Total benchmark time: ${TOTAL_ELAPSED}s"
} >> "$REPORT"

ok "Benchmark complete in ${TOTAL_ELAPSED}s — report at $REPORT"
