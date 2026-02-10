#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_experiments.sh — parameterised Neural LSH experiment runner
#
# Usage:
#   ./run_experiments.sh                        # MNIST defaults
#   ./run_experiments.sh --dataset sift         # SIFT defaults
#   ./run_experiments.sh --k "10 20" --m "64 128" --epochs 10 --probes "1 5 10"
#   ./run_experiments.sh --dataset sift --k "2 4 8" --m "4 8 16" --epochs 2
#
# Every unique (dataset, k, m, epochs) combination produces one index.
# Each index is then searched with every probe value T in --probes.
# Results are appended to a single CSV file (one per dataset).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────

DATASET="mnist"
KNN_VALUES="10 20"
M_VALUES="64 128"
EPOCHS=10
PROBE_VALUES="1 5 10"
NEIGHBORS=10
CSV=""             # auto-generated if empty

# ── Parse flags ───────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)  DATASET="$2";  shift 2 ;;
        --k)        KNN_VALUES="$2"; shift 2 ;;
        --m)        M_VALUES="$2";   shift 2 ;;
        --epochs)   EPOCHS="$2";     shift 2 ;;
        --probes)   PROBE_VALUES="$2"; shift 2 ;;
        -N)         NEIGHBORS="$2";  shift 2 ;;
        --csv)      CSV="$2";        shift 2 ;;
        -h|--help)
            head -15 "$0" | tail -n +2 | sed 's/^# *//'
            exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Dataset-specific paths ────────────────────────────────────────────────────

case "$DATASET" in
    mnist)
        DATA_FILE="data/mnist/train-images.idx3-ubyte"
        QUERY_FILE="data/mnist/t10k-images.idx3-ubyte"
        TYPE="mnist"
        RADIUS=2000
        ;;
    sift)
        DATA_FILE="data/sift/sift_base.fvecs"
        QUERY_FILE="data/sift/sift_query.fvecs"
        TYPE="sift"
        RADIUS=2800
        ;;
    *)
        echo "Unsupported dataset: $DATASET (use mnist or sift)"
        exit 1 ;;
esac

[[ -z "$CSV" ]] && CSV="output/experiments_${DATASET}.csv"
mkdir -p output

# ── CSV header ────────────────────────────────────────────────────────────────

echo "Dataset,Algorithm,k,m,e,T,N,R,AverageAF,Recall@N,QPS,tApproximateAverage,tTrueAverage,Speedup" > "$CSV"

# ── Experiment grid ───────────────────────────────────────────────────────────

echo "========================================================"
echo "  Neural LSH Experiments — ${DATASET^^}"
echo "  k = {${KNN_VALUES}}  m = {${M_VALUES}}  epochs = ${EPOCHS}"
echo "  probes (T) = {${PROBE_VALUES}}  N = ${NEIGHBORS}"
echo "  CSV → ${CSV}"
echo "========================================================"

for K in $KNN_VALUES; do
    for M in $M_VALUES; do
        INDEX_FILE="output/index_${TYPE}_k${K}_m${M}_e${EPOCHS}.pth"

        if [[ ! -f "$INDEX_FILE" ]]; then
            echo "── Building index: k=$K  m=$M  epochs=$EPOCHS ──"
            python3 nlsh_build.py \
                -d "$DATA_FILE" \
                -i "$INDEX_FILE" \
                -type "$TYPE" \
                --knn "$K" \
                -m "$M" \
                --epochs "$EPOCHS"
        else
            echo "── Index exists: $INDEX_FILE (skipping build) ──"
        fi

        for T in $PROBE_VALUES; do
            echo "   Searching: T=$T"
            python3 nlsh_search.py \
                -d "$DATA_FILE" \
                -q "$QUERY_FILE" \
                -i "$INDEX_FILE" \
                -type "$TYPE" \
                -N "$NEIGHBORS" \
                -R "$RADIUS" \
                -T "$T" \
                --range false \
                --minimal-output \
                --csv-output-file "$CSV"
        done
    done
done

echo "========================================================"
echo "  Experiments complete.  Results in $CSV"
echo "========================================================"
