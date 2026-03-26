#!/bin/bash
# Example: Protein similarity search with Neural LSH

set -e

echo "================================================"
echo "Protein Similarity Search Example"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "protein_search.py" ]; then
    echo "Error: Please run this from test_framework-protein_folding/"
    exit 1
fi

# Check if embeddings exist
if [ ! -f "data/protein_vectors.dat" ]; then
    echo "Protein embeddings not found. Running setup..."
    echo "This will download Swiss-Prot and generate embeddings (takes hours)."
    echo ""
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. For a quick test, use:"
        echo "  ./setup.sh --quick"
        exit 0
    fi
    ./setup.sh
fi

echo ""
echo "Running Neural LSH protein search..."
python protein_search.py \
    -d data/protein_vectors.dat \
    -q data/targets.fasta \
    -o output/example_results.txt \
    -method neural \
    -N 50

echo ""
echo "✓ Results saved to output/example_results.txt"
echo ""
echo "To compare all methods:"
echo "  ./run_pipeline.sh --method all"
