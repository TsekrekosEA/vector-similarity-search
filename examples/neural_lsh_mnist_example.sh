#!/bin/bash
# Simple example: Search MNIST dataset with Neural LSH

set -e

echo "================================================"
echo "Neural LSH MNIST Example"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "nlsh_build.py" ]; then
    echo "Error: Please run this from algorithms/neural_lsh/"
    exit 1
fi

# Download small MNIST sample (if not exists)
if [ ! -f "mnist_sample.dat" ]; then
    echo "Note: For a full example, download MNIST data from:"
    echo "http://yann.lecun.com/exdb/mnist/"
    echo ""
    echo "This is a minimal working example showing the commands."
    exit 0
fi

echo "Step 1: Building Neural LSH index..."
python nlsh_build.py \
    -d mnist_sample.dat \
    -i mnist_index.pth \
    -type mnist \
    --knn 10 \
    -m 50 \
    --epochs 5 \
    --seed 42

echo ""
echo "Step 2: Searching with queries..."
python nlsh_search.py \
    -d mnist_sample.dat \
    -q mnist_queries.dat \
    -i mnist_index.pth \
    -o mnist_results.txt \
    -type mnist \
    -N 10 \
    -T 5

echo ""
echo "✓ Results saved to mnist_results.txt"
echo ""
echo "To view results:"
echo "  cat mnist_results.txt"
