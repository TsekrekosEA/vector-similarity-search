# Results and Visualizations

This directory contains benchmark results and performance visualizations from the Vector Similarity Search project.

## Benchmark Images

### MNIST Dataset Results
**MNIST_recall_vs_speedup.png**
- Recall vs. Speedup trade-off for all ANN algorithms on MNIST dataset
- 784-dimensional handwritten digit images (uint8)
- Shows performance comparison of LSH, Hypercube, IVF-Flat, IVF-PQ algorithms

### SIFT Dataset Results
**SIFT_recall_vs_speedup.png**
- Recall vs. Speedup trade-off for all ANN algorithms on SIFT dataset
- 128-dimensional SIFT descriptors (float32)
- Demonstrates algorithm performance on real-world computer vision features

## Generating Visualizations

To regenerate these plots:

```bash
cd scripts
python plot_results.py
```

The script reads benchmark output files and generates publication-quality visualizations using matplotlib and seaborn.

## Understanding the Results

- **X-axis (Speedup)**: Query speed improvement over brute-force search (log scale)
- **Y-axis (Recall@N)**: Fraction of true nearest neighbors found
- **Target**: High recall (≥0.9) with maximum speedup
- **Best performers**: Algorithms in the upper-right region (high recall + high speedup)

See [RESULTS.md](../RESULTS.md) for detailed analysis and interpretation.
