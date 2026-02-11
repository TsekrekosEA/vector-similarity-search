# Neural LSH for Approximate Nearest Neighbor Search

This project implements a Neural Locality Sensitive Hashing (Neural LSH) algorithm for approximate nearest neighbor search in high-dimensional spaces. It uses a graph partitioning approach (KaHIP) to create training labels and a neural network (MLP) to learn the partitioning, enabling fast query routing.

## Files Description

*   **`nlsh_build.py`**: Main script for building the index. It constructs a k-NN graph, partitions it using KaHIP, trains an MLP classifier, and saves the model and inverted index.
*   **`nlsh_search.py`**: Main script for searching. It loads the index, uses the MLP to predict partitions for query points, and performs exact search within the predicted partitions (multi-probe).
*   **`nlsh_build_args.py`**: Argument parsing logic for the build script.
*   **`nlsh_search_args.py`**: Argument parsing logic for the search script.
*   **`nlsh_build_and_search.py`**: Contains the shared `MLPClassifier` PyTorch model definition.
*   **`brute_force.py`**: Implementation of exact brute-force search (using NumPy) for ground truth generation and comparison.
*   **`metrics.py`**: Functions for calculating performance metrics (Accuracy, Recall, QPS, etc.).
*   **`results.py`**: Data structures for storing and manipulating search results.
*   **`output_writer.py`**: Helper to write search results to the specified output file format.
*   **`parse_mnist.py`**: Parser for the MNIST dataset binary format.
*   **`parse_sift.py`**: Parser for the SIFT dataset binary format.
*   **`requirements.txt`**: List of Python dependencies.

## Installation

1.  Ensure you have Python 3.10+ installed.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: `kahip` requires a C++ compiler and CMake to build the underlying C++ library.*

## Usage

### 1. Build Index (`nlsh_build.py`)

This step processes the dataset and creates the index files.

**Syntax:**

```bash
python nlsh_build.py -d <dataset_path> -i <index_output_path> -type <sift|mnist> [options]
```

**Options:**
*   `-d`: Path to the input dataset file.
*   `-i`: Path where the output index (checkpoint) will be saved.
*   `-type`: Dataset type (`sift` or `mnist`).
*   `--knn`: Number of neighbors for the k-NN graph (Default: 10).
*   `-m`: Number of partitions (blocks) for KaHIP (Default: 100).
*   `--imbalance`: KaHIP imbalance parameter (Default: 0.03).
*   `--kahip_mode`: KaHIP mode (0=FAST, 1=ECO, 2=STRONG) (Default: 2).
*   `--layers`: Number of MLP layers (Default: 3).
*   `--nodes`: Number of nodes per hidden layer (Default: 64).
*   `--epochs`: Number of training epochs (Default: 10).
*   `--batch_size`: Training batch size (Default: 128).
*   `--lr`: Learning rate (Default: 0.001).
*   `--seed`: Random seed (Default: 1).

**Example:**

```bash
python nlsh_build.py -d input.dat -i nlsh_index.pth -type sift --knn 15 -m 100 --epochs 10
```

### 2. Search (`nlsh_search.py`)

This step performs queries using the built index.

**Syntax:**

```bash
python nlsh_search.py -d <dataset_path> -q <query_path> -i <index_path> -o <output_path> -type <sift|mnist> [options]
```

**Options:**
*   `-d`: Path to the input dataset file (required for exact distance calculation).
*   `-q`: Path to the query file.
*   `-i`: Path to the index file created by `nlsh_build.py`.
*   `-o`: Path to the output results file.
*   `-type`: Dataset type (`sift` or `mnist`).
*   `-N`: Number of nearest neighbors to find (Default: 1).
*   `-R`: Search radius for range search (Default: 2000 for MNIST, 2800 for SIFT).
*   `-T`: Number of partitions (bins) to probe (Default: 5).
*   `-range`: Enable range search (`true` or `false`) (Default: true).

**Example:**

```bash
python nlsh_search.py -d input.dat -q query.dat -i nlsh_index.pth -o output.txt -type sift -N 10 -T 5
```

## Output Format

The output file follows the format:

```
Neural LSH

Query: <query_id>
Nearest neighbor-1: <id>
distanceApproximate: <dist>
distanceTrue: <dist>
...
R-near neighbors:
<id>
...
Average AF: <value>
Recall@N: <value>
QPS: <value>
tApproximateAverage: <value>
tTrueAverage: <value>
```
