# File Formats

This document describes the binary file formats used throughout the project for storing vectors, indices, and results.

## Binary Vector Format (.dat files)

All vector datasets (MNIST, SIFT, protein embeddings) use a consistent binary format.

### Structure

```
[ Header: 8 bytes ]
[ Vector Data: variable size ]
```

### Header (8 bytes)

| Offset | Type | Field | Description |
|--------|------|-------|-------------|
| 0 | int32 | dimension | Vector dimensionality |
| 4 | int32 | count | Number of vectors |

### Vector Data

Immediately following the header, vectors are stored contiguously:

**For float32 data (SIFT, proteins):**
```
[ vector_0: dimension * 4 bytes ]
[ vector_1: dimension * 4 bytes ]
...
[ vector_n-1: dimension * 4 bytes ]
```

**For uint8 data (MNIST):**
```
[ vector_0: dimension * 1 byte ]
[ vector_1: dimension * 1 byte ]
...
[ vector_n-1: dimension * 1 byte ]
```

### Example

**SIFT dataset (128-dim float32, 10K vectors):**
```
Bytes 0-3:   0x00 0x00 0x00 0x80  → 128 (dimension)
Bytes 4-7:   0x10 0x27 0x00 0x00  → 10000 (count)
Bytes 8-519: [128 * 4 = 512 bytes] → first vector
Bytes 520-1031: → second vector
...
```

### Reading in Python

```python
import struct
import numpy as np

def load_vectors(filename, dtype=np.float32):
    with open(filename, 'rb') as f:
        # Read header
        dimension = struct.unpack('i', f.read(4))[0]
        count = struct.unpack('i', f.read(4))[0]
        
        # Read vectors
        vectors = np.fromfile(f, dtype=dtype, count=dimension * count)
        vectors = vectors.reshape(count, dimension)
        
    return vectors, dimension, count
```

### Reading in C++

```cpp
#include "file_io.hpp"

// Uses the Matrix<T> class
Matrix<float> vectors = load_binary_vectors<float>("dataset.dat");

// Or manually:
std::ifstream file("dataset.dat", std::ios::binary);
int dimension, count;
file.read(reinterpret_cast<char*>(&dimension), sizeof(int));
file.read(reinterpret_cast<char*>(&count), sizeof(int));

std::vector<float> data(dimension * count);
file.read(reinterpret_cast<char*>(data.data()), dimension * count * sizeof(float));
```

## Vector ID Mapping (.json files)

For protein embeddings, vector indices are mapped to UniProt accessions.

### File Naming

If vectors are in `protein_vectors.dat`, IDs are in `protein_vectors_ids.json`

### Format

```json
{
  "0": "P00533",
  "1": "P04637",
  "2": "P00915",
  ...
  "573123": "Q9Y6K9"
}
```

- **Keys:** String representation of vector index (0-based)
- **Values:** UniProt accession or protein identifier

### Reading in Python

```python
import json

def load_vector_ids(dat_filename):
    json_filename = dat_filename.replace('.dat', '_ids.json')
    with open(json_filename, 'r') as f:
        id_map = json.load(f)
    return {int(k): v for k, v in id_map.items()}
```

## Neural LSH Index (.pth files)

PyTorch checkpoint containing trained model and inverted index.

### Structure

```python
{
    'model_state_dict': OrderedDict,  # PyTorch model weights
    'inverted_index': Dict[int, List[int]],  # partition_id -> [point_ids]
    'num_partitions': int,  # Number of KaHIP partitions
    'input_dim': int,  # Vector dimensionality
    'hidden_dim': int,  # MLP hidden layer size
    'hidden_layers': int,  # Number of hidden layers
}
```

### Fields

- **model_state_dict**: MLP classifier weights (can be loaded with `model.load_state_dict()`)
- **inverted_index**: Maps each partition ID to list of point indices in that partition
- **num_partitions**: Total number of partitions (KaHIP parameter `m`)
- **input_dim**: Must match dataset dimensionality
- **hidden_dim**: MLP architecture parameter
- **hidden_layers**: MLP architecture parameter

### Reading

```python
import torch

checkpoint = torch.load('neural_lsh_index.pth')

# Reconstruct model
from nlsh_build_and_search import MLPClassifier
model = MLPClassifier(
    input_dim=checkpoint['input_dim'],
    output_dim=checkpoint['num_partitions'],
    hidden_dim=checkpoint['hidden_dim'],
    hidden_layers=checkpoint['hidden_layers']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Access inverted index
inverted_index = checkpoint['inverted_index']
partition_0_points = inverted_index[0]
```

## C++ Output Format (results.txt)

Standard output from C++ search binary and Neural LSH search.

### Structure

```
<Algorithm Name>

Query: <query_id>
Nearest neighbor-1: <neighbor_id>
distanceApproximate: <approx_distance>
distanceTrue: <true_distance>
Nearest neighbor-2: <neighbor_id>
distanceApproximate: <approx_distance>
distanceTrue: <true_distance>
...

R-near neighbors:
<id>
<id>
...

Average AF: <approximation_factor>
Recall@N: <recall_value>
QPS: <queries_per_second>
tApproximateAverage: <avg_query_time_ms>
tTrueAverage: <avg_true_nn_time_ms>

Query: <next_query_id>
...
```

### Fields

- **Query:** 0-based query index
- **Nearest neighbor-k:** k-th nearest neighbor ID (1-indexed in label)
- **distanceApproximate:** L2 distance computed during ANN search
- **distanceTrue:** True L2 distance (verified with exact computation)
- **R-near neighbors:** List of neighbors within radius R
- **Average AF:** Average approximation factor (distanceApproximate / distanceTrue)
- **Recall@N:** Fraction of true top-N neighbors found
- **QPS:** Queries per second (throughput)
- **tApproximateAverage:** Average query time in milliseconds
- **tTrueAverage:** Average time for exact k-NN in milliseconds

### Parsing in Python

```python
def parse_results(filename):
    results = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    current_query = {}
    for line in lines:
        line = line.strip()
        if line.startswith('Query:'):
            if current_query:
                results.append(current_query)
            current_query = {'query_id': int(line.split(':')[1].strip())}
        elif line.startswith('Recall@N:'):
            current_query['recall'] = float(line.split(':')[1].strip())
        elif line.startswith('QPS:'):
            current_query['qps'] = float(line.split(':')[1].strip())
    
    if current_query:
        results.append(current_query)
    
    return results
```

## Protein Search Output (output/results.txt)

Enhanced format with method comparisons.

### Structure

```
================================================================================
Query Protein: <UniProt_ID>
N = <top_k> (Top-N list size for Recall@N evaluation)

[1] Method Comparison Summary
--------------------------------------------------------------------------------
Method               | Time/query (s)  | QPS        | Recall@N vs BLAST Top-N  
--------------------------------------------------------------------------------
Euclidean LSH        | 3.6585          | 0.4        | 0.11                     
Neural LSH           | 1.0074          | 9.5        | 0.59                     
...

[2] Top-N neighbors per method (showing first 10)
--------------------------------------------------------------------------------
Method: Neural LSH
--------------------------------------------------------------------------------
Rank   | Neighbor ID     | L2 Dist    | BLAST Identity | In BLAST Top-N?
--------------------------------------------------------------------------------
1      | P00533          | 0.00       | 100%           | Yes              
2      | P55245          | 0.04       | 99%            | Yes              
...
```

### Sections

1. **Method Comparison:** Table comparing all methods on speed and recall
2. **Detailed Results:** Per-method top-N neighbors with distances and BLAST comparison
3. **Unique Findings:** Neighbors found by embeddings but not BLAST (or vice versa)

## FASTA Format (input proteins)

Standard bioinformatics format for protein sequences.

### Structure

```
>UniProt_ID|Optional_Description
SEQUENCE_LINE_1
SEQUENCE_LINE_2
...
>Next_Protein_ID
SEQUENCE
```

### Example

```
>P00533|EGFR_HUMAN Epidermal growth factor receptor
MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGN
LEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKT
>P04637|P53_HUMAN Cellular tumor antigen p53
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPE
AAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKT
```

### Parsing

Use BioPython:
```python
from Bio import SeqIO

for record in SeqIO.parse("proteins.fasta", "fasta"):
    protein_id = record.id  # e.g., "P00533"
    sequence = str(record.seq)  # Amino acid sequence
```

## Binary Compatibility

### Endianness

All binary files use **little-endian** byte order (standard on x86/x86_64).

### Type Sizes

- `int32`: 4 bytes signed integer
- `float32`: 4 bytes IEEE 754 floating point
- `uint8`: 1 byte unsigned integer

### Verification

To verify a .dat file is valid:

```python
import struct

with open('dataset.dat', 'rb') as f:
    dim = struct.unpack('<i', f.read(4))[0]  # '<' = little-endian
    count = struct.unpack('<i', f.read(4))[0]
    
    # Check file size matches header
    import os
    expected_size = 8 + dim * count * 4  # for float32
    actual_size = os.path.getsize('dataset.dat')
    
    if expected_size == actual_size:
        print("✓ File format is valid")
    else:
        print(f"✗ Size mismatch: expected {expected_size}, got {actual_size}")
```

## Converting Between Formats

### NumPy array → .dat file

```python
import numpy as np
import struct

def save_vectors(vectors, filename):
    """Save numpy array as .dat file."""
    count, dimension = vectors.shape
    with open(filename, 'wb') as f:
        f.write(struct.pack('i', dimension))
        f.write(struct.pack('i', count))
        vectors.astype(np.float32).tofile(f)
```

### .dat file → NumPy array

See "Reading in Python" section above.

### CSV → .dat file

```python
import pandas as pd

df = pd.read_csv('vectors.csv', header=None)
vectors = df.values.astype(np.float32)
save_vectors(vectors, 'vectors.dat')
```

## File Size Estimates

| Dataset | Dimension | Count | Type | Size |
|---------|-----------|-------|------|------|
| MNIST | 784 | 60K | uint8 | ~47 MB |
| SIFT | 128 | 1M | float32 | ~512 MB |
| Swiss-Prot (proteins) | 320 | 573K | float32 | ~733 MB |

Formula: `size = 8 + dimension * count * sizeof(type)`

## Troubleshooting

### "Invalid file format" error

- Check file is not corrupted (file size matches header)
- Verify endianness (should be little-endian)
- Ensure type matches (float32 vs uint8)

### Dimension mismatch

- Query vectors must match dataset dimension
- Check header values with hex editor or Python script

### Memory issues

- Use memory-mapped files for large datasets:
  ```python
  vectors = np.memmap('large_dataset.dat', dtype=np.float32, mode='r', 
                      offset=8, shape=(count, dimension))
  ```
