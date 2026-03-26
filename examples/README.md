# Examples

This directory contains example scripts demonstrating how to use the various ANN algorithms.

## Quick Examples

### Neural LSH on MNIST
```bash
cd ../algorithms/neural_lsh
bash ../../examples/neural_lsh_mnist_example.sh
```

Shows the complete workflow: build index → search queries → analyze results.

### Protein Similarity Search
```bash
cd ../test_framework-protein_folding
bash ../examples/protein_search_example.sh
```

End-to-end protein search using ESM-2 embeddings and Neural LSH.

## C++ Examples

### Build and Search with LSH

```bash
cd ../algorithms/lsh-hypercube-ivf

# Build the binary
make

# Example: Search SIFT dataset with LSH
./bin/search \
    -d data/sift_base.dat \
    -q data/sift_query.dat \
    -o output/lsh_results.txt \
    -algorithm lsh \
    -k 6 \
    -L 8 \
    -w 1.0 \
    -N 10 \
    -R 2800 \
    -range false
```

### Search with Hypercube

```bash
./bin/search \
    -d data/sift_base.dat \
    -q data/sift_query.dat \
    -o output/hypercube_results.txt \
    -algorithm hypercube \
    -k 12 \
    -w 1.5 \
    -M 5000 \
    -probes 2 \
    -N 10
```

### Search with IVF-Flat

```bash
./bin/search \
    -d data/sift_base.dat \
    -q data/sift_query.dat \
    -o output/ivfflat_results.txt \
    -algorithm ivfflat \
    -nlist 1000 \
    -nprobe 50 \
    -N 10
```

## Python Examples

### Neural LSH: Build Index

```python
#!/usr/bin/env python3
from algorithms.neural_lsh import nlsh_build
import sys

# Parse command-line arguments
args = nlsh_build.parse_arguments([
    '-d', 'dataset.dat',
    '-i', 'index.pth',
    '-type', 'sift',
    '--knn', '10',
    '-m', '100',
    '--epochs', '10'
])

# Build the index
nlsh_build.main(args)
print("✓ Index saved to index.pth")
```

### Neural LSH: Search

```python
#!/usr/bin/env python3
from algorithms.neural_lsh import nlsh_search

args = nlsh_search.parse_arguments([
    '-d', 'dataset.dat',
    '-q', 'queries.dat',
    '-i', 'index.pth',
    '-o', 'results.txt',
    '-type', 'sift',
    '-N', '10',
    '-T', '5'
])

nlsh_search.main(args)
print("✓ Results saved to results.txt")
```

### Generate Protein Embeddings

```python
#!/usr/bin/env python3
import sys
sys.path.append('test_framework-protein_folding')

from protein_embed_backend import embed_proteins
from Bio import SeqIO

# Load FASTA sequences
sequences = []
ids = []
for record in SeqIO.parse("proteins.fasta", "fasta"):
    sequences.append(str(record.seq))
    ids.append(record.id)

# Generate embeddings
embeddings = embed_proteins(
    sequences=sequences,
    batch_size=8,
    device='cuda'  # or 'cpu'
)

# Save as binary .dat file
import numpy as np
import struct

with open('protein_vectors.dat', 'wb') as f:
    dimension = embeddings.shape[1]
    count = len(embeddings)
    f.write(struct.pack('i', dimension))
    f.write(struct.pack('i', count))
    embeddings.astype(np.float32).tofile(f)

# Save ID mapping
import json
with open('protein_vectors_ids.json', 'w') as f:
    json.dump({str(i): id for i, id in enumerate(ids)}, f)

print(f"✓ Saved {count} embeddings ({dimension}D) to protein_vectors.dat")
```

## Creating Your Own Dataset

### From NumPy Array

```python
import numpy as np
import struct

# Your vectors (N x D array)
vectors = np.random.randn(1000, 128).astype(np.float32)

# Save as .dat file
with open('my_dataset.dat', 'wb') as f:
    count, dimension = vectors.shape
    f.write(struct.pack('i', dimension))
    f.write(struct.pack('i', count))
    vectors.tofile(f)

print(f"Saved {count} vectors ({dimension}D)")
```

### From CSV

```python
import pandas as pd
import numpy as np
import struct

# Load CSV (no header, all numeric)
df = pd.read_csv('vectors.csv', header=None)
vectors = df.values.astype(np.float32)

# Save as .dat
with open('dataset.dat', 'wb') as f:
    count, dimension = vectors.shape
    f.write(struct.pack('i', dimension))
    f.write(struct.pack('i', count))
    vectors.tofile(f)
```

## Running Benchmarks

### Full Benchmark Suite

```bash
cd algorithms/lsh-hypercube-ivf
make benchmark
```

This runs all algorithms on both MNIST and SIFT datasets, comparing:
- Recall@N
- Query speed (QPS)
- Approximation factor
- Build time

### Custom Benchmark

```python
#!/usr/bin/env python3
import time
import numpy as np
from algorithms.neural_lsh import nlsh_search, nlsh_build

# 1. Build index
build_start = time.time()
# ... build index ...
build_time = time.time() - build_start

# 2. Search queries
search_start = time.time()
# ... perform searches ...
search_time = time.time() - search_start

# 3. Calculate metrics
qps = num_queries / search_time
print(f"Build time: {build_time:.2f}s")
print(f"QPS: {qps:.1f}")
```

## Troubleshooting

**"Module not found"**
- Ensure you're in the correct directory
- Install requirements: `pip install -r requirements.txt`

**"Binary file format error"**
- Check file with: `hexdump -C dataset.dat | head`
- First 8 bytes should be dimension (int32) + count (int32)

**"Out of memory"**
- Reduce batch size in embedding generation
- Use smaller dataset for testing
- Consider using Neural LSH (lower memory than IVF)

## More Information

- See [docs/FILE_FORMATS.md](../docs/FILE_FORMATS.md) for data format details
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines
- See algorithm-specific READMEs in their directories
