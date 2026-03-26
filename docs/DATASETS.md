# Datasets

This document describes the datasets used in this project for benchmarking ANN algorithms.

## Datasets Overview

| Dataset | Dimension | Count | Type | Size | Domain |
|---------|-----------|-------|------|------|--------|
| **MNIST** | 784 | 60K train + 10K test | uint8 | ~47 MB | Handwritten digits |
| **SIFT** | 128 | 1M base + 10K query | float32 | ~512 MB | Computer vision features |
| **Swiss-Prot** | 320 | 573K | float32 | ~733 MB | Protein embeddings |

---

## MNIST (Modified National Institute of Standards and Technology)

### Description
Handwritten digit images (0-9) converted to 784-dimensional vectors (28×28 pixels flattened).

### Source
- **Original:** Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **URL:** http://yann.lecun.com/exdb/mnist/
- **License:** Public domain (with attribution requested)

### Dataset Details
- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Dimensions:** 784 (28×28 pixel values)
- **Type:** uint8 (0-255 grayscale)
- **Classes:** 10 digits (0-9)

### Download
```bash
# The benchmark scripts automatically download MNIST if needed
cd algorithms/lsh-hypercube-ivf
make benchmark-mnist
```

### Citation
```bibtex
@article{lecun1998gradient,
  title={Gradient-based learning applied to document recognition},
  author={LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  journal={Proceedings of the IEEE},
  volume={86},
  number={11},
  pages={2278--2324},
  year={1998},
  publisher={IEEE}
}
```

### Characteristics
- **Euclidean distances** work well for similarity
- **Low intrinsic dimensionality** despite 784 dimensions
- **Digit classes** provide ground truth for evaluation
- **Widely used benchmark** for nearest neighbor search

---

## SIFT (Scale-Invariant Feature Transform)

### Description
128-dimensional descriptors extracted from images using the SIFT algorithm. Captures local image features invariant to scale and rotation.

### Source
- **Algorithm:** David G. Lowe (1999, 2004)
- **Benchmark datasets:** ANN-Benchmarks, TEXMEX
- **URL:** http://corpus-texmex.irisa.fr/

### Dataset Details
- **Base set:** 1,000,000 vectors
- **Query set:** 10,000 vectors
- **Dimensions:** 128
- **Type:** float32
- **Domain:** Computer vision features

### Download
SIFT datasets can be obtained from:
- **ANN-Benchmarks:** https://github.com/erikbern/ann-benchmarks
- **TEXMEX:** http://corpus-texmex.irisa.fr/

```bash
# Download SIFT1M
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
```

### Citation
```bibtex
@article{lowe2004distinctive,
  title={Distinctive image features from scale-invariant keypoints},
  author={Lowe, David G},
  journal={International journal of computer vision},
  volume={60},
  number={2},
  pages={91--110},
  year={2004},
  publisher={Springer}
}
```

### Characteristics
- **Real-world features** from actual images
- **L2 distance** is standard similarity metric
- **High-dimensional** but structured data
- **Standard benchmark** in computer vision and ANN research

---

## Swiss-Prot (Protein Sequences)

### Description
Manually annotated and reviewed protein sequence database. For this project, sequences are converted to 320-dimensional embeddings using the ESM-2 protein language model.

### Source
- **Database:** UniProt Consortium
- **URL:** https://www.uniprot.org/
- **FTP:** https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/
- **License:** Creative Commons Attribution 4.0

### Dataset Details
- **Count:** ~573,000 proteins (as of 2025)
- **Original format:** FASTA amino acid sequences
- **Embedding model:** ESM-2 (facebook/esm2_t6_8M_UR50D)
- **Dimensions:** 320 (after mean pooling)
- **Type:** float32
- **Size:** ~250 MB (FASTA) → ~733 MB (embeddings)

### Download
```bash
cd test_framework-protein_folding

# Option 1: Full automated setup (downloads + generates embeddings)
./setup.sh

# Option 2: Manual download
wget -O data/uniprot_sprot.fasta.gz \
    https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip data/uniprot_sprot.fasta.gz
```

### Embedding Generation
Swiss-Prot sequences are embedded using ESM-2:

```bash
# Generate embeddings (takes several hours on CPU, ~30 min on GPU)
python protein_embed.py \
    -i data/swissprot.fasta \
    -o data/protein_vectors.dat \
    -b 64  # batch size (adjust based on GPU memory)
```

The embedding process:
1. Load ESM-2 model from HuggingFace
2. Tokenize protein sequences
3. Forward pass through transformer
4. Mean pooling over sequence length
5. Save as 320-dim float32 vectors

### Citation

**Swiss-Prot Database:**
```bibtex
@article{uniprot2023,
  title={UniProt: the universal protein knowledgebase in 2023},
  author={{The UniProt Consortium}},
  journal={Nucleic Acids Research},
  volume={51},
  number={D1},
  pages={D523--D531},
  year={2023},
  publisher={Oxford University Press}
}
```

**ESM-2 Model:**
```bibtex
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yilun and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Salvatore and Rives, Alexander},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```

### Characteristics
- **Semantic embeddings** capture protein function and structure
- **Beyond sequence similarity** - can find remote homologs
- **L2 distance** in embedding space measures functional similarity
- **Complements BLAST** for comprehensive protein search
- **Real-world application** demonstrating ANN in bioinformatics

### Query Proteins
Default queries (`data/targets.fasta`) include 10 human proteins:

| UniProt ID | Protein Name | Function |
|------------|--------------|----------|
| P00533 | EGFR | Epidermal Growth Factor Receptor |
| P04637 | p53 | Tumor suppressor |
| P00915 | CAH1 | Carbonic Anhydrase I |
| P68871 | HBB | Hemoglobin beta |
| P62158 | CALM | Calmodulin |
| P01308 | INS | Insulin |
| P00918 | CAH2 | Carbonic Anhydrase II |
| P69905 | HBA | Hemoglobin alpha |
| P01116 | KRAS | GTPase KRas |
| P02144 | MB | Myoglobin |

---

## Data Preprocessing

All datasets are converted to the same binary `.dat` format for consistency:

### Binary Format
```
[ dimension: int32 ]
[ count: int32 ]
[ vector_0: dimension × sizeof(type) ]
[ vector_1: dimension × sizeof(type) ]
...
```

See [docs/FILE_FORMATS.md](FILE_FORMATS.md) for complete specification.

### Normalization
- **MNIST:** Raw pixel values (0-255), no normalization
- **SIFT:** L2-normalized by SIFT algorithm
- **Proteins:** ESM-2 embeddings are NOT normalized (preserves magnitude information)

---

## Benchmark Methodology

### Ground Truth
For each dataset, ground truth nearest neighbors are computed using:
- **Exact brute-force search** with L2 distance
- Results cached to avoid recomputation
- Used to calculate Recall@N metrics

### Evaluation Metrics
1. **Recall@N** - Fraction of true top-N neighbors found
2. **QPS** - Queries per second (throughput)
3. **Approximation Factor (AF)** - Ratio of approximate to true distance
4. **Build Time** - Index construction time
5. **Memory Usage** - Index size in RAM

### Test Queries
- **MNIST:** 10,000 test set images
- **SIFT:** 10,000 query vectors
- **Proteins:** 10 diverse human proteins (can be customized)

---

## Creating Custom Datasets

### From CSV
```python
import pandas as pd
import numpy as np
import struct

# Load data
df = pd.read_csv('vectors.csv', header=None)
vectors = df.values.astype(np.float32)

# Save as .dat
with open('custom_dataset.dat', 'wb') as f:
    count, dimension = vectors.shape
    f.write(struct.pack('i', dimension))
    f.write(struct.pack('i', count))
    vectors.tofile(f)
```

### From NumPy Arrays
```python
import numpy as np
import struct

vectors = np.random.randn(10000, 128).astype(np.float32)

with open('random_vectors.dat', 'wb') as f:
    count, dimension = vectors.shape
    f.write(struct.pack('i', dimension))
    f.write(struct.pack('i', count))
    vectors.tofile(f)
```

See [examples/README.md](../examples/README.md) for more details.

---

## Dataset Statistics

### MNIST Nearest Neighbor Statistics
- **Average distance to 1st NN:** ~8.5
- **Average distance to 10th NN:** ~12.3
- **Intrinsic dimensionality:** ~12 (estimated)

### SIFT Nearest Neighbor Statistics
- **Average distance to 1st NN:** ~0.42
- **Average distance to 10th NN:** ~0.68
- **High variance** across queries

### Protein Embedding Statistics
- **Average distance to 1st NN:** ~0.02 (exact match)
- **Average distance to 10th NN:** ~0.15
- **Average distance to 50th NN:** ~0.35
- **Functional families** form clusters in embedding space

---

## Troubleshooting

**"Dataset not found"**
- Run setup scripts that download data automatically
- Check paths in configuration files
- Ensure sufficient disk space

**"Out of memory during embedding generation"**
- Reduce batch size: `python protein_embed.py -b 4`
- Use CPU instead of GPU: smaller batches but more stable
- Process in chunks: embed subsets separately

**"Incorrect file format"**
- Verify file header with: `hexdump -C dataset.dat | head`
- Check dimension and count match expectations
- Ensure correct data type (float32 vs uint8)

---

## References

1. LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition.
2. Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints.
3. The UniProt Consortium (2023). UniProt: the universal protein knowledgebase.
4. Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure.
5. ANN-Benchmarks: https://github.com/erikbern/ann-benchmarks
