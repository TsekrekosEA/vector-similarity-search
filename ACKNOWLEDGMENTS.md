# Acknowledgments

This project was completed as a graduate thesis at the **National and Kapodistrian University of Athens (EKPA)**, Department of Informatics and Telecommunications.

## Academic Institution

**National and Kapodistrian University of Athens (ΕΚΠΑ)**
- Department of Informatics and Telecommunications
- Underaduate Program in Computer Science
- Athens, Greece

## Datasets

This research utilized the following publicly available datasets:

### MNIST Database
- **Source:** Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **Description:** Handwritten digit images (60,000 training + 10,000 test)
- **Use:** Benchmarking ANN algorithms on image data
- **Citation:** LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

### SIFT Descriptors
- **Source:** David G. Lowe
- **Description:** Scale-Invariant Feature Transform descriptors for computer vision
- **Use:** Benchmarking on real-world features
- **Citation:** Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2), 91-110.

### Swiss-Prot Protein Database
- **Source:** UniProt Consortium
- **Description:** Manually annotated and reviewed protein sequence database (~573K proteins)
- **Use:** Protein similarity search application
- **URL:** https://www.uniprot.org/
- **Citation:** The UniProt Consortium (2023). UniProt: the Universal Protein Knowledgebase in 2023. Nucleic Acids Research, 51(D1), D523-D531.

## Software Libraries & Tools

### Core Libraries

**PyTorch**
- Deep learning framework for Neural LSH MLP training
- URL: https://pytorch.org/
- License: BSD-3-Clause

**KaHIP (Karlsruhe High Quality Partitioning)**
- Graph partitioning library used in Neural LSH
- URL: https://github.com/KaHIP/KaHIP
- License: MIT
- Citation: Sanders, P., & Schulz, C. (2013). High quality graph partitioning. Graph Partitioning and Graph Clustering, 10, 1-17.

**Transformers (Hugging Face)**
- Pre-trained ESM-2 protein language model
- URL: https://huggingface.co/transformers/
- License: Apache 2.0

**BioPython**
- Biological computation tools for FASTA parsing and BLAST integration
- URL: https://biopython.org/
- License: BSD-3-Clause

**scikit-learn**
- Machine learning utilities for k-NN graph construction
- URL: https://scikit-learn.org/
- License: BSD-3-Clause

### Supporting Libraries

**NumPy** - Numerical computing (BSD License)
**pandas** - Data analysis and manipulation (BSD License)
**matplotlib** - Plotting and visualization (PSF License)
**seaborn** - Statistical data visualization (BSD License)
**tqdm** - Progress bars (MIT/MPL-2.0 License)

## Protein Language Models

**ESM-2 (Evolutionary Scale Modeling)**
- Developed by Meta AI (formerly Facebook AI Research)
- Used for generating protein embeddings
- Citation: Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., ... & Rives, A. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637), 1123-1130.
- Model: facebook/esm2_t6_8M_UR50D

## BLAST (Basic Local Alignment Search Tool)

**NCBI BLAST+**
- Used as baseline comparison for protein similarity search
- URL: https://blast.ncbi.nlm.nih.gov/
- Citation: Camacho, C., Coulouris, G., Avagyan, V., Ma, N., Papadopoulos, J., Bealer, K., & Madden, T. L. (2009). BLAST+: architecture and applications. BMC bioinformatics, 10, 1-9.

## Algorithm Inspirations

### Locality Sensitive Hashing (LSH)
- Indyk, P., & Motwani, R. (1998). Approximate nearest neighbors: towards removing the curse of dimensionality. In Proceedings of the thirtieth annual ACM symposium on Theory of computing (pp. 604-613).

### Inverted File with Product Quantization (IVF-PQ)
- Jégou, H., Douze, M., & Schmid, C. (2011). Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence, 33(1), 117-128.

### Neural Hashing & Learned Index Structures
- Kraska, T., Beutel, A., Chi, E. H., Dean, J., & Polyzotis, N. (2018). The case for learned index structures. In Proceedings of the 2018 International Conference on Management of Data (pp. 489-504).

## Development Tools

- **Git** - Version control
- **GitHub** - Code hosting and collaboration
- **Make** - Build automation for C++ components
- **GCC/G++** - C++ compiler
- **Python** - Primary scripting language
- **Visual Studio Code** - Code editor

## Open Source Community

Special thanks to the open-source community for making high-quality tools and libraries freely available. This project stands on the shoulders of giants.

---

**If I have missed anyone who should be acknowledged, please open an issue or pull request.**
