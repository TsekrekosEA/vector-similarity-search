# Contributing to Vector Similarity Search

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Code of Conduct

- Be respectful and constructive in all interactions
- Focus on improving the project and helping others
- Welcome newcomers and be patient with questions

## Ways to Contribute

### 1. Report Bugs

If you find a bug, please create an issue with:
- **Clear title** describing the problem
- **Steps to reproduce** the issue
- **Expected behavior** vs. actual behavior
- **Environment details** (OS, Python version, compiler, etc.)
- **Error messages** or logs if applicable

### 2. Suggest Features

Feature requests are welcome! Please include:
- **Use case** explaining why this feature would be valuable
- **Proposed approach** if you have ideas on implementation
- **Alternatives considered**

### 3. Improve Documentation

Documentation improvements are always appreciated:
- Fix typos or unclear explanations
- Add examples or tutorials
- Improve API documentation
- Translate documentation

### 4. Submit Code

See the Development Workflow section below.

## Development Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/vector-similarity-search.git
cd vector-similarity-search
git remote add upstream https://github.com/ORIGINAL_OWNER/vector-similarity-search.git
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

**Branch naming conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Adding tests

### 3. Make Changes

Follow the code style guidelines below.

### 4. Test Your Changes

**C++ Components:**
```bash
cd algorithms/lsh-hypercube-ivf
make clean
make
make benchmark  # Ensure benchmarks still work
```

**Python Components:**
```bash
cd algorithms/neural_lsh
# Test basic imports
python -c "import nlsh_build; import nlsh_search"

# Run on small dataset to verify
python nlsh_build.py -d test_data.dat -i test_index.pth -type sift
python nlsh_search.py -d test_data.dat -q test_query.dat -i test_index.pth -o test_out.txt -type sift
```

**Protein Pipeline:**
```bash
cd test_framework-protein_folding
# Test with --quick flag
./run_pipeline.sh --quick --method neural
```

### 5. Commit Changes

**Commit message format:**
```
<type>: <short summary>

<detailed description if needed>

<reference to issue if applicable>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation change
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `perf:` - Performance improvement
- `style:` - Code style changes (formatting, etc.)

**Examples:**
```
feat: Add HNSW algorithm implementation

Implements Hierarchical Navigable Small World graphs for ANN search.
Includes C++ implementation and Python wrapper.

Closes #42
```

```
fix: Resolve segfault in LSH hash computation

The issue occurred when vector dimension exceeded hash table size.
Added bounds checking and proper error handling.

Fixes #38
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- **Clear title** describing the change
- **Description** explaining what and why
- **Testing** notes on how you verified the change
- **References** to related issues

## Code Style Guidelines

### C++ Code Style

**General Principles:**
- Follow existing code style in the project
- Prioritize readability and maintainability
- Comment complex algorithms or non-obvious optimizations

**Naming Conventions:**
```cpp
// Classes and structs: PascalCase
class NearestNeighborSearch { };
struct QueryResult { };

// Functions and variables: snake_case
void build_index();
int num_vectors;

// Constants: UPPER_SNAKE_CASE
const int MAX_ITERATIONS = 1000;

// Template parameters: T for types, N for non-types
template<typename T>
template<int N>
```

**File Organization:**
- **Headers** (`include/`): Interface declarations, inline functions
- **Source** (`src/`): Implementation
- **Use `#pragma once`** for include guards (already used in project)

**Memory Management:**
- Prefer RAII and smart pointers where appropriate
- Document ownership and lifetime
- Use `Matrix<T>` class for contiguous vector storage (project convention)

**Performance:**
- Profile before optimizing
- Document performance-critical sections
- Prefer cache-friendly access patterns (row-major order)

**Example:**
```cpp
// Good: Clear, cache-friendly
void compute_distances(const Matrix<float>& vectors, const float* query) {
    const int dim = vectors.cols();
    const int n = vectors.rows();
    
    for (int i = 0; i < n; ++i) {
        const float* vec = vectors.get_row(i);  // Contiguous access
        float dist = 0.0f;
        for (int j = 0; j < dim; ++j) {
            float diff = vec[j] - query[j];
            dist += diff * diff;
        }
        // Process dist...
    }
}
```

### Python Code Style

**Follow PEP 8** with these specific guidelines:

**Naming Conventions:**
```python
# Modules and packages: lowercase_with_underscores
import nlsh_build

# Classes: PascalCase
class MLPClassifier:

# Functions and variables: lowercase_with_underscores
def build_index():
num_partitions = 100

# Constants: UPPER_SNAKE_CASE
DEFAULT_LEARNING_RATE = 0.001
```

**Type Hints:**
```python
from typing import List, Dict, Optional, Tuple
import numpy as np

def search_index(
    query: np.ndarray,
    index: Dict[int, List[int]],
    top_k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search index for nearest neighbors.
    
    Args:
        query: Query vector (D,)
        index: Inverted index mapping partition -> point IDs
        top_k: Number of neighbors to return
        
    Returns:
        Tuple of (neighbor_ids, distances)
    """
    # Implementation...
```

**Docstrings:**
Use Google-style docstrings:
```python
def function_name(param1: int, param2: str) -> bool:
    """
    Short one-line summary.
    
    Longer description if needed, explaining the function's behavior,
    side effects, and important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When validation fails
    """
```

**Imports:**
```python
# Standard library
import os
import sys
from typing import List

# Third-party
import numpy as np
import torch

# Local
from nlsh_build_args import parse_arguments
import parse_sift
```

**Random Seeds:**
```python
# ALWAYS set random seeds for reproducibility
def set_random_seeds(seed: int):
    """Set seeds for NumPy, PyTorch, and KaHIP."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # KaHIP seed set separately in arguments
```

**Device Handling:**
```python
# Auto-detect GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

## Adding New Algorithms

### C++ Algorithm

1. **Create header** in `algorithms/lsh-hypercube-ivf/include/`
   ```cpp
   #pragma once
   #include "data_types.hpp"
   #include "utils.hpp"
   
   template<typename T>
   class NewAlgorithm {
   public:
       // Constructor, index building, search methods
   };
   ```

2. **Create implementation** in `algorithms/lsh-hypercube-ivf/src/`

3. **Update Makefile** to include new files

4. **Integrate in `main.cpp`** or `benchmark.cpp`

5. **Update documentation** in algorithm README

### Python Algorithm

1. **Create module** in `algorithms/neural_lsh/` or new directory

2. **Follow existing patterns:**
   - Separate `*_args.py` for argument parsing
   - Binary data loading via `parse_*.py`
   - Save index as `.pth` checkpoint with metadata

3. **Integrate with protein search** in `test_framework-protein_folding/protein_search.py`

4. **Update documentation**

## Testing Guidelines

### What to Test

1. **Build succeeds** on clean checkout
2. **Scripts run** without errors on small datasets
3. **Output format** matches expected structure
4. **Reproducibility** - same inputs produce same outputs

### Test Checklist

- [ ] C++ code compiles without warnings
- [ ] Python code has no import errors
- [ ] Benchmark scripts complete successfully
- [ ] Documentation is updated
- [ ] No large binary files committed
- [ ] Git history is clean (squash if needed)

## Performance Considerations

- **Profile before optimizing** - Use `perf`, `valgrind`, or `cProfile`
- **Document trade-offs** - Explain why you chose a particular approach
- **Benchmark changes** - Compare performance before/after
- **Consider memory usage** - Large datasets require efficient memory management

## Documentation Standards

- Update relevant README files when changing functionality
- Keep API documentation in sync with code
- Add examples for new features
- Update ARCHITECTURE.md for structural changes
- English for all primary documentation (Greek academic docs stay in docs/)

## Questions?

- **GitHub Issues** - For bugs and feature requests
- **Discussions** - For general questions (if enabled)
- **Email** - Contact the maintainer for private inquiries

## Recognition

Contributors will be acknowledged in:
- Git commit history
- Future release notes
- Project acknowledgments (for significant contributions)

Thank you for contributing to make this project better! 🎉
