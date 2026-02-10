"""Dataset parsers for MNIST (IDX3) and SIFT (.fvecs) binary formats.

This module mirrors the I/O layer in file_io.cpp from the C++ algorithms but in
Python.  Both formats store dense vectors in compact binary representations and
the parsers read them into contiguous NumPy arrays that the rest of the pipeline
can consume directly.

MNIST IDX3 Format:
The IDX3 file begins with a 16-byte header in big-endian byte order.  The first
four bytes hold a magic number (0x00000803 for unsigned byte data), followed by
three four-byte unsigned integers giving the number of images, the number of
rows (typically 28), and the number of columns (typically 28).  After the header,
pixel data is stored as contiguous uint8 values in row-major order, with one byte
per pixel.  Each image is flattened to a 784-dimensional vector and cast to
float32 for distance computation compatibility with the C++ backend.

SIFT .fvecs Format:
Each record in a .fvecs file is self-describing.  It begins with a four-byte
little-endian uint32 giving the dimension D, followed by D little-endian float32
components.  Every record must share the same dimensionality.  Standard
dimensionalities are 128 for SIFT1M and 320 for the protein embeddings from
Part 3.

Memory Notes:
Both parsers return a single contiguous (N, D) float32 array.  For MNIST-60K
this is roughly 180 MB; for SIFT-1M it is roughly 490 MB.  The parse_sift path
uses a Python loop to skip per-record headers, which is acceptable since it runs
once at startup rather than on the hot path.

Complexity:
Time is O(N * D) for reading plus dtype conversion.  Space is O(N * D) for the
returned array with no intermediate copies for MNIST and one for SIFT.
"""

from __future__ import annotations

import struct
from pathlib import Path

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------

def parse_mnist(path: str | Path) -> np.ndarray:
    """Read an IDX3 file and return an (N, rows*cols) float32 array.

    The IDX3 header contains four big-endian uint32 fields: magic, count, rows,
    and cols.  Pixel data starts at offset 16 and is stored as contiguous
    row-major uint8 values.  We reshape to (count, rows*cols) and upcast to
    float32 so that Euclidean distances are computed in floating point, matching
    the C++ backend behaviour.  Time is O(N * D), dominated by the dtype
    conversion.  Space is O(N * D) for the output array.
    """
    data = Path(path).read_bytes()
    _, count, rows, cols = struct.unpack(">IIII", data[:16])
    pixels = np.frombuffer(data, dtype=np.uint8, offset=16)
    return pixels.reshape(count, rows * cols).astype(np.float32)


# ---------------------------------------------------------------------------
# SIFT (.fvecs)
# ---------------------------------------------------------------------------

def parse_sift(path: str | Path) -> np.ndarray:
    """Read a .fvecs file and return an (N, dim) float32 array.

    Each record starts with a four-byte little-endian uint32 giving the
    dimensionality, followed by dim little-endian float32 values.  The function
    reads the dimension from the first record, computes the number of vectors
    from the total file size, then reinterprets the entire buffer as a flat
    float32 array and reshapes it, skipping the one-element dimension header in
    each record via column slicing.  This vectorised approach avoids the
    per-record Python loop, yielding a significant speedup on large files like
    SIFT-1M.  Time and space are both O(N * D).
    """
    raw = Path(path).read_bytes()
    if len(raw) < 4:
        raise ValueError(f"File too small: {path}")

    dim = struct.unpack_from("<I", raw, 0)[0]
    record_bytes = 4 + dim * 4  # 4-byte header + dim floats
    num_vectors = len(raw) // record_bytes

    # Reinterpret the entire buffer as float32; each record becomes (1 + dim) floats
    # where the first float is the uint32 dimension header reinterpreted as float32.
    # We then slice out the header column to get only the vector data.
    all_floats = np.frombuffer(raw, dtype="<f4").reshape(num_vectors, 1 + dim)
    return all_floats[:, 1:].copy()


# ---------------------------------------------------------------------------
# Dispatcher used by build / search scripts
# ---------------------------------------------------------------------------

def get_parser(is_sift: bool) -> Callable[[str | Path], np.ndarray]:
    """Return the appropriate parse function based on dataset type.

    The -type CLI flag in nlsh_build.py and nlsh_search.py is normalised to a
    boolean is_sift early, so every downstream consumer just calls
    get_parser(args.is_sift) to obtain the right reader.
    """
    return parse_sift if is_sift else parse_mnist
