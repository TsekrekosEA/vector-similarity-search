"""ESM-2 protein embedding backend — transforms amino-acid sequences into dense vectors.

Algorithm Overview:
This module converts protein sequences from a FASTA file into fixed-length dense
vectors using Facebook's ESM-2 (Evolutionary Scale Modeling) protein language
model.  The resulting 320-dimensional embeddings capture evolutionary, structural,
and functional properties of proteins, enabling similarity search with the same
ANN algorithms used for image features (LSH, Hypercube, IVF-Flat, IVF-PQ, Neural
LSH) in Parts 1 and 2.

Theoretical Foundation:
ESM-2 is a masked language model trained on millions of protein sequences
(UniRef50/90).  Each amino acid is mapped to a contextual embedding via a
transformer encoder stack.  The model variant used here (esm2_t6_8M_UR50D) has
6 transformer layers and 8M parameters, small enough to run on a single GPU or
CPU yet expressive enough for similarity tasks.

Embedding Extraction — Mean Pooling:
ESM-2 outputs a per-residue embedding matrix of shape (L, 320) where L is the
sequence length.  To obtain a fixed-length representation we apply mean pooling:
v = (1/L_eff) * sum of h_i over unmasked positions, where h_i is the hidden
state of residue i and L_eff is the number of real (non-pad) residues.  The
attention_mask tensor marks pad positions with 0, so the mean is computed only
over genuine residues.  This is analogous to using the CLS token in BERT but
empirically performs better for protein retrieval.

Output Format:
Embeddings are written in the SIFT .fvecs format (see parsers.py in Part 2).
Each vector is stored as a four-byte LE uint32 (dim=320) followed by 320 float32
LE values, totalling 1284 bytes per vector.  This format is directly readable by
the C++ ANN algorithms and by parse_sift().  Additionally, an _ids.json mapping
file is saved alongside the embeddings, mapping vector index to UniProt accession
(for example index 0 to P12345).  This mapping is used by protein_search.py to
annotate ANN results with protein identifiers and cross-reference with BLAST hits.

Device Selection:
The module auto-detects the best available compute device, preferring CUDA on
NVIDIA GPUs for 10 to 50x speedup over CPU, falling back to XPU for Intel Arc
GPUs via IPEX as experimental support, and finally CPU as the default suitable
for small datasets.

Complexity:
Time is O(N * L_max * model_params), approximately O(N * L_max * 8M) for N
proteins.  Space per batch is O(batch_size * L_max * 320) on the GPU plus
O(N * 320) for the output file.  The max_length=1024 truncation prevents
out-of-memory errors on unusually long proteins.

Batching:
Proteins are processed in batches of batch_size (default 5) to balance GPU
memory usage against throughput.  Padding within a batch is handled by the
tokenizer; the attention mask ensures pad tokens do not affect the mean pooling.
"""

import os
import json
import struct
import time
import logging
from itertools import islice
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel
from Bio import SeqIO

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device.

    Checks for CUDA on NVIDIA GPUs first, then Intel XPU via Intel Extension
    for PyTorch, and falls back to CPU if neither accelerator is available.
    """
    # Check CUDA first (NVIDIA)
    if torch.cuda.is_available():
        return torch.device('cuda')
    
    # Check Intel XPU (if intel_extension_for_pytorch is installed)
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device('xpu')
    except ImportError:
        pass
    except Exception:
        pass  # IPEX may fail if oneAPI not installed
    
    return torch.device('cpu')


def embed(input_file: str, output_file: str, batch_size: int, n_first: int, verbose: bool = False) -> None:
    """Generate ESM-2 embeddings for all proteins in a FASTA file.

    For each batch the function tokenises amino-acid sequences with padding and
    truncation to max_length=1024, runs a forward pass through ESM-2 with
    gradients disabled, mean-pools over the sequence dimension masked by the
    attention_mask to exclude pad tokens, and appends the resulting
    (batch, 320) float32 matrix to the output file in SIFT .fvecs format.
    The output file is removed before writing to avoid appending to stale data,
    and a JSON mapping file is saved alongside.

    The input_file parameter is the path to a FASTA file, output_file is the
    path for the output embeddings, batch_size controls the number of proteins
    per forward pass (trading RAM or VRAM for speed), n_first limits embedding
    to the first N proteins when greater than zero (useful for testing), and
    verbose enables per-batch progress printing instead of the default ten-second
    interval.
    """

    logger.info("Embedding %s -> %s...", input_file, output_file)

    # Get best available device
    device = get_device()
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("  GPU: %s", torch.cuda.get_device_name(0))
    elif device.type == "xpu":
        logger.info("  Intel XPU detected")

    fasta_iterator = SeqIO.parse(input_file, "fasta")
    if n_first != 0:
        fasta_iterator = islice(fasta_iterator, n_first)
    
    records = list(fasta_iterator)
    sequences = [str(r.seq) for r in records]
    protein_ids = [r.id.split('|')[1] if '|' in r.id else r.id for r in records]
    del fasta_iterator, records

    # The following functions take an argument that looks like a filename.
    # It should be noted that it's not a file name, but part of a URL
    # that gets downloaded when the function is called, with caching.
    # On my Linux system, the cache is saved in
    # ~/.cache/huggingface/hub/models--facebook--esm2_t6_8M_UR50D

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = model.to(device)  # Move model to GPU if available
    model.eval()  # Set to evaluation mode

    # remove the output file because it will be appended to later
    # if the above code crashes, the output file won't be affected
    if os.path.exists(output_file):
        os.remove(output_file)

    print_timer_start = time.time()
    start_time = time.time()
    total = len(sequences)
    logger.info("Processing %d proteins...", total)

    for i in tqdm(range(0, len(sequences), batch_size),
                  total=(total + batch_size - 1) // batch_size,
                  desc="Embedding", unit="batch"):

        inputs = tokenizer(
            sequences[i:i+batch_size],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=1024)  # Truncate long sequences to prevent OOM
        
        # Move inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():  # disabling back propagation for efficiency
            outputs = model(**inputs)
        # outputs is a 3d matrix

        # Mean pooling: aggregate per-residue embeddings into a single vector per protein.
        # outputs.last_hidden_state has shape (batch, seq_len, 320).
        # The attention_mask has shape (batch, seq_len) with 1 for real residues, 0 for pads.
        # Unsqueezing adds a third dimension so the mask broadcasts correctly:
        #   mask           : (batch, seq_len, 1)
        #   hidden * mask  : zeros out pad positions
        #   sum(dim=1)     : sums over the sequence axis → (batch, 320)
        #   / mask.sum     : divides by the number of non-pad residues → mean pooling
        mask = inputs["attention_mask"].unsqueeze(-1)
        masked = outputs.last_hidden_state * mask
        mean = masked.sum(dim=1) / mask.sum(dim=1)  # mean = sum / number of elements

        # Move back to CPU for saving
        data = mean.cpu().numpy().astype(np.float32)
        N, D = data.shape

        with open(output_file, "ab") as f:
            for j in range(N):
                f.write(struct.pack("<I", D))
                f.write(data[j].astype("<f4").tobytes())
    
    elapsed_total = time.time() - start_time
    logger.info("Embedding complete! Time: %.1f min (%.1f proteins/sec)",
                elapsed_total / 60, total / elapsed_total)
    
    # Save protein ID mapping file (index -> UniProt ID)
    mapping_file = output_file.replace('.dat', '_ids.json')
    id_mapping = {i: pid for i, pid in enumerate(protein_ids)}
    with open(mapping_file, 'w') as f:
        json.dump(id_mapping, f, indent=2)
    logger.info("Saved protein ID mapping to %s", mapping_file)
