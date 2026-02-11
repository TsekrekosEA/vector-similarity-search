import os
import json
import struct
import time
from itertools import islice
import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel
from Bio import SeqIO


def get_device():
    """Get the best available device.
    
    Checks for:
    1. CUDA (NVIDIA GPUs)
    2. XPU (Intel Arc GPUs via Intel Extension for PyTorch)
    3. Falls back to CPU
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


def embed(input_file, output_file, batch_size, n_first, verbose=False):

    print(f"Embedding {input_file} -> {output_file}...")

    # Get best available device
    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == "xpu":
        print(f"  Intel XPU detected")

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
    print(f"Processing {total} proteins...")

    for i in range(0, len(sequences), batch_size):

        now = time.time()
        if verbose or (now - print_timer_start > 10):
            pct = 100 * i / total if total > 0 else 0
            elapsed = now - start_time
            if i > 0:
                eta = elapsed * (total - i) / i
                eta_str = f", ETA: {eta/60:.1f} min" if eta < 3600 else f", ETA: {eta/3600:.1f} hr"
            else:
                eta_str = ""
            print(f"  {i}/{total} proteins ({pct:.1f}%){eta_str}")
            print_timer_start = now

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

        # mean pooling around dim=1 (second dimension, the first dimension maps proteins)
        mask = inputs["attention_mask"].unsqueeze(-1)  # add a third dimension to the mask
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
    print(f"Embedding complete! Time: {elapsed_total/60:.1f} min ({total/elapsed_total:.1f} proteins/sec)")
    
    # Save protein ID mapping file (index -> UniProt ID)
    mapping_file = output_file.replace('.dat', '_ids.json')
    id_mapping = {i: pid for i, pid in enumerate(protein_ids)}
    with open(mapping_file, 'w') as f:
        json.dump(id_mapping, f, indent=2)
    print(f"Saved protein ID mapping to {mapping_file}")
