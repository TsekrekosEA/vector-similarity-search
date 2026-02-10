"""CLI entry point for ESM-2 protein embedding generation.

This thin front-end parses command-line arguments and defers all heavy work to
protein_embed_backend.embed().  The split is intentional: importing transformers,
torch, and BioPython takes several seconds, so --help and argument validation
remain instant by deferring those imports to after argument parsing.  The
workflow is simple: first parse CLI arguments with no ML imports, then import
protein_embed_backend which triggers model download on first run, and finally
call embed() which reads the FASTA file, generates ESM-2 embeddings, and writes
them in SIFT .fvecs format for consumption by the ANN search algorithms.  See
protein_embed_backend.py for the algorithmic details.
"""

import argparse
import logging
# imports slow down --help and invalid arguments, which is not desired

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ESM-2 embeddings for protein_search.py")
    p.add_argument('-i', '--input', default='data/swissprot.fasta', type=str, metavar='FASTA', 
                   help="default: data/swissprot.fasta, input proteins file")
    p.add_argument('-o', '--output', default='data/protein_vectors.dat', type=str, metavar='OUTPUT', 
                   help="default: data/protein_vectors.dat, embeds file")
    p.add_argument('-b', '--batch', default=5, type=int, metavar='SIZE', 
                   help="default: 5, larger values use more RAM but are faster")
    p.add_argument('-n', '--first', default=0, type=int, metavar='N', 
                   help="default: 0 (all), number of proteins to embed (for testing)")
    p.add_argument('-v', '--verbose', action='store_true',
                   help="print progress every batch")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from protein_embed_backend import embed
    logger.info("Embedding %s -> %s...", args.input, args.output)
    logger.info("Batch size: %d, First N: %s", args.batch, args.first if args.first > 0 else "all")
    embed(
        input_file=args.input, 
        output_file=args.output, 
        batch_size=args.batch, 
        n_first=args.first,
        verbose=args.verbose
    )