import argparse
# imports slow down --help and invalid arguments, which is not desired


def get_args():
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
    from protein_embed_backend import embed
    print(f"Embedding {args.input} -> {args.output}...")
    print(f"Batch size: {args.batch}, First N: {args.first if args.first > 0 else 'all'}")
    embed(
        input_file=args.input, 
        output_file=args.output, 
        batch_size=args.batch, 
        n_first=args.first,
        verbose=args.verbose
    )