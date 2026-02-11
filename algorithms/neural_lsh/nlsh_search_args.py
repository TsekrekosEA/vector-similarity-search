import argparse
from types import SimpleNamespace

def parse_args():
    parser = argparse.ArgumentParser(description='Neural LSH index search')

    parser.add_argument('-d', required=True, metavar='DATASET', help="Path to the input data file")
    parser.add_argument('-q', required=True, metavar='QUERY', help="Path to the query file")
    parser.add_argument('-i', required=True, metavar='INDEX', help="Path to the index created by nlsh_build.py")
    parser.add_argument('-o', required=False, metavar='OUTPUT', help="Path to the output file")
    parser.add_argument('-type', required=True, choices=['sift', 'mnist'], help="Image type")

    parser.add_argument('-N', type=int, default=1, metavar='NEAREST_NEIGHBORS', help="Defaults to 1")
    parser.add_argument('-R', type=float, metavar='SEARCH_RADIUS', help="Defaults to 2800.0 for SIFT or 2000.0 for MNIST")
    parser.add_argument('-T', type=int, default=5, metavar='MULTI_PROBE_BINS', help="Number of parts (bins) to check (multi-probe), defaults to 5")
    parser.add_argument('--range', choices=['true', 'false'], default='true', help="Whether to perform range search, 'true' by default")
    parser.add_argument('--batch_size', type=int, default=128, help="Defaults to 128")

    parser.add_argument('--csv-output-file', metavar='PATH')
    parser.add_argument('--minimal-output', action='store_true', help="If set, skips writing the detailed output file")

    args = parser.parse_args()

    if not args.minimal_output and args.o is None:
        parser.error("the following arguments are required: -o")

    r = SimpleNamespace()
    r.input_dataset_filename = args.d
    r.queries_filename = args.q
    r.input_index_filename = args.i
    r.output_filename = args.o
    r.is_sift = (args.type == 'sift')
    r.num_nearest_neighbors = args.N
    r.search_radius = _get_r(args.R, args.type)
    r.multi_probe_bins = args.T
    r.is_range_search = (args.range == 'true')
    r.output_csv_filename = args.csv_output_file
    r.batch_size = args.batch_size
    r.minimal_output = args.minimal_output

    return r

def _get_r(R, t):

    if R is not None:
        return R

    if t == 'sift':
        return 2800.0
    elif t == 'mnist':
        return 2000.0

    raise Exception(f"R = {R}, t = {t}")

if __name__ == '__main__':
    args = parse_args()
    print(args)