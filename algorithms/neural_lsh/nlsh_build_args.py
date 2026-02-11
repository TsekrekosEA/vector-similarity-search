import argparse
from types import SimpleNamespace

def parse_args():
    parser = argparse.ArgumentParser(description='Neural LSH index builder')

    parser.add_argument('-d', required=True, metavar='DATASET', help="Path to the input file")
    parser.add_argument('-i', required=True, metavar='OUTPUT_INDEX', help="Path to the output file")
    parser.add_argument('-type', required=True, choices=['sift', 'mnist'])

    parser.add_argument('--knn', type=int, default=10, metavar='NEAREST_NEIGHBORS', help="Defaults to 10")
    parser.add_argument('-m', type=int, default=100, metavar='KAHIP_BLOCKS', help="Defaults to 100")
    parser.add_argument('--imbalance', type=float, default=0.03, metavar='KAHIP_IMBALANCE', help="Defaults to 0.03")
    parser.add_argument('--kahip_mode', choices=['0', 'fast', '1', 'eco', '2', 'strong'], default='eco', help="Defaults to 1/eco")
    parser.add_argument('--layers', type=int, default=3, metavar='MLP_LAYERS', help="Defaults to 3")
    parser.add_argument('--nodes', type=int, default=64, metavar='NODES_PER_LAYER', help="Defaults to 64")
    parser.add_argument('--epochs', type=int, default=10, help="Defaults to 10")
    parser.add_argument('--batch_size', type=int, default=128, help="Defaults to 128")
    parser.add_argument('--lr', type=float, default=0.001, metavar='LEARNING_RATE', help="Defaults to 0.001")
    parser.add_argument('--seed', type=int, default=1, metavar='RAND_SEED', help="Defaults to 1")

    args = parser.parse_args()

    r = SimpleNamespace()
    r.input_dataset_filename = args.d
    r.output_index_filename = args.i
    r.is_sift = (args.type == 'sift')
    r.nearest_neighbors_for_knn_graph = args.knn
    r.kahip_blocks = args.m
    r.kahip_imbalance = args.imbalance
    r.kahip_mode = _normalize_kahip_mode(args.kahip_mode)
    r.mlp_layers = args.layers
    r.nodes_per_layer = args.nodes
    r.epochs = args.epochs
    r.batch_size = args.batch_size
    r.learning_rate = args.lr
    r.seed = args.seed

    return r

def _normalize_kahip_mode(m):
    if (m == '0' or m == 'fast'):
        return 0
    elif (m == '1' or m == 'eco'):
        return 1
    elif (m == '2' or m == 'strong'):
        return 2

if __name__ == '__main__':
    args = parse_args()
    print(args)
