#!/usr/bin/env python3
"""Neural LSH index builder.

Algorithm Overview:
Neural LSH replaces the random hash functions of classical Locality-Sensitive Hashing
(see lsh.hpp) with a learned partition function.  Instead of hashing vectors with
random projections, we first discover high-quality partitions through graph-based
clustering via KaHIP, then train a lightweight MLP to predict these partitions at
query time.  The result is an IVF-style index, comparable to ivfflat.hpp, whose
assignment function is a neural network rather than k-means.

Pipeline:
The build process proceeds through five stages.  First, the dataset (MNIST IDX3 or
SIFT .fvecs) is loaded into a contiguous float32 array via parsers.py.  Second, a
mutual nearest-neighbour graph is constructed using scikit-learn's ball-tree, where
edge weights encode neighbourhood symmetry: mutual neighbours (both i to j and j to i)
receive weight 2 while unilateral edges receive weight 1.  This distinction lets KaHIP
prefer cuts that separate weakly-connected regions.  Third, the weighted CSR graph is
fed to KaHIP's kaffpa solver, which finds a balanced M-way partition minimising the
total cut weight, where balanced means each partition has roughly n/M points controlled
by the imbalance factor (approximately 0.03) and "eco" mode provides the default
quality/speed tradeoff.  Fourth, an MLPClassifier (see model.py) is trained via
cross-entropy on the labels produced by KaHIP, amounting to knowledge distillation: the
combinatorial partitioner is the teacher and the network is the student, using Adam
optimiser for typically 10 to 15 epochs.  Finally, model weights, partition labels,
inverted index, and metadata are serialised into a single .pth checkpoint consumed by
nlsh_search.py.

Theoretical Foundation:
The k-NN graph encodes the local neighbourhood structure of the dataset.  A balanced
min-cut partition on this graph groups points that are near each other in Euclidean
space into the same partition, while ensuring that partition sizes are roughly equal.
This is superior to random partitioning (which ignores geometry) and often competitive
with k-means (which assumes convex Voronoi cells).

CSR Representation:
KaHIP expects graph data in Compressed Sparse Row format where xadj[i] to xadj[i+1]
gives the index range into adjncy and adjcwgt for the neighbours of vertex i.  This
mirrors the CSR format used by scipy.sparse and is the standard for large-graph solvers.

Complexity:
The k-NN stage runs in O(n^2 * D / epsilon) via ball-tree approximate search in
practice, or O(n * k * D) amortised with spatial indexing.  KaHIP operates in
O(|E| * log n) with its multi-level scheme.  Training takes O(epochs * n * model_params)
for Adam updates, and serialisation is O(n) to build the inverted index and write the
checkpoint.

Parameter Tuning:
The knn parameter at 10 to 20 controls graph density: higher k gives a denser graph
with more edges for KaHIP to cut, but too high is slow and yields diminishing returns.
The number of partitions m at 50 to 500 affects bucket size: more partitions produce
smaller buckets enabling faster search, but the miss rate increases if T (probes) is
not raised accordingly.  The epochs parameter at 10 to 20 is sufficient because
cross-entropy converges quickly and overfitting is unlikely since the labels are
already noisy due to KaHIP being approximate.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List

import kahip
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import trange

from model import MLPClassifier
from parsers import get_parser

logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> SimpleNamespace:
    ap = argparse.ArgumentParser(description="Neural LSH index builder")

    ap.add_argument("-d", required=True, metavar="DATASET", help="Path to the input dataset file")
    ap.add_argument("-i", required=True, metavar="OUTPUT_INDEX", help="Path to the output index (.pth)")
    ap.add_argument("-type", required=True, choices=["sift", "mnist"])

    ap.add_argument("--knn", type=int, default=10, metavar="K", help="Neighbours for the k-NN graph (default: 10)")
    ap.add_argument("-m", type=int, default=100, metavar="M", help="KaHIP partitions / blocks (default: 100)")
    ap.add_argument("--imbalance", type=float, default=0.03, help="KaHIP imbalance factor (default: 0.03)")
    ap.add_argument(
        "--kahip_mode",
        choices=["0", "fast", "1", "eco", "2", "strong"],
        default="eco",
        help="KaHIP mode: 0/fast, 1/eco (default), 2/strong",
    )
    ap.add_argument("--layers", type=int, default=3, help="Hidden layers in the MLP (default: 3)")
    ap.add_argument("--nodes", type=int, default=64, help="Nodes per hidden layer (default: 64)")
    ap.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    ap.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    ap.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    ap.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")

    raw = ap.parse_args()

    return SimpleNamespace(
        input_dataset_filename=raw.d,
        output_index_filename=raw.i,
        is_sift=(raw.type == "sift"),
        nearest_neighbors_for_knn_graph=raw.knn,
        kahip_blocks=raw.m,
        kahip_imbalance=raw.imbalance,
        kahip_mode=_normalize_kahip_mode(raw.kahip_mode),
        mlp_layers=raw.layers,
        nodes_per_layer=raw.nodes,
        epochs=raw.epochs,
        batch_size=raw.batch_size,
        learning_rate=raw.lr,
        seed=raw.seed,
    )


def _normalize_kahip_mode(m: str) -> int:
    return {"0": 0, "fast": 0, "1": 1, "eco": 1, "2": 2, "strong": 2}[m]


# ── k-NN graph construction ──────────────────────────────────────────────────


def _build_knn_graph(images: np.ndarray, k: int) -> SimpleNamespace:
    """Build a weighted, symmetrised k-NN graph in CSR format for KaHIP.

    The algorithm first fits a ball-tree via scikit-learn's NearestNeighbors on
    the dataset, then queries each point for its k+1 nearest neighbours where
    the +1 accounts for the point itself (always its own nearest neighbour at
    distance 0).  For each directed edge i to j, the edge is recorded in a
    dictionary, and if j to i already exists the edge weight is promoted from 1
    to 2, indicating a mutual neighbour.  This weighting biases KaHIP to cut
    unilateral edges first, preserving tightly-connected local clusters.  The
    adjacency dictionary is then converted into CSR arrays (xadj, adjncy,
    adjcwgt, vwgt) expected by kahip.kaffpa.

    The rationale for symmetrised weights is that in a k-NN graph the relation
    "j is a neighbour of i" is not necessarily symmetric.  Mutual neighbours
    represent a stronger proximity signal, so assigning weight 2 to mutual edges
    communicates to the partitioner that cutting these edges is twice as
    expensive, leading to partitions that better respect the dataset's intrinsic
    neighbourhood structure.

    Time is O(n * k * D) for ball-tree queries plus O(n * k) for CSR
    construction.  Space is O(n * k) for the adjacency representation.
    """
    num_points = images.shape[0]
    if num_points == 0:
        raise ValueError("Dataset is empty; cannot build k-NN graph")

    neighbor_count = min(k + 1, num_points)
    _, neighbor_indexes = (
        NearestNeighbors(n_neighbors=neighbor_count, n_jobs=-1)
        .fit(images)
        .kneighbors(images)
    )

    # Mutual neighbours receive edge weight 2; unilateral neighbours remain 1.
    edge_weights: Dict[tuple[int, int], int] = defaultdict(int)
    for i, neighbours in enumerate(neighbor_indexes):
        for neighbour in neighbours[1:]:
            j = int(neighbour)
            if i == j:
                continue
            edge = (min(i, j), max(i, j))
            edge_weights[edge] += 1

    adjacency: List[List[tuple[int, int]]] = [[] for _ in range(num_points)]
    for (u, v), count in edge_weights.items():
        weight = 2 if count > 1 else 1
        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))

    # Convert to CSR arrays expected by KaHIP.
    vwgt = [1] * num_points
    xadj = [0]
    adjncy: List[int] = []
    adjcwgt: List[int] = []
    for neighbours in adjacency:
        neighbours.sort()
        for vertex, weight in neighbours:
            adjncy.append(vertex)
            adjcwgt.append(weight)
        xadj.append(len(adjncy))

    return SimpleNamespace(vwgt=vwgt, xadj=xadj, adjncy=adjncy, adjcwgt=adjcwgt)


# ── KaHIP partitioning ───────────────────────────────────────────────────────


def _partition(args: SimpleNamespace, graph: SimpleNamespace) -> tuple[int, np.ndarray]:
    """Run KaHIP balanced k-way partitioning on the CSR graph.

    Delegates to kahip.kaffpa which implements a multi-level graph partitioning
    algorithm with local search refinement (Karlsruhe High-Quality Partitioning).
    The kahip_mode parameter selects the quality/speed tradeoff: mode 0 (fast)
    runs a V-cycle with simple refinement, mode 1 (eco) runs a V-cycle with
    flow-based refinement and provides a good balance (this is the default), and
    mode 2 (strong) runs multiple V-cycles with the strongest refinement for the
    best quality at the cost of speed.  Returns a tuple of (edgecut, labels)
    where edgecut is the total weight of edges cut by the partition and labels is
    an (n,) int64 array mapping each point to its partition id in the range
    [0, M).
    """
    edgecut, labels = kahip.kaffpa(
        graph.vwgt,
        graph.xadj,
        graph.adjcwgt,
        graph.adjncy,
        args.kahip_blocks,
        args.kahip_imbalance,
        True,  # suppress_output
        args.seed,
        args.kahip_mode,
    )
    return edgecut, np.asarray(labels, dtype=np.int64)


# ── MLP training ─────────────────────────────────────────────────────────────


def _train_model(
    images: np.ndarray,
    blocks: np.ndarray,
    model: MLPClassifier,
    args: SimpleNamespace,
    device: torch.device,
) -> None:
    """Train the MLP to predict partition labels via cross-entropy.

    This is standard supervised classification where the input is a feature
    vector and the label is the partition id assigned by KaHIP.  The MLP learns
    to approximate the non-linear decision boundary that KaHIP computed
    combinatorially.  CrossEntropyLoss internally applies log-softmax, so the
    model outputs raw logits with no final activation.  Adam is used with a
    fixed learning rate and no scheduler because convergence is generally reached
    within 10 to 15 epochs since the labels are already noisy (KaHIP is a
    heuristic) and the model cannot overfit severely.  The training loop is
    intentionally minimal with no validation split and no early stopping because
    the quality metric that matters is search recall, not classification
    accuracy.

    Time is O(epochs * n / batch_size * forward+backward), approximately
    O(epochs * n * model_params).
    """
    features = torch.from_numpy(images.astype(np.float32))
    labels = torch.from_numpy(blocks.astype(np.int64))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in trange(args.epochs, desc="Training MLP", unit="epoch"):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()


# ── Inverted index ────────────────────────────────────────────────────────────


def _build_inverted_index(labels: np.ndarray) -> Dict[int, List[int]]:
    """Create a mapping from partition id to the list of dataset point indices.

    This is the runtime lookup structure used by nlsh_search.py: for each
    partition predicted by the MLP, the inverted index provides the candidate
    set of dataset points to scan.  It is analogous to the posting lists in
    IVF-Flat (see ivfflat.hpp).  Time is O(n) for a single pass and space is
    O(n) total entries across all lists.
    """
    inv: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        inv[int(label)].append(idx)
    return dict(inv)


# ── Reproducibility ──────────────────────────────────────────────────────────


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# ── Main pipeline ────────────────────────────────────────────────────────────


def main(args: SimpleNamespace | None = None) -> None:
    if args is None:
        args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _set_seeds(args.seed)

    # 1. Load dataset.
    logger.info("Loading dataset...")
    parse = get_parser(args.is_sift)
    images = parse(args.input_dataset_filename)
    logger.info("Dataset loaded. Shape: %s", images.shape)

    # 2. Build k-NN graph and partition with KaHIP.
    logger.info("Building k-NN graph...")
    graph = _build_knn_graph(images, args.nearest_neighbors_for_knn_graph)
    logger.info("Running KaHIP partitioning...")
    edgecut, blocks = _partition(args, graph)
    logger.info("KaHIP finished.")

    # 3. Train MLP to predict partition labels.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(blocks.max() + 1)
    assert num_classes == args.kahip_blocks

    model = MLPClassifier(
        input_dim=images.shape[1],
        output_dim=num_classes,
        hidden_dim=args.nodes_per_layer,
        hidden_layers=args.mlp_layers,
    ).to(device)

    _train_model(images, blocks, model, args, device)

    # 4. Serialise everything the search phase needs.
    state = {
        "model_state_dict": model.state_dict(),
        "partitions": blocks.tolist(),
        "inverted_index": _build_inverted_index(blocks),
        "metadata": {
            "feature_dim": images.shape[1],
            "num_points": images.shape[0],
            "kahip": {
                "blocks": args.kahip_blocks,
                "imbalance": args.kahip_imbalance,
                "mode": args.kahip_mode,
                "edgecut": edgecut,
                "knn_graph_k": args.nearest_neighbors_for_knn_graph,
            },
            "mlp": {
                "layers": args.mlp_layers,
                "hidden_dim": args.nodes_per_layer,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
            "dataset": {
                "is_sift": args.is_sift,
                "path": args.input_dataset_filename,
            },
        },
    }
    torch.save(state, args.output_index_filename)
    logger.info("Index saved to %s", args.output_index_filename)


if __name__ == "__main__":
    main()
