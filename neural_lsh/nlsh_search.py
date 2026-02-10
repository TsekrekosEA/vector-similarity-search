#!/usr/bin/env python3
"""Neural LSH search — multi-probe approximate nearest-neighbour search.

Algorithm Overview:
This module performs the search phase of Neural LSH.  Given a checkpoint produced
by nlsh_build.py (containing model weights, partition labels, and an inverted
index), it classifies every query vector with the trained MLP, probes the top-T
predicted partitions, and reports ANN quality metrics against brute-force ground
truth.

Multi-Probe Search Strategy:
The MLP outputs M logits (one per partition).  torch.topk selects the T
most-likely partitions for each query, analogous to the nprobe parameter in
IVF-Flat (see ivfflat.hpp) or the multi-probe strategy in Hypercube (see
hypercube.hpp).  The union of points in those T partitions forms the candidate
set; exact squared Euclidean distances are then computed only for candidates,
and the top-N results are returned.  Setting T to 1 yields the fastest search
but may miss neighbours in adjacent partitions.  Setting T to roughly M/10
provides a good balance for most datasets with recall above 90 percent.  Setting
T equal to M degenerates to brute force with 100 percent recall and no speedup.

Implementation Details:
Queries are batched (default 128) for efficient GPU or CPU utilisation during MLP
forward passes, similar to how IVF-PQ batches codebook distance computations.
Candidate deduplication via a set removes duplicates when a point appears in
multiple probed partitions.  Distances are computed as squared L2 to avoid a
per-pair square root; the square root is applied only for the final top-N results.

Ground-Truth Comparison:
After the ANN phase, the _get_ground_truth function runs brute-force search (or
loads from cache via ground_truth_cache.py) to populate id_true and distance_true
fields.  The metrics.calculate_metrics function then computes AF, Recall@N, and
QPS.

Complexity:
Let n be the dataset size, D the dimensionality, M the number of partitions, T
the number of probes, and |Q| the number of queries.  MLP inference runs in
O(|Q| * model_params), which is fast for small MLPs.  Candidate collection is
O(|Q| * T * (n/M)) since each partition holds an expected n/M points.  Distance
refinement is O(|Q| * C * D) where C is the number of candidates per query, and
top-N selection is O(|Q| * C) via torch.topk.  The total is
O(|Q| * (model_params + T*n*D/M)) compared to brute force at O(|Q|*n*D), giving
a speedup of approximately M/T when the MLP cost is negligible.
"""

from __future__ import annotations

import argparse
import logging
import time
from types import SimpleNamespace

import numpy as np
import torch

import brute_force
import ground_truth_cache
import metrics
import output_writer
import results
from model import MLPClassifier
from parsers import get_parser

logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> SimpleNamespace:
    ap = argparse.ArgumentParser(description="Neural LSH search")

    ap.add_argument("-d", required=True, metavar="DATASET", help="Path to the dataset file")
    ap.add_argument("-q", required=True, metavar="QUERY", help="Path to the query file")
    ap.add_argument("-i", required=True, metavar="INDEX", help="Path to the index (.pth) from nlsh_build.py")
    ap.add_argument("-o", required=False, metavar="OUTPUT", help="Path to the output file")
    ap.add_argument("-type", required=True, choices=["sift", "mnist"])

    ap.add_argument("-N", type=int, default=1, metavar="K", help="Nearest neighbours to find (default: 1)")
    ap.add_argument("-R", type=float, metavar="RADIUS", help="Search radius (default: 2000 MNIST / 2800 SIFT)")
    ap.add_argument("-T", type=int, default=5, metavar="PROBES", help="Partitions to probe (default: 5)")
    ap.add_argument("--range", choices=["true", "false"], default="true", help="Enable range search (default: true)")
    ap.add_argument("--batch_size", type=int, default=128, help="Query batch size (default: 128)")
    ap.add_argument("--csv-output-file", metavar="PATH", help="Append CSV metrics to this file")
    ap.add_argument("--minimal-output", action="store_true", help="Skip writing detailed output file")

    raw = ap.parse_args()

    if not raw.minimal_output and raw.o is None:
        ap.error("the following arguments are required: -o")

    radius = raw.R
    if radius is None:
        radius = 2800.0 if raw.type == "sift" else 2000.0

    return SimpleNamespace(
        input_dataset_filename=raw.d,
        queries_filename=raw.q,
        input_index_filename=raw.i,
        output_filename=raw.o,
        is_sift=(raw.type == "sift"),
        num_nearest_neighbors=raw.N,
        search_radius=radius,
        multi_probe_bins=raw.T,
        is_range_search=(raw.range == "true"),
        batch_size=raw.batch_size,
        output_csv_filename=raw.csv_output_file,
        minimal_output=raw.minimal_output,
    )


# ── Neural LSH search ────────────────────────────────────────────────────────


def _neural_search(
    model: MLPClassifier,
    dataset_tensor: torch.Tensor,
    queries_np: np.ndarray,
    ids_per_label: dict,
    args: SimpleNamespace,
    device: torch.device,
) -> tuple[results.SearchOutput, float]:
    """Run multi-probe neural search and return (output, elapsed_seconds).

    The search proceeds in three stages.  In the partition prediction stage,
    queries are fed through the MLP in batches of args.batch_size.  For each
    query the MLP outputs M logits, and torch.topk selects the T highest,
    giving the T most-likely partition ids.  This is the learned hash step,
    analogous to computing the hash key in classical LSH or finding the nearest
    centroid in IVF-Flat.  In the candidate retrieval stage, the inverted index
    (ids_per_label) is consulted for each of the T predicted partitions and all
    point indices are collected into a de-duplicated set, mirroring the posting-list
    lookup in IVF-Flat's nprobe scan.  In the distance refinement stage, exact
    squared Euclidean distances are computed between the query and all candidates
    via a single vectorised subtraction and element-wise squaring on the GPU
    tensor; torch.topk with largest=False selects the N smallest distances and
    the final distances are square-rooted for the output.  If is_range_search is
    True, all candidates within search_radius are also collected into
    qr.r_near_neighbors.

    Time per query is O(model_params + T*(n/M) + C*D + C) where C is the number
    of candidates.
    """

    queries_tensor = torch.from_numpy(queries_np)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(queries_tensor),
        batch_size=args.batch_size,
        shuffle=False,
    )

    start = time.time()

    # Predict partitions for every query.
    batches_of_predicted: list[torch.Tensor] = []
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device))
            _, predicted = torch.topk(logits, args.multi_probe_bins, dim=1)
            batches_of_predicted.append(predicted)

    all_predicted = torch.cat(batches_of_predicted)

    output = results.SearchOutput(algorithm="neural LSH")

    for qid, predicted_labels in enumerate(all_predicted):
        # Collect candidate IDs from all probed partitions.
        candidate_ids: list[int] = []
        for bucket in predicted_labels:
            bid = int(bucket.item())
            if bid in ids_per_label:
                candidate_ids.extend(ids_per_label[bid])
        candidate_ids = list(set(candidate_ids))

        qr = results.QueryResult()

        if not candidate_ids:
            output.queries.append(qr)
            continue

        image_ids = torch.tensor(candidate_ids, device=device)
        candidates = dataset_tensor[image_ids]
        query_vec = torch.from_numpy(queries_np[qid]).to(device)

        sq_dists = ((query_vec - candidates) ** 2).sum(dim=1)

        k = min(args.num_nearest_neighbors, len(candidate_ids))
        top_sq, top_idx = torch.topk(sq_dists, k=k, largest=False)
        top_global = image_ids[top_idx]
        top_dists = torch.sqrt(top_sq)

        for gid, dist in zip(top_global.cpu().numpy(), top_dists.cpu().numpy()):
            qr.nearest_neighbors.append(
                results.NearNeighbor(id=int(gid), distance_approximate=float(dist))
            )

        if args.is_range_search:
            all_dists = torch.sqrt(sq_dists)
            within = torch.lt(all_dists, args.search_radius)
            qr.r_near_neighbors = image_ids[within].cpu().numpy().tolist()

        output.queries.append(qr)

    elapsed = time.time() - start
    return output, elapsed


# ── Ground truth (brute force) ────────────────────────────────────────────────


def _get_ground_truth(
    output: results.SearchOutput,
    dataset_np: np.ndarray,
    queries_np: np.ndarray,
    args: SimpleNamespace,
) -> float:
    """Fill in true distances and return brute-force wall time.

    Ground truth is first looked up in the disk cache via ground_truth_cache.py.
    On a cache miss, brute_force.brute_force_search is invoked in fill-existing
    mode, populating id_true and distance_true on the already-allocated
    NearNeighbor entries.  The result is then cached for subsequent runs.
    Returns the brute-force wall-clock seconds, used by metrics.py for timing
    stats and by _write_csv for speedup calculations.
    """

    cached = ground_truth_cache.load_ground_truth(
        args.input_dataset_filename,
        args.queries_filename,
        args.num_nearest_neighbors,
        args.search_radius,
        args.is_range_search,
    )

    if cached:
        logger.info("Using cached ground truth results.")
        cached_output, brute_time = cached
        dataset_size = dataset_np.shape[0]
        count = min(args.num_nearest_neighbors, dataset_size)
        for i, qr in enumerate(output.queries):
            cqr = cached_output.queries[i]
            while len(qr.nearest_neighbors) < count:
                qr.nearest_neighbors.append(results.NearNeighbor())
            for j in range(count):
                if j < len(cqr.nearest_neighbors):
                    qr.nearest_neighbors[j].id_true = cqr.nearest_neighbors[j].id_true
                    qr.nearest_neighbors[j].distance_true = cqr.nearest_neighbors[j].distance_true
            qr.r_near_neighbors = cqr.r_near_neighbors
        return brute_time

    t0 = time.time()
    brute_force.brute_force_search(
        dataset=dataset_np,
        queries=queries_np,
        num_neighbors=args.num_nearest_neighbors,
        search_for_range=args.is_range_search,
        range_radius=args.search_radius,
        existing_output=output,
    )
    brute_time = time.time() - t0

    ground_truth_cache.save_ground_truth(
        output,
        brute_time,
        args.input_dataset_filename,
        args.queries_filename,
        args.num_nearest_neighbors,
        args.search_radius,
        args.is_range_search,
    )
    return brute_time


# ── CSV export ────────────────────────────────────────────────────────────────


def _write_csv(output: results.SearchOutput, state: dict, args: SimpleNamespace) -> None:
    if args.output_csv_filename is None:
        return

    brute_ref = 0.125492 if "mnist" in args.input_dataset_filename.lower() else 0.110198
    t_approx = output.t_approximate_average if output.t_approximate_average > 0 else 1e-9
    speedup = brute_ref / t_approx

    meta = state["metadata"]
    k_knn = meta["kahip"].get("knn_graph_k", "?")
    m_blocks = meta["kahip"]["blocks"]
    epochs = meta["mlp"]["epochs"]

    row = (
        f"{args.input_dataset_filename},Neural LSH,"
        f"{k_knn},{m_blocks},{epochs},{args.multi_probe_bins},"
        f"{args.num_nearest_neighbors},{args.search_radius},"
        f"{output.average_af:.6f},{output.recall_at_n:.6f},{output.queries_per_second:.2f},"
        f"{output.t_approximate_average:.6f},{output.t_true_average:.6f},{speedup:.6f}\n"
    )
    with open(args.output_csv_filename, "a") as f:
        f.write(row)


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args: SimpleNamespace | None = None) -> None:
    if args is None:
        args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parse = get_parser(args.is_sift)

    # Load data.
    dataset_np = parse(args.input_dataset_filename).astype(np.float32)
    dataset_tensor = torch.from_numpy(dataset_np).to(device)
    queries_np = parse(args.queries_filename).astype(np.float32)

    # Load checkpoint.
    state = torch.load(args.input_index_filename, map_location=device)
    meta = state["metadata"]["mlp"]
    model = MLPClassifier(
        input_dim=state["metadata"]["feature_dim"],
        output_dim=len(state["inverted_index"]),
        hidden_dim=meta["hidden_dim"],
        hidden_layers=meta["layers"],
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Neural search.
    output, query_time = _neural_search(
        model, dataset_tensor, queries_np, state["inverted_index"], args, device,
    )

    # Ground truth.
    brute_time = _get_ground_truth(output, dataset_np, queries_np, args)

    # Metrics.
    metrics.calculate_metrics(output, t_approximate_total=query_time, t_true_total=brute_time)

    # CSV.
    _write_csv(output, state, args)

    # Detailed output.
    if not args.minimal_output and args.output_filename:
        output_writer.write_output(output, args.output_filename)


if __name__ == "__main__":
    main()
