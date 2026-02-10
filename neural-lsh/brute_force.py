"""Exact nearest-neighbour search via brute force, serving as the ground truth baseline.

This module mirrors the brute-force path in brute_force.cpp from the C++ algorithms.
It computes exact Euclidean distances between every query and every dataset point,
then selects the top-N nearest neighbours using a partial-sort strategy.

Algorithm:
For each query q in Q the function computes squared Euclidean distances d^2(q, x)
for every dataset point x via a single vectorised operation (dataset - q)^2 in
NumPy.  It then selects the N smallest distances using np.argpartition, which is
an introselect-based partial sort running in O(n) expected time, much faster than
a full O(n log n) sort when N is much smaller than n.  Only the N selected indices
are then sorted by true distance in O(N log N) to produce the final ranked list.
Optionally all points within a radius R are also collected for range search.

Fill-Existing Mode:
When existing_output is provided, the function does not create new NearNeighbor
objects.  Instead it fills in the id_true and distance_true fields on the
already-allocated result slots.  This is the mechanism used by nlsh_search.py:
the ANN algorithm first populates id and distance_approximate, then brute force
adds the ground-truth counterpart so that metrics.py can compute Approximation
Factor.

Complexity:
Time is O(|Q| * n * D) for distance computation as the dominant term, plus
O(|Q| * n) for partial sort per query.  Space is O(n * D) for the dataset
(shared), plus O(n) temporary per query for the distance vector.

Comparison to Part 1:
The C++ implementation parallelises queries with std::for_each and execution
policies and uses a priority queue for top-N selection.  This Python version
instead leverages NumPy's vectorised row-minus-query broadcast, which is
typically fast enough for the 10K-query regime we target and avoids manual
thread management.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from results import NearNeighbor, QueryResult, SearchOutput


def brute_force_search(
    dataset: np.ndarray,
    queries: np.ndarray,
    num_neighbors: int,
    *,
    range_radius: Optional[float] = None,
    search_for_range: bool = True,
    existing_output: Optional[SearchOutput] = None,
) -> SearchOutput:
    """Compute exact distances for each query, mirroring part-1 logic.

    The dataset parameter is an (n, D) float32 array of database vectors and
    queries is a (|Q|, D) float32 array of query vectors.  The num_neighbors
    parameter controls how many nearest neighbours to return per query.  If
    search_for_range is True and range_radius is provided, all points within
    that Euclidean radius are also collected.  When existing_output is provided
    the function operates in fill-existing mode, populating id_true and
    distance_true on the already-allocated NearNeighbor entries rather than
    creating new ones.  Returns the (possibly new) SearchOutput with
    ground-truth fields populated.  Time is O(|Q| * n * D) dominated by the
    distance matrix computation, with O(n) per query for the distance
    work-vector.
    """

    if num_neighbors <= 0:
        raise ValueError("num_neighbors must be positive")

    num_queries = queries.shape[0]
    fill_existing = existing_output is not None

    if fill_existing:
        output = existing_output
        if len(output.queries) != num_queries:
            raise ValueError("existing_output has mismatched query count")
    else:
        output = SearchOutput(algorithm="brute force")
        output.queries = [QueryResult() for _ in range(num_queries)]

    radius_squared = None
    if search_for_range and range_radius is not None:
        radius_squared = range_radius * range_radius

    dataset_size = dataset.shape[0]
    if dataset_size == 0:
        raise ValueError("dataset is empty; brute force search is undefined")

    for i in range(num_queries):
        query_vec = queries[i]
        # Broadcast subtraction: (n, D) - (D,) → (n, D), then row-wise sum of squares → (n,)
        diff = dataset - query_vec
        dist_squared = np.sum(diff * diff, axis=1)

        count = min(num_neighbors, dataset_size)
        kth = count - 1
        if kth >= 0:
            # argpartition performs an introselect (O(n) expected) so that the first
            # `count` entries are the smallest — but *not* sorted among themselves.
            # A subsequent argsort on only those `count` entries is O(N log N).
            nearest_idx = np.argpartition(dist_squared, kth)[:count]
            nearest_idx = nearest_idx[np.argsort(dist_squared[nearest_idx])]
        else:
            nearest_idx = np.array([], dtype=int)

        row = output.queries[i]
        if fill_existing:
            if len(row.nearest_neighbors) < count:
                row.nearest_neighbors.extend(
                    NearNeighbor() for _ in range(count - len(row.nearest_neighbors))
                )
        else:
            row.nearest_neighbors = [NearNeighbor() for _ in range(len(nearest_idx))]

        for j, idx in enumerate(nearest_idx):
            distance_true = float(np.sqrt(dist_squared[idx]))
            row.nearest_neighbors[j].id_true = int(idx)
            row.nearest_neighbors[j].distance_true = distance_true
            if not fill_existing:
                row.nearest_neighbors[j].distance_approximate = 0.0
        # Range search output mirrors part 1: fill only if enabled
        if search_for_range and radius_squared is not None:
            within_radius = np.where(dist_squared <= radius_squared)[0]
            within_radius = within_radius[np.argsort(dist_squared[within_radius])]
            row.r_near_neighbors = [int(idx) for idx in within_radius]
        elif not search_for_range:
            row.r_near_neighbors = []
        else:
            row.r_near_neighbors = []

    return output
