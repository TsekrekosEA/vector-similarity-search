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
    """Compute exact distances for each query, mirroring part-1 logic."""

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
        diff = dataset - query_vec
        dist_squared = np.sum(diff * diff, axis=1)

        count = min(num_neighbors, dataset_size)
        kth = count - 1
        if kth >= 0:
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
