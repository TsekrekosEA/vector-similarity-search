from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np

from results import NearNeighbor


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Return the Euclidean distance between two vectors."""
    diff = vec1 - vec2
    return float(np.linalg.norm(diff))


def euclidean_distance_squared(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Return the squared Euclidean distance between two vectors."""
    diff = vec1 - vec2
    return float(np.dot(diff, diff))


def eucl_d_sq_if_smaller_else_inf(
    vec1: np.ndarray, vec2: np.ndarray, other_distance: float
) -> float:
    """Return squared distance if it is smaller than other_distance, else inf."""
    diff = vec1 - vec2
    sum_of_squares = 0.0
    for value in diff:
        sum_of_squares += float(value * value)
        if sum_of_squares > other_distance:
            return math.inf
    return sum_of_squares


def find_k_nearest_from_candidates(
    candidate_ids: Sequence[int],
    query_point: np.ndarray,
    dataset: np.ndarray,
    k: int,
) -> List[NearNeighbor]:
    """Return the k closest candidates according to true distance."""
    if k <= 0 or len(candidate_ids) == 0:
        return []

    import heapq

    max_heap: List[tuple[float, int]] = []
    for idx in candidate_ids:
        distance = euclidean_distance(query_point, dataset[idx])
        if len(max_heap) < k:
            heapq.heappush(max_heap, (-distance, idx))
        elif distance < -max_heap[0][0]:
            heapq.heapreplace(max_heap, (-distance, idx))

    neighbors: List[NearNeighbor] = []
    while max_heap:
        distance, idx = heapq.heappop(max_heap)
        neighbors.append(NearNeighbor(id=idx, distance_approximate=-distance))
    neighbors.reverse()
    return neighbors


def find_in_range_from_candidates(
    candidate_ids: Iterable[int],
    query_point: np.ndarray,
    dataset: np.ndarray,
    radius: float,
) -> List[int]:
    """Return all candidate ids whose distance from the query is within radius."""
    if radius < 0:
        return []

    result: List[int] = []
    for idx in candidate_ids:
        if euclidean_distance(query_point, dataset[idx]) <= radius:
            result.append(idx)
    return result
