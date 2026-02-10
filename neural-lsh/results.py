"""Core data structures for ANN search results and quality metrics.

This module is the Python counterpart of data_types.hpp in the C++ algorithms.
It defines the three-level result hierarchy that every search algorithm in this
project populates and that every metric and output routine consumes.

Data Flow:
The search algorithm populates a SearchOutput containing a list of QueryResult
objects, each holding a list of NearNeighbor entries.  The brute-force pass then
fills in the id_true and distance_true fields for ground-truth comparison.  The
metrics module reads both approximate and true fields to compute AF, Recall@N, and
QPS.  Finally the output_writer serialises the whole structure to the
assignment-mandated text format.

Design Decisions:
The dual-distance representation means each NearNeighbor stores both the
approximate distance returned by the ANN algorithm and the exact distance computed
by brute force.  This lets us compute the Approximation Factor
AF = d_approx / d_true per query without a second traversal, matching the
ANearNeighbor struct in data_types.hpp.  Default-zero fields on the dataclass
allow algorithms to populate only the fields they know: the ANN pass fills id and
distance_approximate, while the brute-force pass later fills id_true and
distance_true.  SearchOutput is self-contained: once calculate_metrics from
metrics.py enriches it, the object carries the algorithm name, per-query results,
and aggregate quality numbers, ready for serialisation or programmatic inspection.

Complexity Note:
These are pure data containers; all computation happens in brute_force.py and
metrics.py.  Memory footprint per query is O(N) where N is the number of requested
neighbours, plus O(R) for the optional range-search list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class NearNeighbor:
    """A single neighbour entry carrying both approximate and true distances.

    This mirrors the ANearNeighbor struct in Part 1's data_types.hpp.  The two-id,
    two-distance design enables the Approximation Factor calculation
    AF = distance_approximate / distance_true where an ideal exact algorithm yields
    AF = 1.0 and higher values indicate worse approximation.  The id field holds the
    index of this neighbour in the dataset as reported by the approximate search
    algorithm, while id_true holds the index of the true nearest neighbour at this
    rank as filled by the brute-force pass.  The distance_approximate field stores
    the Euclidean distance returned by the ANN algorithm, and distance_true stores
    the exact Euclidean distance computed by brute force.
    """

    id: int = 0
    id_true: int = 0
    distance_approximate: float = 0.0
    distance_true: float = 0.0


@dataclass
class QueryResult:
    """Results for a single query vector.

    This mirrors OutputForOneQuery in data_types.hpp.  It contains the top-N nearest
    neighbours sorted by approximate distance and an optional list of all points within
    a given radius for range search.  The nearest_neighbors list is ordered by ascending
    approximate distance, while r_near_neighbors holds indices of dataset points within
    the search radius, ordered by ascending true distance to facilitate ground-truth
    comparison.  The range list is populated only when range search is enabled.
    """

    nearest_neighbors: List[NearNeighbor] = field(default_factory=list)
    r_near_neighbors: List[int] = field(default_factory=list)


@dataclass
class SearchOutput:
    """Aggregate search output for all queries plus summary quality metrics.

    This mirrors Output in data_types.hpp.  After construction by a search algorithm
    and enrichment by calculate_metrics, this single object contains everything needed
    to produce the assignment output file and the CSV benchmark row.

    The average_af field holds the mean Approximation Factor across valid queries,
    computed as AF_q = d_approx(q, nn_approx) / d_true(q, nn_true).  The recall_at_n
    field records the fraction of queries where the true 1-NN appears anywhere in the
    top-N approximate result list.  The queries_per_second field equals
    num_queries / t_approximate_total.  The t_approximate_average and t_true_average
    fields hold the mean wall-clock seconds per query for the ANN algorithm and the
    brute-force ground truth respectively.  The algorithm field stores a human-readable
    name such as "neural LSH", and the queries list holds per-query results whose
    length equals the number of query vectors.
    """

    algorithm: str = ""
    queries: List[QueryResult] = field(default_factory=list)
    average_af: float = 0.0
    recall_at_n: float = 0.0
    queries_per_second: float = 0.0
    t_approximate_average: float = 0.0
    t_true_average: float = 0.0
