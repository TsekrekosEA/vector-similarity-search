from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class NearNeighbor:
    id: int = 0
    id_true: int = 0
    distance_approximate: float = 0.0
    distance_true: float = 0.0


@dataclass
class QueryResult:
    nearest_neighbors: List[NearNeighbor] = field(default_factory=list)
    r_near_neighbors: List[int] = field(default_factory=list)


@dataclass
class SearchOutput:
    algorithm: str = ""
    queries: List[QueryResult] = field(default_factory=list)
    average_af: float = 0.0
    recall_at_n: float = 0.0
    queries_per_second: float = 0.0
    t_approximate_average: float = 0.0
    t_true_average: float = 0.0
