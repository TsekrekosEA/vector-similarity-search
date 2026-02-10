"""Serialise a SearchOutput to the assignment-mandated text format.

The output format mirrors the one produced by main.cpp in the C++ algorithms.
It is designed for human readability and automated parsing by the university
grading scripts.

Output Format:
The first line contains the algorithm name (for example "neural LSH"), followed
by a blank line.  For each query, a "Query: <id>" header is printed, then for
each of the top-N neighbours the dataset index, the approximate distance, and
the true distance are printed on successive lines.  If range search was enabled,
a section listing all dataset indices within the search radius follows.  After
all queries, summary metrics are printed: Average AF, Recall@N, QPS,
tApproximateAverage, and tTrueAverage.  All floating-point values are formatted
to six decimal places for consistency with the C++ output.
"""

from __future__ import annotations

from pathlib import Path

from results import SearchOutput


def _format_float(value: float) -> str:
    """Format a float to six decimal places, matching the C++ std::setprecision(6)."""
    return f"{value:.6f}"


def write_output(output: SearchOutput, file_path: str | Path) -> None:
    """Write the search output in the assignment text format.

    The output should be a fully populated SearchOutput that has already been
    passed through calculate_metrics.  The file_path specifies the destination
    and its parent directories must already exist.
    """
    path = Path(file_path)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{output.algorithm}\n\n")
        for query_id, query in enumerate(output.queries):
            f.write(f"Query: {query_id}\n")
            for rank, neighbor in enumerate(query.nearest_neighbors, start=1):
                f.write(f"Nearest neighbor-{rank}: {neighbor.id}\n")
                f.write(f"distanceApproximate: {_format_float(neighbor.distance_approximate)}\n")
                f.write(f"distanceTrue: {_format_float(neighbor.distance_true)}\n")
            if query.r_near_neighbors:
                f.write("R-near neighbors:\n")
                for neighbor_id in query.r_near_neighbors:
                    f.write(f"{neighbor_id}\n")
            f.write("\n")

        f.write(f"Average AF: {_format_float(output.average_af)}\n")
        f.write(f"Recall@N: {_format_float(output.recall_at_n)}\n")
        f.write(f"QPS: {_format_float(output.queries_per_second)}\n")
        f.write(f"tApproximateAverage: {_format_float(output.t_approximate_average)}\n")
        f.write(f"tTrueAverage: {_format_float(output.t_true_average)}\n")
