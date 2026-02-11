from __future__ import annotations

from pathlib import Path

from results import SearchOutput


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def write_output(output: SearchOutput, file_path: str | Path) -> None:
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
