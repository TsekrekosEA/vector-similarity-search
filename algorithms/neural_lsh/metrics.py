from __future__ import annotations

from results import QueryResult, SearchOutput


def _is_query_valid(query: QueryResult) -> bool:
    if not query.nearest_neighbors:
        return False
    first = query.nearest_neighbors[0]
    return not (first.distance_approximate == 0.0 and first.id == 0)


def calculate_metrics(
    output: SearchOutput,
    t_approximate_total: float | None = None,
    t_true_total: float | None = None,
) -> SearchOutput:
    """Populate AF, Recall@N, QPS, and timing stats onto the output object."""
    if not output.queries:
        output.average_af = 0.0
        output.recall_at_n = 0.0
        output.queries_per_second = 0.0
        output.t_approximate_average = 0.0
        output.t_true_average = 0.0
        return output

    total_af = 0.0
    queries_with_true_nn = 0
    valid_queries = 0

    for query in output.queries:
        if not _is_query_valid(query):
            continue
        first = query.nearest_neighbors[0]
        valid_queries += 1

        if first.distance_true > 0.0:
            total_af += first.distance_approximate / first.distance_true
        else:
            total_af += 1.0

        true_id = first.id_true
        if any(neighbor.id == true_id for neighbor in query.nearest_neighbors):
            queries_with_true_nn += 1

    if valid_queries > 0:
        output.average_af = total_af / valid_queries
        output.recall_at_n = queries_with_true_nn / valid_queries
    else:
        output.average_af = 0.0
        output.recall_at_n = 0.0

    num_queries = len(output.queries)
    if t_approximate_total and t_approximate_total > 0.0:
        output.t_approximate_average = t_approximate_total / num_queries
        output.queries_per_second = num_queries / t_approximate_total
    else:
        output.t_approximate_average = 0.0
        output.queries_per_second = 0.0

    if t_true_total and t_true_total > 0.0:
        output.t_true_average = t_true_total / num_queries
    else:
        output.t_true_average = 0.0

    return output
