"""Quality metrics for approximate nearest-neighbour search evaluation.

This module computes the same evaluation metrics produced by benchmark.cpp in
Part 1, ensuring that results across the three project parts are directly
comparable.

Metric Definitions:

The Approximation Factor (AF) for a query q is defined as
AF_q = d_approx(q, nn_approx) / d_true(q, nn_true) and measures how much
farther the approximate first nearest neighbour is compared to the true first
nearest neighbour.  The ideal value is 1.0 indicating an exact match, and the
average_af field reports the arithmetic mean across all valid queries.

Recall at N is the fraction of queries where the true first nearest neighbour
appears anywhere in the top-N approximate result list.  A recall of 1.0 means
the algorithm never misses the true nearest neighbour.

Queries Per Second (QPS) equals the number of queries divided by the total
approximate search time.  Higher values are faster.

The timing averages t_approximate_average and t_true_average are simply the
total wall-clock time for the approximate or brute-force pass divided by the
number of queries.

Query Validity Filter:
A query is considered invalid and excluded from AF and Recall if its first
neighbour has both distance_approximate equal to zero and id equal to zero.
This guards against uninitialised or empty result slots that would pollute
aggregate statistics.
"""

from __future__ import annotations

from results import QueryResult, SearchOutput


def _is_query_valid(query: QueryResult) -> bool:
    """Return False for uninitialised or empty result slots.

    A query whose first neighbour has default-zero id and default-zero
    distance_approximate is treated as not having been answered by the ANN
    algorithm.
    """
    if not query.nearest_neighbors:
        return False
    first = query.nearest_neighbors[0]
    return not (first.distance_approximate == 0.0 and first.id == 0)


def calculate_metrics(
    output: SearchOutput,
    t_approximate_total: float | None = None,
    t_true_total: float | None = None,
) -> SearchOutput:
    """Populate AF, Recall@N, QPS, and timing stats onto the output in-place.

    This is the single point where all quality numbers are computed, analogous to
    the summary block printed by the C++ benchmark.cpp.  The function iterates
    over all queries, skipping invalid ones as determined by _is_query_valid.
    For each valid query it computes AF as d_approx divided by d_true (clamped to
    1.0 when d_true is zero) and checks whether id_true appears in the approximate
    neighbour list for the Recall@N contribution.  The accumulated AF and recall
    values are then averaged over valid queries, and QPS plus per-query timing are
    derived from the total wall-clock seconds.

    The t_approximate_total parameter gives wall-clock seconds for the ANN search
    phase across all queries, and t_true_total gives the corresponding time for
    the brute-force ground-truth phase.  Returns the same output object, now
    enriched with average_af, recall_at_n, queries_per_second,
    t_approximate_average, and t_true_average.
    """
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
