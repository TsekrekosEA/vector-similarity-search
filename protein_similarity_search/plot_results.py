#!/usr/bin/env python3
"""Protein similarity search result visualisation.

This script reads the structured text report produced by protein_search.py and
generates four publication-quality figures that summarise the performance and
biological relevance of each ANN method compared to BLAST.

Plot Descriptions:
    recall_vs_qps       Recall@N plotted against queries per second for every method.
                        Points further to the upper-right represent better
                        speed-accuracy trade-offs.  BLAST, the ground-truth
                        reference, is drawn as a dashed horizontal line at
                        recall 1.0.

    time_comparison      Horizontal bar chart of total wall-clock time per method,
                        making it easy to compare the end-to-end cost of each
                        algorithm including index construction overhead.

    recall_comparison    Grouped bar chart of Recall@N for every method across all
                        query proteins, showing both per-query bars and the mean
                        line so variance is visible at a glance.

    blast_overlap        Stacked bar chart per method showing the fraction of
                        returned neighbours that also appear in the BLAST top-N
                        versus those that do not, providing a biological-level
                        sanity check beyond pure distance recall.

Usage:
    python plot_results.py results.txt              Reads results.txt, writes PNGs
    python plot_results.py results.txt -o figures/  Writes PNGs to figures/
"""

import argparse
import os
import re
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MethodSummary:
    """Aggregated metrics for a single ANN method."""
    name: str
    total_time: float = 0.0
    qps: float = 0.0
    recall: float = 0.0
    per_query_recall: List[float] = field(default_factory=list)


@dataclass
class NeighborInfo:
    """Parsed row from the per-query neighbour table."""
    rank: int
    protein_id: str
    l2_dist: float
    blast_identity: Optional[float]
    in_blast_top_n: bool


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_results(filepath: str) -> tuple[List[MethodSummary], Optional[MethodSummary], Dict[str, Dict[str, List[NeighborInfo]]]]:
    """Parse the structured text report into method summaries and neighbour tables.

    Returns a tuple of (ann_methods, blast_method, neighbors_by_method_and_query).
    The neighbours dict is keyed by method name, then query protein id, yielding
    the list of NeighborInfo entries for that combination.
    """
    methods: Dict[str, MethodSummary] = {}
    blast: Optional[MethodSummary] = None
    neighbors: Dict[str, Dict[str, List[NeighborInfo]]] = {}
    current_query_id = ""
    current_method_name = ""
    in_neighbor_table = False
    in_summary_table = False

    with open(filepath, "r") as f:
        for raw_line in f:
            line = raw_line.rstrip()

            # Detect query header
            m = re.match(r"^Query Protein:\s+(\S+)", line)
            if m:
                current_query_id = m.group(1)
                in_neighbor_table = False
                in_summary_table = False
                continue

            # Detect summary table
            if line.startswith("[1] Method Comparison Summary"):
                in_summary_table = True
                in_neighbor_table = False
                continue

            if line.startswith("[2]"):
                in_summary_table = False
                continue

            # Parse summary rows
            if in_summary_table and "|" in line and not line.startswith("-"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4 and parts[0] != "Method":
                    name = parts[0]
                    try:
                        time_val = float(parts[1])
                        qps_val = float(parts[2])
                    except ValueError:
                        continue
                    recall_str = parts[3].split("(")[0].strip()
                    try:
                        recall_val = float(recall_str)
                    except ValueError:
                        recall_val = 1.0

                    if "BLAST" in name:
                        if blast is None:
                            blast = MethodSummary(name="BLAST", total_time=time_val, qps=qps_val, recall=1.0)
                        blast.per_query_recall.append(1.0)
                    else:
                        if name not in methods:
                            methods[name] = MethodSummary(name=name)
                        methods[name].qps = qps_val
                        methods[name].per_query_recall.append(recall_val)
                continue

            # Detect neighbour table for a method
            m = re.match(r"^Method:\s+(.+)$", line)
            if m:
                current_method_name = m.group(1).strip()
                in_neighbor_table = True
                if current_method_name not in neighbors:
                    neighbors[current_method_name] = {}
                if current_query_id not in neighbors[current_method_name]:
                    neighbors[current_method_name][current_query_id] = []
                continue

            # Parse neighbour rows
            if in_neighbor_table and "|" in line and not line.startswith("-"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 6 and parts[0] != "Rank":
                    try:
                        rank = int(parts[0])
                        pid = parts[1]
                        dist = float(parts[2])
                        identity_str = parts[3].replace("%", "").strip()
                        identity = float(identity_str) if identity_str != "N/A" else None
                        in_blast = parts[4].strip().lower() == "yes"
                        ni = NeighborInfo(rank=rank, protein_id=pid, l2_dist=dist,
                                          blast_identity=identity, in_blast_top_n=in_blast)
                        neighbors[current_method_name][current_query_id].append(ni)
                    except (ValueError, IndexError):
                        pass
                continue

            # Detect overall summary
            if "OVERALL SUMMARY" in line:
                in_summary_table = False
                in_neighbor_table = False
                continue

    # Compute average recall from per-query values
    for ms in methods.values():
        if ms.per_query_recall:
            ms.recall = float(np.mean(ms.per_query_recall))

    # Read total time and QPS from OVERALL SUMMARY if present
    _parse_overall_summary(filepath, methods, blast)

    return list(methods.values()), blast, neighbors


def _parse_overall_summary(filepath: str, methods: Dict[str, MethodSummary],
                           blast: Optional[MethodSummary]) -> None:
    """Second pass to capture total time and average QPS from the OVERALL SUMMARY block."""
    in_overall = False
    with open(filepath, "r") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            if "OVERALL SUMMARY" in line:
                in_overall = True
                continue
            if in_overall and "|" in line and not line.startswith("-"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4 and parts[0] not in ("Method", ""):
                    name = parts[0]
                    try:
                        total_t = float(parts[1])
                        avg_qps = float(parts[2])
                    except ValueError:
                        continue
                    if "BLAST" in name:
                        if blast is not None:
                            blast.total_time = total_t
                            blast.qps = avg_qps
                    elif name in methods:
                        methods[name].total_time = total_t
                        methods[name].qps = avg_qps


# ── Plotting ──────────────────────────────────────────────────────────────────

COLOURS = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974", "#64b5cd"]


def plot_recall_vs_qps(methods: List[MethodSummary], blast: Optional[MethodSummary],
                       out_dir: str) -> None:
    """Scatter: Recall@N vs Queries Per Second."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, m in enumerate(methods):
        ax.scatter(m.qps, m.recall, s=120, color=COLOURS[i % len(COLOURS)],
                   zorder=3, label=m.name, edgecolors="white", linewidths=0.8)

    if blast:
        ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=1, label="BLAST (Recall=1.0)")
        ax.scatter(blast.qps, 1.0, s=120, marker="D", color="grey",
                   zorder=3, edgecolors="white", linewidths=0.8)

    ax.set_xlabel("Queries Per Second (QPS)", fontsize=11)
    ax.set_ylabel("Recall@N vs BLAST", fontsize=11)
    ax.set_title("Recall vs Throughput", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "recall_vs_qps.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_time_comparison(methods: List[MethodSummary], blast: Optional[MethodSummary],
                         out_dir: str) -> None:
    """Horizontal bar chart of total wall-clock time."""
    all_methods = list(methods)
    if blast:
        all_methods.append(blast)

    names = [m.name for m in all_methods]
    times = [m.total_time for m in all_methods]
    colours = [COLOURS[i % len(COLOURS)] if m != blast else "grey" for i, m in enumerate(all_methods)]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.6)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, times, color=colours, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Total Time (seconds)", fontsize=11)
    ax.set_title("Wall-Clock Time Comparison", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "time_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_recall_comparison(methods: List[MethodSummary], out_dir: str) -> None:
    """Bar chart of Recall@N per method, with per-query dots for variance."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = [m.name for m in methods]
    recalls = [m.recall for m in methods]
    x = np.arange(len(names))

    bars = ax.bar(x, recalls, color=[COLOURS[i % len(COLOURS)] for i in range(len(names))],
                  edgecolor="white", linewidth=0.8, width=0.6)

    # Overlay per-query points if there are multiple queries
    for i, m in enumerate(methods):
        if len(m.per_query_recall) > 1:
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(m.per_query_recall))
            ax.scatter(x[i] + jitter, m.per_query_recall, color="black",
                       alpha=0.5, s=20, zorder=4)

    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=1, label="BLAST (Recall=1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Recall@N", fontsize=11)
    ax.set_title("Recall@N Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "recall_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_blast_overlap(methods: List[MethodSummary],
                       neighbors: Dict[str, Dict[str, List[NeighborInfo]]],
                       out_dir: str) -> None:
    """Stacked bar chart: fraction of neighbours also in BLAST top-N."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = []
    in_blast_fracs = []
    not_in_blast_fracs = []

    for m in methods:
        method_neighbors = neighbors.get(m.name, {})
        total = 0
        in_blast = 0
        for query_nbrs in method_neighbors.values():
            for nb in query_nbrs:
                total += 1
                if nb.in_blast_top_n:
                    in_blast += 1
        if total > 0:
            names.append(m.name)
            in_blast_fracs.append(in_blast / total)
            not_in_blast_fracs.append(1.0 - in_blast / total)

    if not names:
        logger.warning("No neighbour data available for BLAST overlap chart")
        return

    x = np.arange(len(names))
    width = 0.6

    ax.bar(x, in_blast_fracs, width, label="In BLAST Top-N", color="#55a868", edgecolor="white")
    ax.bar(x, not_in_blast_fracs, width, bottom=in_blast_fracs,
           label="Not in BLAST Top-N", color="#c44e52", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Fraction of Returned Neighbours", fontsize=11)
    ax.set_title("BLAST Top-N Overlap per Method", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "blast_overlap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Plot protein search results")
    parser.add_argument("results_file", help="Path to results.txt from protein_search.py")
    parser.add_argument("-o", "--output-dir", default=".", help="Directory for output PNGs")
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        logger.error("Results file not found: %s", args.results_file)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Parsing %s ...", args.results_file)
    methods, blast, neighbors = parse_results(args.results_file)

    if not methods:
        logger.error("No method results found in %s", args.results_file)
        sys.exit(1)

    logger.info("Found %d ANN methods: %s", len(methods), ", ".join(m.name for m in methods))

    plot_recall_vs_qps(methods, blast, args.output_dir)
    plot_time_comparison(methods, blast, args.output_dir)
    plot_recall_comparison(methods, args.output_dir)
    plot_blast_overlap(methods, neighbors, args.output_dir)

    logger.info("All plots saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
