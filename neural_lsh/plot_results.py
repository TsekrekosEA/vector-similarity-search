#!/usr/bin/env python3
"""Visualise Neural LSH experiment results from CSV files.

Usage:
    python plot_results.py              # reads output/experiments_*.csv
    python plot_results.py path/to.csv  # explicit file
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_recall_vs_speedup(df: pd.DataFrame, dataset: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    df["BuildParams"] = df.apply(lambda r: f"k={r['k']}, m={r['m']}", axis=1)

    for cfg in df["BuildParams"].unique():
        sub = df[df["BuildParams"] == cfg].sort_values("T")
        ax.plot(sub["Speedup"], sub["Recall@N"], marker="o", label=cfg, linewidth=2, markersize=8, alpha=0.8)
        for _, row in sub.iterrows():
            ax.annotate(f"T={int(row['T'])}", (row["Speedup"], row["Recall@N"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8, alpha=0.7)

    ax.set_xlabel("Speedup vs Brute Force", fontsize=13, fontweight="bold")
    ax.set_ylabel("Recall@N", fontsize=13, fontweight="bold")
    ax.set_title(f"{dataset}: Recall vs Speedup (Neural LSH)", fontsize=15, fontweight="bold")
    ax.axhline(y=0.9, color="g", linestyle="--", alpha=0.3, label="Target: 90%")
    ax.legend(title="Build Config", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{dataset}_neural_recall_vs_speedup.png")


def plot_parameter_effects(df: pd.DataFrame, dataset: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, metric, ylabel in zip(axes, ["Recall@N", "Speedup"], ["Recall@N", "Speedup"]):
        for k_val in sorted(df["k"].unique()):
            for m_val in sorted(df["m"].unique()):
                sub = df[(df["k"] == k_val) & (df["m"] == m_val)].sort_values("T")
                if not sub.empty:
                    ax.plot(sub["T"], sub[metric], marker="o", label=f"k={k_val}, m={m_val}")
        ax.set_xlabel("T (Multi-probe bins)", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(f"Effect of T on {ylabel}", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{dataset}: Neural LSH Parameter Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    _save(fig, f"{dataset}_neural_parameter_effects.png")


def plot_3d_interaction(df: pd.DataFrame, dataset: str) -> None:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    k_values = sorted(df["k"].unique())
    colour_map = {k: i for i, k in enumerate(k_values)}
    colours = df["k"].map(colour_map)

    ax.scatter(df["m"], df["T"], df["Recall@N"], c=colours, s=100, cmap="viridis", alpha=0.8, edgecolors="black")
    ax.set_xlabel("m (Partitions)", fontweight="bold")
    ax.set_ylabel("T (Probes)", fontweight="bold")
    ax.set_zlabel("Recall@N", fontweight="bold")
    ax.set_title(f"{dataset}: m x T vs Recall (colour = k)", fontweight="bold")

    import matplotlib.cm as cm
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize

    cmap = cm.viridis
    norm = Normalize(vmin=0, vmax=max(len(k_values) - 1, 1))
    patches = [mpatches.Patch(color=cmap(norm(i)), label=f"k={k}") for i, k in enumerate(k_values)]
    ax.legend(handles=patches, title="KNN Graph k")

    _save(fig, f"{dataset}_neural_3d_interaction.png")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("k", "m", "T", "Recall@N", "Speedup"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["k", "m", "T"])


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("NEURAL LSH RESULTS VISUALISATION")
    print("=" * 60)

    if len(sys.argv) > 1:
        csv_files = [(os.path.basename(p).replace(".csv", "").upper(), p) for p in sys.argv[1:]]
    else:
        csv_files = [
            ("MNIST", "output/experiments_mnist.csv"),
            ("SIFT", "output/experiments_sift.csv"),
        ]

    for label, csv_path in csv_files:
        if not os.path.exists(csv_path):
            print(f"\n  Skipping {label} (no data at {csv_path})")
            continue
        print(f"\nProcessing {label}...")
        try:
            df = _load(csv_path)
            plot_recall_vs_speedup(df, label)
            plot_parameter_effects(df, label)
            plot_3d_interaction(df, label)
        except Exception as exc:
            print(f"  Error: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
