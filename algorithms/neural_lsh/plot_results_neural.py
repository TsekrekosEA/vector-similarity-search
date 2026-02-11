#!/usr/bin/env python3
"""
Neural LSH results visualization
Usage: python3 plot_results_neural.py
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
import numpy as np
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def plot_recall_vs_speedup(df, dataset):
    """Recall vs Speedup trade-off analysis"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by build parameters (k, m, e) to see how search parameter T affects the curve
    # We create a label for each unique build configuration
    df['BuildParams'] = df.apply(lambda row: f"k={row['k']}, m={row['m']}", axis=1)
    
    for build_config in df['BuildParams'].unique():
        data = df[df['BuildParams'] == build_config].sort_values('T')
        ax.plot(data['Speedup'], data['Recall@N'], marker='o', label=build_config, linewidth=2, markersize=8, alpha=0.8)
        
        # Annotate T values
        for _, row in data.iterrows():
            ax.annotate(f"T={int(row['T'])}", (row['Speedup'], row['Recall@N']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

    ax.set_xlabel('Speedup vs Brute Force', fontsize=13, fontweight='bold')
    ax.set_ylabel('Recall@N', fontsize=13, fontweight='bold')
    ax.set_title(f'{dataset}: Recall vs Speedup (Neural LSH)\n(Higher & Righter = Better)', 
                fontsize=15, fontweight='bold')
    # ax.set_xscale('log') # Speedup might not span orders of magnitude like in Part 1, but we can check
    ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.3, label='Target: 90%')
    ax.legend(title="Build Config", fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f'output/{dataset}_neural_recall_vs_speedup.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: output/{dataset}_neural_recall_vs_speedup.png")
    plt.close()

def plot_parameter_effects(df, dataset):
    """
    Plots the effect of T (probes) on Recall and Speedup for different m (partitions).
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Panel 1: T vs Recall
    ax1 = axes[0]
    for k_val in sorted(df['k'].unique()):
        for m_val in sorted(df['m'].unique()):
            data = df[(df['k'] == k_val) & (df['m'] == m_val)].sort_values('T')
            if not data.empty:
                ax1.plot(data['T'], data['Recall@N'], marker='o', label=f'k={k_val}, m={m_val}')
    
    ax1.set_xlabel('T (Multi-probe bins)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Recall@N', fontsize=12, fontweight='bold')
    ax1.set_title('Effect of Probes (T) on Recall', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: T vs Speedup
    ax2 = axes[1]
    for k_val in sorted(df['k'].unique()):
        for m_val in sorted(df['m'].unique()):
            data = df[(df['k'] == k_val) & (df['m'] == m_val)].sort_values('T')
            if not data.empty:
                ax2.plot(data['T'], data['Speedup'], marker='s', label=f'k={k_val}, m={m_val}')
    
    ax2.set_xlabel('T (Multi-probe bins)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax2.set_title('Effect of Probes (T) on Speedup', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{dataset}: Neural LSH Parameter Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'output/{dataset}_neural_parameter_effects.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: output/{dataset}_neural_parameter_effects.png")
    plt.close()

def plot_3d_interaction(df, dataset):
    """3D visualization of m, T, and Recall"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by k to see if graph construction affects it
    # Map k to colors
    k_values = sorted(df['k'].unique())
    color_map = {k: i for i, k in enumerate(k_values)}
    colors = df['k'].map(color_map)
    
    scatter = ax.scatter(df['m'], df['T'], df['Recall@N'], 
                        c=colors, s=100, cmap='viridis', 
                        alpha=0.8, edgecolors='black')
    
    ax.set_xlabel('m (Partitions)', fontweight='bold')
    ax.set_ylabel('T (Probes)', fontweight='bold')
    ax.set_zlabel('Recall@N', fontweight='bold')
    ax.set_title(f'{dataset}: m × T vs Recall (Color=k)', fontweight='bold')
    
    # Create a custom legend for k
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    cmap = cm.viridis
    norm = Normalize(vmin=0, vmax=len(k_values)-1)
    patches = [mpatches.Patch(color=cmap(norm(i)), label=f'k={k}') for i, k in enumerate(k_values)]
    ax.legend(handles=patches, title="KNN Graph k")

    plt.savefig(f'output/{dataset}_neural_3d_interaction.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: output/{dataset}_neural_3d_interaction.png")
    plt.close()

def main():
    print("=" * 60)
    print("NEURAL LSH RESULTS VISUALIZATION")
    print("=" * 60)
    
    for dataset in ['mnist', 'sift']:
        csv_file = f'output/experiments_{dataset}.csv'
        if not os.path.exists(csv_file):
            print(f"\n⚠ Skipping {dataset} (no data found at {csv_file})")
            continue
        
        print(f"\nProcessing {dataset}...")
        try:
            df = pd.read_csv(csv_file)
            
            # Convert columns to numeric just in case
            for col in ['k', 'm', 'T', 'Recall@N', 'Speedup']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with missing values
            df = df.dropna(subset=['k', 'm', 'T'])

            plot_recall_vs_speedup(df, dataset.upper())
            plot_parameter_effects(df, dataset.upper())
            plot_3d_interaction(df, dataset.upper())
            
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All visualizations saved to output/")
    print("=" * 60)

if __name__ == "__main__":
    main()
