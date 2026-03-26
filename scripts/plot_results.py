#!/usr/bin/env python3
"""
Benchmark results visualization - pandas optimized
Usage: python3 plot_results.py
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
import numpy as np
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

COLORS = {'LSH': '#1f77b4', 'Hypercube': '#ff7f0e', 'IVF-Flat': '#2ca02c', 'IVFFLAT': '#2ca02c', 'IVFPQ': '#d62728', 'IVF-PQ': '#d62728'}
MARKERS = {'LSH': 'o', 'Hypercube': 's', 'IVF-Flat': '^', 'IVFFLAT': '^', 'IVFPQ': 'D', 'IVF-PQ': 'D'}

def plot_recall_vs_speedup(df, dataset):
    """Recall vs Speedup trade-off analysis"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for algo in df['Algorithm'].unique():
        data = df[df['Algorithm'] == algo]
        ax.scatter(data['Speedup'], data['Recall'], label=algo, s=150, alpha=0.7,
                  color=COLORS.get(algo, '#333'), marker=MARKERS.get(algo, 'o'),
                  edgecolors='black', linewidths=1.5)
        
        # Annotate best recall point
        best_idx = data['Recall'].idxmax()
        ax.annotate(f'{algo}\n(best)', (data.loc[best_idx, 'Speedup'], data.loc[best_idx, 'Recall']),
                   xytext=(10, 10), textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS.get(algo, '#333'), alpha=0.3))
    
    ax.set_xlabel('Speedup vs Brute Force (log scale)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Recall@N', fontsize=13, fontweight='bold')
    ax.set_title(f'{dataset}: Recall vs Speedup\n(Higher & Righter = Better)', 
                fontsize=15, fontweight='bold')
    ax.set_xscale('log')
    ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.3, label='Target: 90%')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_recall_vs_speedup.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {dataset}_recall_vs_speedup.png")
    plt.close()

def plot_af_vs_recall(df, dataset):
    """Approximation Factor vs Recall"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for algo in df['Algorithm'].unique():
        data = df[df['Algorithm'] == algo]
        ax.scatter(data['AvgAF'], data['Recall'], label=algo, s=150, alpha=0.7,
                  color=COLORS.get(algo, '#333'), marker=MARKERS.get(algo, 'o'),
                  edgecolors='black', linewidths=1.5)
    
    ax.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='Perfect AF=1.0', alpha=0.6)
    ax.axvline(x=1.1, color='orange', linestyle=':', linewidth=1.5, label='Good AF≤1.1', alpha=0.5)
    ax.set_xlabel('Average Approximation Factor', fontsize=13, fontweight='bold')
    ax.set_ylabel('Recall@N', fontsize=13, fontweight='bold')
    ax.set_title(f'{dataset}: Quality Analysis\n(Left & Higher = Better)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_quality_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {dataset}_quality_analysis.png")
    plt.close()

def plot_algorithm_comparison(df, dataset):
    """6-panel algorithm comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    stats = df.groupby('Algorithm').agg({
        'Recall': ['mean', 'max'],
        'AvgAF': 'mean',
        'Speedup': ['mean', 'max'],
        'TimePerQuery': 'mean'
    })
    
    algos = stats.index.tolist()
    colors = [COLORS.get(a, '#333') for a in algos]
    
    # Average Recall
    axes[0].bar(algos, stats['Recall']['mean'], color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Average Recall@N', fontweight='bold')
    axes[0].set_title('Accuracy: Average Recall', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Best Recall
    axes[1].bar(algos, stats['Recall']['max'], color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Best Recall@N', fontweight='bold')
    axes[1].set_title('Accuracy: Peak Recall', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Average AF
    axes[2].bar(algos, stats['AvgAF']['mean'], color=colors, alpha=0.7, edgecolor='black')
    axes[2].axhline(y=1.0, color='r', linestyle='--', linewidth=2, alpha=0.6)
    axes[2].set_ylabel('Average AF', fontweight='bold')
    axes[2].set_title('Quality: AF (lower=better)', fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].tick_params(axis='x', rotation=45)
    
    # Average Speedup
    axes[3].bar(algos, stats['Speedup']['mean'], color=colors, alpha=0.7, edgecolor='black')
    axes[3].set_ylabel('Average Speedup', fontweight='bold')
    axes[3].set_title('Speed: Average Speedup', fontweight='bold')
    axes[3].grid(True, alpha=0.3, axis='y')
    axes[3].tick_params(axis='x', rotation=45)
    
    # Best Speedup
    axes[4].bar(algos, stats['Speedup']['max'], color=colors, alpha=0.7, edgecolor='black')
    axes[4].set_ylabel('Max Speedup', fontweight='bold')
    axes[4].set_title('Speed: Peak Speedup', fontweight='bold')
    axes[4].grid(True, alpha=0.3, axis='y')
    axes[4].tick_params(axis='x', rotation=45)
    
    # Average Query Time
    axes[5].bar(algos, stats['TimePerQuery']['mean'] * 1000, color=colors, alpha=0.7, edgecolor='black')
    axes[5].set_ylabel('Time (ms)', fontweight='bold')
    axes[5].set_title('Speed: Query Time', fontweight='bold')
    axes[5].grid(True, alpha=0.3, axis='y')
    axes[5].tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'{dataset} Comprehensive Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {dataset}_comprehensive_comparison.png")
    plt.close()

def plot_lsh_parameter_interaction(df, dataset):
    """3D visualization of LSH parameters (w, L, k) vs Recall"""
    lsh_data = df[df['Algorithm'] == 'LSH'].copy()
    if lsh_data.empty:
        print(f"  ⚠ No LSH data for {dataset}")
        return
    
    # Extract parameters from the Parameters string
    def extract_params(param_str):
        params = {'k': None, 'L': None, 'w': None}
        for part in param_str.split():
            if 'k=' in part:
                params['k'] = int(part.split('=')[1])
            elif 'L=' in part:
                params['L'] = int(part.split('=')[1])
            elif 'w=' in part:
                params['w'] = float(part.split('=')[1])
        return params
    
    # Parse parameters
    param_data = lsh_data['Parameters'].apply(extract_params).apply(pd.Series)
    lsh_data = pd.concat([lsh_data, param_data], axis=1)
    lsh_data = lsh_data.dropna(subset=['k', 'L', 'w'])
    
    if lsh_data.empty:
        print(f"  ⚠ Could not parse LSH parameters for {dataset}")
        return
    
    # Create figure with subplots for different views
    fig = plt.figure(figsize=(20, 12))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(lsh_data['k'], lsh_data['L'], lsh_data['w'], 
                         c=lsh_data['Recall'], s=200, cmap='RdYlGn', 
                         alpha=0.7, edgecolors='black', linewidths=1.5)
    ax1.set_xlabel('k (hash functions)', fontweight='bold', fontsize=10)
    ax1.set_ylabel('L (hash tables)', fontweight='bold', fontsize=10)
    ax1.set_zlabel('w (bucket width)', fontweight='bold', fontsize=10)
    ax1.set_title('3D: k × L × w vs Recall', fontweight='bold', fontsize=12)
    plt.colorbar(scatter, ax=ax1, label='Recall', pad=0.1)
    
    # k vs Recall (grouped by L)
    ax2 = fig.add_subplot(2, 3, 2)
    for L_val in sorted(lsh_data['L'].unique()):
        data = lsh_data[lsh_data['L'] == L_val]
        ax2.scatter(data['k'], data['Recall'], s=150, alpha=0.7, 
                   label=f'L={int(L_val)}', edgecolors='black', linewidths=1)
        # Fit line
        if len(data) > 1:
            z = np.polyfit(data['k'], data['Recall'], 1)
            p = np.poly1d(z)
            ax2.plot(data['k'].sort_values(), p(data['k'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax2.set_xlabel('k (hash functions)', fontweight='bold')
    ax2.set_ylabel('Recall', fontweight='bold')
    ax2.set_title('Effect of k on Recall (by L)', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # L vs Recall (grouped by k)
    ax3 = fig.add_subplot(2, 3, 3)
    for k_val in sorted(lsh_data['k'].unique()):
        data = lsh_data[lsh_data['k'] == k_val]
        ax3.scatter(data['L'], data['Recall'], s=150, alpha=0.7,
                   label=f'k={int(k_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['L'], data['Recall'], 1)
            p = np.poly1d(z)
            ax3.plot(data['L'].sort_values(), p(data['L'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax3.set_xlabel('L (hash tables)', fontweight='bold')
    ax3.set_ylabel('Recall', fontweight='bold')
    ax3.set_title('Effect of L on Recall (by k)', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # w vs Recall (colored by k×L product)
    ax4 = fig.add_subplot(2, 3, 4)
    lsh_data['kL_product'] = lsh_data['k'] * lsh_data['L']
    scatter = ax4.scatter(lsh_data['w'], lsh_data['Recall'], 
                         c=lsh_data['kL_product'], s=150, cmap='viridis',
                         alpha=0.7, edgecolors='black', linewidths=1)
    ax4.set_xlabel('w (bucket width)', fontweight='bold')
    ax4.set_ylabel('Recall', fontweight='bold')
    ax4.set_title('Effect of w on Recall (colored by k×L)', fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='k × L')
    ax4.grid(True, alpha=0.3)
    
    # Heatmap: k vs L (average Recall)
    ax5 = fig.add_subplot(2, 3, 5)
    pivot = lsh_data.pivot_table(values='Recall', index='k', columns='L', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax5, 
                cbar_kws={'label': 'Recall'}, linewidths=0.5)
    ax5.set_xlabel('L (hash tables)', fontweight='bold')
    ax5.set_ylabel('k (hash functions)', fontweight='bold')
    ax5.set_title('k × L Heatmap: Average Recall', fontweight='bold')
    
    # Parameter correlation with Recall
    ax6 = fig.add_subplot(2, 3, 6)
    correlations = lsh_data[['k', 'L', 'w', 'Recall']].corr()['Recall'].drop('Recall')
    colors_corr = ['green' if x > 0 else 'red' for x in correlations.values]
    bars = ax6.barh(correlations.index, correlations.values, color=colors_corr, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Correlation with Recall', fontweight='bold')
    ax6.set_title('Parameter Correlation Analysis', fontweight='bold')
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, correlations.values)):
        ax6.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}', 
                va='center', ha='left' if val > 0 else 'right', fontweight='bold')
    
    plt.suptitle(f'{dataset}: LSH Parameter Effects on Recall (k, L, w)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_lsh_parameter_interaction.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {dataset}_lsh_parameter_interaction.png")
    plt.close()

def plot_hypercube_parameter_interaction(df, dataset):
    """Multi-panel visualization of Hypercube parameters (kproj, M, probes) vs Recall"""
    hc_data = df[df['Algorithm'] == 'Hypercube'].copy()
    if hc_data.empty:
        print(f"  ⚠ No Hypercube data for {dataset}")
        return
    
    # Extract parameters from the Parameters string
    def extract_params(param_str):
        params = {'kproj': None, 'M': None, 'probes': None}
        for part in param_str.split():
            if 'kproj=' in part:
                params['kproj'] = int(part.split('=')[1])
            elif 'M=' in part:
                params['M'] = int(part.split('=')[1])
            elif 'probes=' in part:
                params['probes'] = int(part.split('=')[1])
        return params
    
    # Parse parameters
    param_data = hc_data['Parameters'].apply(extract_params).apply(pd.Series)
    hc_data = pd.concat([hc_data, param_data], axis=1)
    hc_data = hc_data.dropna(subset=['kproj', 'M', 'probes'])
    
    if hc_data.empty:
        print(f"  ⚠ Could not parse Hypercube parameters for {dataset}")
        return
    
    # Create figure with 2x3 grid (reduced from 3x3 since no w parameter)
    fig = plt.figure(figsize=(20, 12))
    
    # === Row 1: 3D visualization and 2D effects ===
    
    # 3D: kproj × M × probes (colored by Recall)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(hc_data['kproj'], hc_data['M'], hc_data['probes'], 
                         c=hc_data['Recall'], s=200, cmap='RdYlGn', 
                         alpha=0.7, edgecolors='black', linewidths=1.5)
    ax1.set_xlabel('kproj (dimensions)', fontweight='bold', fontsize=10)
    ax1.set_ylabel('M (max candidates)', fontweight='bold', fontsize=10)
    ax1.set_zlabel('probes (vertices)', fontweight='bold', fontsize=10)
    ax1.set_title('3D: kproj × M × probes vs Recall', fontweight='bold', fontsize=12)
    plt.colorbar(scatter, ax=ax1, label='Recall', pad=0.1, shrink=0.8)
    
    # kproj vs Recall (grouped by probes)
    ax2 = fig.add_subplot(2, 3, 2)
    for probes_val in sorted(hc_data['probes'].unique()):
        data = hc_data[hc_data['probes'] == probes_val]
        ax2.scatter(data['kproj'], data['Recall'], s=150, alpha=0.7, 
                   label=f'probes={int(probes_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['kproj'], data['Recall'], 1)
            p = np.poly1d(z)
            ax2.plot(data['kproj'].sort_values(), p(data['kproj'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax2.set_xlabel('kproj (projection dimensions)', fontweight='bold')
    ax2.set_ylabel('Recall', fontweight='bold')
    ax2.set_title('Effect of kproj on Recall (by probes)', fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # M vs Recall (grouped by probes)
    ax3 = fig.add_subplot(2, 3, 3)
    for probes_val in sorted(hc_data['probes'].unique()):
        data = hc_data[hc_data['probes'] == probes_val]
        ax3.scatter(data['M'], data['Recall'], s=150, alpha=0.7,
                   label=f'probes={int(probes_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['M'], data['Recall'], 1)
            p = np.poly1d(z)
            ax3.plot(data['M'].sort_values(), p(data['M'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax3.set_xlabel('M (max candidates to check)', fontweight='bold')
    ax3.set_ylabel('Recall', fontweight='bold')
    ax3.set_title('Effect of M on Recall (by probes)', fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # === Row 2: Heatmaps and correlations ===
    
    # probes vs Recall (grouped by kproj)
    ax4 = fig.add_subplot(2, 3, 4)
    for kproj_val in sorted(hc_data['kproj'].unique()):
        data = hc_data[hc_data['kproj'] == kproj_val]
        ax4.scatter(data['probes'], data['Recall'], s=150, alpha=0.7,
                   label=f'kproj={int(kproj_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['probes'], data['Recall'], 1)
            p = np.poly1d(z)
            ax4.plot(data['probes'].sort_values(), p(data['probes'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax4.set_xlabel('probes (hypercube vertices to check)', fontweight='bold')
    ax4.set_ylabel('Recall', fontweight='bold')
    ax4.set_title('Effect of probes on Recall (by kproj)', fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    
    # Heatmap: kproj vs probes (average Recall)
    ax5 = fig.add_subplot(2, 3, 5)
    pivot = hc_data.pivot_table(values='Recall', index='kproj', columns='probes', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax5, 
                cbar_kws={'label': 'Recall'}, linewidths=0.5)
    ax5.set_xlabel('probes (vertices)', fontweight='bold')
    ax5.set_ylabel('kproj (dimensions)', fontweight='bold')
    ax5.set_title('kproj × probes Heatmap: Average Recall', fontweight='bold')
    
    # Parameter correlation with Recall
    ax6 = fig.add_subplot(2, 3, 6)
    correlations = hc_data[['kproj', 'M', 'probes', 'Recall']].corr()['Recall'].drop('Recall')
    colors_corr = ['green' if x > 0 else 'red' for x in correlations.values]
    bars = ax6.barh(correlations.index, correlations.values, color=colors_corr, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Correlation with Recall', fontweight='bold')
    ax6.set_title('Parameter Correlation Analysis', fontweight='bold')
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, correlations.values)):
        ax6.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}', 
                va='center', ha='left' if val > 0 else 'right', fontweight='bold')
    
    plt.suptitle(f'{dataset}: Hypercube Parameter Effects on Recall (kproj, M, probes)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_hypercube_parameter_interaction.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {dataset}_hypercube_parameter_interaction.png")
    plt.close()

def plot_lsh_parameter_interaction_speedup(df, dataset):
    """3D visualization of LSH parameters (w, L, k) vs Speedup"""
    lsh_data = df[df['Algorithm'] == 'LSH'].copy()
    if lsh_data.empty:
        print(f"  ⚠ No LSH data for {dataset}")
        return
    
    # Extract parameters from the Parameters string
    def extract_params(param_str):
        params = {'k': None, 'L': None, 'w': None}
        for part in param_str.split():
            if 'k=' in part:
                params['k'] = int(part.split('=')[1])
            elif 'L=' in part:
                params['L'] = int(part.split('=')[1])
            elif 'w=' in part:
                params['w'] = float(part.split('=')[1])
        return params
    
    # Parse parameters
    param_data = lsh_data['Parameters'].apply(extract_params).apply(pd.Series)
    lsh_data = pd.concat([lsh_data, param_data], axis=1)
    lsh_data = lsh_data.dropna(subset=['k', 'L', 'w'])
    
    if lsh_data.empty:
        print(f"  ⚠ Could not parse LSH parameters for {dataset}")
        return
    
    # Create figure with subplots for different views
    fig = plt.figure(figsize=(20, 12))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(lsh_data['k'], lsh_data['L'], lsh_data['w'], 
                         c=lsh_data['Speedup'], s=200, cmap='viridis', 
                         alpha=0.7, edgecolors='black', linewidths=1.5)
    ax1.set_xlabel('k (hash functions)', fontweight='bold', fontsize=10)
    ax1.set_ylabel('L (hash tables)', fontweight='bold', fontsize=10)
    ax1.set_zlabel('w (bucket width)', fontweight='bold', fontsize=10)
    ax1.set_title('3D: k × L × w vs Speedup', fontweight='bold', fontsize=12)
    plt.colorbar(scatter, ax=ax1, label='Speedup', pad=0.1)
    
    # k vs Speedup (grouped by L)
    ax2 = fig.add_subplot(2, 3, 2)
    for L_val in sorted(lsh_data['L'].unique()):
        data = lsh_data[lsh_data['L'] == L_val]
        ax2.scatter(data['k'], data['Speedup'], s=150, alpha=0.7, 
                   label=f'L={int(L_val)}', edgecolors='black', linewidths=1)
        # Fit line
        if len(data) > 1:
            z = np.polyfit(data['k'], data['Speedup'], 1)
            p = np.poly1d(z)
            ax2.plot(data['k'].sort_values(), p(data['k'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax2.set_xlabel('k (hash functions)', fontweight='bold')
    ax2.set_ylabel('Speedup', fontweight='bold')
    ax2.set_title('Effect of k on Speedup (by L)', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # L vs Speedup (grouped by k)
    ax3 = fig.add_subplot(2, 3, 3)
    for k_val in sorted(lsh_data['k'].unique()):
        data = lsh_data[lsh_data['k'] == k_val]
        ax3.scatter(data['L'], data['Speedup'], s=150, alpha=0.7,
                   label=f'k={int(k_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['L'], data['Speedup'], 1)
            p = np.poly1d(z)
            ax3.plot(data['L'].sort_values(), p(data['L'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax3.set_xlabel('L (hash tables)', fontweight='bold')
    ax3.set_ylabel('Speedup', fontweight='bold')
    ax3.set_title('Effect of L on Speedup (by k)', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # w vs Speedup (colored by k×L product)
    ax4 = fig.add_subplot(2, 3, 4)
    lsh_data['kL_product'] = lsh_data['k'] * lsh_data['L']
    scatter = ax4.scatter(lsh_data['w'], lsh_data['Speedup'], 
                         c=lsh_data['kL_product'], s=150, cmap='plasma',
                         alpha=0.7, edgecolors='black', linewidths=1)
    ax4.set_xlabel('w (bucket width)', fontweight='bold')
    ax4.set_ylabel('Speedup', fontweight='bold')
    ax4.set_title('Effect of w on Speedup (colored by k×L)', fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='k × L')
    ax4.grid(True, alpha=0.3)
    
    # Heatmap: k vs L (average Speedup)
    ax5 = fig.add_subplot(2, 3, 5)
    pivot = lsh_data.pivot_table(values='Speedup', index='k', columns='L', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax5, 
                cbar_kws={'label': 'Speedup'}, linewidths=0.5)
    ax5.set_xlabel('L (hash tables)', fontweight='bold')
    ax5.set_ylabel('k (hash functions)', fontweight='bold')
    ax5.set_title('k × L Heatmap: Average Speedup', fontweight='bold')
    
    # Parameter correlation with Speedup
    ax6 = fig.add_subplot(2, 3, 6)
    correlations = lsh_data[['k', 'L', 'w', 'Speedup']].corr()['Speedup'].drop('Speedup')
    colors_corr = ['green' if x > 0 else 'red' for x in correlations.values]
    bars = ax6.barh(correlations.index, correlations.values, color=colors_corr, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Correlation with Speedup', fontweight='bold')
    ax6.set_title('Parameter Correlation Analysis', fontweight='bold')
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, correlations.values)):
        ax6.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}', 
                va='center', ha='left' if val > 0 else 'right', fontweight='bold')
    
    plt.suptitle(f'{dataset}: LSH Parameter Effects on Speedup (k, L, w)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_lsh_parameter_interaction_speedup.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {dataset}_lsh_parameter_interaction_speedup.png")
    plt.close()

def plot_hypercube_parameter_interaction_speedup(df, dataset):
    """Multi-panel visualization of Hypercube parameters (kproj, M, probes) vs Speedup"""
    hc_data = df[df['Algorithm'] == 'Hypercube'].copy()
    if hc_data.empty:
        print(f"  ⚠ No Hypercube data for {dataset}")
        return
    
    # Extract parameters from the Parameters string
    def extract_params(param_str):
        params = {'kproj': None, 'M': None, 'probes': None}
        for part in param_str.split():
            if 'kproj=' in part:
                params['kproj'] = int(part.split('=')[1])
            elif 'M=' in part:
                params['M'] = int(part.split('=')[1])
            elif 'probes=' in part:
                params['probes'] = int(part.split('=')[1])
        return params
    
    # Parse parameters
    param_data = hc_data['Parameters'].apply(extract_params).apply(pd.Series)
    hc_data = pd.concat([hc_data, param_data], axis=1)
    hc_data = hc_data.dropna(subset=['kproj', 'M', 'probes'])
    
    if hc_data.empty:
        print(f"  ⚠ Could not parse Hypercube parameters for {dataset}")
        return
    
    # Create figure with 2x3 grid (reduced from 3x3 since no w parameter)
    fig = plt.figure(figsize=(20, 12))
    
    # === Row 1: 3D visualization and 2D effects ===
    
    # 3D: kproj × M × probes (colored by Speedup)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(hc_data['kproj'], hc_data['M'], hc_data['probes'], 
                         c=hc_data['Speedup'], s=200, cmap='viridis', 
                         alpha=0.7, edgecolors='black', linewidths=1.5)
    ax1.set_xlabel('kproj (dimensions)', fontweight='bold', fontsize=10)
    ax1.set_ylabel('M (max candidates)', fontweight='bold', fontsize=10)
    ax1.set_zlabel('probes (vertices)', fontweight='bold', fontsize=10)
    ax1.set_title('3D: kproj × M × probes vs Speedup', fontweight='bold', fontsize=12)
    plt.colorbar(scatter, ax=ax1, label='Speedup', pad=0.1, shrink=0.8)
    
    # kproj vs Speedup (grouped by probes)
    ax2 = fig.add_subplot(2, 3, 2)
    for probes_val in sorted(hc_data['probes'].unique()):
        data = hc_data[hc_data['probes'] == probes_val]
        ax2.scatter(data['kproj'], data['Speedup'], s=150, alpha=0.7, 
                   label=f'probes={int(probes_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['kproj'], data['Speedup'], 1)
            p = np.poly1d(z)
            ax2.plot(data['kproj'].sort_values(), p(data['kproj'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax2.set_xlabel('kproj (projection dimensions)', fontweight='bold')
    ax2.set_ylabel('Speedup', fontweight='bold')
    ax2.set_title('Effect of kproj on Speedup (by probes)', fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # M vs Speedup (grouped by probes)
    ax3 = fig.add_subplot(2, 3, 3)
    for probes_val in sorted(hc_data['probes'].unique()):
        data = hc_data[hc_data['probes'] == probes_val]
        ax3.scatter(data['M'], data['Speedup'], s=150, alpha=0.7,
                   label=f'probes={int(probes_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['M'], data['Speedup'], 1)
            p = np.poly1d(z)
            ax3.plot(data['M'].sort_values(), p(data['M'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax3.set_xlabel('M (max candidates to check)', fontweight='bold')
    ax3.set_ylabel('Speedup', fontweight='bold')
    ax3.set_title('Effect of M on Speedup (by probes)', fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # === Row 2: Heatmaps and correlations ===
    
    # probes vs Speedup (grouped by kproj)
    ax4 = fig.add_subplot(2, 3, 4)
    for kproj_val in sorted(hc_data['kproj'].unique()):
        data = hc_data[hc_data['kproj'] == kproj_val]
        ax4.scatter(data['probes'], data['Speedup'], s=150, alpha=0.7,
                   label=f'kproj={int(kproj_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['probes'], data['Speedup'], 1)
            p = np.poly1d(z)
            ax4.plot(data['probes'].sort_values(), p(data['probes'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax4.set_xlabel('probes (hypercube vertices to check)', fontweight='bold')
    ax4.set_ylabel('Speedup', fontweight='bold')
    ax4.set_title('Effect of probes on Speedup (by kproj)', fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    
    # Heatmap: kproj vs probes (average Speedup)
    ax5 = fig.add_subplot(2, 3, 5)
    pivot = hc_data.pivot_table(values='Speedup', index='kproj', columns='probes', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax5, 
                cbar_kws={'label': 'Speedup'}, linewidths=0.5)
    ax5.set_xlabel('probes (vertices)', fontweight='bold')
    ax5.set_ylabel('kproj (dimensions)', fontweight='bold')
    ax5.set_title('kproj × probes Heatmap: Average Speedup', fontweight='bold')
    
    # Parameter correlation with Speedup
    ax6 = fig.add_subplot(2, 3, 6)
    correlations = hc_data[['kproj', 'M', 'probes', 'Speedup']].corr()['Speedup'].drop('Speedup')
    colors_corr = ['green' if x > 0 else 'red' for x in correlations.values]
    bars = ax6.barh(correlations.index, correlations.values, color=colors_corr, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Correlation with Speedup', fontweight='bold')
    ax6.set_title('Parameter Correlation Analysis', fontweight='bold')
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, correlations.values)):
        ax6.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}', 
                va='center', ha='left' if val > 0 else 'right', fontweight='bold')
    
    plt.suptitle(f'{dataset}: Hypercube Parameter Effects on Speedup (kproj, M, probes)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_hypercube_parameter_interaction_speedup.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {dataset}_hypercube_parameter_interaction_speedup.png")
    plt.close()

def plot_ivfflat_parameter_interaction(df, dataset):
    """Simplified visualization of IVF-Flat parameters (k_clusters, nprobe) - 2x2 layout"""
    ivf_data = df[df['Algorithm'] == 'IVF-Flat'].copy()
    if ivf_data.empty:
        print(f"  ⚠ No IVF-Flat data for {dataset}")
        return
    
    # Extract parameters from the Parameters string
    def extract_params(param_str):
        params = {'k_clusters': None, 'nprobe': None}
        for part in param_str.split():
            if 'k_clusters=' in part:
                params['k_clusters'] = int(part.split('=')[1])
            elif 'nprobe=' in part:
                params['nprobe'] = int(part.split('=')[1])
        return params
    
    # Parse parameters
    param_data = ivf_data['Parameters'].apply(extract_params).apply(pd.Series)
    ivf_data = pd.concat([ivf_data, param_data], axis=1)
    ivf_data = ivf_data.dropna(subset=['k_clusters', 'nprobe'])
    
    if ivf_data.empty:
        print(f"  ⚠ Could not parse IVF-Flat parameters for {dataset}")
        return
    
    # Create simplified 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # === Panel 1: k_clusters vs Recall (grouped by nprobe) ===
    ax1 = axes[0, 0]
    for nprobe_val in sorted(ivf_data['nprobe'].unique()):
        data = ivf_data[ivf_data['nprobe'] == nprobe_val]
        ax1.scatter(data['k_clusters'], data['Recall'], s=150, alpha=0.7, 
                   label=f'nprobe={int(nprobe_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['k_clusters'], data['Recall'], 1)
            p = np.poly1d(z)
            ax1.plot(data['k_clusters'].sort_values(), p(data['k_clusters'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax1.set_xlabel('k_clusters (number of centroids)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Recall', fontweight='bold', fontsize=12)
    ax1.set_title('Effect of k_clusters on Recall', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # === Panel 2: nprobe vs Recall (grouped by k_clusters) ===
    ax2 = axes[0, 1]
    for k_val in sorted(ivf_data['k_clusters'].unique()):
        data = ivf_data[ivf_data['k_clusters'] == k_val]
        ax2.scatter(data['nprobe'], data['Recall'], s=150, alpha=0.7,
                   label=f'k={int(k_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['nprobe'], data['Recall'], 1)
            p = np.poly1d(z)
            ax2.plot(data['nprobe'].sort_values(), p(data['nprobe'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax2.set_xlabel('nprobe (clusters to search)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Recall', fontweight='bold', fontsize=12)
    ax2.set_title('Effect of nprobe on Recall', fontweight='bold', fontsize=13)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # === Panel 3: Heatmap k_clusters vs nprobe (Recall) ===
    ax3 = axes[1, 0]
    pivot = ivf_data.pivot_table(values='Recall', index='k_clusters', columns='nprobe', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax3, 
                cbar_kws={'label': 'Recall'}, linewidths=0.5, vmin=0, vmax=1)
    ax3.set_xlabel('nprobe (clusters searched)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('k_clusters (centroids)', fontweight='bold', fontsize=12)
    ax3.set_title('Recall Heatmap: k_clusters × nprobe', fontweight='bold', fontsize=13)
    
    # === Panel 4: Heatmap k_clusters vs nprobe (Speedup) ===
    ax4 = axes[1, 1]
    pivot_speedup = ivf_data.pivot_table(values='Speedup', index='k_clusters', columns='nprobe', aggfunc='mean')
    sns.heatmap(pivot_speedup, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax4, 
                cbar_kws={'label': 'Speedup'}, linewidths=0.5)
    ax4.set_xlabel('nprobe (clusters searched)', fontweight='bold', fontsize=12)
    ax4.set_ylabel('k_clusters (centroids)', fontweight='bold', fontsize=12)
    ax4.set_title('Speedup Heatmap: k_clusters × nprobe', fontweight='bold', fontsize=13)
    
    plt.suptitle(f'{dataset}: IVF-Flat Parameter Effects (k_clusters, nprobe)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_ivfflat_parameter_interaction.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {dataset}_ivfflat_parameter_interaction.png")
    plt.close()

def plot_ivfflat_parameter_interaction_speedup(df, dataset):
    """Simplified visualization of IVF-Flat parameters (k_clusters, nprobe) vs Speedup - 2x2 layout"""
    ivf_data = df[df['Algorithm'] == 'IVF-Flat'].copy()
    if ivf_data.empty:
        print(f"  ⚠ No IVF-Flat data for {dataset}")
        return
    
    # Extract parameters from the Parameters string
    def extract_params(param_str):
        params = {'k_clusters': None, 'nprobe': None}
        for part in param_str.split():
            if 'k_clusters=' in part:
                params['k_clusters'] = int(part.split('=')[1])
            elif 'nprobe=' in part:
                params['nprobe'] = int(part.split('=')[1])
        return params
    
    # Parse parameters
    param_data = ivf_data['Parameters'].apply(extract_params).apply(pd.Series)
    ivf_data = pd.concat([ivf_data, param_data], axis=1)
    ivf_data = ivf_data.dropna(subset=['k_clusters', 'nprobe'])
    
    if ivf_data.empty:
        print(f"  ⚠ Could not parse IVF-Flat parameters for {dataset}")
        return
    
    # Create simplified 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # === Panel 1: k_clusters vs Speedup (grouped by nprobe) ===
    ax1 = axes[0, 0]
    for nprobe_val in sorted(ivf_data['nprobe'].unique()):
        data = ivf_data[ivf_data['nprobe'] == nprobe_val]
        ax1.scatter(data['k_clusters'], data['Speedup'], s=150, alpha=0.7, 
                   label=f'nprobe={int(nprobe_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['k_clusters'], data['Speedup'], 1)
            p = np.poly1d(z)
            ax1.plot(data['k_clusters'].sort_values(), p(data['k_clusters'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax1.set_xlabel('k_clusters (number of centroids)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Speedup', fontweight='bold', fontsize=12)
    ax1.set_title('Effect of k_clusters on Speedup', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # === Panel 2: nprobe vs Speedup (grouped by k_clusters) ===
    ax2 = axes[0, 1]
    for k_val in sorted(ivf_data['k_clusters'].unique()):
        data = ivf_data[ivf_data['k_clusters'] == k_val]
        ax2.scatter(data['nprobe'], data['Speedup'], s=150, alpha=0.7,
                   label=f'k={int(k_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['nprobe'], data['Speedup'], 1)
            p = np.poly1d(z)
            ax2.plot(data['nprobe'].sort_values(), p(data['nprobe'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax2.set_xlabel('nprobe (clusters to search)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Speedup', fontweight='bold', fontsize=12)
    ax2.set_title('Effect of nprobe on Speedup', fontweight='bold', fontsize=13)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # === Panel 3: Heatmap k_clusters vs nprobe (Speedup) ===
    ax3 = axes[1, 0]
    pivot_speedup = ivf_data.pivot_table(values='Speedup', index='k_clusters', columns='nprobe', aggfunc='mean')
    sns.heatmap(pivot_speedup, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3, 
                cbar_kws={'label': 'Speedup'}, linewidths=0.5)
    ax3.set_xlabel('nprobe (clusters searched)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('k_clusters (centroids)', fontweight='bold', fontsize=12)
    ax3.set_title('Speedup Heatmap: k_clusters × nprobe', fontweight='bold', fontsize=13)
    
    # === Panel 4: Parameter correlation with Speedup ===
    ax4 = axes[1, 1]
    correlations = ivf_data[['k_clusters', 'nprobe', 'Speedup']].corr()['Speedup'].drop('Speedup')
    colors_corr = ['green' if x > 0 else 'red' for x in correlations.values]
    bars = ax4.barh(correlations.index, correlations.values, color=colors_corr, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_xlabel('Correlation with Speedup', fontweight='bold', fontsize=12)
    ax4.set_title('Parameter Correlation Analysis', fontweight='bold', fontsize=13)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1.2)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, correlations.values)):
        ax4.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}', 
                va='center', ha='left' if val > 0 else 'right', fontweight='bold', fontsize=11)
    
    plt.suptitle(f'{dataset}: IVF-Flat Parameter Effects on Speedup (k_clusters, nprobe)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_ivfflat_parameter_interaction_speedup.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {dataset}_ivfflat_parameter_interaction_speedup.png")
    plt.close()

def plot_ivfpq_parameter_interaction(df, dataset):
    """Multi-panel visualization of IVF-PQ parameters (k, nprobe, m, nbits) vs Recall"""
    ivfpq_data = df[df['Algorithm'].isin(['IVF-PQ', 'IVFPQ'])].copy()
    if ivfpq_data.empty:
        print(f"  ⚠ No IVF-PQ data for {dataset}")
        return
    
    # Extract parameters from the Parameters string
    def extract_params(param_str):
        params = {'k': None, 'nprobe': None, 'm': None, 'nbits': None}
        for part in param_str.split():
            if 'k=' in part:
                params['k'] = int(part.split('=')[1])
            elif 'nprobe=' in part:
                params['nprobe'] = int(part.split('=')[1])
            elif 'm=' in part:
                params['m'] = int(part.split('=')[1])
            elif 'nbits=' in part:
                params['nbits'] = int(part.split('=')[1])
        return params
    
    # Parse parameters
    param_data = ivfpq_data['Parameters'].apply(extract_params).apply(pd.Series)
    ivfpq_data = pd.concat([ivfpq_data, param_data], axis=1)
    ivfpq_data = ivfpq_data.dropna(subset=['k', 'nprobe', 'm'])
    
    if ivfpq_data.empty:
        print(f"  ⚠ Could not parse IVF-PQ parameters for {dataset}")
        return
    
    # Create figure with 3x3 grid
    fig = plt.figure(figsize=(24, 18))
    
    # === Row 1: 3D visualizations ===
    
    # 3D: k × nprobe × m (colored by Recall)
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    scatter = ax1.scatter(ivfpq_data['k'], ivfpq_data['nprobe'], ivfpq_data['m'], 
                         c=ivfpq_data['Recall'], s=200, cmap='RdYlGn', 
                         alpha=0.7, edgecolors='black', linewidths=1.5)
    ax1.set_xlabel('k (coarse clusters)', fontweight='bold', fontsize=10)
    ax1.set_ylabel('nprobe (clusters searched)', fontweight='bold', fontsize=10)
    ax1.set_zlabel('m (subvectors)', fontweight='bold', fontsize=10)
    ax1.set_title('3D: k × nprobe × m vs Recall', fontweight='bold', fontsize=12)
    plt.colorbar(scatter, ax=ax1, label='Recall', pad=0.1, shrink=0.8)
    
    # 3D: k × m × nprobe (colored by Recall, different angle)
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    scatter = ax2.scatter(ivfpq_data['k'], ivfpq_data['m'], ivfpq_data['nprobe'], 
                         c=ivfpq_data['Recall'], s=200, cmap='RdYlGn', 
                         alpha=0.7, edgecolors='black', linewidths=1.5)
    ax2.set_xlabel('k (coarse clusters)', fontweight='bold', fontsize=10)
    ax2.set_ylabel('m (subvectors)', fontweight='bold', fontsize=10)
    ax2.set_zlabel('nprobe (clusters searched)', fontweight='bold', fontsize=10)
    ax2.set_title('3D: k × m × nprobe vs Recall', fontweight='bold', fontsize=12)
    plt.colorbar(scatter, ax=ax2, label='Recall', pad=0.1, shrink=0.8)
    
    # k vs Recall (grouped by nprobe)
    ax3 = fig.add_subplot(3, 3, 3)
    for nprobe_val in sorted(ivfpq_data['nprobe'].unique()):
        data = ivfpq_data[ivfpq_data['nprobe'] == nprobe_val]
        ax3.scatter(data['k'], data['Recall'], s=150, alpha=0.7, 
                   label=f'nprobe={int(nprobe_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['k'], data['Recall'], 1)
            p = np.poly1d(z)
            ax3.plot(data['k'].sort_values(), p(data['k'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax3.set_xlabel('k (coarse clusters)', fontweight='bold')
    ax3.set_ylabel('Recall', fontweight='bold')
    ax3.set_title('Effect of k on Recall (by nprobe)', fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # === Row 2: Parameter effects ===
    
    # nprobe vs Recall (grouped by k)
    ax4 = fig.add_subplot(3, 3, 4)
    for k_val in sorted(ivfpq_data['k'].unique()):
        data = ivfpq_data[ivfpq_data['k'] == k_val]
        ax4.scatter(data['nprobe'], data['Recall'], s=150, alpha=0.7,
                   label=f'k={int(k_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['nprobe'], data['Recall'], 1)
            p = np.poly1d(z)
            ax4.plot(data['nprobe'].sort_values(), p(data['nprobe'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax4.set_xlabel('nprobe (clusters to search)', fontweight='bold')
    ax4.set_ylabel('Recall', fontweight='bold')
    ax4.set_title('Effect of nprobe on Recall (by k)', fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    
    # m vs Recall (grouped by nprobe)
    ax5 = fig.add_subplot(3, 3, 5)
    for nprobe_val in sorted(ivfpq_data['nprobe'].unique()):
        data = ivfpq_data[ivfpq_data['nprobe'] == nprobe_val]
        ax5.scatter(data['m'], data['Recall'], s=150, alpha=0.7,
                   label=f'nprobe={int(nprobe_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1 and len(data['m'].unique()) > 1:
            z = np.polyfit(data['m'], data['Recall'], 1)
            p = np.poly1d(z)
            ax5.plot(data['m'].sort_values(), p(data['m'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax5.set_xlabel('m (number of subvectors)', fontweight='bold')
    ax5.set_ylabel('Recall', fontweight='bold')
    ax5.set_title('Effect of m on Recall (by nprobe)', fontweight='bold')
    ax5.legend(fontsize=9, loc='best')
    ax5.grid(True, alpha=0.3)
    
    # m vs Recall (grouped by k)
    ax6 = fig.add_subplot(3, 3, 6)
    for k_val in sorted(ivfpq_data['k'].unique()):
        data = ivfpq_data[ivfpq_data['k'] == k_val]
        ax6.scatter(data['m'], data['Recall'], s=150, alpha=0.7,
                   label=f'k={int(k_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1 and len(data['m'].unique()) > 1:
            z = np.polyfit(data['m'], data['Recall'], 1)
            p = np.poly1d(z)
            ax6.plot(data['m'].sort_values(), p(data['m'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax6.set_xlabel('m (number of subvectors)', fontweight='bold')
    ax6.set_ylabel('Recall', fontweight='bold')
    ax6.set_title('Effect of m on Recall (by k)', fontweight='bold')
    ax6.legend(fontsize=9, loc='best')
    ax6.grid(True, alpha=0.3)
    
    # === Row 3: Heatmaps and correlations ===
    
    # Heatmap: k vs nprobe (average Recall)
    ax7 = fig.add_subplot(3, 3, 7)
    pivot = ivfpq_data.pivot_table(values='Recall', index='k', columns='nprobe', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax7, 
                cbar_kws={'label': 'Recall'}, linewidths=0.5)
    ax7.set_xlabel('nprobe (clusters searched)', fontweight='bold')
    ax7.set_ylabel('k (coarse clusters)', fontweight='bold')
    ax7.set_title('k × nprobe Heatmap: Recall', fontweight='bold')
    
    # Heatmap: k vs m (average Recall, averaged over nprobe)
    ax8 = fig.add_subplot(3, 3, 8)
    if len(ivfpq_data['m'].unique()) > 1:
        pivot_m = ivfpq_data.pivot_table(values='Recall', index='k', columns='m', aggfunc='mean')
        sns.heatmap(pivot_m, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax8, 
                    cbar_kws={'label': 'Recall'}, linewidths=0.5)
        ax8.set_xlabel('m (subvectors)', fontweight='bold')
        ax8.set_ylabel('k (coarse clusters)', fontweight='bold')
        ax8.set_title('k × m Heatmap: Recall', fontweight='bold')
    else:
        ax8.text(0.5, 0.5, 'Single m value\n(no variation)', 
                ha='center', va='center', transform=ax8.transAxes, fontsize=14)
        ax8.axis('off')
    
    # Parameter correlation with Recall
    ax9 = fig.add_subplot(3, 3, 9)
    correlations = ivfpq_data[['k', 'nprobe', 'm', 'Recall']].corr()['Recall'].drop('Recall')
    colors_corr = ['green' if x > 0 else 'red' for x in correlations.values]
    bars = ax9.barh(correlations.index, correlations.values, color=colors_corr, alpha=0.7, edgecolor='black')
    ax9.set_xlabel('Correlation with Recall', fontweight='bold')
    ax9.set_title('Parameter Correlation Analysis', fontweight='bold')
    ax9.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax9.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, correlations.values)):
        ax9.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}', 
                va='center', ha='left' if val > 0 else 'right', fontweight='bold')
    
    plt.suptitle(f'{dataset}: IVF-PQ Parameter Effects on Recall (k, nprobe, m, nbits)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_ivfpq_parameter_interaction.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {dataset}_ivfpq_parameter_interaction.png")
    plt.close()

def plot_ivfpq_parameter_interaction_speedup(df, dataset):
    """Multi-panel visualization of IVF-PQ parameters (k, nprobe, m, nbits) vs Speedup"""
    ivfpq_data = df[df['Algorithm'].isin(['IVF-PQ', 'IVFPQ'])].copy()
    if ivfpq_data.empty:
        print(f"  ⚠ No IVF-PQ data for {dataset}")
        return
    
    # Extract parameters from the Parameters string
    def extract_params(param_str):
        params = {'k': None, 'nprobe': None, 'm': None, 'nbits': None}
        for part in param_str.split():
            if 'k=' in part:
                params['k'] = int(part.split('=')[1])
            elif 'nprobe=' in part:
                params['nprobe'] = int(part.split('=')[1])
            elif 'm=' in part:
                params['m'] = int(part.split('=')[1])
            elif 'nbits=' in part:
                params['nbits'] = int(part.split('=')[1])
        return params
    
    # Parse parameters
    param_data = ivfpq_data['Parameters'].apply(extract_params).apply(pd.Series)
    ivfpq_data = pd.concat([ivfpq_data, param_data], axis=1)
    ivfpq_data = ivfpq_data.dropna(subset=['k', 'nprobe', 'm'])
    
    if ivfpq_data.empty:
        print(f"  ⚠ Could not parse IVF-PQ parameters for {dataset}")
        return
    
    # Create figure with 3x3 grid
    fig = plt.figure(figsize=(24, 18))
    
    # === Row 1: 3D visualizations ===
    
    # 3D: k × nprobe × m (colored by Speedup)
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    scatter = ax1.scatter(ivfpq_data['k'], ivfpq_data['nprobe'], ivfpq_data['m'], 
                         c=ivfpq_data['Speedup'], s=200, cmap='viridis', 
                         alpha=0.7, edgecolors='black', linewidths=1.5)
    ax1.set_xlabel('k (coarse clusters)', fontweight='bold', fontsize=10)
    ax1.set_ylabel('nprobe (clusters searched)', fontweight='bold', fontsize=10)
    ax1.set_zlabel('m (subvectors)', fontweight='bold', fontsize=10)
    ax1.set_title('3D: k × nprobe × m vs Speedup', fontweight='bold', fontsize=12)
    plt.colorbar(scatter, ax=ax1, label='Speedup', pad=0.1, shrink=0.8)
    
    # 3D: k × m × nprobe (colored by Speedup, different angle)
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    scatter = ax2.scatter(ivfpq_data['k'], ivfpq_data['m'], ivfpq_data['nprobe'], 
                         c=ivfpq_data['Speedup'], s=200, cmap='viridis', 
                         alpha=0.7, edgecolors='black', linewidths=1.5)
    ax2.set_xlabel('k (coarse clusters)', fontweight='bold', fontsize=10)
    ax2.set_ylabel('m (subvectors)', fontweight='bold', fontsize=10)
    ax2.set_zlabel('nprobe (clusters searched)', fontweight='bold', fontsize=10)
    ax2.set_title('3D: k × m × nprobe vs Speedup', fontweight='bold', fontsize=12)
    plt.colorbar(scatter, ax=ax2, label='Speedup', pad=0.1, shrink=0.8)
    
    # k vs Speedup (grouped by nprobe)
    ax3 = fig.add_subplot(3, 3, 3)
    for nprobe_val in sorted(ivfpq_data['nprobe'].unique()):
        data = ivfpq_data[ivfpq_data['nprobe'] == nprobe_val]
        ax3.scatter(data['k'], data['Speedup'], s=150, alpha=0.7, 
                   label=f'nprobe={int(nprobe_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['k'], data['Speedup'], 1)
            p = np.poly1d(z)
            ax3.plot(data['k'].sort_values(), p(data['k'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax3.set_xlabel('k (coarse clusters)', fontweight='bold')
    ax3.set_ylabel('Speedup', fontweight='bold')
    ax3.set_title('Effect of k on Speedup (by nprobe)', fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # === Row 2: Parameter effects ===
    
    # nprobe vs Speedup (grouped by k)
    ax4 = fig.add_subplot(3, 3, 4)
    for k_val in sorted(ivfpq_data['k'].unique()):
        data = ivfpq_data[ivfpq_data['k'] == k_val]
        ax4.scatter(data['nprobe'], data['Speedup'], s=150, alpha=0.7,
                   label=f'k={int(k_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1:
            z = np.polyfit(data['nprobe'], data['Speedup'], 1)
            p = np.poly1d(z)
            ax4.plot(data['nprobe'].sort_values(), p(data['nprobe'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax4.set_xlabel('nprobe (clusters to search)', fontweight='bold')
    ax4.set_ylabel('Speedup', fontweight='bold')
    ax4.set_title('Effect of nprobe on Speedup (by k)', fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    
    # m vs Speedup (grouped by nprobe)
    ax5 = fig.add_subplot(3, 3, 5)
    for nprobe_val in sorted(ivfpq_data['nprobe'].unique()):
        data = ivfpq_data[ivfpq_data['nprobe'] == nprobe_val]
        ax5.scatter(data['m'], data['Speedup'], s=150, alpha=0.7,
                   label=f'nprobe={int(nprobe_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1 and len(data['m'].unique()) > 1:
            z = np.polyfit(data['m'], data['Speedup'], 1)
            p = np.poly1d(z)
            ax5.plot(data['m'].sort_values(), p(data['m'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax5.set_xlabel('m (number of subvectors)', fontweight='bold')
    ax5.set_ylabel('Speedup', fontweight='bold')
    ax5.set_title('Effect of m on Speedup (by nprobe)', fontweight='bold')
    ax5.legend(fontsize=9, loc='best')
    ax5.grid(True, alpha=0.3)
    
    # m vs Speedup (grouped by k)
    ax6 = fig.add_subplot(3, 3, 6)
    for k_val in sorted(ivfpq_data['k'].unique()):
        data = ivfpq_data[ivfpq_data['k'] == k_val]
        ax6.scatter(data['m'], data['Speedup'], s=150, alpha=0.7,
                   label=f'k={int(k_val)}', edgecolors='black', linewidths=1)
        if len(data) > 1 and len(data['m'].unique()) > 1:
            z = np.polyfit(data['m'], data['Speedup'], 1)
            p = np.poly1d(z)
            ax6.plot(data['m'].sort_values(), p(data['m'].sort_values()), 
                    '--', alpha=0.5, linewidth=2)
    ax6.set_xlabel('m (number of subvectors)', fontweight='bold')
    ax6.set_ylabel('Speedup', fontweight='bold')
    ax6.set_title('Effect of m on Speedup (by k)', fontweight='bold')
    ax6.legend(fontsize=9, loc='best')
    ax6.grid(True, alpha=0.3)
    
    # === Row 3: Heatmaps and trade-offs ===
    
    # Heatmap: k vs nprobe (average Speedup)
    ax7 = fig.add_subplot(3, 3, 7)
    pivot = ivfpq_data.pivot_table(values='Speedup', index='k', columns='nprobe', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax7, 
                cbar_kws={'label': 'Speedup'}, linewidths=0.5)
    ax7.set_xlabel('nprobe (clusters searched)', fontweight='bold')
    ax7.set_ylabel('k (coarse clusters)', fontweight='bold')
    ax7.set_title('k × nprobe Heatmap: Speedup', fontweight='bold')
    
    # Recall vs Speedup scatter (colored by k)
    ax8 = fig.add_subplot(3, 3, 8)
    scatter = ax8.scatter(ivfpq_data['Speedup'], ivfpq_data['Recall'], 
                         c=ivfpq_data['k'], s=200, alpha=0.7,
                         cmap='coolwarm', edgecolors='black', linewidths=1.5)
    plt.colorbar(scatter, ax=ax8, label='k (clusters)')
    ax8.set_xlabel('Speedup', fontweight='bold')
    ax8.set_ylabel('Recall', fontweight='bold')
    ax8.set_title('Recall vs Speedup\n(colored by k)', fontweight='bold')
    ax8.set_xscale('log')
    ax8.grid(True, alpha=0.3)
    
    # Parameter correlation with Speedup
    ax9 = fig.add_subplot(3, 3, 9)
    correlations = ivfpq_data[['k', 'nprobe', 'm', 'Speedup']].corr()['Speedup'].drop('Speedup')
    colors_corr = ['green' if x > 0 else 'red' for x in correlations.values]
    bars = ax9.barh(correlations.index, correlations.values, color=colors_corr, alpha=0.7, edgecolor='black')
    ax9.set_xlabel('Correlation with Speedup', fontweight='bold')
    ax9.set_title('Parameter Correlation Analysis', fontweight='bold')
    ax9.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax9.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, correlations.values)):
        ax9.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}', 
                va='center', ha='left' if val > 0 else 'right', fontweight='bold')
    
    plt.suptitle(f'{dataset}: IVF-PQ Parameter Effects on Speedup (k, nprobe, m, nbits)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_ivfpq_parameter_interaction_speedup.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {dataset}_ivfpq_parameter_interaction_speedup.png")
    plt.close()

def main():
    """Main plotting function"""
    print("=" * 60)
    print("BENCHMARK RESULTS VISUALIZATION")
    print("=" * 60)
    
    for dataset in ['MNIST', 'SIFT']:
        csv_file = f'results/benchmark_{dataset}.csv'
        if not os.path.exists(csv_file):
            print(f"\n⚠ Skipping {dataset} (no data)")
            continue
        
        print(f"\nProcessing {dataset}...")
        df = pd.read_csv(csv_file)
        
        plot_recall_vs_speedup(df, dataset)
        plot_af_vs_recall(df, dataset)
        plot_algorithm_comparison(df, dataset)
        plot_lsh_parameter_interaction(df, dataset)
        plot_hypercube_parameter_interaction(df, dataset)
        plot_ivfflat_parameter_interaction(df, dataset)
        plot_ivfpq_parameter_interaction(df, dataset)
        plot_lsh_parameter_interaction_speedup(df, dataset)
        plot_hypercube_parameter_interaction_speedup(df, dataset)
        plot_ivfflat_parameter_interaction_speedup(df, dataset)
        plot_ivfpq_parameter_interaction_speedup(df, dataset)
        
        print(f"✓ {dataset} complete")
    
    print("\n" + "=" * 60)
    print("All visualizations saved to results/")
    print("=" * 60)

if __name__ == "__main__":
    main()
