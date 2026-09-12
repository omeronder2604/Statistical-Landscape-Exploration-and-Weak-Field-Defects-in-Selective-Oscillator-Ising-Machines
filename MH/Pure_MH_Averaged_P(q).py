# -*- coding: utf-8 -*-
"""
LOCAL Execution: Plot & Analyze Pure MH Averaged P(q), Energy, Local Fields, and History
Reads the generated pure MH .npz file, computes seed-averaged analytics,
and plots the distributions with fixed axes and LaTeX-formatted numeration, without graph titles.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import sys
import time
from scipy.cluster.hierarchy import linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import entropy

# ==========================================
# 1. STANDARDIZED FONT & PLOT CONFIGURATION
# ==========================================
# Switch the main text to serif (classic academic look)
plt.rc('font', family='serif', size=14)

# Force the math engine to use the classic LaTeX font (Computer Modern)
plt.rc('mathtext', fontset='cm') 

plt.rc('axes', labelsize=16, labelweight='normal', linewidth=1.5)
plt.rc('xtick', labelsize=14, direction='in', top=True)    # Ticks point inward
plt.rc('ytick', labelsize=14, direction='in', right=True)  # Ticks point inward
plt.rc('legend', fontsize=16, framealpha=1.0, edgecolor='black')

# Common LaTeX formatter for tick numbers
latex_fmt = FuncFormatter(lambda x, pos: f"${x:g}$")

# ==========================================
# 2. USER CONTROL PANEL
# ==========================================
# Ensure this matches the exact filename output by your TRUBA Pure MH run
TARGET_NPZ = "AllToAll_PureMH_MPI_SeedAvg_AveragedData.npz"
SAMPLE_RATE = 500
STEPS_ANNEAL = int(10e4) # Based on the standard Pure MH script

# ==========================================
# 3. HELPER FUNCTIONS (ANALYTICS)
# ==========================================
def calculate_pq_metrics_global(overlaps):
    hist, bin_edges = np.histogram(overlaps, bins=50, range=(-1, 1))
    p_i = hist / np.sum(hist)
    p_i = p_i[p_i > 0] 
    
    clutter_fraction = np.sum([hist[i] for i in range(len(hist)) if -0.9 <= bin_edges[i] <= 0.9]) / len(overlaps)
    shannon_entropy = entropy(p_i, base=2)
    
    return clutter_fraction, shannon_entropy

def calculate_valley_diversity(spins, dynamic_threshold=0.30):
    num_reps, N = spins.shape
    
    # Calculate standard Hamming distance
    dist_vec_H = pdist(spins, metric='hamming')
    # Quotient out the Z_2 symmetry: d_Z2 = min(d_H, 1 - d_H)
    dist_vec = np.minimum(dist_vec_H, 1.0 - dist_vec_H)
    
    Z = linkage(dist_vec, method='average')
    
    merge_heights = Z[:, 2]
    if len(merge_heights) > 1:
        gaps = np.diff(merge_heights)
        largest_gap_idx = np.argmax(gaps)
        auto_threshold = merge_heights[largest_gap_idx] + (gaps[largest_gap_idx] / 2.0)
    else:
        auto_threshold = dynamic_threshold
        
    max_hamming = dist_vec.max()
    mean_hamming = dist_vec.mean()
    cross_valley_fraction = np.mean(dist_vec > auto_threshold)
    
    labels = fcluster(Z, t=auto_threshold, criterion='distance')
    num_clusters = len(np.unique(labels))
    final_merge_height = Z[-1, 2] if len(Z) > 0 else 0.0
    
    intra_dists, inter_dists = [], []
    dist_matrix = np.zeros((num_reps, num_reps))
    idx = 0
    for i in range(num_reps):
        for j in range(i+1, num_reps):
            dist_matrix[i,j] = dist_matrix[j,i] = dist_vec[idx]
            idx += 1
            
    for i in range(num_reps):
        for j in range(i+1, num_reps):
            if labels[i] == labels[j]:
                intra_dists.append(dist_matrix[i,j])
            else:
                inter_dists.append(dist_matrix[i,j])
                
    intra_mean = np.mean(intra_dists) if intra_dists else 0.0
    inter_mean = np.mean(inter_dists) if inter_dists else 0.0
    separation_ratio = inter_mean / intra_mean if intra_mean > 0 else np.nan

    return {
        "max_hamming": max_hamming,
        "mean_hamming": mean_hamming,
        "cross_valley_fraction": cross_valley_fraction,
        "num_distinct_clusters": num_clusters,
        "final_merge_height": final_merge_height,
        "intra_cluster_mean": intra_mean,
        "inter_cluster_mean": inter_mean,
        "separation_ratio": separation_ratio
    }

def get_averaged_valley_metrics(metrics_list):
    avg_metrics = {}
    keys = ["max_hamming", "mean_hamming", "cross_valley_fraction", 
            "num_distinct_clusters", "final_merge_height", 
            "intra_cluster_mean", "inter_cluster_mean", "separation_ratio"]
    for k in keys:
        valid_vals = [m[k] for m in metrics_list if not np.isnan(m[k])]
        avg_metrics[k] = np.mean(valid_vals) if valid_vals else np.nan
    return avg_metrics

def print_valley_diversity(metrics, label=""):
    print("\n" + "="*60)
    if label:
        print(f"SEED-AVERAGED VALLEY DIVERSITY METRICS (Z_2-Quotiented): {label}")
    else:
        print("SEED-AVERAGED VALLEY DIVERSITY METRICS (Z_2-Quotiented)")
    print(" * Note: Clustering performed modulo global spin inversion.")
    print(f"  Max pairwise Z_2 dist:            {metrics['max_hamming']:.4f}")
    print(f"  Mean pairwise Z_2 dist:           {metrics['mean_hamming']:.4f}")
    print(f"  Cross-valley replica pairs:       {metrics['cross_valley_fraction']*100:.1f}%")
    print(f"  Distinct valleys (clusters):      {metrics['num_distinct_clusters']:.1f} (Average)")
    print(f"  Final dendrogram merge height:    {metrics['final_merge_height']:.4f}")
    print(f"  Mean intra-cluster Z_2 dist:      {metrics['intra_cluster_mean']:.4f}")
    print(f"  Mean inter-cluster Z_2 dist:      {metrics['inter_cluster_mean']:.4f}")
    print(f"  Inter/Intra separation ratio:     {metrics['separation_ratio']:.4f}")
    print("="*60)

# ==========================================
# 4. HELPER FUNCTIONS (PLOTTING)
# ==========================================
def plot_energy_progression(h_e, h_t, filename):
    print(f"Generating {filename}...")
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    
    step_axis = np.arange(len(h_e)) * SAMPLE_RATE
    
    # Plot Energy
    ax1.plot(step_axis, h_e, color='crimson', linewidth=2, label='System Energy', zorder=2)
        
    ax1.set_xlim(0, step_axis[-1])
    
    # Custom formatter: Appends '\times 10^4' strictly to the very last tick on the axis
    def custom_x_formatter(x, pos):
        val = x / 1e4
        locs = ax1.xaxis.get_majorticklocs()
        xlim = ax1.get_xlim()
        visible_locs = [loc for loc in locs if xlim[0] <= loc <= xlim[1]]
        
        if len(visible_locs) > 0 and np.isclose(x, visible_locs[-1]):
            return f"${val:g} \\times 10^4$"
        return f"${val:g}$"

    ax1.xaxis.set_major_formatter(FuncFormatter(custom_x_formatter))
    ax1.yaxis.set_major_formatter(latex_fmt)
    ax1.set_xlabel(r"Simulation Steps ($\times 10^4$)", fontsize=20, fontweight='normal')
    ax1.set_ylabel(r"Energy $E$", color='black', fontsize=20, fontweight='normal')
    ax1.tick_params(axis='both', labelcolor='black', labelsize=18)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot Temperature
    ax2 = ax1.twinx()
    ax2.set_ylabel("Temperature $T$", color='black', fontsize=20, fontweight='normal')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=18)
    ax2.yaxis.set_major_formatter(latex_fmt)
    
    ax2.plot(step_axis, h_t, color='navy', linestyle='--', linewidth=2.5, label='Temperature $T$', alpha=0.8, zorder=1)
    
    # Anneal End Marker
    ax1.axvline(x=STEPS_ANNEAL, color='black', linestyle=':', linewidth=2.5, label='Anneal End', zorder=0)
        
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    
    # Opaque Legend in upper right
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, 
               loc='upper right', ncol=1, framealpha=1.0, edgecolor='black', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

def plot_pq_distribution(overlaps, filename):
    print(f"Generating {filename}...")
    # Locked size to 8x5 to match OIM histograms exactly
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    
    bins = np.linspace(-1.0, 1.0, 51)
    
    # Using Probability Density
    ax.hist(overlaps, bins=bins, color='darkorchid', edgecolor='black', alpha=0.7, density=True)
    
    ax.xaxis.set_major_formatter(latex_fmt)
    ax.yaxis.set_major_formatter(latex_fmt)
    
    ax.set_xlabel(r"Overlap $q$", fontsize=20, fontweight='normal')
    ax.set_ylabel("Probability Density", fontsize=20, fontweight='normal')
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    ax.set_ylim(0, 2.5)
    
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

def plot_energy_distribution(energies, filename):
    print(f"Generating {filename}...")
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    
    bins = np.arange(-112, -95 + 0.1, 0.2)
    
    # Using Probability Density
    ax.hist(energies, bins=bins, color='darkorange', edgecolor='black', alpha=0.8, density=True)
        
    mean_e = np.mean(energies)
    min_e = np.min(energies)
    
    ax.axvline(mean_e, color='black', linestyle='dashed', linewidth=2.5, label=f'Mean: ${mean_e:.1f}$')
    ax.axvline(min_e, color='red', linestyle='solid', linewidth=2.5, label=f'Min: ${min_e:.1f}$')
    
    ax.xaxis.set_major_formatter(latex_fmt)
    ax.yaxis.set_major_formatter(latex_fmt)
    
    ax.set_xlabel(r"Energy $E$", fontsize=20, fontweight='normal')
    ax.set_ylabel("Probability Density", fontsize=20, fontweight='normal')
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=16, framealpha=1.0, edgecolor='black')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Locked x-limit and y-limit identically to the Hybrid plot for visual parity
    ax.set_xlim(-110, -95)
    ax.set_ylim(0, 0.4)
    
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()
    
def plot_local_fields(fields, filename):
    print(f"Generating {filename}...")
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    
    f_max = fields.max()
    bins = np.linspace(0, f_max, 60)
    
    # Using Probability Density
    ax.hist(fields, bins=bins, color='#66b3ff', edgecolor='black', alpha=0.75, density=True, label='Pure MH Baseline')
    
    ax.xaxis.set_major_formatter(latex_fmt)
    ax.yaxis.set_major_formatter(latex_fmt)
    ax.set_xlabel(r"Absolute Ising local field $|h_i|$", fontsize=20, fontweight='normal')
    ax.set_ylabel("Probability Density", fontsize=20, fontweight='normal')
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(axis='y', linestyle=':', alpha=0.7, zorder=0)
    
    ax.legend(loc='upper right', fontsize=16, framealpha=1.0)
    
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

# ==========================================
# 5. EXECUTION LOGIC
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("LOCAL EXECUTION: Pure MH P(q), Energy & Field Analysis")
    print("="*60)
    
    if not os.path.exists(TARGET_NPZ):
        print(f"[ERROR] The file '{TARGET_NPZ}' was not found in the current directory.")
        sys.exit(1)
        
    print(f"Loading data from {TARGET_NPZ}...\n")
    start_time = time.time()
    
    # Load the compressed data
    loader = np.load(TARGET_NPZ)
    
    # Extract spins, energies, matrices, and history
    master_final_spins = loader['master_final_spins']
    master_energies = loader['master_energies']
    master_J = loader['master_J']
    avg_h_e = loader['avg_h_e']
    h_t = loader['h_t']
    
    num_seeds, num_reps, N = master_final_spins.shape
    print(f"Data Extracted: {num_seeds} Seeds, {num_reps} Replicas per seed, N={N} Spins.")
    
    print("Calculating intra-seed pairwise overlaps and local fields...")
    master_mh_overlaps = []
    master_mh_fields = []
    
    mh_valley_accum = []
    mh_coph_accum = []
    
    # Calculate metrics independently for each seed, then pool them
    for s_idx in range(num_seeds):
        mh_spins = master_final_spins[s_idx]
        J = master_J[s_idx]
        
        # 1. Overlaps
        ovl_mh = [np.dot(mh_spins[i], mh_spins[j]) / N for i in range(num_reps) for j in range(i+1, num_reps)]
        master_mh_overlaps.extend(ovl_mh)
        
        # 2. Per-Seed Landscape/Dendrogram Metrics (Using Z_2 Quotiented distance)
        d_vec_mh_H = pdist(mh_spins, metric='hamming')
        d_vec_mh = np.minimum(d_vec_mh_H, 1.0 - d_vec_mh_H)
        Z_mh = linkage(d_vec_mh, method='average')
        c_mh, _ = cophenet(Z_mh, d_vec_mh)
        mh_coph_accum.append(c_mh)
        mh_valley_accum.append(calculate_valley_diversity(mh_spins))
        
        # 3. Local Fields (|h_i|) using exact binarized states
        bin_mh_spins = np.where(mh_spins >= 0, 1.0, -1.0)
        fields = np.abs(bin_mh_spins @ J).flatten()
        master_mh_fields.extend(fields)
        
    master_mh_fields = np.array(master_mh_fields)
    
    print(f"Total overlap pairs calculated: {len(master_mh_overlaps)}")
    print(f"Total local fields calculated: {len(master_mh_fields)}")
    
    # --------------------------------------------------------
    # PRINT ANALYTICS PANELS
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("STATE ENERGY STATISTICS (Pure MH)")
    print(f"  - Mean Energy: {np.mean(master_energies):.4f}")
    print(f"  - Min Energy:  {np.min(master_energies):.4f}")
    print("="*60)
    
    mh_clutter, mh_ent = calculate_pq_metrics_global(master_mh_overlaps)
    print("\nP(q) OVERLAP DISTRIBUTION QUALITY (Pure MH)")
    print(f"  - Clutter Fraction (|q| <= 0.9): {mh_clutter:.4f} ({mh_clutter*100:.1f}%)")
    print(f"  - Shannon Entropy:               {mh_ent:.4f} bits")
    print("="*60)
    
    print("\n" + "="*60)
    print("ULTRAMETRICITY & RSB QUALITY (Pure MH, Z_2-Quotiented)")
    avg_coph_mh = np.mean(mh_coph_accum)
    print(f"  - Averaged Cophenetic Correlation: {avg_coph_mh:.4f}")
    print("="*60)
    
    avg_mh_valleys = get_averaged_valley_metrics(mh_valley_accum)
    print_valley_diversity(avg_mh_valleys, "Pure MH Baseline")

    # --------------------------------------------------------
    # PLOTTING
    # --------------------------------------------------------
    print("\nPlotting Distributions...")
    plot_energy_progression(avg_h_e, h_t, "Avg_Plot_1_PureMH_EnergyProg.pdf")
    plot_pq_distribution(master_mh_overlaps, "Avg_Plot_PureMH_Overlap.pdf")
    plot_energy_distribution(master_energies, "Avg_Plot_PureMH_EnergyDist.pdf")
    
    plot_local_fields(master_mh_fields, "Avg_Plot_PureMH_LocalFields.pdf")
    
    print("="*60)
    print("CAPTION DATA: EXACT ENERGY MINIMA")
    print("="*60)
    print(f"Pure MH Pooled Minimum Energy: {np.min(master_energies):.4f}")
    print("="*60)
    
    print(f"Analysis & Plotting Complete in {time.time() - start_time:.2f} seconds.")
    print("All 4 plots saved successfully.")
    print("="*60)