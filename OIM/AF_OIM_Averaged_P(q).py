# -*- coding: utf-8 -*-
"""
LOCAL Execution: Plot & Analyze Post-Oscillator and Post-MH Averaged Data
Reads the generated .npz file, computes seed-averaged analytics (P(q), RSB, Valley Diversity),
evaluates Weak-Field Mechanisms across all seeds,
and plots the distributions with fixed axes and LaTeX-formatted numeration.
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
plt.rc('font', family='serif', size=14)
plt.rc('mathtext', fontset='cm') 
plt.rc('axes', labelsize=16, labelweight='normal', linewidth=1.5)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14, framealpha=1.0, edgecolor='black')

latex_fmt = FuncFormatter(lambda x, pos: f"${x:g}$")

# ==========================================
# 2. USER CONTROL PANEL
# ==========================================
TARGET_NPZ = "AF_AllToAll_Hybrid_EA_SeedAvg_AveragedData.npz"

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

def calculate_weak_field_metrics(J, bin_spins, changed_mask):
    """
    Evaluates the weak-field cancellation mechanism on exact binary spins.
    Returns the mean metrics strictly for this single disorder realization.
    """
    # h_i = \sum_{j \neq i} J_{ij} s_j  (Matrix diagonal is 0, so direct dot product works)
    h = bin_spins @ J 
    abs_h = np.abs(h)
    
    # L_i = \sum_{j \neq i} |J_{ij}|  (Row incident magnitude)
    L = np.sum(np.abs(J), axis=1)
    L_broadcast = np.broadcast_to(L, abs_h.shape)
    
    # r_i = |h_i| / L_i  (Cancellation ratio)
    r = abs_h / (L_broadcast + 1e-12)
    
    flipped_h = abs_h[changed_mask]
    unflipped_h = abs_h[~changed_mask]
    
    flipped_L = L_broadcast[changed_mask]
    unflipped_L = L_broadcast[~changed_mask]
    
    flipped_r = r[changed_mask]
    unflipped_r = r[~changed_mask]
    
    return {
        "mean_abs_h_flipped": np.mean(flipped_h) if len(flipped_h) > 0 else np.nan,
        "mean_abs_h_unflipped": np.mean(unflipped_h) if len(unflipped_h) > 0 else np.nan,
        "mean_L_flipped": np.mean(flipped_L) if len(flipped_L) > 0 else np.nan,
        "mean_L_unflipped": np.mean(unflipped_L) if len(unflipped_L) > 0 else np.nan,
        "mean_r_flipped": np.mean(flipped_r) if len(flipped_r) > 0 else np.nan,
        "mean_r_unflipped": np.mean(unflipped_r) if len(unflipped_r) > 0 else np.nan
    }

def print_weak_field_analysis(seed_metrics, label=""):
    print("\n" + "="*60)
    if label:
        print(f"WEAK-FIELD CANCELLATION DIAGNOSTIC: {label}")
    else:
        print("WEAK-FIELD CANCELLATION DIAGNOSTIC")
        
    def get_stats(key):
        vals = [m[key] for m in seed_metrics if not np.isnan(m[key])]
        if not vals: return np.nan, np.nan
        return np.mean(vals), np.std(vals)
    
    mh_f, std_h_f = get_stats("mean_abs_h_flipped")
    mh_u, std_h_u = get_stats("mean_abs_h_unflipped")
    
    mL_f, std_L_f = get_stats("mean_L_flipped")
    mL_u, std_L_u = get_stats("mean_L_unflipped")
    
    mr_f, std_r_f = get_stats("mean_r_flipped")
    mr_u, std_r_u = get_stats("mean_r_unflipped")
    
    print("Disorder-Averaged Statistics (Mean ± Std Dev across 30 seeds):")
    print(f"  |h_i| (Absolute Local Field):")
    print(f"      Corrected spins: {mh_f:.4f} ± {std_h_f:.4f}")
    print(f"      Unchanged spins: {mh_u:.4f} ± {std_h_u:.4f}")
    print(f"  L_i (Total Incident Magnitude / Row Norm):")
    print(f"      Corrected spins: {mL_f:.4f} ± {std_L_f:.4f}")
    print(f"      Unchanged spins: {mL_u:.4f} ± {std_L_u:.4f}")
    print(f"  r_i (|h_i| / L_i, Cancellation Ratio):")
    print(f"      Corrected spins: {mr_f:.4f} ± {std_r_f:.4f}")
    print(f"      Unchanged spins: {mr_u:.4f} ± {std_r_u:.4f}")
    print("="*60)

# ==========================================
# 4. HELPER FUNCTIONS (PLOTTING)
# ==========================================
def plot_energy_progression(h_e, h_t, h_ks, pts, filename):
    print(f"Generating {filename}...")
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    
    for i in range(len(pts)):
        start = pts[i]
        end = pts[i+1] if i + 1 < len(pts) else len(h_e)
        label = f"Epoch {i+1}" if i < len(pts) - 1 else f"Epoch {len(pts)} (MH)"
        ax1.plot(np.arange(start, end) * 500, h_e[start:end], label=label, linewidth=2)
        
    if len(pts) > 2:
        for pt in pts[1:-1]:
            ax1.axvline(x=pt * 500, color='gray', linestyle='--', linewidth=1.5, zorder=2)
            
        ax1.axvline(x=pts[-1] * 500, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='MH Takeover')
        
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x/1e4:g}$"))
    ax1.yaxis.set_major_formatter(latex_fmt)
    ax1.set_xlabel(r"Simulation Steps ($\times 10^4$)", fontsize=20, fontweight='normal')
    ax1.set_ylabel(r"Energy $E$", color='black', fontsize=20, fontweight='normal')
    ax1.tick_params(axis='both', labelcolor='black', labelsize=18)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Noise & SHIL Amplitude", color='black', fontsize=20, fontweight='normal')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=18)
    ax2.yaxis.set_major_formatter(latex_fmt)
    
    h_kn = np.sqrt(2 * np.array(h_t))
    ax2.plot(np.arange(len(h_kn)) * 500, h_kn, color='navy', linestyle='--', alpha=0.8, linewidth=2, label='Noise $K_n$')
    
    if len(h_ks) > 0 and len(h_ks) == len(h_t):
        ax2.plot(np.arange(len(h_ks)) * 500, h_ks, color='forestgreen', linestyle=':', alpha=0.8, linewidth=2, label='SHIL Amp $K_s$')
        
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, 
               loc='upper right', ncol=2, framealpha=1.0, edgecolor='black', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

def plot_comparative_pq_distribution(osc_overlaps, mh_overlaps, filename):
    print(f"Generating {filename}...")
    # Standardized figsize to match Energy & Field distributions
    plt.figure(figsize=(8, 5)) 
    ax = plt.gca()
    
    ax.hist(osc_overlaps, bins=50, range=(-1, 1), color='#ffaa66', edgecolor='black', 
            alpha=0.85, density=True, label='OIM+')
            
    ax.hist(mh_overlaps, bins=50, range=(-1, 1), color='#66b3ff', edgecolor='black', 
            alpha=0.75, density=True, label='DH-OIM+')
            
    ax.xaxis.set_major_formatter(latex_fmt)
    ax.yaxis.set_major_formatter(latex_fmt)
    ax.set_xlabel(r"Overlap $q$", fontsize=20, fontweight='normal')
    ax.set_ylabel("Probability Density", fontsize=20, fontweight='normal')
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)
    # Standardized legend fontsize & edgecolor
    ax.legend(loc='upper right', fontsize=16, framealpha=1.0, edgecolor='black')
    
    ax.set_ylim(0, 2.5)
    
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

def plot_comparative_energy_distribution(osc_energies, mh_energies, filename):
    print(f"Generating {filename}...")
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    
    bins = np.arange(-112, -95 + 0.1, 0.2)
        
    ax.hist(osc_energies, bins=bins, color='#ffaa66', edgecolor='black', 
            alpha=0.85, density=True, label='OIM+')
            
    ax.hist(mh_energies, bins=bins, color='#66b3ff', edgecolor='black', 
            alpha=0.75, density=True, label='DH-OIM+')
            
    mean_osc = np.mean(osc_energies)
    mean_mh = np.mean(mh_energies)
    min_e = np.min(mh_energies)
    
    ax.axvline(mean_osc, color='#cc5500', linestyle='dashed', linewidth=2.5, 
               label=f'OIM+ Mean: ${mean_osc:.1f}$')
    ax.axvline(mean_mh, color='#0044aa', linestyle='dashed', linewidth=2.5, 
               label=f'DH-OIM+ Mean: ${mean_mh:.1f}$')
    ax.axvline(min_e, color='red', linestyle='solid', linewidth=2.5, 
               label=f'Min: ${min_e:.1f}$')
               
    ax.xaxis.set_major_formatter(latex_fmt)
    ax.yaxis.set_major_formatter(latex_fmt)
    ax.set_xlabel(r"Energy $E$", fontsize=20, fontweight='normal')
    ax.set_ylabel("Probability Density", fontsize=20, fontweight='normal')
    ax.tick_params(axis='both', labelsize=18)
    
    ax.legend(fontsize=16, framealpha=1.0, edgecolor='black')
    ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)
    
    ax.set_xlim(-112, -95)
    ax.set_ylim(0, 0.4)
    
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

def plot_flipped_fields(flipped_fields, all_osc_fields, filename):
    print(f"Generating {filename}...")
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    f_max = max(all_osc_fields.max(), flipped_fields.max() if len(flipped_fields) > 0 else 1.0)
    bins = np.linspace(0, f_max, 60)
    
    ax.hist(all_osc_fields, bins=bins, color='#ffaa66', alpha=0.85, edgecolor='black', density=True, label='OIM+')
    if len(flipped_fields) > 0:
        ax.hist(flipped_fields, bins=bins, color='#882255', alpha=0.8, edgecolor='black', density=True, label='Spins corrected by diagnostic MH')
        
    ax.xaxis.set_major_formatter(latex_fmt)
    ax.yaxis.set_major_formatter(latex_fmt)
    ax.set_xlabel(r"Absolute Ising local field $|h_i|$", fontsize=20, fontweight='normal')
    ax.set_ylabel("Probability Density", fontsize=20, fontweight='normal')
    ax.tick_params(axis='both', labelsize=18)
    # Standardized legend fontsize & edgecolor
    ax.legend(fontsize=16, framealpha=1.0, edgecolor='black')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()
    

def plot_comparative_fields(osc_fields_all, mh_fields_all, filename):
    print(f"Generating {filename}...")
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    
    f_max_all = max(osc_fields_all.max(), mh_fields_all.max())
    bins_all = np.linspace(0, f_max_all, 60)
    
    ax.hist(osc_fields_all, bins=bins_all, color='#ffaa66', edgecolor='black', alpha=0.85, density=True, label='OIM+')
    ax.hist(mh_fields_all, bins=bins_all, color='#66b3ff', edgecolor='black', alpha=0.75, density=True, label='DH-OIM+')
            
    ax.xaxis.set_major_formatter(latex_fmt)
    ax.yaxis.set_major_formatter(latex_fmt)
    ax.set_xlabel(r"Absolute Ising local field $|h_i|$", fontsize=20, fontweight='normal')
    ax.set_ylabel("Probability Density", fontsize=20, fontweight='normal')
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(axis='y', linestyle=':', alpha=0.7, zorder=0)
    # Standardized legend fontsize & edgecolor
    ax.legend(loc='upper right', fontsize=16, framealpha=1.0, edgecolor='black')
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

# ==========================================
# 5. EXECUTION LOGIC
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("LOCAL EXECUTION: P(q), Energy, and Local Field Comparative Analysis")
    print("="*60)
    
    if not os.path.exists(TARGET_NPZ):
        print(f"[ERROR] The file '{TARGET_NPZ}' was not found in the current directory.")
        sys.exit(1)
        
    print(f"Loading data from {TARGET_NPZ}...\n")
    start_time = time.time()
    
    # Load the compressed data
    loader = np.load(TARGET_NPZ)
    
    # Extract arrays
    master_J = loader['master_J']
    master_osc_spins = loader['master_osc_spins']
    master_final_spins = loader['master_final_spins']
    master_osc_energies = loader['master_osc_energies']
    master_energies = loader['master_energies']
    
    # Track History
    avg_h_e = loader['avg_h_e']
    h_t = loader['h_t']
    pts = loader['pts']
    
    num_seeds, num_reps, N = master_osc_spins.shape
    
    # --------------------------------------------------------
    # Reconstruct h_ks from original MPI epochs
    # --------------------------------------------------------
    epochs_cfg = [
        {"anneal": int(15e4), "settle": int(0.0), "Ks_s": 0.0, "Ks_f": 0.4},
        {"anneal": int(15e4), "settle": int(0.0), "Ks_s": 0.4, "Ks_f": 0.6},
        {"anneal": int(25e4), "settle": int(0.0), "Ks_s": 0.6, "Ks_f": 0.8},
        {"anneal": int(25e4), "settle": int(0.0), "Ks_s": 0.8, "Ks_f": 0.8}
    ]
    sample_rate = 500
    h_ks = []
    for cfg in epochs_cfg:
        total_steps = cfg["anneal"] + cfg["settle"]
        for step in range(0, total_steps, sample_rate):
            p_total = step / (total_steps - 1) if total_steps > 1 else 1.0
            Ks = cfg["Ks_s"] * (1 - p_total) + cfg["Ks_f"] * p_total
            h_ks.append(Ks)
            
    mh_len = len(avg_h_e) - len(h_ks)
    if mh_len > 0:
        h_ks.extend(np.full(mh_len, epochs_cfg[-1]["Ks_f"]))
    h_ks = np.array(h_ks)
    
    master_osc_overlaps = []
    master_mh_overlaps = []
    
    osc_valley_accum = []
    mh_valley_accum = []
    osc_coph_accum = []
    mh_coph_accum = []
    
    all_flipped_fields = []
    all_osc_fields_flat = []
    all_mh_fields_flat = []
    total_z2_flips = []
    
    wf_accum = []
    
    # Process Metrics Per Seed
    for s_idx in range(num_seeds):
        J = master_J[s_idx]
        osc_spins = master_osc_spins[s_idx]
        mh_spins = master_final_spins[s_idx]
        
        # 1. Per-Seed Overlaps
        ovl_osc = [np.dot(osc_spins[i], osc_spins[j]) / N for i in range(num_reps) for j in range(i+1, num_reps)]
        ovl_mh = [np.dot(mh_spins[i], mh_spins[j]) / N for i in range(num_reps) for j in range(i+1, num_reps)]
        master_osc_overlaps.extend(ovl_osc)
        master_mh_overlaps.extend(ovl_mh)
        
        # 2. Per-Seed Landscape/Dendrogram Metrics (Using Z_2 Quotiented distance)
        d_vec_osc_H = pdist(osc_spins, metric='hamming')
        d_vec_osc = np.minimum(d_vec_osc_H, 1.0 - d_vec_osc_H)
        Z_osc = linkage(d_vec_osc, method='average')
        c_osc, _ = cophenet(Z_osc, d_vec_osc)
        osc_coph_accum.append(c_osc)
        osc_valley_accum.append(calculate_valley_diversity(osc_spins))
        
        d_vec_mh_H = pdist(mh_spins, metric='hamming')
        d_vec_mh = np.minimum(d_vec_mh_H, 1.0 - d_vec_mh_H)
        Z_mh = linkage(d_vec_mh, method='average')
        c_mh, _ = cophenet(Z_mh, d_vec_mh)
        mh_coph_accum.append(c_mh)
        mh_valley_accum.append(calculate_valley_diversity(mh_spins))

        # 3. Local Fields (Explicitly calculated using binarized spins as |h_i| = |\sum_j J_{ij}s_j|)
        bin_osc_spins = np.where(osc_spins >= 0, 1.0, -1.0)
        bin_mh_spins = np.where(mh_spins >= 0, 1.0, -1.0)
        
        osc_fields = np.abs(bin_osc_spins @ J)
        mh_fields = np.abs(bin_mh_spins @ J)
        all_osc_fields_flat.extend(osc_fields.flatten())
        all_mh_fields_flat.extend(mh_fields.flatten())
        
        # 4. Z2 Corrected Flips & Balance
        raw_flips = np.sum(bin_osc_spins != bin_mh_spins, axis=1)
        inverted_replicas = raw_flips > (N / 2)
        true_flips = np.minimum(raw_flips, N - raw_flips)
        total_z2_flips.extend(true_flips)
        
        aligned_mh_spins = bin_mh_spins.copy()
        aligned_mh_spins[inverted_replicas] *= -1
        changed_mask_matrix = (bin_osc_spins != aligned_mh_spins)
        
        # Calculate Weak-Field / Cancellation Diagnostic purely on binarized states
        seed_metrics = calculate_weak_field_metrics(J, bin_osc_spins, changed_mask_matrix)
        wf_accum.append(seed_metrics)
        
        all_flipped_fields.extend(osc_fields[changed_mask_matrix])
        
    # Finalize Aggregations
    all_flipped_fields = np.array(all_flipped_fields)
    all_osc_fields_flat = np.array(all_osc_fields_flat)
    all_mh_fields_flat = np.array(all_mh_fields_flat)

    # --------------------------------------------------------
    # PRINT ANALYTICS PANELS
    # --------------------------------------------------------
    
    # POST-OSCILLATOR
    print("="*60)
    print("STATE ENERGY STATISTICS (Post-Oscillator)")
    print(f"  - Mean Energy: {np.mean(master_osc_energies):.4f}")
    print(f"  - Min Energy:  {np.min(master_osc_energies):.4f}")
    print("="*60)
    
    osc_clutter, osc_ent = calculate_pq_metrics_global(master_osc_overlaps)
    print("\nP(q) OVERLAP DISTRIBUTION QUALITY (Post-Oscillator)")
    print(f"  - Clutter Fraction (|q| <= 0.9): {osc_clutter:.4f} ({osc_clutter*100:.1f}%)")
    print(f"  - Shannon Entropy:               {osc_ent:.4f} bits")
    print("="*60)
    
    print("\n" + "="*60)
    print("ULTRAMETRICITY & RSB QUALITY (Post-Oscillator, Z_2-Quotiented)")
    avg_coph_osc = np.mean(osc_coph_accum)
    print(f"  - Averaged Cophenetic Correlation: {avg_coph_osc:.4f}")
    print("="*60)
    
    avg_osc_valleys = get_averaged_valley_metrics(osc_valley_accum)
    print_valley_diversity(avg_osc_valleys, "Post-Oscillator")

    # METROPOLIS-HASTINGS WORKLOAD
    mean_flips = np.mean(total_z2_flips)
    max_flips = np.max(total_z2_flips)
    percent_changed = (mean_flips / N) * 100
    
    print("\n" + "="*60)
    print("METROPOLIS-HASTINGS WORKLOAD ANALYSIS (Z_2 Corrected)")
    print(f"Average structural spins flipped by MH: {mean_flips:.2f} out of {N} ({percent_changed:.2f}%)")
    print(f"Max structural spins flipped in any replica: {max_flips}")
    print("="*60)

    # WEAK-ROW VS CANCELLATION DIAGNOSTIC
    print_weak_field_analysis(wf_accum, label="OIM+ vs Corrected Spins")

    # POST-MH REFINEMENT
    print("\n" + "="*60)
    print("STATE ENERGY STATISTICS (Post-MH Refinement)")
    print(f"  - Mean Energy: {np.mean(master_energies):.4f}")
    print(f"  - Min Energy:  {np.min(master_energies):.4f}")
    print("="*60)
    
    mh_clutter, mh_ent = calculate_pq_metrics_global(master_mh_overlaps)
    print("\nP(q) OVERLAP DISTRIBUTION QUALITY (Post-MH Refinement)")
    print(f"  - Clutter Fraction (|q| <= 0.9): {mh_clutter:.4f} ({mh_clutter*100:.1f}%)")
    print(f"  - Shannon Entropy:               {mh_ent:.4f} bits")
    print("="*60)
    
    print("\n" + "="*60)
    print("ULTRAMETRICITY & RSB QUALITY (Post-MH Refinement, Z_2-Quotiented)")
    avg_coph_mh = np.mean(mh_coph_accum)
    print(f"  - Averaged Cophenetic Correlation: {avg_coph_mh:.4f}")
    print("="*60)
    
    avg_mh_valleys = get_averaged_valley_metrics(mh_valley_accum)
    print_valley_diversity(avg_mh_valleys, "Post-MH Refinement")

    # --------------------------------------------------------
    # PLOTTING
    # --------------------------------------------------------
    print("\nPlotting Distributions...")
    
    plot_energy_progression(avg_h_e, h_t, h_ks, pts, "Avg_Plot_1_EnergyProg.pdf")
    
    plot_comparative_pq_distribution(master_osc_overlaps, master_mh_overlaps, "Avg_Plot_2_Comparative_Overlap.pdf")
    plot_comparative_energy_distribution(master_osc_energies, master_energies, "Avg_Plot_3_Comparative_Energy.pdf")
    
    plot_flipped_fields(all_flipped_fields, all_osc_fields_flat, "Avg_Plot_4_Flipped_Fields.pdf")
    plot_comparative_fields(all_osc_fields_flat, all_mh_fields_flat, "Avg_Plot_5_Comparative_Fields.pdf")
    
    print("="*60)
    print("CAPTION DATA: EXACT ENERGY MINIMA")
    print("="*60)
    print(f"OIM+ Pooled Minimum Energy:    {np.min(master_osc_energies):.4f}")
    print(f"DH-OIM+ Pooled Minimum Energy: {np.min(master_energies):.4f}")
    print("="*60)
    
    print(f"Analysis & Plotting Complete in {time.time() - start_time:.2f} seconds.")
    print("All plots saved successfully.")
    print("="*60)