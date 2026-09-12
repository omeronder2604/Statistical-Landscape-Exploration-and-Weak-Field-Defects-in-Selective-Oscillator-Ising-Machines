# -*- coding: utf-8 -*-
"""
TRUBA Execution: All-to-All Gaussian Spin Glass 
Hybrid Oscillator + MH Refinement using 1D MPI Replica Distribution.
(Updated with C-Wrapper ThreadPoolExecutor for On-the-Fly Generation)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import time
import sys
from numba import njit
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import entropy

# ==========================================
# 1. STANDARDIZED FONT & PLOT CONFIGURATION
# ==========================================
plt.rc('font', family='sans-serif', size=14)
plt.rc('axes', labelsize=16, labelweight='bold', linewidth=1.5)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14, framealpha=0.95, edgecolor='black')

# ==========================================
# 2. USER CONTROL PANEL
# ==========================================
GLOBAL_SEED = 52 #52
N = 200
TOTAL_REPLICAS = 402  
FILE_BASE = "AllToAll_Hybrid_EA"
sample_rate = 500

# Central control for the mean of the structural disorder matrix
J0_VAL = 0.0 

# MH Tuning Parameters
MH_STEPS = int(1e4)
T_MH_S = 0.2
T_MH_E = 0.1

# ==========================================
# 3. CENTRALIZED RNG & STATE GENERATION
# ==========================================
def setup_master_rng(global_seed, num_epochs):
    """Initializes the master seed sequence and branches it for all stochastic needs."""
    master_seq = np.random.SeedSequence(global_seed)
    seq_J, seq_init, seq_epochs_branch, seq_mh_branch = master_seq.spawn(4)
    epoch_sequences = seq_epochs_branch.spawn(num_epochs)
    return seq_J, seq_init, epoch_sequences, seq_mh_branch

def generate_j_matrix(seq_J, N, J0=0.0, J_std=1.0):
    """Generates the exact structural disorder matrix."""
    gen_J = np.random.default_rng(seq_J)
    J = gen_J.normal(J0/N, J_std/np.sqrt(N), (N, N))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    return J

def generate_initial_phases(seq_init, total_replicas, N, start_idx, local_reps):
    """Generates phase states globally to maintain topology independence, then slices locally."""
    gen_init = np.random.default_rng(seq_init)
    all_phis = gen_init.uniform(0, 2 * np.pi, (total_replicas, N))
    return all_phis[start_idx : start_idx + local_reps].copy()

def get_local_generators(seq_branch, total_replicas, start_idx, local_reps):
    """Spawns unique PCG64 C-engines mapped exactly to the active local replicas."""
    replica_seqs = seq_branch.spawn(total_replicas)
    local_seqs = replica_seqs[start_idx : start_idx + local_reps]
    return [np.random.default_rng(seq) for seq in local_seqs]

# ==========================================
# 4. NUMBA DYNAMICS (Oscillator & MH)
# ==========================================
# By using nogil=True, Numba releases the Python Lock. This lets ThreadPoolExecutor
# run these C-compiled functions on multiple cores simultaneously in parallel.
@njit(nogil=True, fastmath=True)
def run_osc_epoch(J, N, S_A, S_S, Kn_s, Kn_m, Ks_s, Ks_f, K_c, dt, gen, initial_phi):
    phi = initial_phi.copy()
    sqrt_dt, total_steps = np.sqrt(dt), S_A + S_S
    h_size = int(total_steps // sample_rate)
    e_hist, t_hist, ks_hist = np.zeros(h_size), np.zeros(h_size), np.zeros(h_size)
    
    # Pre-allocate tiny array to hold single-step noise and stay in L1 cache
    noise = np.zeros(N)
    
    for step in range(total_steps):
        p_total = step / (total_steps - 1) if total_steps > 1 else 1.0
        Ks = Ks_s * (1 - p_total) + Ks_f * p_total
        
        if step < S_A:
            p = step / S_A
            Kn = Kn_s * (1 - p) + Kn_m * p
        else:
            Kn = Kn_m

        sa, ca = np.sin(phi), np.cos(phi)
        
        # Pull directly from C-engine wrapper on the fly
        for i in range(N):
            noise[i] = gen.normal(0.0, 1.0)
            
        phi += (0.5 * (ca*(J@sa) - sa*(J@ca)) - Ks*np.sin(2*phi))*dt + Kn*sqrt_dt*noise

        if step % sample_rate == 0:
            idx = step // sample_rate
            if idx < h_size:
                sp = np.sign(np.cos(phi))
                for i in range(N):
                    if sp[i] == 0: sp[i] = 1.0
                e_hist[idx] = -0.5 * np.dot(sp, J @ sp)
                t_hist[idx], ks_hist[idx] = (Kn**2)/2, Ks
                
    return phi, e_hist, t_hist, ks_hist

@njit(nogil=True, fastmath=True)
def run_mh_epoch(J, N, steps, T_start, T_end, start_spins, gen):
    spins = start_spins.copy()
    h_size = int(steps // sample_rate)
    mh_e, mh_t = np.zeros(h_size), np.zeros(h_size)
    
    for s in range(steps):
        p = s / (steps - 1) if steps > 1 else 1.0
        T = T_start * (1 - p) + T_end * p
        
        # Generate random values on the fly without heavy memory allocation
        i = int(gen.random() * N)
        if i == N: i = N - 1 # Safety bounds check
        
        delta_E = 2.0 * spins[i] * np.dot(J[i], spins)
        
        if delta_E <= 0 or gen.random() < np.exp(-delta_E / T):
            spins[i] *= -1.0
            
        if s % sample_rate == 0:
            idx = s // sample_rate
            if idx < h_size:
                mh_e[idx] = -0.5 * np.dot(spins, J @ spins)
                mh_t[idx] = T
    return spins, mh_e, mh_t

def parallel_osc_stage_threads(J, N, num_replicas, S_A, S_S, Kn_s, Kn_m, Ks_s, Ks_f, epoch_dt, local_gens, phis):
    new_phis = np.zeros((num_replicas, N))
    h_len = int((S_A + S_S) // sample_rate)
    e_b, t_b, ks_b = np.zeros((num_replicas, h_len)), np.zeros((num_replicas, h_len)), np.zeros((num_replicas, h_len))
    
    def task(r):
        p, eh, th, ksh = run_osc_epoch(J, N, S_A, S_S, Kn_s, Kn_m, Ks_s, Ks_f, 0.5, epoch_dt, local_gens[r], phis[r])
        new_phis[r], e_b[r], t_b[r], ks_b[r] = p, eh, th, ksh

    with ThreadPoolExecutor() as executor:
        executor.map(task, range(num_replicas))
        
    return new_phis, e_b, t_b, ks_b

def parallel_mh_stage_threads(J, N, num_replicas, steps, T_s, T_e, start_phis, local_gens):
    final_spins = np.zeros((num_replicas, N))
    h_len = int(steps // sample_rate)
    e_b, t_b = np.zeros((num_replicas, h_len)), np.zeros((num_replicas, h_len))
    
    def task(r):
        start_spins = np.sign(np.cos(start_phis[r]))
        for i in range(N):
            if start_spins[i] == 0: start_spins[i] = 1.0
        fs, mhe, mht = run_mh_epoch(J, N, steps, T_s, T_e, start_spins, local_gens[r])
        final_spins[r], e_b[r], t_b[r] = fs, mhe, mht

    with ThreadPoolExecutor() as executor:
        executor.map(task, range(num_replicas))
        
    return final_spins, e_b, t_b

# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================
def calculate_pq_metrics(spins):
    num_reps = spins.shape[0]
    N = spins.shape[1]
    
    ovl = [np.dot(spins[i], spins[j])/N for i in range(num_reps) for j in range(i+1, num_reps)]
    hist, bin_edges = np.histogram(ovl, bins=50, range=(-1, 1))
    
    p_i = hist / np.sum(hist)
    p_i = p_i[p_i > 0] 
    
    clutter_fraction = np.sum([hist[i] for i in range(len(hist)) if -0.9 <= bin_edges[i] <= 0.9]) / len(ovl)
    shannon_entropy = entropy(p_i, base=2)
    
    return clutter_fraction, shannon_entropy

def calculate_valley_diversity(spins, hamming_threshold=0.30):
    num_reps, N = spins.shape
    dist_vec = pdist(spins, metric='hamming')
    Z = linkage(dist_vec, method='average')
    
    max_hamming = dist_vec.max()
    mean_hamming = dist_vec.mean()
    cross_valley_fraction = np.mean(dist_vec > hamming_threshold)
    labels = fcluster(Z, t=hamming_threshold, criterion='distance')
    num_clusters = len(np.unique(labels))
    final_merge_height = Z[-1, 2]
    
    intra_dists = []
    inter_dists = []
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
        "max_hamming":             max_hamming,
        "mean_hamming":            mean_hamming,
        "cross_valley_fraction":  cross_valley_fraction,
        "num_distinct_clusters":  num_clusters,
        "final_merge_height":     final_merge_height,
        "intra_cluster_mean":     intra_mean,
        "inter_cluster_mean":     inter_mean,
        "separation_ratio":       separation_ratio,
    }

def print_valley_diversity(metrics, label=""):
    print("\n" + "="*60)
    if label:
        print(f"VALLEY DIVERSITY METRICS: {label}")
    else:
        print("VALLEY DIVERSITY METRICS")
    print(f"  Max pairwise Hamming dist:        {metrics['max_hamming']:.4f}")
    print(f"  Mean pairwise Hamming dist:       {metrics['mean_hamming']:.4f}")
    print(f"  Cross-valley replica pairs:       {metrics['cross_valley_fraction']*100:.1f}%")
    print(f"  Distinct valleys (clusters):      {metrics['num_distinct_clusters']}")
    print(f"  Final dendrogram merge height:    {metrics['final_merge_height']:.4f}")
    print(f"  Mean intra-cluster Hamming:       {metrics['intra_cluster_mean']:.4f}")
    print(f"  Mean inter-cluster Hamming:       {metrics['inter_cluster_mean']:.4f}")
    print(f"  Inter/Intra separation ratio:     {metrics['separation_ratio']:.4f}")
    print("="*60)

def calculate_field_balance(J, osc_phis, changed_mask, balance_quantile=0.25, eps=1e-12):
    cos_phis = np.cos(osc_phis)
    osc_field = cos_phis @ J.T
    abs_osc_field = np.abs(osc_field)
    contribution_l1 = np.abs(cos_phis[:, None, :] * J[None, :, :]).sum(axis=2)
    row_j_l1 = np.sum(np.abs(J), axis=1)
    row_j_l2 = np.sqrt(np.sum(J * J, axis=1))
    row_j_l1 = np.broadcast_to(row_j_l1, osc_field.shape)
    row_j_l2 = np.broadcast_to(row_j_l2, osc_field.shape)
    balance_ratio = abs_osc_field / (contribution_l1 + eps)
    balance_threshold = float(np.quantile(balance_ratio, balance_quantile))

    flipped_balance_ratio = balance_ratio[changed_mask]
    unflipped_balance_ratio = balance_ratio[~changed_mask]

    flipped_balanced_fraction = (
        float(np.mean(flipped_balance_ratio < balance_threshold))
        if flipped_balance_ratio.size > 0 else 0.0
    )
    unflipped_balanced_fraction = (
        float(np.mean(unflipped_balance_ratio < balance_threshold))
        if unflipped_balance_ratio.size > 0 else 0.0
    )
    balanced_enrichment = (
        flipped_balanced_fraction / unflipped_balanced_fraction
        if unflipped_balanced_fraction > 0 else np.inf
    )

    return {
        "osc_field": osc_field,
        "abs_osc_field": abs_osc_field,
        "contribution_l1": contribution_l1,
        "row_j_l1": row_j_l1,
        "row_j_l2": row_j_l2,
        "balance_ratio": balance_ratio,
        "flipped_contribution_l1": contribution_l1[changed_mask],
        "unflipped_contribution_l1": contribution_l1[~changed_mask],
        "flipped_row_j_l2": row_j_l2[changed_mask],
        "unflipped_row_j_l2": row_j_l2[~changed_mask],
        "flipped_balance_ratio": flipped_balance_ratio,
        "unflipped_balance_ratio": unflipped_balance_ratio,
        "balance_threshold": balance_threshold,
        "flipped_balanced_fraction": flipped_balanced_fraction,
        "unflipped_balanced_fraction": unflipped_balanced_fraction,
        "balanced_enrichment": balanced_enrichment,
    }

def print_field_balance(fb, label=""):
    def safe_mean(x):
        return float(np.mean(x)) if x.size > 0 else 0.0

    print("\n" + "="*60)
    if label:
        print(f"WEAK-ROW VS CANCELLATION DIAGNOSTIC (Balanced Field Analysis): {label}")
    else:
        print("WEAK-ROW VS CANCELLATION DIAGNOSTIC (Balanced Field Analysis)")
    print(f"  mean sum |J_ij cos(phi_j)| flipped:   {safe_mean(fb['flipped_contribution_l1']):.6g}")
    print(f"  mean sum |J_ij cos(phi_j)| unflipped: {safe_mean(fb['unflipped_contribution_l1']):.6g}")
    print(f"  mean row ||J_i||_2 flipped:           {safe_mean(fb['flipped_row_j_l2']):.6g}")
    print(f"  mean row ||J_i||_2 unflipped:         {safe_mean(fb['unflipped_row_j_l2']):.6g}")
    print(f"  mean balance ratio flipped:           {safe_mean(fb['flipped_balance_ratio']):.6g}")
    print(f"  mean balance ratio unflipped:         {safe_mean(fb['unflipped_balance_ratio']):.6g}")
    print(f"  balance threshold, q25:               {fb['balance_threshold']:.6g}")
    print(f"  balanced fraction flipped:            {100.0*fb['flipped_balanced_fraction']:.1f}%")
    print(f"  balanced fraction unflipped:          {100.0*fb['unflipped_balanced_fraction']:.1f}%")
    print(f"  balanced-site enrichment:             {fb['balanced_enrichment']:.3g}x")
    print("="*60)

def plot_analytics(J, spins, h_e, h_t, h_ks, pts, phase_name, file_prefix, J0=0.0, flipped_fields=None, all_osc_fields=None):
    num_reps = spins.shape[0]
    N = spins.shape[1]
    
    energies = np.zeros(num_reps)
    for r in range(num_reps):
        energies[r] = -0.5 * np.dot(spins[r], J @ spins[r])
        
    mh_fields_flat = np.abs(spins @ J).flatten()
    std_J = np.std(J)
    threshold = std_J * 2.5 if std_J > 0 else 0.01 

    ovl = [np.dot(spins[i], spins[j])/N for i in range(num_reps) for j in range(i+1, num_reps)]
    
    title_str = f"J0={J0}, N={N}, Seed={GLOBAL_SEED}, {phase_name}"
    
    plt.figure(figsize=(8, 5))
    ax1 = plt.gca()
    ax1.set_title(f"Energy Progression, {title_str}", fontsize=14, pad=15)
    for i in range(len(pts)):
        start = pts[i]
        end = pts[i+1] if i + 1 < len(pts) else len(h_e)
        label = f"Epoch {i+1}" if i < len(pts) - 1 else f"Epoch {len(pts)} (MH)"
        ax1.plot(np.arange(start, end) * sample_rate, h_e[start:end], label=label)
    if len(pts) > 2:
        ax1.axvline(x=pts[-1] * sample_rate, color='red', linestyle='--', alpha=0.7, label='MH Takeover')
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1e4:g}"))
    ax1.set_xlabel(r"Simulation Steps ($\times 10^4$)")
    ax1.set_ylabel("Energy", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Noise Amplitude & SHIL Amplitude", color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    h_kn = np.sqrt(2 * np.array(h_t))
    ax2.plot(np.arange(len(h_kn)) * sample_rate, h_kn, color='navy', linestyle='--', alpha=0.8, linewidth=2, label='Noise $K_n$')
    if len(h_ks) > 0 and len(h_ks) == len(h_t):
        ax2.plot(np.arange(len(h_ks)) * sample_rate, h_ks, color='forestgreen', linestyle=':', alpha=0.8, linewidth=2, label='SHIL Amp $K_s$')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', ncol=1, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_Plot_1_Energy.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    
    plt.figure(figsize=(8, 5))
    ax3 = plt.gca()
    ax3.set_title(f"Overlap P(q), {title_str}", fontsize=14, pad=15)
    ax3.hist(ovl, bins=50, range=(-1, 1), color='darkorchid', edgecolor='black', alpha=0.7)
    ax3.set_xlabel("Overlap", fontsize=20, fontweight='bold')
    ax3.set_ylabel("Frequency", fontsize=20, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=18)
    ax3.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_Plot_2_Overlap.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    
    plt.figure(figsize=(8, 5))
    ax4 = plt.gca()
    ax4.set_title(f"Dendrogram Analysis, {title_str}", fontsize=14, pad=15)
    dist_vec = pdist(spins, metric='hamming')
    Z = linkage(dist_vec, method='average')
    ax4.set_ylabel("Hamming Distance", fontsize=20, fontweight='bold')
    dendrogram(Z, truncate_mode='lastp', p=40, leaf_rotation=90, no_labels=True, ax=ax4)
    ax4.set_xlabel("Replicas", fontsize=20, fontweight='bold')
    ax4.tick_params(axis='y', labelsize=18)
    ax4.grid(True, which='both', axis='y', linestyle='--', alpha=0.4)
    ax4.axhline(y=0.5, color='red', linestyle='-', alpha=0.3, label='Random (0.5)')
    ax4.legend(loc='upper right', fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_Plot_3_Dendrogram.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    ax6 = plt.gca()
    ax6.set_title(f"Local Field Magnitude, {title_str}", fontsize=14, pad=15)
    if flipped_fields is not None and all_osc_fields is not None:
        f_max = max(all_osc_fields.max(), flipped_fields.max() if len(flipped_fields) > 0 else 1.0)
        bins = np.linspace(0, f_max, 60)
        ax6.hist(all_osc_fields, bins=bins, color='dodgerblue', alpha=0.5, edgecolor='black', density=True, label='All Osc Fields')
        if len(flipped_fields) > 0:
            ax6.hist(flipped_fields, bins=bins, color='crimson', alpha=0.7, edgecolor='black', density=True, label='Flipped by MH')
        ax6.axvline(x=threshold, color='red', linestyle='-', linewidth=2, label=f'Threshold ({threshold:.2f})')
        ax6.axvspan(0, threshold, color='red', alpha=0.1)
        ax6.set_xlabel("Oscillator Field Magnitude $|h_i|$", fontweight='bold')
        ax6.set_ylabel("Density (Frequency / Total)", fontweight='bold')
    else:
        f_min, f_max = mh_fields_flat.min(), mh_fields_flat.max()
        if np.isclose(f_min, f_max):
            ax6.hist(mh_fields_flat, bins=[f_min - 0.5, f_max + 0.5], color='dodgerblue', alpha=0.7, edgecolor='black')
        else:
            ax6.hist(mh_fields_flat, bins=60, color='dodgerblue', alpha=0.7, edgecolor='black')
        ax6.axvline(x=threshold, color='red', linestyle='-', linewidth=2, label=f'Threshold ({threshold:.2f})')
        ax6.axvspan(0, threshold, color='red', alpha=0.1)
        ax6.set_xlabel("Field Magnitude $|h_i|$", fontweight='bold')
        ax6.set_ylabel("Frequency", fontweight='bold')
    ax6.tick_params(axis='both', labelsize=18)
    ax6.legend(fontsize=16)
    ax6.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_Plot_4_Fields.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    ax9 = plt.gca()
    ax9.set_title(f"Energy Distribution, {title_str}", fontsize=14, pad=15)
    e_min, e_max = energies.min(), energies.max()
    if np.isclose(e_min, e_max):
        ax9.hist(energies, bins=[e_min - 0.5, e_max + 0.5], color='darkorange', edgecolor='black', alpha=0.8)
    else:
        bin_width = 0.2
        start_bin = np.floor(e_min / bin_width) * bin_width
        end_bin = np.ceil(e_max / bin_width) * bin_width + bin_width
        custom_bins = np.arange(start_bin, end_bin + 0.1 * bin_width, bin_width)
        ax9.hist(energies, bins=custom_bins, color='darkorange', edgecolor='black', alpha=0.8)
    mean_e = np.mean(energies)
    min_e = np.min(energies)
    ax9.axvline(mean_e, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {mean_e:.1f}')
    ax9.axvline(min_e, color='red', linestyle='solid', linewidth=2, label=f'Min: {min_e:.1f}')
    ax9.set_xlabel("Energy", fontsize=20, fontweight='bold')
    ax9.set_ylabel("Frequency", fontsize=20, fontweight='bold')
    ax9.tick_params(axis='x', labelsize=20)
    ax9.tick_params(axis='y', labelsize=18)
    ax9.legend(fontsize=20, framealpha=0.95, edgecolor='black')
    ax9.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_Plot_5_EnergyDist.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    
# ==========================================
# 6. MPI EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    base_reps = TOTAL_REPLICAS // size
    remainder = TOTAL_REPLICAS % size
    start_idx = rank * base_reps + min(rank, remainder)
    local_reps = base_reps + (1 if rank < remainder else 0)
    
    if rank == 0:
        print("="*60)
        print(f"TRUBA MPI EXECUTION: All-to-All Spin Glass")
        print(f"Nodes (MPI Ranks): {size}")
        print(f"Total Replicas: {TOTAL_REPLICAS}")
        print(f"N (Spins): {N}")
        print("="*60, flush=True)

    epochs = [
        {"anneal": int(15e4), "settle": int(0.0), "T_s": 0.7, "T_m": 0.5, "Ks_s": 0.0, "Ks_f": 0.4, "dt": 1e-3},
        {"anneal": int(15e4), "settle": int(0.0), "T_s": 0.5, "T_m": 0.3, "Ks_s": 0.4, "Ks_f": 0.6, "dt": 1e-2},
        {"anneal": int(25e4), "settle": int(0.0), "T_s": 0.3, "T_m": 0.2, "Ks_s": 0.6, "Ks_f": 0.8, "dt": 1e-2},
        {"anneal": int(25e4), "settle": int(0.0), "T_s": 0.2, "T_m": 0.1, "Ks_s": 0.8, "Ks_f": 0.8, "dt": 5e-2}
    ]

    # Initialize Master Sequence Architecture
    seq_J, seq_init, epoch_sequences, seq_mh_branch = setup_master_rng(GLOBAL_SEED, len(epochs))

    # --- 1a. J Matrix Generation ---
    J = generate_j_matrix(seq_J, N, J0=J0_VAL, J_std=1.0)
    
    # --- 1b. Initialize Local States ---
    local_phis = generate_initial_phases(seq_init, TOTAL_REPLICAS, N, start_idx, local_reps)
    
    h_e, h_t, h_ks, pts = [], [], [], []
    replica_h_e = None  # To capture the 2D energy matrix
    curr_step = 0
    start_time = time.time()
    
    for idx, cfg in enumerate(epochs):
        if rank == 0:
            print(f"\nProcessing Epoch {idx+1}/{len(epochs)}...", flush=True)
            
        # Spawn pure C-wrapper generator instances
        local_gens = get_local_generators(epoch_sequences[idx], TOTAL_REPLICAS, start_idx, local_reps)
            
        # Run multithreaded nogil integration wrapper
        local_phis, eb, tb, kb = parallel_osc_stage_threads(
            J, N, local_reps, int(cfg["anneal"]), int(cfg["settle"]), 
            np.sqrt(2*cfg["T_s"]), np.sqrt(2*cfg["T_m"]), 
            cfg["Ks_s"], cfg["Ks_f"], 
            cfg["dt"], 
            local_gens, local_phis
        )
        
        # Gather all 2D eb arrays from all nodes
        gathered_eb = comm.gather(eb, root=0)
        
        if rank == 0:
            epoch_eb_all = np.vstack(gathered_eb)
            if replica_h_e is None:
                replica_h_e = epoch_eb_all
            else:
                replica_h_e = np.hstack((replica_h_e, epoch_eb_all))
                
            h_e.extend(eb[0]); h_t.extend(tb[0]); h_ks.extend(kb[0])
            pts.append(curr_step); curr_step += len(eb[0])

        comm.Barrier()
        
        # ====================================================
        # EVOLUTIONARY ALGORITHM: SELECTION & REPLICATION
        # ====================================================
        all_phis_gathered = comm.gather(local_phis, root=0)

        if rank == 0:
            flat_phis = np.vstack(all_phis_gathered)
            temp_spins = np.sign(np.cos(flat_phis))
            temp_spins[temp_spins == 0.0] = 1.0
            
            temp_energies = np.zeros(TOTAL_REPLICAS)
            for r in range(TOTAL_REPLICAS):
                temp_energies[r] = -0.5 * np.dot(temp_spins[r], J @ temp_spins[r])
            
            sorted_indices = np.argsort(temp_energies)
            num_keep = TOTAL_REPLICAS // 3
            
            best_energy = temp_energies[sorted_indices[0]]
            worst_kept = temp_energies[sorted_indices[num_keep - 1]]
            worst_energy = temp_energies[sorted_indices[-1]]
            
            print(f"  EA Selection (Top 33.3%) -> Best: {best_energy:.4f} | Worst Kept: {worst_kept:.4f} | Eliminated: {worst_energy:.4f}")
            
            best_indices = sorted_indices[:num_keep]
            best_phis = flat_phis[best_indices]
            new_flat_phis = np.vstack((best_phis, best_phis, best_phis))
        else:
            new_flat_phis = None

        new_flat_phis = bcast_data = comm.bcast(new_flat_phis, root=0)
        local_phis = new_flat_phis[start_idx : start_idx + local_reps].copy()
        comm.Barrier()
    
    osc_spins = np.sign(np.cos(local_phis))
    osc_spins[osc_spins == 0.0] = 1.0
            
    all_osc_spins = comm.gather(osc_spins, root=0)
    all_osc_phis = comm.gather(local_phis, root=0)
    
    if rank == 0:
        flat_osc_spins = np.vstack(all_osc_spins)
        flat_osc_phis = np.vstack(all_osc_phis)
        
        osc_energies = np.zeros(TOTAL_REPLICAS)
        for r in range(TOTAL_REPLICAS):
            osc_energies[r] = -0.5 * np.dot(flat_osc_spins[r], J @ flat_osc_spins[r])
            
        print("\n" + "="*60)
        print("STATE ENERGY STATISTICS (Post-Oscillator)")
        print(f"  - Mean Energy: {np.mean(osc_energies):.4f}")
        print(f"  - Min Energy:  {np.min(osc_energies):.4f}")
        print("="*60)
        
        osc_clutter, osc_entropy = calculate_pq_metrics(flat_osc_spins)
        print("\nP(q) OVERLAP DISTRIBUTION QUALITY (Post-Oscillator)")
        print(f"  - Clutter Fraction (|q| <= 0.9): {osc_clutter:.4f} ({osc_clutter*100:.1f}%)")
        print(f"  - Shannon Entropy:               {osc_entropy:.4f} bits")
        print("="*60)
        
        dist_vec_osc = pdist(flat_osc_spins, metric='hamming')
        Z_osc = linkage(dist_vec_osc, method='average')
        coph_corr_osc, _ = cophenet(Z_osc, dist_vec_osc)

        print("\n" + "="*60)
        print("ULTRAMETRICITY & RSB QUALITY (Post-Oscillator)")
        print(f"  - Cophenetic Correlation: {coph_corr_osc:.4f}")
        if coph_corr_osc > 0.8:
            print("  - Status: Strong ultrametric structure detected (High RSB).")
        elif coph_corr_osc > 0.6:
            print("  - Status: Weak ultrametricity. System may be trapped in metastable states.")
        else:
            print("  - Status: Poor ultrametricity. Geometry does not resemble a true spin glass.")
        print("="*60)

        osc_valley_metrics = calculate_valley_diversity(flat_osc_spins)
        print_valley_diversity(osc_valley_metrics, "Post-Oscillator")

        print("\nPlotting Intermediate Oscillator Analytics...")
        plot_analytics(J, flat_osc_spins, h_e, h_t, h_ks, pts, "Post-Oscillator", f"{FILE_BASE}_PostOsc", J0=J0_VAL)

    # --- 3. Metropolis-Hastings Takeover ---
    if rank == 0:
        pts.append(curr_step)
        print(f"\nInitiating MH Refinement ({MH_STEPS} steps)...", flush=True)
        
    local_mh_gens = get_local_generators(seq_mh_branch, TOTAL_REPLICAS, start_idx, local_reps)
        
    local_final_spins, mhe, mht = parallel_mh_stage_threads(
        J, N, local_reps, MH_STEPS, T_MH_S, T_MH_E, local_phis, local_mh_gens
    )
    
    # Gather MH energies from all nodes
    gathered_mhe = comm.gather(mhe, root=0)
    
    if rank == 0:
        mh_eb_all = np.vstack(gathered_mhe)
        replica_h_e = np.hstack((replica_h_e, mh_eb_all))
        
        h_e.extend(mhe[0]); h_t.extend(mht[0]); h_ks.extend(np.full_like(mht[0], epochs[-1]["Ks_f"]))

    comm.Barrier()
    all_final_spins = comm.gather(local_final_spins, root=0)
    
    # --- 4. Final Aggregation & Saving ---
    if rank == 0:
        flat_final_spins = np.vstack(all_final_spins)
        total_time = time.time() - start_time
        
        raw_flips = np.sum(flat_osc_spins != flat_final_spins, axis=1)
        inverted_replicas = raw_flips > (N / 2)
        true_flips = np.minimum(raw_flips, N - raw_flips)
        
        mean_flips = np.mean(true_flips)
        max_flips = np.max(true_flips)
        percent_changed = (mean_flips / N) * 100
        
        print("\n" + "="*60)
        print("METROPOLIS-HASTINGS WORKLOAD ANALYSIS (Z2 Corrected)")
        print(f"Average structural spins flipped by MH: {mean_flips:.2f} out of {N} ({percent_changed:.2f}%)")
        print(f"Max structural spins flipped in any replica: {max_flips}")
        
        std_J = np.std(J)
        threshold = std_J * 2.5 if std_J > 0 else 0.01 
        
        flipped_fields = []
        unflipped_fields = []
        
        for r in range(TOTAL_REPLICAS):
            osc_fields = np.abs(flat_osc_spins[r] @ J)
            if inverted_replicas[r]:
                changed_mask = (flat_osc_spins[r] == flat_final_spins[r])
            else:
                changed_mask = (flat_osc_spins[r] != flat_final_spins[r])
                
            flipped_fields.extend(osc_fields[changed_mask])
            unflipped_fields.extend(osc_fields[~changed_mask])
            
        flipped_fields = np.array(flipped_fields)
        unflipped_fields = np.array(unflipped_fields)
        all_osc_fields = np.concatenate((flipped_fields, unflipped_fields))
        
        total_flips = len(flipped_fields)
        defect_flips = np.sum(flipped_fields < threshold)
        targeted_percentage = (defect_flips / total_flips * 100) if total_flips > 0 else 0.0
        
        mean_flipped_force = np.mean(flipped_fields) if total_flips > 0 else 0.0
        mean_unflipped_force = np.mean(unflipped_fields) if len(unflipped_fields) > 0 else 0.0
        overall_mean_force = np.mean(all_osc_fields)
        
        print(f"\nOSCILLATOR FORCE TARGETING ANALYSIS:")
        print(f"Average structural force (|h_i|) on ALL spins:         {overall_mean_force:.4f}")
        print(f"Average structural force (|h_i|) on FLIPPED spins:     {mean_flipped_force:.4f}")
        print(f"Average structural force (|h_i|) on UNFLIPPED spins: {mean_unflipped_force:.4f}")
        print("-" * 60)
        print(f"Total structural spins flipped across all replicas: {total_flips}")
        print(f"Flips that targeted weak fields (|h_i| < {threshold:.2f}): {defect_flips}")
        print(f"Percentage of MH workload spent fixing weak forces: {targeted_percentage:.1f}%")
        print("="*60)

        aligned_final_spins = flat_final_spins.copy()
        aligned_final_spins[inverted_replicas] *= -1
        changed_mask_matrix = (flat_osc_spins != aligned_final_spins)

        field_balance = calculate_field_balance(J, flat_osc_phis, changed_mask_matrix)
        print_field_balance(field_balance, label="Post-MH Refinement")

        final_energies = np.zeros(TOTAL_REPLICAS)
        for r in range(TOTAL_REPLICAS):
            final_energies[r] = -0.5 * np.dot(flat_final_spins[r], J @ flat_final_spins[r])
            
        print("\n" + "="*60)
        print("STATE ENERGY STATISTICS (Post-MH Refinement)")
        print(f"  - Mean Energy: {np.mean(final_energies):.4f}")
        print(f"  - Min Energy:  {np.min(final_energies):.4f}")
        print("="*60)
        
        final_clutter, final_entropy = calculate_pq_metrics(flat_final_spins)

        print("\nP(q) OVERLAP DISTRIBUTION QUALITY (Post-MH Refinement)")
        print(f"  - Clutter Fraction (|q| <= 0.9): {final_clutter:.4f} ({final_clutter*100:.1f}%)")
        print(f"  - Shannon Entropy:               {final_entropy:.4f} bits")
        print("="*60)
        
        dist_vec_final = pdist(flat_final_spins, metric='hamming')
        Z_final = linkage(dist_vec_final, method='average')
        coph_corr, _ = cophenet(Z_final, dist_vec_final)

        print("\n" + "="*60)
        print("ULTRAMETRICITY & RSB QUALITY (Post-MH Refinement)")
        print(f"  - Cophenetic Correlation: {coph_corr:.4f}")
        if coph_corr > 0.8:
            print("  - Status: Strong ultrametric structure detected (High RSB).")
        elif coph_corr > 0.6:
            print("  - Status: Weak ultrametricity. System may be trapped in metastable states.")
        else:
            print("  - Status: Poor ultrametricity. Geometry does not resemble a true spin glass.")
        print("="*60)

        valley_metrics = calculate_valley_diversity(flat_final_spins, hamming_threshold=0.30)
        print_valley_diversity(valley_metrics, "Post-MH Refinement")
        
        print("\nPlotting Final Analytics...")
        plot_analytics(J, flat_final_spins, h_e, h_t, h_ks, pts, "Post-MH Refinement", f"{FILE_BASE}_PostMH", 
                       flipped_fields=flipped_fields, all_osc_fields=all_osc_fields, J0=J0_VAL)
        
        print("Generating Plot 6: Comparative Local Field Distributions...")
        plt.figure(figsize=(12, 6))
        ax10 = plt.gca()
        
        osc_fields_all = np.abs(flat_osc_spins @ J).flatten()
        mh_fields_all = np.abs(flat_final_spins @ J).flatten()
        
        defects_osc = np.sum(osc_fields_all < threshold)
        defects_mh = np.sum(mh_fields_all < threshold)
        
        f_max_all = max(osc_fields_all.max(), mh_fields_all.max())
        bins_all = np.linspace(0, f_max_all, 60)
        
        ax10.hist(osc_fields_all, bins=bins_all, color='#ffaa66', edgecolor='black', alpha=0.85, 
                label=f'After Oscillator (Singular: {defects_osc})')
                
        ax10.hist(mh_fields_all, bins=bins_all, color='#66b3ff', edgecolor='black', alpha=0.75, 
                label=f'After MH Polish (Singular: {defects_mh})')
                
        ax10.axvline(x=threshold, color='red', linestyle='-', linewidth=2.5, 
                   label=f'Defect Threshold ({threshold:.3f})', zorder=5)
        ax10.axvspan(0, threshold, color='red', alpha=0.1, zorder=1)
        
        ax10.set_title(f"Comparative Local Fields, J0={J0_VAL}, N={N}, Seed={GLOBAL_SEED}, Post-MH Refinement", fontsize=14, pad=15)
        
        ax10.set_xlabel("Absolute Local Field Magnitude", fontsize=22, fontweight='bold')
        ax10.set_ylabel("Frequency", fontsize=22, fontweight='bold')
        ax10.tick_params(axis='both', labelsize=20)
        ax10.grid(axis='y', linestyle=':', alpha=0.7, zorder=0)
        ax10.legend(loc='upper right', fontsize=20)
        
        plt.tight_layout()
        plt.savefig(f"{FILE_BASE}_Plot_6_Comparative_Fields.pdf", format="pdf", bbox_inches="tight")
        plt.close()

        print("Saving NPZ Data...")
        np.savez(f"{FILE_BASE}.npz", 
                 J=J, 
                 osc_spins=flat_osc_spins,
                 final_spins=flat_final_spins, 
                 h_e=h_e, h_t=h_t, pts=pts,
                 replica_h_e=replica_h_e,
                 avg_h_e=np.mean(replica_h_e, axis=0))
        
        print("="*60)
        print(f"Simulation Complete in {total_time:.2f} seconds.")
        print(f"Saved: {FILE_BASE}.npz, and independent PDF plots for both phases.")
        print("="*60)