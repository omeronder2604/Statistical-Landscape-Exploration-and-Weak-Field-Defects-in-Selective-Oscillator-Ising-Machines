# -*- coding: utf-8 -*-
"""
TRUBA Execution: All-to-All Gaussian Spin Glass 
Hybrid Oscillator + MH Refinement using 1D MPI Replica Distribution.
(Updated with Master Seed Averaging, Ultrametricity %, Dynamic Valley Analysis, and Full State Storage)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import time
import sys
from numba import njit, prange
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
GLOBAL_SEED = 42
NUM_SEEDS = 30      # Number of independent disorder realizations to average over
N = 200
TOTAL_REPLICAS = 402  
FILE_BASE = "AllToAll_Hybrid_EA_SeedAvg"
sample_rate = 500

# MH Tuning Parameters
MH_STEPS = int(1e4)
T_MH_S = 0.2
T_MH_E = 0.1

# ==========================================
# 3. CENTRALIZED RNG & STATE GENERATION
# ==========================================
def setup_master_rng(global_seed, num_epochs):
    """Initializes the master seed sequence and spawns independent sequences for each realization."""
    master_seq = np.random.SeedSequence(global_seed)
    return master_seq.spawn(num_epochs)

def generate_j_matrix(seq_J, N, J0=0, J_std=1.0):
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
@njit(nogil=True, fastmath=True)
def run_osc_epoch(J, N, S_A, S_S, Kn_s, Kn_m, Ks_s, Ks_f, K_c, dt, gen, initial_phi):
    phi = initial_phi.copy()
    sqrt_dt, total_steps = np.sqrt(dt), S_A + S_S
    h_size = int(total_steps // sample_rate)
    e_hist, t_hist, ks_hist = np.zeros(h_size), np.zeros(h_size), np.zeros(h_size)
    
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
        
        i = int(gen.random() * N)
        if i == N: i = N - 1 
        
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

    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=cpus) as executor:
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

    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=cpus) as executor:
        executor.map(task, range(num_replicas))
        
    return final_spins, e_b, t_b

# ==========================================
# 5. HELPER FUNCTIONS & AVERAGED PLOTTING
# ==========================================
def calculate_valley_diversity(spins):
    num_reps, N = spins.shape
    dist_vec = pdist(spins, metric='hamming')
    Z = linkage(dist_vec, method='average')
    
    merge_heights = Z[:, 2]
    
    if len(merge_heights) > 1:
        gaps = np.diff(merge_heights)
        largest_gap_idx = np.argmax(gaps)
        dynamic_threshold = merge_heights[largest_gap_idx] + (gaps[largest_gap_idx] / 2.0)
        top_branches = merge_heights[-3:] 
    else:
        dynamic_threshold = 0.30
        top_branches = np.array([])
        
    max_hamming = dist_vec.max()
    mean_hamming = dist_vec.mean()
    cross_valley_fraction = np.mean(dist_vec > dynamic_threshold)
    
    labels = fcluster(Z, t=dynamic_threshold, criterion='distance')
    num_clusters = len(np.unique(labels))
    final_merge_height = Z[-1, 2] if len(Z) > 0 else 0.0
    
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
        "max_hamming":            max_hamming,
        "mean_hamming":           mean_hamming,
        "dynamic_threshold":      dynamic_threshold,
        "top_branch_distances":   top_branches,
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
        
    print(f"  Natural Branching Threshold (Auto-detected): {metrics['dynamic_threshold']:.4f}")
    if len(metrics['top_branch_distances']) > 0:
        top_b = ", ".join([f"{x:.4f}" for x in metrics['top_branch_distances']])
        print(f"  Top Branch Merge Distances:          [{top_b}]")
    print("-" * 60)
    print(f"  Max pairwise Hamming dist:           {metrics['max_hamming']:.4f}")
    print(f"  Mean pairwise Hamming dist:          {metrics['mean_hamming']:.4f}")
    print(f"  Cross-valley replica pairs:          {metrics['cross_valley_fraction']*100:.1f}%")
    print(f"  Distinct valleys (clusters):         {metrics['num_distinct_clusters']}")
    print(f"  Final dendrogram merge height:       {metrics['final_merge_height']:.4f}")
    print(f"  Mean intra-cluster Hamming:          {metrics['intra_cluster_mean']:.4f}")
    print(f"  Mean inter-cluster Hamming:          {metrics['inter_cluster_mean']:.4f}")
    print(f"  Inter/Intra separation ratio:        {metrics['separation_ratio']:.4f}")
    print("="*60)

@njit(parallel=True, fastmath=True)
def calculate_ultrametric_percentage(spins, tol):
    """
    Checks all triplets to find the percentage that satisfy the 
    strict ultrametric condition: d_max - d_mid <= tol
    """
    R = spins.shape[0]
    N = spins.shape[1]
    
    dist = np.zeros((R, R))
    for i in prange(R):
        for j in range(i+1, R):
            d = 0.0
            for k in range(N):
                if spins[i, k] != spins[j, k]:
                    d += 1.0
            d /= N
            dist[i, j] = d
            dist[j, i] = d
            
    valid = 0
    total = 0
    for i in range(R):
        for j in range(i+1, R):
            for k in range(j+1, R):
                d1, d2, d3 = dist[i, j], dist[i, k], dist[j, k]
                
                # Sort: d1 <= d2 <= d3
                if d1 > d2: d1, d2 = d2, d1
                if d2 > d3: d2, d3 = d3, d2
                if d1 > d2: d1, d2 = d2, d1
                
                # Condition: two largest distances must be equal up to tolerance
                if (d3 - d2) <= tol:
                    valid += 1
                total += 1
                
    return (valid / total) * 100.0 if total > 0 else 0.0

def plot_averaged_pq(overlaps, phase_name, file_prefix):
    title_str = f"J0=0, N={N}, Seeds={NUM_SEEDS}, {phase_name}"
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.set_title(f"Averaged Overlap P(q), {title_str}", fontsize=14, pad=15)
    
    ax.hist(overlaps, bins=50, range=(-1, 1), color='darkorchid', edgecolor='black', alpha=0.7, density=True)
    ax.set_xlabel("Overlap", fontsize=20, fontweight='bold')
    ax.set_ylabel("Probability Density", fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_Avg_Plot_2_Overlap.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def plot_averaged_energy_progression(h_e, h_t, h_ks, pts, phase_name, file_prefix):
    title_str = f"J0=0, N={N}, Seeds={NUM_SEEDS}, {phase_name}"
    plt.figure(figsize=(8, 5))
    ax1 = plt.gca()
    ax1.set_title(f"Averaged Energy Progression, {title_str}", fontsize=14, pad=15)
    
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
    plt.savefig(f"{file_prefix}_Avg_Plot_1_EnergyProg.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def plot_averaged_energy_dist(energies, phase_name, file_prefix):
    title_str = f"J0=0, N={N}, Seeds={NUM_SEEDS}, {phase_name}"
    plt.figure(figsize=(8, 5))
    ax9 = plt.gca()
    ax9.set_title(f"Averaged Energy Distribution, {title_str}", fontsize=14, pad=15)
    
    e_min, e_max = np.min(energies), np.max(energies)
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
    plt.savefig(f"{file_prefix}_Avg_Plot_5_EnergyDist.pdf", format="pdf", bbox_inches="tight")
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
        print(f"Number of Seeds: {NUM_SEEDS}")
        print("="*60, flush=True)

    epochs = [
        {"anneal": int(15e4), "settle": int(0.0), "T_s": 0.7, "T_m": 0.5, "Ks_s": 0.0, "Ks_f": 0.4, "dt": 1e-3},
        {"anneal": int(15e4), "settle": int(0.0), "T_s": 0.5, "T_m": 0.3, "Ks_s": 0.4, "Ks_f": 0.6, "dt": 1e-2},
        {"anneal": int(25e4), "settle": int(0.0), "T_s": 0.3, "T_m": 0.2, "Ks_s": 0.6, "Ks_f": 0.8, "dt": 1e-2},
        {"anneal": int(25e4), "settle": int(0.0), "T_s": 0.2, "T_m": 0.1, "Ks_s": 0.8, "Ks_f": 0.8, "dt": 5e-2}
    ]

    seed_sequences = setup_master_rng(GLOBAL_SEED, NUM_SEEDS)
    
    # Grand Accumulators across all seeds
    master_J = []
    master_overlaps = []
    master_energies = []
    master_osc_energies = []
    master_osc_spins = []
    master_final_spins = []
    master_ultra_pcts = []
    master_h_e_hist = []
    master_h_t = None
    master_h_ks = None
    master_pts = None

    grand_start_time = time.time()

    # Iterate over every independent realization
    for s_idx, current_seed_seq in enumerate(seed_sequences):
        if rank == 0:
            print(f"\n" + "="*50)
            print(f" BEGINNING SEED REALIZATION {s_idx + 1}/{NUM_SEEDS}")
            print("="*50, flush=True)
            
        seq_J, seq_init, seq_epochs_branch, seq_mh_branch = current_seed_seq.spawn(4)
        epoch_sequences = seq_epochs_branch.spawn(len(epochs))

        # --- Generate realization-specific matrices and states ---
        J = generate_j_matrix(seq_J, N, J0=0, J_std=1.0)
        
        if rank == 0:
            master_J.append(J)
            
        local_phis = generate_initial_phases(seq_init, TOTAL_REPLICAS, N, start_idx, local_reps)
        
        h_e, h_t, h_ks, pts = [], [], [], []
        curr_step = 0
        
        # --- Oscillator Pipeline ---
        for idx, cfg in enumerate(epochs):
            if rank == 0:
                print(f"  Processing Epoch {idx+1}/{len(epochs)}...", flush=True)
                
            local_gens = get_local_generators(epoch_sequences[idx], TOTAL_REPLICAS, start_idx, local_reps)
                
            local_phis, eb, tb, kb = parallel_osc_stage_threads(
                J, N, local_reps, int(cfg["anneal"]), int(cfg["settle"]), 
                np.sqrt(2*cfg["T_s"]), np.sqrt(2*cfg["T_m"]), 
                cfg["Ks_s"], cfg["Ks_f"], 
                cfg["dt"], 
                local_gens, local_phis
            )
            
            if rank == 0:
                h_e.extend(eb[0]); h_t.extend(tb[0]); h_ks.extend(kb[0])
                pts.append(curr_step); curr_step += len(eb[0])

            comm.Barrier()
            
            # --- EA Selection ---
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
                print(f"    EA Selection -> Best: {best_energy:.4f} | Worst Kept: {worst_kept:.4f}")
                
                best_indices = sorted_indices[:num_keep]
                best_phis = flat_phis[best_indices]
                new_flat_phis = np.vstack((best_phis, best_phis, best_phis))
            else:
                new_flat_phis = None

            new_flat_phis = comm.bcast(new_flat_phis, root=0)
            local_phis = new_flat_phis[start_idx : start_idx + local_reps].copy()
            comm.Barrier()
            
        # --- Evaluate Post-Oscillator States ---
        osc_spins = np.sign(np.cos(local_phis))
        osc_spins[osc_spins == 0.0] = 1.0
        all_osc_spins = comm.gather(osc_spins, root=0)

        if rank == 0:
            flat_osc_spins = np.vstack(all_osc_spins)
            master_osc_spins.append(flat_osc_spins)
            
            osc_energies = np.zeros(TOTAL_REPLICAS)
            for r in range(TOTAL_REPLICAS):
                osc_energies[r] = -0.5 * np.dot(flat_osc_spins[r], J @ flat_osc_spins[r])
            master_osc_energies.extend(osc_energies)

        # --- MH Takeover ---
        if rank == 0:
            pts.append(curr_step)
            print(f"  Initiating MH Refinement ({MH_STEPS} steps)...", flush=True)
            
        local_mh_gens = get_local_generators(seq_mh_branch, TOTAL_REPLICAS, start_idx, local_reps)
            
        local_final_spins, mhe, mht = parallel_mh_stage_threads(
            J, N, local_reps, MH_STEPS, T_MH_S, T_MH_E, local_phis, local_mh_gens
        )
        
        if rank == 0:
            h_e.extend(mhe[0]); h_t.extend(mht[0]); h_ks.extend(np.full_like(mht[0], epochs[-1]["Ks_f"]))

        comm.Barrier()
        all_final_spins = comm.gather(local_final_spins, root=0)
        
        # --- Single Seed Evaluation & Accumulation ---
        if rank == 0:
            flat_final_spins = np.vstack(all_final_spins)
            master_final_spins.append(flat_final_spins)
            
            # Record final energies
            final_energies = np.zeros(TOTAL_REPLICAS)
            for r in range(TOTAL_REPLICAS):
                final_energies[r] = -0.5 * np.dot(flat_final_spins[r], J @ flat_final_spins[r])
            master_energies.extend(final_energies)
            
            # Record overlaps P(q)
            ovl = [np.dot(flat_final_spins[i], flat_final_spins[j])/N for i in range(TOTAL_REPLICAS) for j in range(i+1, TOTAL_REPLICAS)]
            master_overlaps.extend(ovl)
            
            # Check Ultrametricity percentage (Tolerance: 4/N allows up to 2 spin difference)
            ultra_tol = 4.0 / N
            u_pct = calculate_ultrametric_percentage(flat_final_spins, ultra_tol)
            master_ultra_pcts.append(u_pct)
            
            # Record historical progression paths
            master_h_e_hist.append(h_e)
            if master_h_t is None: 
                master_h_t, master_h_ks, master_pts = h_t, h_ks, pts
                
            print(f"  Seed {s_idx + 1} Results:")
            print(f"    - Mean Final Energy: {np.mean(final_energies):.4f}")
            print(f"    - Triplet Ultrametricity Pass Rate: {u_pct:.2f}%")
            
            valley_metrics = calculate_valley_diversity(flat_final_spins)
            print_valley_diversity(valley_metrics, label=f"Seed {s_idx + 1} Post-MH")

    # ==========================================
    # 7. GRAND AVERAGED ANALYTICS & PLOTTING
    # ==========================================
    if rank == 0:
        total_time = time.time() - grand_start_time
        
        print("\n" + "="*60)
        print("GRAND SEED-AVERAGED SUMMARY STATISTICS")
        print("="*60)
        
        avg_overall_energy = np.mean(master_energies)
        std_overall_energy = np.std(master_energies)
        print(f"Averaged Energy per Replica: {avg_overall_energy:.4f} +/- {std_overall_energy:.4f}")
        
        avg_ultra_pct = np.mean(master_ultra_pcts)
        print(f"Averaged Ultrametricity Percentage: {avg_ultra_pct:.2f}% of triplets satisfy condition")
        print("="*60)

        # Plotting the grand averages
        print("Generating Averaged Plots...")
        avg_h_e = np.mean(master_h_e_hist, axis=0)
        
        plot_averaged_energy_progression(avg_h_e, master_h_t, master_h_ks, master_pts, "Post-MH", FILE_BASE)
        plot_averaged_pq(master_overlaps, "Post-MH", FILE_BASE)
        plot_averaged_energy_dist(master_energies, "Post-MH", FILE_BASE)
        
        # Save grand data bundle
        np.savez(f"{FILE_BASE}_AveragedData.npz", 
                 master_J=np.array(master_J),
                 master_osc_spins=np.array(master_osc_spins),
                 master_final_spins=np.array(master_final_spins),
                 master_overlaps=np.array(master_overlaps),
                 master_energies=np.array(master_energies),
                 master_osc_energies=np.array(master_osc_energies),
                 master_ultra_pcts=np.array(master_ultra_pcts),
                 avg_h_e=avg_h_e, 
                 h_t=master_h_t, 
                 pts=master_pts)
        
        print(f"All {NUM_SEEDS} Seeds Complete in {total_time:.2f} seconds.")
        print(f"Averaged metrics and PDF plots saved using prefix: {FILE_BASE}")
        print("="*60)