# -*- coding: utf-8 -*-
"""
TRUBA Execution: All-to-All Gaussian Spin Glass 
Pure Metropolis-Hastings (Seed-Averaged, Step-Based, Full State Storage, MPI Parallelized)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import time
from numba import njit, prange
from mpi4py import MPI
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

latex_fmt = FuncFormatter(lambda x, pos: f"${x:g}$")

# ==========================================
# 2. USER CONTROL PANEL
# ==========================================
GLOBAL_SEED = 42
NUM_SEEDS = 30      # Number of independent disorder realizations to average over
N = 200
num_replicas = 402  
FILE_BASE = "AllToAll_PureMH_MPI_SeedAvg"
sample_rate = 500
J0 = -4.0

# Original Step/dt Logic
dt = 1e-3
Anneal_Time = 10e4 * dt
Settle_Time = 1e4 * dt
Measure_Time = 1e4 * dt  

Steps_Anneal = int(Anneal_Time / dt)
Steps_Settle = int(Settle_Time / dt)
Steps_Measure = int(Measure_Time / dt)

# MH Thermodynamic Parameters
T_start = 0.7   
T_measure = 0.1 

# ==========================================
# 3. CENTRALIZED RNG & STATE GENERATION
# ==========================================
def setup_master_rng(global_seed, num_epochs):
    master_seq = np.random.SeedSequence(global_seed)
    return master_seq.spawn(num_epochs)

def generate_j_matrix(seq_J, N, J0=0, J_std=1.0):
    gen_J = np.random.default_rng(seq_J)
    J = gen_J.normal(J0/N, J_std/np.sqrt(N), (N, N))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    return J

# ==========================================
# 4. NUMBA DYNAMICS (Metropolis-Hastings)
# ==========================================
@njit(fastmath=True)
def run_metropolis_full_simulation(J, N, Steps_Anneal, Steps_Settle, Steps_Measure, T_start, T_measure, seed, sample_rate):
    np.random.seed(int(seed))
    spins = np.where(np.random.random(N) > 0.5, 1.0, -1.0)
    spin_sum_accumulator = np.zeros(N)
    
    total_steps = Steps_Anneal + Steps_Settle + Steps_Measure
    energy_history = np.zeros(total_steps // sample_rate)
    temp_history = np.zeros(total_steps // sample_rate) 

    for step in range(total_steps):
        if step < Steps_Anneal:
            progress = step / Steps_Anneal
            current_T = T_start * (1 - progress) + T_measure * progress
        else:
            current_T = T_measure
            
        beta = 1.0 / current_T
        
        if step % sample_rate == 0:
            idx = step // sample_rate
            energy_history[idx] = -0.5 * np.dot(spins, J @ spins)
            temp_history[idx] = current_T 
        
        i = np.random.randint(0, N)
        
        # Vectorized local field evaluation
        h_i = np.dot(J[i], spins) 
        delta_E = 2.0 * spins[i] * h_i
        
        if delta_E <= 0 or np.random.random() < np.exp(-delta_E * beta):
            spins[i] *= -1.0
            
        if step >= (Steps_Anneal + Steps_Settle):
            spin_sum_accumulator += spins

    return spin_sum_accumulator / Steps_Measure, energy_history, temp_history

@njit(parallel=True)
def run_local_replicas_parallel(J, N, local_reps, Steps_Anneal, Steps_Settle, Steps_Measure, T_start, T_measure, local_seeds, sample_rate):
    spin_results = np.zeros((local_reps, N))
    
    # Run the first local replica separately to extract the history arrays
    avg_spins_0, sample_e_hist, sample_t_hist = run_metropolis_full_simulation(
        J, N, Steps_Anneal, Steps_Settle, Steps_Measure, T_start, T_measure, local_seeds[0], sample_rate
    )
    spin_results[0] = avg_spins_0
    
    for r in prange(1, local_reps):
        avg_spins, _, _ = run_metropolis_full_simulation(
            J, N, Steps_Anneal, Steps_Settle, Steps_Measure, T_start, T_measure, local_seeds[r], sample_rate
        )
        spin_results[r] = avg_spins
        
    return spin_results, sample_e_hist, sample_t_hist

# ==========================================
# 5. HELPER FUNCTIONS & AVERAGED PLOTTING
# ==========================================
@njit(parallel=True, fastmath=True)
def calculate_ultrametric_percentage(spins, tol):
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
                
                if d1 > d2: d1, d2 = d2, d1
                if d2 > d3: d2, d3 = d3, d2
                if d1 > d2: d1, d2 = d2, d1
                
                if (d3 - d2) <= tol:
                    valid += 1
                total += 1
                
    return (valid / total) * 100.0 if total > 0 else 0.0

def plot_averaged_pq(overlaps, phase_name, file_prefix):
    title_str = f"J0={J0}, N={N}, Seeds={NUM_SEEDS}, {phase_name}"
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

def plot_averaged_energy_progression(h_e, h_t, phase_name, file_prefix):
    title_str = f"J0={J0}, N={N}, Seeds={NUM_SEEDS}, {phase_name}"
    plt.figure(figsize=(8, 5))
    ax1 = plt.gca()
    ax1.set_title(f"Averaged Energy Progression, {title_str}", fontsize=14, pad=15)
    
    step_axis = np.arange(len(h_e)) * sample_rate
    ax1.plot(step_axis, h_e, color='crimson', linewidth=2, label='System Energy', zorder=2)
        
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1e4:g}"))
    ax1.set_xlabel(r"Simulation Steps ($\times 10^4$)")
    ax1.set_ylabel("Energy", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Temperature $T$", color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.plot(step_axis, h_t, color='navy', linestyle='--', linewidth=2.5, label='Temperature $T$', alpha=0.8, zorder=1)
    
    ax1.axvline(x=Steps_Anneal, color='black', linestyle=':', linewidth=2.5, label='Anneal End', zorder=0)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', ncol=1, framealpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_Avg_Plot_1_EnergyProg.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def plot_averaged_energy_dist(energies, phase_name, file_prefix):
    title_str = f"J0={J0}, N={N}, Seeds={NUM_SEEDS}, {phase_name}"
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
    
    base_reps = num_replicas // size
    remainder = num_replicas % size
    start_idx = rank * base_reps + min(rank, remainder)
    local_reps = base_reps + (1 if rank < remainder else 0)

    if rank == 0:
        print("="*60)
        print(f"TRUBA MPI EXECUTION: All-to-All Spin Glass")
        print(f"Algorithm: Pure Metropolis-Hastings (MPI + Numba prange)")
        print(f"Nodes (MPI Ranks): {size}")
        print(f"Total Replicas: {num_replicas}")
        print(f"N (Spins): {N}")
        print(f"Number of Seeds: {NUM_SEEDS}")
        print("="*60, flush=True)

    seed_sequences = setup_master_rng(GLOBAL_SEED, NUM_SEEDS)
    
    # Grand Accumulators across all seeds (only tracked on rank 0)
    master_J = []
    master_overlaps = []
    master_energies = []
    master_final_spins = []
    master_ultra_pcts = []
    master_h_e_hist = []
    master_h_t = None

    grand_start_time = time.time()

    # Iterate over every independent realization
    for s_idx, current_seed_seq in enumerate(seed_sequences):
        seed_display = current_seed_seq.generate_state(1)[0]
        
        if rank == 0:
            print(f"\n" + "="*50)
            print(f" BEGINNING SEED REALIZATION {s_idx + 1}/{NUM_SEEDS}, SEED = {seed_display}")
            print("="*50, flush=True)
            
        seq_J, seq_reps = current_seed_seq.spawn(2)

        # --- Generate realization-specific matrix on all ranks (deterministic) ---
        J = generate_j_matrix(seq_J, N, J0=J0, J_std=1.0)
        
        if rank == 0:
            master_J.append(J)
        
        # Generate raw integer states to safely seed np.random inside Numba per replica
        # Generated identically on all ranks, then sliced
        replica_seeds = seq_reps.generate_state(num_replicas)
        local_seeds = replica_seeds[start_idx : start_idx + local_reps]
        
        if rank == 0:
            print(f"  Initiating Steps... (Anneal: {Steps_Anneal}, Settle: {Steps_Settle}, Measure: {Steps_Measure})", flush=True)
        
        # --- Run Parallel Numba Execution Locally ---
        start_time = time.time()
        local_replicas_avg, e_hist_full, t_hist_full = run_local_replicas_parallel(
            J, N, local_reps, Steps_Anneal, Steps_Settle, Steps_Measure, 
            T_start, T_measure, local_seeds, sample_rate
        )
        
        if rank == 0:
            print(f"  Simulation finished in {time.time() - start_time:.2f} seconds.")
        
        # Ensure spins are correctly signed locally
        local_frozen_states = np.sign(local_replicas_avg)
        for r in range(local_reps):
            for i in range(N):
                if local_frozen_states[r, i] == 0: local_frozen_states[r, i] = 1.0

        # --- Gather back to rank 0 ---
        comm.Barrier()
        all_final_spins = comm.gather(local_frozen_states, root=0)
        
        if rank == 0:
            flat_final_spins = np.vstack(all_final_spins)
            master_final_spins.append(flat_final_spins)
            
            # Record final energies
            final_energies = np.zeros(num_replicas)
            for r in range(num_replicas):
                final_energies[r] = -0.5 * np.dot(flat_final_spins[r], J @ flat_final_spins[r])
            master_energies.extend(final_energies)
            
            # Record overlaps P(q)
            ovl = [np.dot(flat_final_spins[i], flat_final_spins[j])/N for i in range(num_replicas) for j in range(i+1, num_replicas)]
            master_overlaps.extend(ovl)
            
            # Check Ultrametricity percentage
            ultra_tol = 4.0 / N
            u_pct = calculate_ultrametric_percentage(flat_final_spins, ultra_tol)
            master_ultra_pcts.append(u_pct)
            
            # Record historical progression paths
            master_h_e_hist.append(e_hist_full)
            if master_h_t is None: 
                master_h_t = t_hist_full
                
            print(f"  Seed {s_idx + 1} Results:")
            print(f"    - Mean Final Energy: {np.mean(final_energies):.4f}")
            print(f"    - Triplet Ultrametricity Pass Rate: {u_pct:.2f}%")

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
        
        plot_averaged_energy_progression(avg_h_e, master_h_t, "Pure MH Baseline", FILE_BASE)
        plot_averaged_pq(master_overlaps, "Pure MH Baseline", FILE_BASE)
        plot_averaged_energy_dist(master_energies, "Pure MH Baseline", FILE_BASE)
        
        # Save grand data bundle
        np.savez(f"{FILE_BASE}_AveragedData.npz", 
                 master_J=np.array(master_J),
                 master_final_spins=np.array(master_final_spins),
                 master_overlaps=np.array(master_overlaps),
                 master_energies=np.array(master_energies),
                 master_ultra_pcts=np.array(master_ultra_pcts),
                 avg_h_e=avg_h_e, 
                 h_t=master_h_t)
        
        print(f"All {NUM_SEEDS} Seeds Complete in {total_time:.2f} seconds.")
        print(f"Averaged metrics and PDF plots saved using prefix: {FILE_BASE}")
        print("="*60)