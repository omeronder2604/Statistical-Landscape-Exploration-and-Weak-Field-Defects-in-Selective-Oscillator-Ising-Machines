# -*- coding: utf-8 -*-
"""
Generates individual Comparative Dendrograms as separate PDF files.
Loads the exact J matrix from the Hybrid NPZ data to ensure physics parity,
runs the Pure MH Baseline dynamically via Numba, then plots 
Pure MH, Post-Oscillator, and Post-MH Refinement topologies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from numba import njit, prange
import time
import os
import sys

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# ==========================================
# 1. STANDARDIZED FONT & PLOT CONFIGURATION
# ==========================================
plt.rc('font', family='serif', size=14)
plt.rc('mathtext', fontset='cm') 
plt.rc('axes', labelsize=16, labelweight='normal', linewidth=1.5)
plt.rc('xtick', labelsize=14, direction='in', top=True)    # Ticks point inward
plt.rc('ytick', labelsize=14, direction='in', right=True)  # Ticks point inward
plt.rc('legend', fontsize=18, framealpha=1.0, edgecolor='black')

latex_fmt = FuncFormatter(lambda x, pos: f"${x:g}$")

# ==========================================
# 2. CONFIGURATION & MH ENGINE
# ==========================================
GLOBAL_SEED = 52
N = 200
dt = 1e-3
Anneal_Time = 10e4 * dt  
Settle_Time = 1e4 * dt
Measure_Time = 1e4 * dt  
num_replicas = 402
sample_rate = 500  

Steps_Anneal = int(Anneal_Time / dt)
Steps_Settle = int(Settle_Time / dt)
Steps_Measure = int(Measure_Time / dt)

T_start = 0.7
T_measure = 0.1

# Target NPZ file from the Hybrid run (Single Seed)
TARGET_NPZ = "AllToAll_Hybrid_EA.npz"

@njit(fastmath=True)
def run_metropolis_full_simulation(J, N, Steps_Anneal, Steps_Settle, Steps_Measure, T_start, T_measure, seed, sample_rate):
    np.random.seed(seed)
    spins = np.where(np.random.random(N) > 0.5, 1.0, -1.0)
    spin_sum_accumulator = np.zeros(N)
    
    total_steps = Steps_Anneal + Steps_Settle + Steps_Measure

    for step in range(total_steps):
        if step < Steps_Anneal:
            progress = step / Steps_Anneal
            current_T = T_start * (1 - progress) + T_measure * progress
        else:
            current_T = T_measure
            
        beta = 1.0 / current_T
        i = np.random.randint(0, N)
        h_i = np.dot(J[i], spins)
        
        delta_E = 2.0 * spins[i] * h_i
        if delta_E <= 0 or np.random.random() < np.exp(-delta_E * beta):
            spins[i] *= -1.0
            
        if step >= (Steps_Anneal + Steps_Settle):
            spin_sum_accumulator += spins

    return spin_sum_accumulator / Steps_Measure

@njit(parallel=True)
def run_all_replicas_parallel(J, N, num_replicas, Steps_Anneal, Steps_Settle, Steps_Measure, T_start, T_measure, replica_seeds, sample_rate):
    spin_results = np.zeros((num_replicas, N))
    for r in prange(num_replicas):
        avg_spins = run_metropolis_full_simulation(J, N, Steps_Anneal, Steps_Settle, Steps_Measure, T_start, T_measure, replica_seeds[r], sample_rate)
        spin_results[r] = avg_spins
        
    return spin_results

# ==========================================
# 3. INDIVIDUAL PLOTTING FUNCTION
# ==========================================
def plot_dendrogram(spins, filename):
    print(f"Generating {filename}...")
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Calculate standard Hamming distance
    dist_vec_H = pdist(spins, metric='hamming')
    # Quotient out the Z_2 symmetry: d_Z2 = min(d_H, 1 - d_H)
    dist_vec = np.minimum(dist_vec_H, 1.0 - dist_vec_H)
    
    Z = linkage(dist_vec, method='average')

    # no_labels=True entirely suppresses the tiny leaf enumerations
    dendrogram(Z, truncate_mode='lastp', p=40, no_labels=True, ax=ax)
    
    ax.set_ylabel(r"$\mathbb{Z}_2$-Quotiented Distance", fontsize=26, fontweight='normal')
    
    ax.yaxis.set_major_formatter(latex_fmt)
    ax.tick_params(axis='y', labelsize=22)
    
    # Remove x-axis tick marks entirely for the clean look
    ax.set_xticks([]) 
    
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
    
    # Lock the Y-axis so all 3 plots scale identically for visual comparison
    ax.set_ylim(0, 0.55)
    
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

# ==========================================
# 4. EXECUTION LOGIC
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("Generating Separate Dendrogram Plots")
    print("="*60)
    
    # 1. Load Hybrid Run Data to extract the TRUE physics matrix
    if not os.path.exists(TARGET_NPZ):
        print(f"[ERROR] Could not find Hybrid NPZ file: {TARGET_NPZ}")
        print("Please ensure you have run the Hybrid simulation for this seed first.")
        sys.exit(1)
        
    print(f"Loading Hybrid NPZ Data from '{TARGET_NPZ}'...")
    data = np.load(TARGET_NPZ)
    J = data['J']  # Extracting exact mathematical disorder matrix
    osc_spins = data['osc_spins']
    final_spins = data['final_spins']
    
    # 2. Generate Pure MH Baseline using the extracted J
    print("Running Pure MH Baseline Simulation...")
    
    # Using GLOBAL_SEED + 1 to perfectly match energy_prog_3.py methodology
    master_seq = np.random.SeedSequence(GLOBAL_SEED + 1)
    replica_seeds = master_seq.generate_state(num_replicas)
    
    start_time = time.time()
    mh_avg_spins = run_all_replicas_parallel(
        J, N, num_replicas, Steps_Anneal, Steps_Settle, Steps_Measure, 
        T_start, T_measure, replica_seeds, sample_rate
    )
    
    # Discretize MH states
    mh_frozen_states = np.sign(mh_avg_spins)
    mh_frozen_states[mh_frozen_states == 0.0] = 1.0
    print(f"MH Simulation finished in {time.time() - start_time:.2f} seconds.")
    
    # 3. Generate the 3 separate plots
    print("\nRendering individual plots...")
    
    plot_dendrogram(
        mh_frozen_states, 
        "Dendrogram_1_PureMH.pdf"
    )
    
    plot_dendrogram(
        osc_spins, 
        "Dendrogram_2_PostOscillator.pdf"
    )
    
    plot_dendrogram(
        final_spins, 
        "Dendrogram_3_PostMH.pdf"
    )
    
    print("="*60)
    print("Success! All 3 PDF plots saved independently with exact physics parity.")
    print("="*60)