# -*- coding: utf-8 -*-
"""
LOCAL Execution: Pure Metropolis-Hastings G-Set Max-Cut Solver
Loads a single G-Set .txt file and solves using exact step-based MH logic.
"""

import numpy as np
import time
import os
import sys
from numba import njit, prange

# ==========================================
# 1. USER CONTROL PANEL
# ==========================================
# Type the exact name of the G-set text file you want to solve
TARGET_TXT = "G36.txt"  

GLOBAL_SEED = 42
num_replicas = 448  # Match the archived OIM checkpoint population
sample_rate = 500  

# Exact Step/dt Logic Preserved
dt = 1e-3
Anneal_Time = 10e4 * dt  
Settle_Time = 1e4 * dt
Measure_Time = 1e4 * dt  

Steps_Anneal = int(Anneal_Time / dt)
Steps_Settle = int(Settle_Time / dt)
Steps_Measure = int(Measure_Time / dt)

# MH Thermodynamic Parameters
T_start = 0.7
T_measure = 0.01

# ==========================================
# 2. HELPER FUNCTIONS (G-Set Loader & Cuts)
# ==========================================
def load_gset(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    n_nodes, n_edges = map(int, lines[0].split())
    J = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            u, v, w = int(parts[0])-1, int(parts[1])-1, float(parts[2])
            J[u, v] = -w
            J[v, u] = -w
            
    return J, n_nodes

def calculate_cuts(spins, J):
    W = -J 
    cuts = np.zeros(spins.shape[0])
    for r in range(spins.shape[0]):
        s = spins[r]
        cuts[r] = 0.25 * np.sum(W * (1 - np.outer(s, s)))
    return cuts

# ==========================================
# 3. NUMBA DYNAMICS (Metropolis-Hastings)
# ==========================================
@njit(fastmath=True)
def run_metropolis_full_simulation(J, N, Steps_Anneal, Steps_Settle, Steps_Measure, T_start, T_measure, seed):
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
        h_i = 0.0
        for j in range(N):
            h_i += J[i, j] * spins[j]
        
        delta_E = 2.0 * spins[i] * h_i
        if delta_E <= 0 or np.random.random() < np.exp(-delta_E * beta):
            spins[i] *= -1.0
            
        if step >= (Steps_Anneal + Steps_Settle):
            spin_sum_accumulator += spins

    return spin_sum_accumulator / Steps_Measure

@njit(parallel=True)
def run_all_replicas_parallel(J, N, num_replicas, Steps_Anneal, Steps_Settle, Steps_Measure, T_start, T_measure, base_seed):
    spin_results = np.zeros((num_replicas, N))
    
    for r in prange(num_replicas):
        avg_spins = run_metropolis_full_simulation(J, N, Steps_Anneal, Steps_Settle, Steps_Measure, T_start, T_measure, base_seed + r)
        spin_results[r] = avg_spins
        
    return spin_results

# ==========================================
# 4. EXECUTION CONTROLLER
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("LOCAL EXECUTION: Pure MH G-Set Max-Cut Solver")
    print("="*60, flush=True)
    
    if not os.path.exists(TARGET_TXT):
        print(f"[ERROR] The file '{TARGET_TXT}' was not found in the current directory.")
        sys.exit(1)
        
    print(f"\n--- Loading and Solving {TARGET_TXT} ---")
    
    # Load Graph
    J, N = load_gset(TARGET_TXT)
    print(f"Graph loaded successfully: {N} nodes.")
    
    print(f"Starting Metropolis simulation: N={N}, Replicas={num_replicas}, Seed={GLOBAL_SEED}")
    start_time = time.time()
    
    # Run Parallel MH Numba Execution
    all_replicas_avg = run_all_replicas_parallel(
        J, N, num_replicas, Steps_Anneal, Steps_Settle, Steps_Measure, T_start, T_measure, GLOBAL_SEED
    )
    
    print(f"Simulation finished in {time.time() - start_time:.2f} seconds.")

    # Convert accumulated averages to discrete spins
    frozen_states = np.sign(all_replicas_avg)
    for r in range(num_replicas):
        for i in range(N):
            if frozen_states[r, i] == 0: 
                frozen_states[r, i] = 1.0

    # Calculate Max-Cut values
    cuts = calculate_cuts(frozen_states, J)
    max_cut = int(np.max(cuts))
    mean_cut = np.mean(cuts)

    print("\n" + "="*60)
    print(f"MAX-CUT RESULTS FOR {TARGET_TXT}")
    print("="*60)
    print(f"  -> Max Cut Found: {max_cut}")
    print(f"  -> Mean Cut:      {mean_cut:.2f}")
    print("="*60)
