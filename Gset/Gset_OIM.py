# -*- coding: utf-8 -*-
"""
LOCAL Execution: G-Set Max-Cut Solver (Tuning Phase Only)
Loads a single TRUBA checkpoint .npz file, evaluates OIM baseline, 
and refines states using multithreaded MH.
"""

import numpy as np
import time
import os
import sys
from numba import njit, prange

# ==========================================
# 1. USER CONTROL PANEL
# ==========================================
# Just type the exact name of the npz file you downloaded from TRUBA
TARGET_NPZ = "checkpoint_G39.txt.npz"  

GLOBAL_SEED = 42

# MH Tuning Parameters 
MH_STEPS = int(1e4)
T_MH_S = 0.1
T_MH_E = 0.01

# ==========================================
# 2. NUMBA DYNAMICS (Metropolis-Hastings)
# ==========================================
@njit(fastmath=True)
def run_mh_epoch(J, N, steps, T_start, T_end, start_spins, seed):
    np.random.seed(int(seed))
    spins = start_spins.copy()
    
    for s in range(steps):
        p = s / (steps - 1) if steps > 1 else 1.0
        T = T_start * (1 - p) + T_end * p
        
        # 1 Step = 1 random spin flip attempt
        i = np.random.randint(0, N)
        
        # Vectorized local field evaluation
        h_i = np.dot(J[i], spins)
        delta_E = 2.0 * spins[i] * h_i
        
        if delta_E <= 0 or np.random.random() < np.exp(-delta_E / T):
            spins[i] *= -1.0
                
    return spins

@njit(parallel=True)
def parallel_mh_stage(J, N, num_replicas, steps, T_s, T_e, start_phis, replica_seeds):
    final_spins = np.zeros((num_replicas, N))
    for r in prange(num_replicas):
        # Convert continuous oscillator phi to discrete spins
        start_spins = np.sign(np.cos(start_phis[r]))
        for i in range(N):
            if start_spins[i] == 0: start_spins[i] = 1.0
        final_spins[r] = run_mh_epoch(J, N, steps, T_s, T_e, start_spins, replica_seeds[r])
    return final_spins

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def calculate_cuts(spins, J):
    W = -J 
    cuts = np.zeros(spins.shape[0])
    for r in range(spins.shape[0]):
        s = spins[r]
        cuts[r] = 0.25 * np.sum(W * (1 - np.outer(s, s)))
    return cuts

# ==========================================
# 4. EXECUTION CONTROLLER
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("LOCAL EXECUTION: Metropolis-Hastings Tuning Phase")
    print("="*60, flush=True)
    
    if not os.path.exists(TARGET_NPZ):
        print(f"[ERROR] The file '{TARGET_NPZ}' was not found in the current directory.")
        sys.exit(1)
        
    print(f"\n--- Loading and Tuning {TARGET_NPZ} ---")
    start_time = time.time()
    
    # Load Checkpoint
    loader = np.load(TARGET_NPZ)
    J = loader['J']
    N = int(loader['N'])
    states = loader['replica_states']
    num_reps = states.shape[0]
    
    # Evaluate Initial OIM State (Pre-MH)
    oim_spins = np.sign(np.cos(states))
    oim_spins[oim_spins == 0] = 1.0
    oim_cuts = calculate_cuts(oim_spins, J)
    oim_max_cut = int(np.max(oim_cuts))
    
    # Initialize deterministic seed sequence isolated for the tuning phase
    tuning_seq = np.random.SeedSequence(GLOBAL_SEED + 5000)
    replica_seeds = tuning_seq.generate_state(num_reps)
    
    # Run MH (Numba prange will utilize all local CPU cores)
    final_spins = parallel_mh_stage(
        J, N, num_reps, MH_STEPS, T_MH_S, T_MH_E, states, replica_seeds
    )
    
    # Calculate Final Cuts
    cuts = calculate_cuts(final_spins, J)
    max_cut = int(np.max(cuts))
    mean_cut = np.mean(cuts)
    
    # Calculate Contribution Percentage
    if max_cut > 0:
        oim_contribution = (oim_max_cut / max_cut) * 100
    else:
        oim_contribution = 0.0
    
    print(f"Results for {TARGET_NPZ} (Tuning time: {time.time() - start_time:.2f}s):")
    print(f"  -> OIM Max Cut (Pre-MH):  {oim_max_cut}")
    print(f"  -> Final Max Cut (MH):    {max_cut}")
    print(f"  -> OIM Contribution:      {oim_contribution:.4f}%")
    print(f"  -> Mean Cut (Final):      {mean_cut:.2f}")
    print("-" * 40)