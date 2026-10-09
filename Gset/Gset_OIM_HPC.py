# -*- coding: utf-8 -*-
"""
TRUBA / LOCAL Execution: G-Set Max-Cut Solver
Split Workflow: MPI Oscillator Search (TRUBA) -> Local MH Tuning (PC)
(Fully Updated with Hybrid Optimization Schedule, Linear Interpolation, and 4-Epoch Dict)
"""

import numpy as np
import time
import os
import sys
from numba import njit, prange
from mpi4py import MPI

# ==========================================
# 1. USER CONTROL PANEL
# ==========================================
RUN_MODE = "search"  # Set to "search" for TRUBA, "tuning" for Local PC

TARGET_FILES = [
    "G1.txt", "G6.txt", "G11.txt", "G14.txt", 
    "G15.txt", "G22.txt", "G23.txt", "G36.txt", "G39.txt"
]

GLOBAL_SEED = 42
TOTAL_REPLICAS = 448  # Scaled to perfectly saturate 8 nodes (32 tasks * 14 CPUs)

# MH Tuning Parameters (Hybrid Logic)
MH_STEPS = int(1e4)
T_MH_S = 0.2
T_MH_E = 0.1

# ==========================================
# 2. NUMBA DYNAMICS (Oscillator & MH)
# ==========================================
@njit(fastmath=True)
def run_osc_epoch(J, N, S_A, S_S, Kn_s, Kn_m, Ks_s, Ks_f, dt, seed, initial_phi):
    np.random.seed(int(seed))
    phi = initial_phi.copy()
    sqrt_dt, total_steps = np.sqrt(dt), S_A + S_S
    
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
            noise[i] = np.random.normal(0.0, 1.0)
            
        phi += (0.5 * (ca*(J@sa) - sa*(J@ca)) - Ks*np.sin(2*phi))*dt + Kn*sqrt_dt*noise

    return phi

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
def parallel_osc_stage(J, N, local_reps, S_A, S_S, Kn_s, Kn_m, Ks_s, Ks_f, dt, replica_seeds, phis):
    new_phis = np.zeros((local_reps, N))
    for r in prange(local_reps):
        new_phis[r] = run_osc_epoch(J, N, S_A, S_S, Kn_s, Kn_m, Ks_s, Ks_f, dt, replica_seeds[r], phis[r])
    return new_phis

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
# 4. EXECUTION CONTROLLER
# ==========================================
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    master_seq = np.random.SeedSequence(GLOBAL_SEED)
    
    # ---------------------------------------------------------
    # MODE A: SEARCH (Runs on TRUBA cluster with MPI)
    # ---------------------------------------------------------
    if RUN_MODE == "search":
        base_reps = TOTAL_REPLICAS // size
        remainder = TOTAL_REPLICAS % size
        start_idx = rank * base_reps + min(rank, remainder)
        local_reps = base_reps + (1 if rank < remainder else 0)
        
        if rank == 0:
            print("="*60)
            print(f"TRUBA MPI EXECUTION: Oscillator Search Phase")
            print(f"Nodes (MPI Ranks): {size} | Total Replicas: {TOTAL_REPLICAS}")
            print("="*60, flush=True)

        epochs = [
            {"anneal": int(15e4), "settle": int(0.0), "T_s": 0.7, "T_m": 0.5, "Ks_s": 0.0, "Ks_f": 0.4, "dt": 1e-3},
            {"anneal": int(15e4), "settle": int(0.0), "T_s": 0.5, "T_m": 0.3, "Ks_s": 0.4, "Ks_f": 0.6, "dt": 1e-2},
            {"anneal": int(25e4), "settle": int(0.0), "T_s": 0.3, "T_m": 0.2, "Ks_s": 0.6, "Ks_f": 0.8, "dt": 1e-2},
            {"anneal": int(25e4), "settle": int(0.0), "T_s": 0.2, "T_m": 0.1, "Ks_s": 0.8, "Ks_f": 0.8, "dt": 5e-2}
        ]

        file_seqs = master_seq.spawn(len(TARGET_FILES))

        for f_idx, g_file in enumerate(TARGET_FILES):
            if not os.path.exists(g_file):
                if rank == 0: print(f"[WARNING] {g_file} not found. Skipping.")
                continue
                
            J, N = load_gset(g_file)
            file_seq = file_seqs[f_idx]
            seq_init, seq_epochs_branch = file_seq.spawn(2)
            
            # Initialize phases correctly across nodes without overlap
            gen_init = np.random.default_rng(seq_init)
            global_phis = gen_init.uniform(0, 2 * np.pi, (TOTAL_REPLICAS, N))
            local_phis = global_phis[start_idx : start_idx + local_reps].copy()
            
            epoch_seqs = seq_epochs_branch.spawn(len(epochs))
            
            if rank == 0:
                print(f"--- Simulating {g_file} (N={N}) ---", flush=True)
            
            epoch_phis = []

            for idx, cfg in enumerate(epochs):
                # Generate deterministic, non-overlapping random states for all nodes
                replica_seeds = epoch_seqs[idx].generate_state(TOTAL_REPLICAS)
                local_seeds = replica_seeds[start_idx : start_idx + local_reps]
                
                local_phis = parallel_osc_stage(
                    J, N, local_reps, int(cfg["anneal"]), int(cfg["settle"]), 
                    np.sqrt(2*cfg["T_s"]), np.sqrt(2*cfg["T_m"]), 
                    cfg["Ks_s"], cfg["Ks_f"], cfg["dt"], 
                    local_seeds, local_phis
                )
                epoch_phis.append(local_phis.copy())

            comm.Barrier()
            all_phis = comm.gather(local_phis, root=0)
            all_epoch_phis = comm.gather(epoch_phis, root=0)
            
            if rank == 0:
                flat_phis = np.vstack(all_phis)
                flat_epoch_phis = [
                    np.vstack([rank_epochs[e] for rank_epochs in all_epoch_phis])
                    for e in range(len(epochs))
                ]
                save_name = f"checkpoint_{g_file}.npz"
                np.savez(
                    save_name,
                    J=J,
                    N=N,
                    replica_states=flat_phis,
                    epoch_1_phis=flat_epoch_phis[0],
                    epoch_2_phis=flat_epoch_phis[1],
                    epoch_3_phis=flat_epoch_phis[2],
                    epoch_4_phis=flat_epoch_phis[3]
                )
                print(f"Saved: {save_name}")

    # ---------------------------------------------------------
    # MODE B: TUNING (Runs Locally, Multithreaded via Numba)
    # ---------------------------------------------------------
    elif RUN_MODE == "tuning":
        # Confine to rank 0 to prevent issues if accidentally run with mpirun locally
        if rank == 0:
            print("="*60)
            print(f"LOCAL EXECUTION: Metropolis-Hastings Tuning Phase")
            print("="*60, flush=True)
            
            tuning_seq = np.random.SeedSequence(GLOBAL_SEED + 5000)
            file_tuning_seqs = tuning_seq.spawn(len(TARGET_FILES))
            
            for f_idx, g_file in enumerate(TARGET_FILES):
                cp_file = f"checkpoint_{g_file}.npz"
                if not os.path.exists(cp_file):
                    continue
                    
                print(f"\n--- Tuning {g_file} ---")
                start_time = time.time()
                
                # Load Checkpoint
                loader = np.load(cp_file)
                J = loader['J']
                N = int(loader['N'])
                states = loader['replica_states']
                num_reps = states.shape[0]
                
                # Generate exact seeds for the local MH replicas
                replica_seeds = file_tuning_seqs[f_idx].generate_state(num_reps)
                
                # Run MH (Numba prange will utilize all local CPU cores)
                final_spins = parallel_mh_stage(
                    J, N, num_reps, MH_STEPS, T_MH_S, T_MH_E, states, replica_seeds
                )
                
                # Calculate Cuts
                cuts = calculate_cuts(final_spins, J)
                max_cut = int(np.max(cuts))
                mean_cut = np.mean(cuts)
                
                print(f"Results for {g_file} (Tuning time: {time.time() - start_time:.2f}s):")
                print(f"  -> Max Cut Found: {max_cut}")
                print(f"  -> Mean Cut:      {mean_cut:.2f}")
                print("-" * 40)