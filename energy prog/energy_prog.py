# -*- coding: utf-8 -*-
"""
LOCAL Execution: Single Seed All-to-All Gaussian Spin Glass 
Pure Metropolis-Hastings (Step-Based, Full State Storage, Multi-core Parallelized)
Generates targeted plots: Oscillator-Only Progression & Comparative MH Polish (with Inset)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, LogFormatterMathtext
import time
import os
from numba import njit, prange

# ==========================================
# 1. STANDARDIZED FONT & PLOT CONFIGURATION
# ==========================================
plt.rc('font', family='serif', size=14)
plt.rc('mathtext', fontset='cm') 
plt.rc('axes', labelsize=16, labelweight='normal', linewidth=1.5)
plt.rc('xtick', labelsize=14, direction='in', top=True)
plt.rc('ytick', labelsize=14, direction='in', right=True)
plt.rc('legend', fontsize=14, framealpha=1.0, edgecolor='black')

latex_fmt = FuncFormatter(lambda x, pos: f"${x:g}$")

# ==========================================
# 2. USER CONTROL PANEL
# ==========================================
GLOBAL_SEED = 52 
N = 200
num_replicas = 402  
FILE_BASE = f"AllToAll_PureMH_Seed{GLOBAL_SEED}"
HYBRID_NPZ = "AllToAll_Hybrid_EA.npz" # Must be in the same folder matching your seed run
sample_rate = 500
J0_VAL = 0.0

# Original Step/dt Logic
dt = 1e-3
Anneal_Time = 10e4 * dt
Settle_Time = 1e4 * dt
Measure_Time = 1e4 * dt  

Steps_Anneal = int(Anneal_Time / dt)
Steps_Settle = int(Settle_Time / dt)
Steps_Measure = int(Measure_Time / dt)
total_steps = Steps_Anneal + Steps_Settle + Steps_Measure

# MH Thermodynamic Parameters
T_start = 0.7   
T_measure = 0.1 

# ==========================================
# 3. CENTRALIZED RNG & STATE GENERATION (Synced with EA.py style)
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

# ==========================================
# 4. NUMBA DYNAMICS (Metropolis-Hastings Core)
# ==========================================
@njit(fastmath=True)
def run_metropolis_full_simulation(J, N, Steps_Anneal, Steps_Settle, Steps_Measure, T_start, T_measure, seed, sample_rate):
    np.random.seed(int(seed))
    spins = np.where(np.random.random(N) > 0.5, 1.0, -1.0)
    spin_sum_accumulator = np.zeros(N)

    t_steps = Steps_Anneal + Steps_Settle + Steps_Measure
    energy_history = np.zeros(t_steps // sample_rate)
    temp_history = np.zeros(t_steps // sample_rate) 

    for step in range(t_steps):
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
        h_i = np.dot(J[i], spins) 
        delta_E = 2.0 * spins[i] * h_i

        if delta_E <= 0 or np.random.random() < np.exp(-delta_E * beta):
            spins[i] *= -1.0

        if step >= (Steps_Anneal + Steps_Settle):
            spin_sum_accumulator += spins

    return spin_sum_accumulator / Steps_Measure, energy_history, temp_history

@njit(parallel=True)
def run_local_replicas_parallel(J, N, num_reps, Steps_Anneal, Steps_Settle, Steps_Measure, T_start, T_measure, local_seeds, sample_rate):
    spin_results = np.zeros((num_reps, N))
    
    t_steps = Steps_Anneal + Steps_Settle + Steps_Measure
    h_len = t_steps // sample_rate
    e_b = np.zeros((num_reps, h_len))
    t_b = np.zeros((num_reps, h_len))

    for r in prange(num_reps):
        avg_spins, sample_e_hist, sample_t_hist = run_metropolis_full_simulation(
            J, N, Steps_Anneal, Steps_Settle, Steps_Measure, T_start, T_measure, local_seeds[r], sample_rate
        )
        spin_results[r] = avg_spins
        e_b[r] = sample_e_hist
        t_b[r] = sample_t_hist

    return spin_results, e_b, t_b

# ==========================================
# 5. TARGETED PLOTTING FUNCTIONS
# ==========================================
def plot_oscillator_only(h_e_data, h_t, h_ks, pts, filename):
    print(f"Generating {filename}...")
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    
    end_idx = pts[-1]
    
    dt_list = [1e-3, 1e-2, 1e-2, 5e-2]
    osc_time_axis = np.zeros(end_idx)
    current_time = 0.0
    
    for i in range(len(pts) - 1):
        start = pts[i]
        end = pts[i+1]
        dt_val = dt_list[i] if i < len(dt_list) else 1e-3
        t_per_sample = sample_rate * dt_val
        osc_time_axis[start:end] = current_time + np.arange(end - start) * t_per_sample
        current_time += (end - start) * t_per_sample
    
    for i in range(len(pts) - 1):
        start = pts[i]
        end = pts[i+1]
        label = f"Epoch {i+1}"
        x_vals = osc_time_axis[start:end]
        
        if h_e_data.ndim == 2:
            y_mean = np.mean(h_e_data[:, start:end], axis=0)
            line_plot = ax1.plot(x_vals, y_mean, label=label, linewidth=2.5, zorder=4)
            epoch_color = line_plot[0].get_color()
            
            y_min = np.min(h_e_data[:, start:end], axis=0)
            y_max = np.max(h_e_data[:, start:end], axis=0)
            ax1.fill_between(x_vals, y_min, y_max, color=epoch_color, alpha=0.25, edgecolor='none', zorder=3)
        else:
            ax1.plot(x_vals, h_e_data[start:end], label=label, linewidth=2.5, zorder=4)

    for pt in pts[1:-1]:
        ax1.axvline(x=osc_time_axis[pt], color='gray', linestyle='--', linewidth=1.5, zorder=2)

    ax1.set_xlim(0, osc_time_axis[-1])
    
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
    
    ax1.set_xlabel(r"Dimensionless oscillator time $t$", fontsize=20, fontweight='normal')
    ax1.set_ylabel(r"Energy $E$", color='black', fontsize=20, fontweight='normal')
    ax1.tick_params(axis='both', labelcolor='black', labelsize=18)
    
    # Horizontal grid enabled cleanly
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5) 
    
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Amplitudes $K_n, K_s$", color='black', fontsize=20, fontweight='normal')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=18)
    ax2.yaxis.set_major_formatter(latex_fmt)
    
    h_kn = np.sqrt(2 * np.array(h_t[:end_idx]))
    ax2.plot(osc_time_axis, h_kn, color='navy', linestyle='--', alpha=0.8, linewidth=2, label='Noise $K_n$')
    
    if len(h_ks) > 0 and len(h_ks) >= end_idx:
        ax2.plot(osc_time_axis, h_ks[:end_idx], color='forestgreen', linestyle=':', alpha=0.8, linewidth=2, label='SHIL $K_s$')
        
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', ncol=2, 
               frameon=True, facecolor='white', framealpha=1.0, edgecolor='black', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def plot_mh_comparison(pure_mh_e, pure_mh_t, hybrid_mh_e, filename):
    print(f"Generating {filename}...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    pure_is_2d = (pure_mh_e.ndim == 2)
    p_num_steps = pure_mh_e.shape[-1] if pure_is_2d else len(pure_mh_e)
    p_step_axis = np.arange(p_num_steps) * sample_rate
    
    if pure_is_2d:
        p_mean = np.mean(pure_mh_e, axis=0)
        ax1.plot(p_step_axis, p_mean, color='crimson', linewidth=2.5, label='Pure MH Baseline', zorder=4)
        p_min, p_max = np.min(pure_mh_e, axis=0), np.max(pure_mh_e, axis=0)
        ax1.fill_between(p_step_axis, p_min, p_max, color='crimson', alpha=0.20, edgecolor='none', zorder=3)
    else:
        ax1.plot(p_step_axis, pure_mh_e, color='crimson', linewidth=2.5, label='Pure MH Baseline', zorder=4)

    ax1.set_xscale('log')
    ax1.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax1.xaxis.set_major_formatter(LogFormatterMathtext())
    ax1.yaxis.set_major_formatter(latex_fmt)
    
    ax1.set_xlabel(r"MH proposals per replica", fontsize=20, fontweight='normal')
    ax1.set_ylabel(r"Energy $E$", color='black', fontsize=20, fontweight='normal')
    ax1.tick_params(axis='both', labelcolor='black', labelsize=18)
    
    # Horizontal grid enabled cleanly
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5) 
    
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Temperature $T_{\rm MH}$", color='black', fontsize=20, fontweight='normal')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=18)
    ax2.yaxis.set_major_formatter(latex_fmt)
    ax2.plot(p_step_axis, pure_mh_t, color='navy', linestyle='--', linewidth=2.5, label=r'Temp $T_{\rm MH}$', alpha=0.8, zorder=1)
        
    if hybrid_mh_e is not None:
        axin = ax1.inset_axes([0.08, 0.48, 0.35, 0.35])
        
        hybrid_is_2d = (hybrid_mh_e.ndim == 2)
        h_num_steps = hybrid_mh_e.shape[-1] if hybrid_is_2d else len(hybrid_mh_e)
        
        h_step_axis = np.arange(h_num_steps) * sample_rate 
        
        if hybrid_is_2d:
            h_mean = np.mean(hybrid_mh_e, axis=0)
            axin.plot(h_step_axis, h_mean, color='dodgerblue', linewidth=2.5, label='Hybrid MH Polish')
            h_min, h_max = np.min(hybrid_mh_e, axis=0), np.max(hybrid_mh_e, axis=0)
            axin.fill_between(h_step_axis, h_min, h_max, color='dodgerblue', alpha=0.25, edgecolor='none')
        else:
            axin.plot(h_step_axis, hybrid_mh_e, color='dodgerblue', linewidth=2.5, label='Hybrid MH Polish')

        abs_min, abs_max = np.min(hybrid_mh_e), np.max(hybrid_mh_e)
        
        v_pad = (abs_max - abs_min) * 0.6 
        if v_pad == 0: v_pad = 5.0
        axin.set_ylim(abs_min - v_pad, abs_max + v_pad)
        
        x_pad = (h_step_axis[-1] - h_step_axis[0]) * 0.05
        if x_pad > 0:
            axin.set_xlim(h_step_axis[0] - x_pad, h_step_axis[-1] + x_pad)

        text_bbox = dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.3')
        axin.text(0.95, 0.95, "Defect-Heal Polish", transform=axin.transAxes, 
                  ha='right', va='top', fontsize=12, bbox=text_bbox)
        
        axin.set_xlabel("MH proposals", fontsize=12, fontweight='normal')
        
        def inset_x_formatter(x, pos):
            val = x / 1e4
            locs = axin.xaxis.get_majorticklocs()
            xlim = axin.get_xlim()
            visible_locs = [loc for loc in locs if xlim[0] <= loc <= xlim[1]]
            
            if len(visible_locs) > 0 and np.isclose(x, visible_locs[-1]):
                return f"${val:g} \\times 10^4$"
            return f"${val:g}$"
            
        axin.xaxis.set_major_formatter(FuncFormatter(inset_x_formatter))
        axin.yaxis.set_major_formatter(latex_fmt)
        axin.tick_params(axis='both', labelsize=12)
        
        # Horizontal grid for inset
        axin.grid(True, axis='y', linestyle=':', alpha=0.6)
        
        lines_in, labels_in = axin.get_legend_handles_labels()
    else:
        lines_in, labels_in = [], []

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    
    ax2.legend(lines_1 + lines_2 + lines_in, labels_1 + labels_2 + labels_in, 
               loc='upper right', ncol=1, frameon=True, facecolor='white', framealpha=1.0, edgecolor='black', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

# ==========================================
# 6. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("LOCAL MULTI-CORE EXECUTION: Pure Metropolis-Hastings (Seed 52)")
    print("="*60, flush=True)

    # Use the robust EA.py style Master Seed Architecture
    seq_J, seq_init, epoch_sequences, seq_mh_branch = setup_master_rng(GLOBAL_SEED, 4)
    J = generate_j_matrix(seq_J, N, J0=J0_VAL, J_std=1.0)
    
    # Generate replica seeds matching the layout
    master_seq_reps = np.random.SeedSequence(GLOBAL_SEED + 1)
    replica_seeds = master_seq_reps.generate_state(num_replicas)
    
    grand_start_time = time.time()
    print(f"Initiating Pure MH Steps... (Anneal: {Steps_Anneal}, Settle: {Steps_Settle}, Measure: {Steps_Measure})")
    
    local_replicas_avg, pure_replica_h_e, pure_replica_h_t = run_local_replicas_parallel(
        J, N, num_replicas, Steps_Anneal, Steps_Settle, Steps_Measure, 
        T_start, T_measure, replica_seeds, sample_rate
    )
    print(f"Pure MH Simulation finished in {time.time() - grand_start_time:.2f} seconds.")

    pure_h_t = pure_replica_h_t[0]

    # --- 7. LOAD HYBRID DATA & PLOT ---
    print("\n" + "="*60)
    print("EXTRACTING HYBRID DATA & GENERATING PLOTS")
    print("="*60)
    
    if os.path.exists(HYBRID_NPZ):
        hybrid_loader = np.load(HYBRID_NPZ)
        
        if 'replica_h_e' in hybrid_loader:
            hybrid_h_e_full = hybrid_loader['replica_h_e']
        elif 'all_seeds_replica_h_e' in hybrid_loader:
            hybrid_h_e_full = hybrid_loader['all_seeds_replica_h_e']
        else:
            hybrid_h_e_full = hybrid_loader['avg_h_e']
            
        hybrid_h_t = hybrid_loader['h_t']
        hybrid_pts = hybrid_loader['pts']
        
        epochs_cfg = [
            {"anneal": int(15e4), "settle": int(0.0), "Ks_s": 0.0, "Ks_f": 0.4},
            {"anneal": int(15e4), "settle": int(0.0), "Ks_s": 0.4, "Ks_f": 0.6},
            {"anneal": int(25e4), "settle": int(0.0), "Ks_s": 0.6, "Ks_f": 0.8},
            {"anneal": int(25e4), "settle": int(0.0), "Ks_s": 0.8, "Ks_f": 0.8}
        ]
        hybrid_h_ks = []
        for cfg in epochs_cfg:
            for step in range(0, cfg["anneal"] + cfg["settle"], sample_rate):
                p_tot = step / ((cfg["anneal"] + cfg["settle"]) - 1) if (cfg["anneal"] + cfg["settle"]) > 1 else 1.0
                hybrid_h_ks.append(cfg["Ks_s"] * (1 - p_tot) + cfg["Ks_f"] * p_tot)
        hybrid_h_ks = np.array(hybrid_h_ks)
        
        start_mh_idx = hybrid_pts[-1]
        if hybrid_h_e_full.ndim == 2:
            osc_h_e = hybrid_h_e_full[:, :start_mh_idx]
            hybrid_mh_e = hybrid_h_e_full[:, start_mh_idx:]
        else:
            osc_h_e = hybrid_h_e_full[:start_mh_idx]
            hybrid_mh_e = hybrid_h_e_full[start_mh_idx:]
            
        print(f"Successfully sliced Hybrid Data. (Oscillator steps: {start_mh_idx * sample_rate}, MH Steps: {hybrid_mh_e.shape[-1] * sample_rate})")

        plot_oscillator_only(osc_h_e, hybrid_h_t, hybrid_h_ks, hybrid_pts, f"{FILE_BASE}_Plot_1_Oscillator_Only.pdf")
        plot_mh_comparison(pure_replica_h_e, pure_h_t, hybrid_mh_e, f"{FILE_BASE}_Plot_2_MH_Comparison.pdf")
        
        print("\nAll target plots generated successfully!")
        
        # --- OUTPUT MINIMUM ENERGIES FOR CAPTION ---
        print("\n" + "="*60)
        print("CAPTION DATA: MINIMUM ENERGIES REACHED")
        print("="*60)
        print(f"OIM (Oscillator) Minimum Energy : {np.min(osc_h_e):.4f}")
        print(f"Pure MH Baseline Minimum Energy : {np.min(pure_replica_h_e):.4f}")
        print(f"Hybrid MH Polish Minimum Energy : {np.min(hybrid_mh_e):.4f}")
        print("="*60 + "\n")
        
    else:
        print(f"\n[ERROR] '{HYBRID_NPZ}' not found. Cannot generate comparative plots. Please place the file in the directory and re-run.")