import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ==========================================
# 1. STANDARDIZED FONT & PLOT CONFIGURATION
# ==========================================
plt.rc('font', family='serif', size=16)
plt.rc('mathtext', fontset='cm') 
plt.rc('axes', labelsize=20, labelweight='normal', linewidth=1.5)
plt.rc('xtick', labelsize=18, direction='in', top=True)    # PRE inward ticks
plt.rc('ytick', labelsize=18, direction='in', right=True)  # PRE inward ticks
plt.rc('legend', fontsize=16, framealpha=0.95, edgecolor='black')

latex_fmt = FuncFormatter(lambda x, pos: f"${x:g}$")

# ==========================================
# 2. GENERATE SCHEDULE DATA
# ==========================================
epochs = [
    {"anneal": 15e4 * 1e-3, "T_s": 0.7, "T_m": 0.5, "Ks_s": 0.0, "Ks_f": 0.4}, # 150
    {"anneal": 15e4 * 1e-2, "T_s": 0.5, "T_m": 0.3, "Ks_s": 0.4, "Ks_f": 0.6}, # 1500
    {"anneal": 25e4 * 1e-2, "T_s": 0.3, "T_m": 0.2, "Ks_s": 0.6, "Ks_f": 0.8}, # 2500
    {"anneal": 25e4 * 5e-2, "T_s": 0.2, "T_m": 0.1, "Ks_s": 0.8, "Ks_f": 0.8}  # 12500
]

T_all, Ks_all, time_all = [], [], []
epoch_boundaries = []
current_time = 0.0

# Number of points per epoch to ensure a smooth curve
num_points = 1000 

# Oscillator Phases
for cfg in epochs:
    anneal_duration = cfg["anneal"]
    if anneal_duration == 0:
        continue
        
    # p_total goes from 0 to 1 over the epoch
    p_total = np.linspace(0, 1, num_points, endpoint=False)
    
    # Linear transition for Ks
    Ks_epoch = cfg["Ks_s"] * (1 - p_total) + cfg["Ks_f"] * p_total
    
    # Direct quadratic decay for T_JN to ensure a visible curve
    T_epoch = (cfg["T_s"] - cfg["T_m"]) * (1 - p_total)**2 + cfg["T_m"]
    
    # Direct time vector calculation for the epoch based on the anneal duration
    time_epoch = np.linspace(current_time, current_time + anneal_duration, num_points, endpoint=False)
    
    T_all.extend(T_epoch)
    Ks_all.extend(Ks_epoch)
    time_all.extend(time_epoch)
    
    current_time += anneal_duration
    epoch_boundaries.append(current_time)

# ==========================================
# 3. PLOTTING
# ==========================================
fig, ax = plt.subplots(figsize=(18, 5))

# Plot the main lines against the dimensionless time
ax.plot(time_all, T_all, color='coral', linewidth=3, label=r'$T_{\rm JN}$', zorder=3)
ax.plot(time_all, Ks_all, color='royalblue', linewidth=3, label=r'$K_s$', zorder=3)

# Shading Phase Backgrounds and Annotations
last_bound = 0.0
for i, cfg in enumerate(epochs):
    anneal_duration = cfg["anneal"]
    anneal_end = last_bound + anneal_duration
        
    # Epoch separator
    ax.axvline(x=anneal_end, color='gray', linestyle='--', linewidth=1.5, zorder=2)
    
    # Add S/R (Selection/Repopulation) marker after each epoch
    ax.text(anneal_end, 0.92, "S/R", transform=ax.get_xaxis_transform(), 
            ha='center', va='center', fontsize=14, color='black',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2))
    
    # Add Text above plot
    center_x = last_bound + (anneal_duration / 2)
    
    # Shift text manually for Epoch 1 and 2 to prevent overlapping
    if i == 0:
        center_x = -300
    elif i == 1:
        center_x = 1200
        
    ax.text(center_x, 1.05, f"Epoch {i+1}", transform=ax.get_xaxis_transform(), 
            ha='center', va='bottom', fontsize=15)
    
    last_bound = anneal_end

# Formatting Limits
ax.set_xlim(-1500, epoch_boundaries[-1] + 500)

# Custom formatter: Appends '\times 10^4' strictly to the very last tick on the axis
def custom_x_formatter(x, pos):
    val = x / 1e4
    locs = ax.xaxis.get_majorticklocs()
    xlim = ax.get_xlim()
    visible_locs = [loc for loc in locs if xlim[0] <= loc <= xlim[1]]
    
    if len(visible_locs) > 0 and np.isclose(x, visible_locs[-1]):
        return f"${val:g} \\times 10^4$"
    return f"${val:g}$"

# Apply formatting
ax.xaxis.set_major_formatter(FuncFormatter(custom_x_formatter))
ax.yaxis.set_major_formatter(latex_fmt)

# Set strictly dimensionless label (multiplier moved to tick)
ax.set_xlabel(r"Dimensionless oscillator time $t$")
ax.set_ylabel("Annealing parameter")

ax.grid(True, linestyle='-', alpha=0.3, zorder=0)

# Legend Setup
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9), framealpha=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Export
filename = "OIM+_Schedule_Normalized.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
plt.close()

print(f"Plot successfully generated and saved as: {filename}")