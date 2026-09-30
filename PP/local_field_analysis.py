#!/usr/bin/env python3
"""
Reproduce Fig. 6-style local-field and correction-kernel analysis
from the averaged SK/OIM NPZ bundle.

Required NPZ keys:
    master_J
    master_osc_spins
    master_final_spins

Main outputs:
    - local-field statistics (panel a)
    - spin-correction statistics
    - empirical correction kernel K_emp(lambda) with bootstrap CIs (panel b)
    - frozen-field model K_frz(lambda)
    - panel (b) figure
    - combined two-panel Fig. 6-style figure
    - CSV tables for all headline numbers

Example:
    python PP/local_field_analysis.py raw_data/AllToAll_Hybrid_EA_SeedAvg_AveragedData.npz --outdir PP/outputs
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Panel (a) settings
# ---------------------------------------------------------------------
SIGMA_PP_ORIGINAL = 1.17
FIELD_SCALE = 1.0 / np.sqrt(2.0)

PLOT_BINS = 50
PLOT_XMAX = 4.0
OIM_COLOR = "tab:orange"
DH_OIM_COLOR = "tab:blue"

FIT_BIN_WIDTH = 0.001
FIT_HMAX = 0.4

# ---------------------------------------------------------------------
# Panel (b) settings
# ---------------------------------------------------------------------
# Fixed bins, denser near lambda = 0 where the correction curve changes rapidly.
KERNEL_BIN_EDGES = np.array([
    -1.10, -0.80, -0.60, -0.45, -0.30, -0.18, -0.08, 0.00,
     0.10,  0.22,  0.38,  0.58,  0.82,  1.10,  1.45,  2.05
], dtype=float)

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 123456

U_MAX = 50.0
T_START = 0.2
T_END = 0.1
FROZEN_FIELD_XMIN = -1.05
FROZEN_FIELD_XMAX = 2.05
FROZEN_FIELD_NPTS = 400
FROZEN_FIELD_NSTEPS = 5000
K_FRZ_COLOR = "tab:orange"
K_EMP_COLOR = "tab:blue"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def rayleigh_pdf(h, sigma):
    h = np.asarray(h, dtype=float)
    return (h / sigma**2) * np.exp(-h**2 / (2.0 * sigma**2))


def load_data(npz_path):
    d = np.load(npz_path)
    required = ("master_J", "master_osc_spins", "master_final_spins")
    missing = [k for k in required if k not in d.files]
    if missing:
        raise KeyError(f"Missing required NPZ keys: {missing}; available: {d.files}")

    J = np.asarray(d["master_J"], dtype=float)
    S_oim = np.asarray(d["master_osc_spins"], dtype=float)
    S_mh = np.asarray(d["master_final_spins"], dtype=float)

    if J.ndim != 3 or J.shape[1] != J.shape[2]:
        raise ValueError(f"master_J must have shape (seeds,N,N); got {J.shape}")
    if S_oim.ndim != 3 or S_mh.ndim != 3:
        raise ValueError("spin arrays must have shape (seeds,replicas,N)")
    if S_oim.shape != S_mh.shape:
        raise ValueError(
            f"OIM and DH-OIM+ spin arrays must have identical shapes; "
            f"got {S_oim.shape} and {S_mh.shape}"
        )
    if J.shape[0] != S_oim.shape[0] or J.shape[0] != S_mh.shape[0]:
        raise ValueError("seed dimension mismatch")
    if J.shape[1] != S_oim.shape[2] or J.shape[1] != S_mh.shape[2]:
        raise ValueError("spin dimension mismatch")

    return J, S_oim, S_mh


def local_fields(J, spins):
    return np.einsum("sij,srj->sri", J, spins, optimize=True)


def rayleigh_mle_sigma(abs_h):
    x = np.asarray(abs_h, dtype=float)
    return np.sqrt(np.mean(x**2) / 2.0)


def small_h_fit_density(values, fit_bin_width=FIT_BIN_WIDTH, fit_hmax=FIT_HMAX, xmax=PLOT_XMAX):
    edges = np.arange(0.0, xmax + fit_bin_width, fit_bin_width)
    hist, edges = np.histogram(values, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mask = (centers >= 0.0) & (centers <= fit_hmax)
    slope, intercept = np.polyfit(centers[mask], hist[mask], deg=1)
    yhat = intercept + slope * centers[mask]
    ss_res = np.sum((hist[mask] - yhat) ** 2)
    ss_tot = np.sum((hist[mask] - np.mean(hist[mask])) ** 2)
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": float(r2),
        "hist": hist,
        "edges": edges,
        "centers": centers,
    }


def summarize_fields(H, spins):
    abs_h = np.abs(H).ravel()
    lam = (spins * H).ravel()
    fit = small_h_fit_density(abs_h)

    summary = {
        "n_fields": int(abs_h.size),
        "mean_abs_h": float(np.mean(abs_h)),
        "std_abs_h": float(np.std(abs_h)),
        "rayleigh_sigma_mle": float(rayleigh_mle_sigma(abs_h)),
        "fraction_abs_h_lt_0p1": float(np.mean(abs_h < 0.1)),
        "fraction_abs_h_lt_0p2": float(np.mean(abs_h < 0.2)),
        "fraction_lambda_lt_0": float(np.mean(lam < 0.0)),
        "fraction_0_le_lambda_lt_0p1": float(np.mean((lam >= 0.0) & (lam < 0.1))),
        "small_h_intercept": fit["intercept"],
        "small_h_slope": fit["slope"],
        "small_h_r2": fit["r2"],
    }
    return summary, fit


# ---------------------------------------------------------------------
# Correction statistics / empirical kernel
# ---------------------------------------------------------------------
def compute_change_statistics(lambda_init, spins_oim, spins_mh):
    changed = (spins_oim != spins_mh)
    total_sites = int(changed.size)
    n_changed = int(np.sum(changed))

    unstable = (lambda_init < 0.0)
    stable = ~unstable

    n_unstable = int(np.sum(unstable))
    n_stable = int(np.sum(stable))

    changed_unstable = int(np.sum(changed & unstable))
    changed_stable = int(np.sum(changed & stable))

    return {
        "total_sites": total_sites,
        "n_changed": n_changed,
        "fraction_changed": n_changed / total_sites,
        "n_unstable": n_unstable,
        "fraction_unstable": n_unstable / total_sites,
        "n_stable": n_stable,
        "fraction_stable": n_stable / total_sites,
        "n_changed_unstable": changed_unstable,
        "n_changed_stable": changed_stable,
        "p_changed_given_unstable": changed_unstable / n_unstable if n_unstable else np.nan,
        "p_changed_given_stable": changed_stable / n_stable if n_stable else np.nan,
        "fraction_changed_from_unstable": changed_unstable / n_changed if n_changed else np.nan,
        "fraction_changed_from_stable": changed_stable / n_changed if n_changed else np.nan,
    }


def kernel_from_flat(lambda_flat, changed_flat, bin_edges):
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (lambda_flat >= lo) & (lambda_flat < hi)
        n = int(np.sum(mask))
        if n == 0:
            x = 0.5 * (lo + hi)
            p = np.nan
        else:
            x = float(np.mean(lambda_flat[mask]))
            p = float(np.mean(changed_flat[mask]))
        rows.append({
            "lambda_lo": float(lo),
            "lambda_hi": float(hi),
            "lambda_x": float(x),
            "count": n,
            "K_emp": p,
        })
    return rows


def bootstrap_kernel_by_seed(lambda_init, changed, bin_edges, n_boot=N_BOOTSTRAP, rng_seed=BOOTSTRAP_SEED):
    """
    Bootstrap by resampling the disorder realizations (seed index) with replacement.
    """
    rng = np.random.default_rng(rng_seed)
    n_seeds = lambda_init.shape[0]

    # Precompute seedwise counts in each bin.
    n_bins = len(bin_edges) - 1
    seed_counts = np.zeros((n_seeds, n_bins), dtype=np.int64)
    seed_changed = np.zeros((n_seeds, n_bins), dtype=np.int64)

    for s in range(n_seeds):
        lam_s = lambda_init[s].ravel()
        chg_s = changed[s].ravel().astype(np.int64)
        inds = np.digitize(lam_s, bin_edges) - 1
        valid = (inds >= 0) & (inds < n_bins)
        inds = inds[valid]
        chg_s = chg_s[valid]
        for b in range(n_bins):
            mask = (inds == b)
            seed_counts[s, b] = np.sum(mask)
            if seed_counts[s, b] > 0:
                seed_changed[s, b] = np.sum(chg_s[mask])

    boot = np.full((n_boot, n_bins), np.nan, dtype=float)
    for k in range(n_boot):
        picks = rng.integers(0, n_seeds, size=n_seeds)
        counts = np.sum(seed_counts[picks], axis=0)
        changed_counts = np.sum(seed_changed[picks], axis=0)
        valid = counts > 0
        boot[k, valid] = changed_counts[valid] / counts[valid]

    ci_lo = np.nanpercentile(boot, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot, 97.5, axis=0)
    mean_boot = np.nanmean(boot, axis=0)
    return mean_boot, ci_lo, ci_hi


# ---------------------------------------------------------------------
# Frozen-field model
# ---------------------------------------------------------------------
def acceptance_plus(lam, T):
    return np.minimum(1.0, np.exp(-2.0 * lam / T))


def acceptance_minus(lam, T):
    return np.minimum(1.0, np.exp(2.0 * lam / T))


def T_of_u(u):
    return T_START - (T_START - T_END) * (u / U_MAX)


def dqdu(q, lam, u):
    T = T_of_u(u)
    a_plus = acceptance_plus(lam, T)
    a_minus = acceptance_minus(lam, T)
    return a_plus * (1.0 - q) - a_minus * q


def frozen_field_kernel(lam_values, n_steps=FROZEN_FIELD_NSTEPS):
    """
    Solve dq/du = a_+(1-q) - a_- q with q(0)=0 by RK4.
    """
    lam_values = np.asarray(lam_values, dtype=float)
    q = np.zeros_like(lam_values)
    du = U_MAX / n_steps
    u = 0.0

    for _ in range(n_steps):
        k1 = dqdu(q, lam_values, u)
        k2 = dqdu(q + 0.5 * du * k1, lam_values, u + 0.5 * du)
        k3 = dqdu(q + 0.5 * du * k2, lam_values, u + 0.5 * du)
        k4 = dqdu(q + du * k3, lam_values, u + du)
        q = q + (du / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        u += du

    return q


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def make_panel_a(abs_h_oim, abs_h_mh, sigma_pp, output_stem):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    edges = np.linspace(0.0, PLOT_XMAX, PLOT_BINS + 1)
    ax.hist(abs_h_oim, bins=edges, density=True, histtype="step",
            linewidth=1.6, color=OIM_COLOR, label="OIM+")
    ax.hist(abs_h_mh, bins=edges, density=True, histtype="step",
            linewidth=1.6, color=DH_OIM_COLOR, label="DH-OIM+")

    x = np.linspace(0.0, PLOT_XMAX, 800)
    ax.plot(x, rayleigh_pdf(x, sigma_pp), "k--", linewidth=1.8,
            label="Palmer–Pond (fixed scale)")

    ax.set_xlim(0.0, PLOT_XMAX)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"Absolute Ising local field $|h_i|$", fontsize=13)
    ax.set_ylabel("Probability density", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(str(output_stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(output_stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_panel_b(kernel_rows, ci_lo, ci_hi, lam_curve, kfrz_curve, output_stem):
    fig, ax = plt.subplots(figsize=(6.0, 4.8))

    x = np.array([r["lambda_x"] for r in kernel_rows], dtype=float)
    y = np.array([r["K_emp"] for r in kernel_rows], dtype=float)
    yerr = np.vstack((y - ci_lo, ci_hi - y))

    valid = np.isfinite(x) & np.isfinite(y)
    ax.plot(lam_curve, kfrz_curve, "--", linewidth=1.4, color=K_FRZ_COLOR,
            label=r"$K_{\rm frz}(\lambda)$")
    ax.errorbar(
        x[valid], y[valid],
        yerr=yerr[:, valid],
        fmt="o-", linewidth=1.3, markersize=3.5, capsize=2.5, color=K_EMP_COLOR,
        label=r"$K_{\rm emp}(\lambda)$"
    )

    ax.axvline(0.0, linestyle=":", linewidth=0.9, color="0.5")
    ax.set_xlim(FROZEN_FIELD_XMIN, FROZEN_FIELD_XMAX)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"Initial signed local stability $\lambda_i = s_i h_i$", fontsize=13)
    ax.set_ylabel("End-point correction probability", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(str(output_stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(output_stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig6_combined(abs_h_oim, abs_h_mh, sigma_pp, kernel_rows, ci_lo, ci_hi,
                       lam_curve, kfrz_curve, output_stem):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    ax1, ax2 = axes

    # Panel (a)
    edges = np.linspace(0.0, PLOT_XMAX, PLOT_BINS + 1)
    ax1.hist(abs_h_oim, bins=edges, density=True, histtype="step",
             linewidth=1.6, color=OIM_COLOR, label="OIM+")
    ax1.hist(abs_h_mh, bins=edges, density=True, histtype="step",
             linewidth=1.6, color=DH_OIM_COLOR, label="DH-OIM+")
    x = np.linspace(0.0, PLOT_XMAX, 800)
    ax1.plot(x, rayleigh_pdf(x, sigma_pp), "k--", linewidth=1.8,
             label="Palmer–Pond (fixed scale)")
    ax1.set_xlim(0.0, PLOT_XMAX)
    ax1.set_ylim(bottom=0.0)
    ax1.set_xlabel(r"Absolute Ising local field $|h_i|$", fontsize=12)
    ax1.set_ylabel("Probability density", fontsize=12)
    ax1.tick_params(labelsize=10)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=True, fontsize=9)

    # Panel (b)
    xk = np.array([r["lambda_x"] for r in kernel_rows], dtype=float)
    yk = np.array([r["K_emp"] for r in kernel_rows], dtype=float)
    yerr = np.vstack((yk - ci_lo, ci_hi - yk))
    valid = np.isfinite(xk) & np.isfinite(yk)

    ax2.plot(lam_curve, kfrz_curve, "--", linewidth=1.4, color=K_FRZ_COLOR,
             label=r"$K_{\rm frz}(\lambda)$")
    ax2.errorbar(
        xk[valid], yk[valid],
        yerr=yerr[:, valid],
        fmt="o-", linewidth=1.3, markersize=3.2, capsize=2.3, color=K_EMP_COLOR,
        label=r"$K_{\rm emp}(\lambda)$"
    )
    ax2.axvline(0.0, linestyle=":", linewidth=0.9, color="0.5")
    ax2.set_xlim(FROZEN_FIELD_XMIN, FROZEN_FIELD_XMAX)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel(r"Initial signed local stability $\lambda_i = s_i h_i$", fontsize=12)
    ax2.set_ylabel("End-point correction probability", fontsize=12)
    ax2.tick_params(labelsize=10)
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=True, fontsize=9)

    # panel labels
    ax1.text(0.5, -0.27, "(a)", transform=ax1.transAxes, ha="center", va="top", fontsize=12)
    ax2.text(0.5, -0.27, "(b)", transform=ax2.transAxes, ha="center", va="top", fontsize=12)

    fig.tight_layout()
    fig.savefig(str(output_stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(output_stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Path to the averaged-data NPZ file")
    parser.add_argument("--outdir", default="fig6_analysis", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    J, S_oim, S_mh = load_data(args.npz)
    H_oim = local_fields(J, S_oim)
    H_mh = local_fields(J, S_mh)

    abs_h_oim = np.abs(H_oim).ravel()
    abs_h_mh = np.abs(H_mh).ravel()
    lambda_init = S_oim * H_oim

    sigma_pp = SIGMA_PP_ORIGINAL * FIELD_SCALE
    mean_pp = sigma_pp * np.sqrt(np.pi / 2.0)

    # Panel (a) summaries
    oim_summary, oim_fit = summarize_fields(H_oim, S_oim)
    mh_summary, mh_fit = summarize_fields(H_mh, S_mh)
    oim_summary["relative_sigma_difference_vs_pp"] = (
        oim_summary["rayleigh_sigma_mle"] / sigma_pp - 1.0
    )
    mh_summary["relative_sigma_difference_vs_pp"] = (
        mh_summary["rayleigh_sigma_mle"] / sigma_pp - 1.0
    )

    # Panel (b) summaries
    change_stats = compute_change_statistics(lambda_init, S_oim, S_mh)
    changed = (S_oim != S_mh)

    kernel_rows = kernel_from_flat(lambda_init.ravel(), changed.ravel(), KERNEL_BIN_EDGES)
    mean_boot, ci_lo, ci_hi = bootstrap_kernel_by_seed(lambda_init, changed, KERNEL_BIN_EDGES)

    for i, row in enumerate(kernel_rows):
        row["K_emp_bootstrap_mean"] = float(mean_boot[i]) if np.isfinite(mean_boot[i]) else np.nan
        row["K_emp_ci_low"] = float(ci_lo[i]) if np.isfinite(ci_lo[i]) else np.nan
        row["K_emp_ci_high"] = float(ci_hi[i]) if np.isfinite(ci_hi[i]) else np.nan

    lam_curve = np.linspace(FROZEN_FIELD_XMIN, FROZEN_FIELD_XMAX, FROZEN_FIELD_NPTS)
    kfrz_curve = frozen_field_kernel(lam_curve)

    # Console output
    print("\n" + "=" * 78)
    print("FIG. 6 LOCAL-FIELD AND CORRECTION-KERNEL ANALYSIS")
    print("=" * 78)
    print(f"J shape:                           {J.shape}")
    print(f"OIM spin shape:                    {S_oim.shape}")
    print(f"DH-OIM+ spin shape:                {S_mh.shape}")
    print(f"Palmer sigma, rescaled:            {sigma_pp:.8f}")
    print(f"Palmer predicted mean |h|:         {mean_pp:.8f}")

    print("\nPanel (a): local-field distribution")
    print(f"  OIM mean |h|:                    {oim_summary['mean_abs_h']:.8f}")
    print(f"  OIM Rayleigh sigma:              {oim_summary['rayleigh_sigma_mle']:.8f}")
    print(f"  OIM sigma excess vs Palmer:      {100*oim_summary['relative_sigma_difference_vs_pp']:.6f}%")
    print(f"  OIM P(|h|<0.1):                  {100*oim_summary['fraction_abs_h_lt_0p1']:.4f}%")
    print(f"  OIM P(|h|<0.2):                  {100*oim_summary['fraction_abs_h_lt_0p2']:.4f}%")
    print(f"  DH-OIM+ P(|h|<0.1):              {100*mh_summary['fraction_abs_h_lt_0p1']:.4f}%")
    print(f"  DH-OIM+ P(|h|<0.2):              {100*mh_summary['fraction_abs_h_lt_0p2']:.4f}%")
    print(f"  OIM small-h fit:                 P(h) = {oim_fit['intercept']:.6f} + {oim_fit['slope']:.6f} h")
    print(f"  DH-OIM+ small-h fit:             P(h) = {mh_fit['intercept']:.6f} + {mh_fit['slope']:.6f} h")

    print("\nPanel (b): endpoint-correction statistics")
    print(f"  Total changed endpoints:         {change_stats['n_changed']} of {change_stats['total_sites']}")
    print(f"  Fraction changed:                {100*change_stats['fraction_changed']:.4f}%")
    print(f"  Initially unstable fraction:     {100*change_stats['fraction_unstable']:.4f}%")
    print(f"  P(changed | unstable):           {100*change_stats['p_changed_given_unstable']:.4f}%")
    print(f"  P(changed | stable):             {100*change_stats['p_changed_given_stable']:.4f}%")
    print(f"  Fraction changed from unstable:  {100*change_stats['fraction_changed_from_unstable']:.4f}%")
    print(f"  Fraction changed from stable:    {100*change_stats['fraction_changed_from_stable']:.4f}%")
    print("=" * 78)

    # Save summary CSVs
    summary_rows = [
        {"population": "OIM+", "sigma_pp_rescaled": sigma_pp, "mean_pp_predicted": mean_pp, **oim_summary},
        {"population": "DH-OIM+", "sigma_pp_rescaled": sigma_pp, "mean_pp_predicted": mean_pp, **mh_summary},
    ]
    save_csv(outdir / "local_field_summary.csv", summary_rows, list(summary_rows[0].keys()))

    change_row = [{
        "total_sites": change_stats["total_sites"],
        "n_changed": change_stats["n_changed"],
        "fraction_changed": change_stats["fraction_changed"],
        "n_unstable": change_stats["n_unstable"],
        "fraction_unstable": change_stats["fraction_unstable"],
        "n_stable": change_stats["n_stable"],
        "fraction_stable": change_stats["fraction_stable"],
        "n_changed_unstable": change_stats["n_changed_unstable"],
        "n_changed_stable": change_stats["n_changed_stable"],
        "p_changed_given_unstable": change_stats["p_changed_given_unstable"],
        "p_changed_given_stable": change_stats["p_changed_given_stable"],
        "fraction_changed_from_unstable": change_stats["fraction_changed_from_unstable"],
        "fraction_changed_from_stable": change_stats["fraction_changed_from_stable"],
    }]
    save_csv(outdir / "correction_statistics.csv", change_row, list(change_row[0].keys()))

    save_csv(outdir / "empirical_correction_kernel.csv", kernel_rows, list(kernel_rows[0].keys()))
    save_csv(
        outdir / "frozen_field_kernel.csv",
        [{"lambda": float(x), "K_frz": float(y)} for x, y in zip(lam_curve, kfrz_curve)],
        ["lambda", "K_frz"]
    )

    # Save plots
    make_panel_a(abs_h_oim, abs_h_mh, sigma_pp, outdir / "Main_LocalFields_Palmer_J0_0")
    make_panel_b(kernel_rows, ci_lo, ci_hi, lam_curve, kfrz_curve, outdir / "Main_Kemp_vs_Kfrz_J0_0")
    make_fig6_combined(abs_h_oim, abs_h_mh, sigma_pp, kernel_rows, ci_lo, ci_hi,
                       lam_curve, kfrz_curve, outdir / "Figure_6_combined")

    print("\nSaved outputs:")
    print(outdir / "local_field_summary.csv")
    print(outdir / "correction_statistics.csv")
    print(outdir / "empirical_correction_kernel.csv")
    print(outdir / "frozen_field_kernel.csv")
    print(outdir / "Main_LocalFields_Palmer_J0_0.pdf")
    print(outdir / "Main_LocalFields_Palmer_J0_0.png")
    print(outdir / "Main_Kemp_vs_Kfrz_J0_0.pdf")
    print(outdir / "Main_Kemp_vs_Kfrz_J0_0.png")
    print(outdir / "Figure_6_combined.pdf")
    print(outdir / "Figure_6_combined.png")


if __name__ == "__main__":
    main()
