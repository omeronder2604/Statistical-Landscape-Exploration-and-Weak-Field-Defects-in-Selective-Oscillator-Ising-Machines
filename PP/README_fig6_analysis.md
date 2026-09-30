# Figure 6 weak-field and correction-kernel analysis

This folder contains the post-processing used to reproduce the weak-field and endpoint-correction analysis from the saved 30-seed SK/OIM data bundle.

## Input

Expected NPZ file:

```text
AllToAll_Hybrid_EA_SeedAvg_AveragedData.npz
```

Required keys:

```text
master_J
master_osc_spins
master_final_spins
```

## Run

From the repository root, for example:

```bash
python PP/local_field_analysis.py raw_data/AllToAll_Hybrid_EA_SeedAvg_AveragedData.npz --outdir PP/outputs
```

The first positional argument is the NPZ path. The optional `--outdir` argument
selects the output directory. If it is omitted, outputs are written to
`fig6_analysis/` relative to the current working directory.

The repository copy of the archived NPZ is in `raw_data/`. No user-specific
absolute path is required.

## Panel (a): local-field distribution

The local field and signed stability are

```text
h_i      = sum_j J_ij s_j
lambda_i = s_i h_i
```

The Palmer-Pond reference is rescaled using

```text
sigma_PP = 1.17 / sqrt(2)
```

because the symmetrized couplings in the simulation have variance `1/(2N)`, whereas Palmer and Pond use `1/N`.

The broad-distribution quantities `<|h|>` and the Rayleigh MLE `sigma_hat` are computed directly from the unbinned local-field samples.
The cumulative probabilities `P(|h|<0.1)` and `P(|h|<0.2)` are likewise
computed directly from the raw fields, not from plotted histogram bins.

The quoted small-field linear regression uses:

```text
histogram type: probability density
fit bin width:  Delta h = 0.001
fit interval:   0 <= |h| <= 0.4
```

The histogram shown in the figure is intentionally coarser (50 bins on `0 <= |h| <= 4`) for visual clarity. Its binning is not used to obtain the reported small-h coefficients.

## Panel (b): endpoint-correction kernel

The empirical kernel is

```text
K_emp(lambda) = P(endpoint spin changed after MH | initial lambda)
```

The initial stability is always computed from the pre-MH OIM+ state. The endpoint-change indicator compares `master_osc_spins` with `master_final_spins` site by site.

Bootstrap confidence intervals resample the disorder realizations (the seed axis) rather than individual spin endpoints.

The frozen-field reference solves

```text
dq/du = a_plus(lambda,T(u)) (1-q) - a_minus(lambda,T(u)) q
q(0)  = 0
```

with

```text
a_plus  = min(1, exp(-2 lambda / T))
a_minus = min(1, exp(+2 lambda / T))
T(u)    = 0.2 - (0.1/50) u
0 <= u <= 50
```

and defines `K_frz(lambda) = q(50;lambda)`.

### Important plotting detail

The empirical-kernel bin edges are explicitly defined by `KERNEL_BIN_EDGES`
near the top of the script. This fixed array is the canonical reproducible
binning used by the repository implementation, with finer resolution near
`lambda = 0`. The implementation reproduces the manuscript analysis and
figure structure. The headline endpoint-count statistics are calculated
directly and do not depend on these plotting bins.

## Expected headline values for the current 30-seed data

```text
Palmer sigma, rescaled            0.82731493
Palmer predicted mean |h|         1.03688550
OIM mean |h|                      1.03655448
OIM Rayleigh sigma                0.83243480
OIM sigma excess over Palmer      0.618853 %
OIM P(|h|<0.1)                    2.4995 %
OIM P(|h|<0.2)                    5.3346 %
DH-OIM+ P(|h|<0.1)                1.6034 %
DH-OIM+ P(|h|<0.2)                3.9045 %
OIM small-h fit                   0.210924 + 0.569487 h
DH-OIM+ small-h fit               0.105103 + 0.920376 h

Changed endpoints                 142700 / 2412000
Changed fraction                  5.9163 %
Initially unstable fraction       1.9256 %
P(changed | unstable)             65.8227 %
P(changed | stable)               4.7400 %
Changed endpoints from stable     78.5760 %
```

These reproduce the manuscript values at the stated precision. The small
difference in the OIM+ small-h slope (`0.569487` versus quoted `0.5692`) is
an implementation/rounding-level difference. The current script is the
canonical repository source for both panels and their scalar results.

## Outputs

The script writes:

```text
local_field_summary.csv
correction_statistics.csv
empirical_correction_kernel.csv
frozen_field_kernel.csv
Main_LocalFields_Palmer_J0_0.pdf
Main_LocalFields_Palmer_J0_0.png
Main_Kemp_vs_Kfrz_J0_0.pdf
Main_Kemp_vs_Kfrz_J0_0.png
Figure_6_combined.pdf
Figure_6_combined.png
```

The first two PDF names match the manuscript `\includegraphics` inputs.
