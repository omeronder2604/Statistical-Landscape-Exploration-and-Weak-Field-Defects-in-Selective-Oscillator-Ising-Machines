# Statistical Landscape Exploration and Weak-Field Defects in Selective Oscillator Ising Machines

This repository contains a Python implementation of a hybrid oscillator Ising machine workflow for dense spin-glass optimization experiments.

The codebase simulates ensembles of noisy, coupled phase oscillators, maps their continuous phases to binary Ising spins, and subsequently applies a Metropolis-Hastings (MH) refinement stage. The workflow is intended for studying the statistical structure of oscillator Ising machine outputs, especially the weak-local-field defects that remain after the continuous oscillator stage and can be healed by short Monte Carlo refinement.

The current version supports:

- CPU execution with Numba-parallel kernels.
- Distributed execution across high-performance computing (HPC) nodes using MPI (mpi4py).
- Multi-epoch annealing schedules with linearly interpolated noise amplitudes.
- Sherrington-Kirkpatrick-style dense random coupling matrices and G-set graphs.
- Post-run analysis and plotting of energies, overlaps, basins, local fields, and spin flips.

---

## Repository layout

```text

.
├── Gset/             # Graph datasets for max-cut benchmarking
├── HPC/              # Codes and Slurm batch scripts for HPC cluster submission
├── MH/               # Pure-Metropolis-Hastings plots and data analysis such as min/mean energy, Z_2-Hamming distance, and basin clustering
├── OIM/              # DH-OIM+ plots and data analysis such as min/mean energy, Z_2-Hamming distance, and basin clustering
├── dendrogram/       # Dendrogram plots for pure-MH, OIM+ and DH-OIM+ algorithms via Z_2-Hamming distance
├── energy prog/      # Energy progression tracking and history arrays for the single J realization with J_0=0
├── raw_data/         # Output directory for generated .npz archive data
├── schedule/         # Multi-epoch annealing and parameter timing schedule plot for the OIM+
└── .gitignore

```

## Model

The oscillator stage evolves phase variables \(\phi_i\) under noisy coupled dynamics with all-to-all Ising couplings \(J_{ij}\), second-harmonic pinning, and an annealed noise scale. Spins are read out from phases by

$$
s_i = \mathrm{sign}(\cos \phi_i).
$$

The Ising energy is

$$
E = -\frac{1}{2}\sum_{i,j} J_{ij}s_i s_j,
$$

with zero diagonal in \(J\).

For the default SK-style dense problem, the coupling matrix is sampled as a symmetric Gaussian matrix with approximately

$$
J_{ij} \sim \mathrm{Normal}\left(\frac{J_0}{N}, \frac{\sigma_J^2}{2N}\right)
$$

for off-diagonal entries, where `J0_VAL` controls \(J_0\) and `J_std` controls \(\sigma_J\).


## Analysis and plotting
Main plotted quantities include:

- full annealing schedule,
- energy over schedule,
- oscillator and post-MH energy histograms,
- overlap matrices,
- overlap distribution `P(q)`,
- basin dendrograms,
- local-field distributions before and after MH,
- spin flips induced by MH.



## Scientific purpose

This code is built around the hypothesis that the oscillator dynamics rapidly finds a useful low-energy basin, while a short Metropolis-Hastings stage efficiently heals weak-field defects that remain hard for the continuous oscillator dynamics to resolve.

In this interpretation:

- the oscillator stage provides fast global organization,
- the MH stage performs local defect healing,
- local-field and overlap diagnostics reveal where the oscillator dynamics succeeds or fails.



## Citation

If this code is used in academic work, cite the associated manuscript when available:

```text
Ö. Önder and A. C. Keser,
"Statistical Landscape Exploration and Weak-Field Defects in Selective Oscillator Ising Machines,"
Arxiv id: XXXXXXXX
```

---

## License

Add a license before making the repository public.

Common choices for academic research code are:

```text
MIT License
BSD-3-Clause
Apache-2.0
```

---

## Contact

Ömer Önder
Department of Physics, Bilkent University  
Email: omer.onder@bilkent.edu.tr


