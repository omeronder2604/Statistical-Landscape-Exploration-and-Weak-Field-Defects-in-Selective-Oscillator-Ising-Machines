# Statistical Landscape Exploration and Weak-Field Defects in Selective Oscillator Ising Machines

Research code and analysis data accompanying:

> Ömer Önder and Aydın Cem Keser, “Statistical Landscape Exploration and Weak-Field Defects in Selective Oscillator Ising Machines,” [arXiv:2609.14550](https://arxiv.org/abs/2609.14550) (2026).

## Overview

This repository contains the current Python implementation and analysis artifacts for selective oscillator Ising machine experiments. The principal workflow is:

1. **OIM+** evolves a population of noisy coupled phase oscillators through four annealing epochs, with energy-based selection and repopulation after each epoch.
2. **DH-OIM+** applies a short Metropolis-Hastings (MH) refinement to the selected OIM+ population.
3. **Pure MH** provides an independent local-update baseline initialized from random spins.

The primary experiments use dense Sherrington-Kirkpatrick (SK) coupling matrices. The repository also contains local post-processing programs and archived oscillator checkpoints for selected G-set Max-Cut instances. Analyses cover energy distributions, signed overlaps, Z2-quotiented Hamming distances, hierarchical clustering, cophenetic correlation, and local-field/cancellation diagnostics. These analyses describe the sampled configuration ensembles; they do not by themselves establish thermodynamic pure states, dynamical basins, or replica-symmetry breaking.

## Repository layout

```text
.
├── OIM/                 # OIM+/DH-OIM+ analysis; normal script is J0=0, AF-prefixed script is J0=-4
├── MH/                  # Pure-MH analysis; normal script is J0=0, AF-prefixed script is J0=-4
├── Gset/                # Archived-checkpoint DH-OIM+ refinement and pure-MH G-set baseline
├── PP/                  # Palmer-Pond weak-field and frozen-field comparison analysis
├── HPC/
│   ├── Single_OIM/      # Single-realization OIM+/DH-OIM+ MPI HPC driver (J0=0)
│   ├── Seed_Avg_OIM/    # 30-realization OIM+/DH-OIM+ MPI HPC driver
│   │   └── AF/          # J0=-4 MPI HPC driver
│   └── Seed_Avg_MH/     # 30-realization pure-MH MPI HPC driver
│       └── AF/          # J0=-4 MPI HPC driver
├── dendrogram/          # Representative J0=0 dendrogram generation for single-realization OIM+/DH-OIM+
├── energy_prog/         # Representative J0=0 energy-evolution comparison for single-realization OIM+/DH-OIM+
├── schedule/            # Four-epoch OIM+ schedule plot
├── raw_data/            # Archived SK arrays and G-set graph/checkpoint data
├── requirements.txt
├── repository_submission_audit.md
├── LICENSE
└── .gitignore
```

The curated `OIM/` and `MH/` directories use AF-prefixed filenames rather than `AF/` subdirectories. The `HPC/` tree retains explicit `AF/` subdirectories.

## Installation

The exact Python version and package versions used for the original production runs were not archived. Use Python 3.x; exact-version confirmation remains a reproducibility task.

```bash
git clone https://github.com/omeronder2604/Statistical-Landscape-Exploration-and-Weak-Field-Defects-in-Selective-Oscillator-Ising-Machines.git
cd Statistical-Landscape-Exploration-and-Weak-Field-Defects-in-Selective-Oscillator-Ising-Machines
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Activate the virtual environment using the command appropriate for your shell. MPI runs additionally require a working system MPI implementation compatible with `mpi4py`. The original TRUBA scripts loaded `miniconda3` and `openmpi` and activated a conda environment named `spinglass`; no reliable environment export or version lock was available, so this repository does not fabricate an `environment.yml`.

## Running the code

Most programs use configuration constants near the top of the source file rather than command-line arguments. Inspect and deliberately set variables such as `GLOBAL_SEED`, `J0_VAL`/`J0`, `TARGET_NPZ`, `TARGET_TXT`, replica counts, and output prefixes before a production run.

### Lightweight local plot

The schedule plot has no external data dependency:

```bash
cd schedule
python Schedule.py
```

It writes both the legacy `OIM+_Schedule_Normalized.pdf` and the manuscript-named `OIM_Schedule.pdf` in the current directory.

### Local analysis of archived SK data

The analysis programs resolve input files relative to the current working directory. Run them from `raw_data/`, where the supplied archives reside:

```bash
cd raw_data
python "../OIM/OIM_Averaged_P(q).py"          # J0 = 0 OIM+ / DH-OIM+
python "../OIM/AF_OIM_Averaged_P(q).py"       # J0 = -4 OIM+ / DH-OIM+
python "../MH/Pure_MH_Averaged_P(q).py"       # J0 = 0 pure MH
python "../MH/AF_Pure_MH_Averaged_P(q).py"    # J0 = -4 pure MH
```

These commands write PDFs to the current directory. The AF scripts use
`AF_`-prefixed output names so they do not overwrite the `J0=0` figures. The
scripts perform millions of pairwise operations and are not instant, even
though they do not rerun the production simulations.

The representative trajectory and dendrogram programs also run from
`raw_data/`:

```bash
python ../energy_prog/energy_prog.py
python ../dendrogram/dendograms.py
```

Both regenerate a pure-MH comparison and can be expensive. The energy program
writes the manuscript-named `Osc_energy_prog.pdf` and `MH_energy_prog.pdf`
alongside its legacy-named copies.

### Local G-set programs

Run from the G-set data directory so the hard-coded target basename resolves:

```bash
cd raw_data/Gset
python ../../Gset/Gset_OIM.py
python ../../Gset/Gset_MH.py
```

Before running, edit `TARGET_NPZ` in `Gset_OIM.py` or `TARGET_TXT` in `Gset_MH.py` to choose an instance. `Gset_OIM.py` starts from a fixed archived 448-replica continuous-phase checkpoint and performs only the short MH refinement; the active repository does not contain the production oscillator-checkpoint generator. `Gset_MH.py` provides the independently runnable 448-replica pure-MH comparison from graph text input. Both programs may be computationally expensive. See [`Gset/README.md`](Gset/README.md) for the precise scope and commands.

### Single-realization MPI run

From the script directory:

```bash
cd HPC/Single_OIM
mpirun -np 8 python single_EA.py
```

This is the seed-52, `J0=0` representative OIM+/DH-OIM+ production run and is expensive. It writes `AllToAll_Hybrid_EA.npz` and several PDFs in the working directory.

### Averaged MPI runs

The following are production-scale commands and were not executed during repository preparation:

```bash
cd HPC/Seed_Avg_OIM
mpirun -np 8 python avg.py

cd AF
mpirun -np 8 python AF_avg.py
```

```bash
cd HPC/Seed_Avg_MH
mpirun -np 8 python MH.py

cd AF
mpirun -np 8 python AF_MH.py
```

The normal directories use `J0=0`; the `AF/` drivers use `J0=-4.0`.

### Slurm submission

The files under `HPC/` are documented TRUBA-oriented examples. Adapt the partition, account, modules, environment name, wall time, and resources for another cluster. From the corresponding directory:

```bash
sbatch selective.slurm   # Single_OIM or Seed_Avg_OIM[/AF]
sbatch MH.slurm          # Seed_Avg_MH[/AF]
```

The checked examples use partition `hamsi`, eight MPI ranks, `NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK`, `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`, `module load miniconda3`, `conda activate spinglass`, `module load openmpi`, and `mpirun -np 8`. Account selection is intentionally left to the user. These are site-specific examples that require adaptation and were not production-tested during this repository audit.

## Reproducibility

The canonical SK drivers use a fixed `GLOBAL_SEED` and NumPy `SeedSequence` branching:

- the root sequence spawns independent disorder-realization sequences;
- each realization has separate branches for the coupling matrix, initial phases, oscillator epochs, and post-OIM MH refinement;
- each oscillator epoch and the refinement stage spawn replica-specific generators;
- phases and generators are assigned by global replica index before contiguous MPI slicing;
- rank zero performs selection in gathered global order, then broadcasts the repopulated phases;
- cloned phases subsequently receive distinct stochastic streams.

The archived paired OIM and pure-MH datasets contain exactly matching disorder matrices. The `J0=-4` matrices are the corresponding `J0=0` centered realizations with the intended `-4/N` off-diagonal shift.

This supports deterministic reruns under a fixed codebase, dependency stack, numerical backend, compiler/runtime configuration, and hardware environment. It does not guarantee bitwise identity across machines. Numba `fastmath`, BLAS reductions, compiler/CPU differences, and library versions may change floating-point trajectories. Full RNG states, generated child seeds, code hashes, and environment versions were not archived.

## Model and conventions

Oscillator phases are read out as

$$
s_i = \mathrm{sign}(\cos \phi_i)
$$

with exact zeros assigned to `+1`. For a symmetric zero-diagonal coupling matrix, the reported Ising energy is

$$
E(\mathbf{s}) = -\sum_{i \lt j} J_{ij}s_i s_j
= -\frac{1}{2}\mathbf{s}^{T} J \mathbf{s}
$$

The SK driver samples raw matrix entries with mean `J0/N` and standard deviation `J_std/sqrt(N)`, symmetrizes with `(J + J.T)/2`, and zeros the diagonal. Consequently the off-diagonal variance is `J_std**2/(2*N)`.

The current SK protocol uses `N=200`, 402 replicas, four oscillator epochs, and a two-thirds cull after each epoch. The short DH-OIM+ refinement uses 10,000 single-spin proposals per replica. These are experiment-specific settings, not universal requirements of the algorithm.

## Analysis and plotting

The repository contains code for:

- final and evolving Ising energies;
- signed within-realization overlap distributions `P(q)`;
- Z2-quotiented Hamming distances;
- average-linkage hierarchical clustering and dendrogram fidelity via cophenetic correlation;
- intra- and inter-cluster distance summaries;
- globally aligned pre/post-refinement spin-change masks;
- local fields, incident coupling magnitudes, and cancellation ratios;
- G-set weighted Max-Cut values.

Most analysis scripts write PDFs to the current working directory. Run the
`J0=0` and `J0=-4` analyses in separate locations if additional exploratory
outputs are added. The maintained AF scripts now use `AF_`-prefixed output
names, and `PP/outputs/` is ignored for generated Palmer-Pond artifacts.

### Palmer-Pond weak-field analysis

`PP/local_field_analysis.py` regenerates the local-field distribution, endpoint-correction statistics, empirical correction kernel, and frozen-field reference from the archived 30-seed OIM archive:

```bash
python PP/local_field_analysis.py \
    raw_data/AllToAll_Hybrid_EA_SeedAvg_AveragedData.npz \
    --outdir PP/outputs
```

The script writes manuscript-named PDFs, PNG companions, a combined figure, and four CSV tables. See [`PP/README_fig6_analysis.md`](PP/README_fig6_analysis.md) for the precise field definitions, fine fitting bins versus coarse plotting bins, unbinned statistics, seed-level bootstrap, and expected numerical results.

## J0 = 0 and J0 = -4 datasets

- `J0=0` is the unbiased/zero-mean SK case.
- `J0=-4` is the antiferromagnetically biased case.
- Normal OIM/MH analysis scripts and unprefixed SK archives correspond to `J0=0`.
- `AF_` analysis scripts and `AF_` raw-data archives correspond to `J0=-4`.
- `HPC/Seed_Avg_OIM/AF/` and `HPC/Seed_Avg_MH/AF/` contain the production AF drivers.
- Maintained AF analysis scripts emit `AF_`-prefixed figure names, preventing collision with the `J0=0` outputs.

## Reproducing manuscript figures

The status describes repository coverage, not whether a production-scale command was rerun during this audit.

| Manuscript item | Method | J0 | Script | Input data | Script output | Status |
|---|---|---:|---|---|---|---|
| `OIM_Schedule.pdf` | OIM+ schedule | 0 / schedule | `schedule/Schedule.py` | none | exact manuscript basename plus legacy copy | PASS |
| `Osc_energy_prog.pdf` | Representative OIM+ trajectory | 0 | `energy_prog/energy_prog.py` | `raw_data/AllToAll_Hybrid_EA.npz` | exact manuscript basename plus legacy copy | PASS |
| `MH_energy_prog.pdf` | Representative pure-MH trajectory and DH-OIM+ inset | 0 | `energy_prog/energy_prog.py` | same archive; script also runs pure MH | exact manuscript basename plus legacy copy | PASS |
| `Avg_Plot_3_Comparative_Energy.pdf` | OIM+ vs DH-OIM+ energy | 0 | `OIM/OIM_Averaged_P(q).py` | unprefixed OIM archive | same basename | PASS |
| `Avg_Plot_PureMH_EnergyDist.pdf` | Pure-MH energy | 0 | `MH/Pure_MH_Averaged_P(q).py` | unprefixed MH archive | same basename | PASS |
| `Avg_Plot_2_Comparative_Overlap.pdf` | OIM+ vs DH-OIM+ overlap | 0 | `OIM/OIM_Averaged_P(q).py` | unprefixed OIM archive | same basename | PASS |
| `Avg_Plot_PureMH_Overlap.pdf` | Pure-MH overlap | 0 | `MH/Pure_MH_Averaged_P(q).py` | unprefixed MH archive | same basename | PASS |
| `Dendrogram_1_PureMH.pdf` | Pure-MH clustering | 0 | `dendrogram/dendograms.py` | `raw_data/AllToAll_Hybrid_EA.npz`; script runs pure MH | same basename | PASS |
| `Dendrogram_2_PostOscillator.pdf` | OIM+ clustering | 0 | `dendrogram/dendograms.py` | same representative archive | same basename | PASS |
| `Dendrogram_3_PostMH.pdf` | DH-OIM+ clustering | 0 | `dendrogram/dendograms.py` | same representative archive | same basename | PASS |
| `Main_LocalFields_Palmer_J0_0.pdf` | Local-field/Palmer-Pond comparison | 0 | `PP/local_field_analysis.py` | `raw_data/AllToAll_Hybrid_EA_SeedAvg_AveragedData.npz` | exact manuscript basename | PASS |
| `Main_Kemp_vs_Kfrz_J0_0.pdf` | Empirical/frozen-field correction kernel | 0 | `PP/local_field_analysis.py` | same 30-seed archive | exact manuscript basename | PASS |
| `AF_Avg_Plot_2_Comparative_Overlap.pdf` | OIM+ vs DH-OIM+ overlap | -4 | `OIM/AF_OIM_Averaged_P(q).py` | AF OIM archive | exact manuscript basename | PASS |
| `AF_Avg_Plot_PureMH_Overlap.pdf` | Pure-MH overlap | -4 | `MH/AF_Pure_MH_Averaged_P(q).py` | AF MH archive | exact manuscript basename | PASS |
| `AF_Avg_Plot_3_Comparative_Energy.pdf` | OIM+ vs DH-OIM+ energy | -4 | `OIM/AF_OIM_Averaged_P(q).py` | AF OIM archive | exact manuscript basename | PASS |
| `AF_Avg_Plot_PureMH_EnergyDist.pdf` | Pure-MH energy | -4 | `MH/AF_Pure_MH_Averaged_P(q).py` | AF MH archive | exact manuscript basename | PASS |

Exact evidence and table mappings are recorded in `repository_submission_audit.md`.

## Data size

`raw_data/` contains 23 files totaling approximately 416.7 MiB. Two OIM seed-averaged archives are approximately 64.6 MiB each and exceed GitHub's 50 MiB warning threshold; two pure-MH archives are approximately 46.1 MiB each, and four 2,000-node G-set checkpoints are approximately 37.4 MiB each. No individual file exceeds GitHub's 100 MiB hard limit. For a durable public release, consider Git LFS or an external research-data archive such as Zenodo; migrating already-committed files requires an explicit history-management decision.

## Citation

```bibtex
@article{Onder2026StatisticalLandscape,
  author        = {Ömer Önder and Aydın Cem Keser},
  title         = {Statistical Landscape Exploration and Weak-Field Defects in Selective Oscillator Ising Machines},
  year          = {2026},
  eprint        = {2609.14550},
  archivePrefix = {arXiv}
}
```

No DOI was found in the inspected project metadata.

## License

This repository is released under the [MIT License](LICENSE).

## Known release limitations

- The exact original Python/package/compiler environment is not preserved.
- G-set oscillator checkpoint generation is absent from the active repository; the supplied 448-replica checkpoints are fixed archived inputs.
- The representative final OIM+ endpoint minimum and a transient trajectory minimum are distinguishable in the archive; see the audit before reusing the manuscript wording.
- On Windows, clone to a shorter parent directory if possible. For long checkouts, use `git config --global core.longpaths true` or run individual commands as `git -c core.longpaths=true ...`.

See [repository_submission_audit.md](repository_submission_audit.md) for the full submission checklist.
