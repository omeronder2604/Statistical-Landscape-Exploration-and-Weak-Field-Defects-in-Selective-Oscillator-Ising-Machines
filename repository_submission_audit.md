# Repository Submission Audit

Audit date: 2026-09-26
Last updated: 2026-10-01

## Scope and safeguards

This was a public-release and reproducibility audit of the curated Git repository, the active manuscript ZIP, the current production/analysis scripts, Slurm examples, and archived data. It did not include expensive simulations, MPI production runs, cluster jobs, a formal mathematical proof, or exhaustive scientific validation. No scientific algorithm, archived result, or manuscript conclusion was changed. The G-set pure-MH comparison population was set to 448 replicas to match the supplied archived OIM checkpoint population.

## Repository inventory

Present at audit start:

- `README.md`, `LICENSE`, `.gitignore`
- `OIM/`, `MH/`, `Gset/`, `HPC/`, `dendrogram/`, `energy_prog/`, `schedule/`, `raw_data/`
- 14 active Python files
- five public-repository Slurm examples before restoration of the G-set generator workflow

Absent at audit start:

- `requirements.txt`
- `environment.yml`
- `repository_submission_audit.md`

Added after the initial audit:

- `PP/local_field_analysis.py` and `PP/README_fig6_analysis.md`, providing the Palmer-Pond / frozen-field post-processing workflow.
- `Gset/Gset_OIM_HPC.py` and `Gset/Gset.slurm`, restoring the MPI oscillator checkpoint-generation workflow.

The active repository now contains 16 Python files and six public Slurm examples.

The active manuscript is `OIMp_v3.tex` inside the adjacent manuscript ZIP, not inside the Git repository. It references 13 distinct figure PDFs (17 panel files when AF and multi-panel components are counted individually).

## Primary runnable and analysis scripts

Commands are shown from the directory stated in the Working directory column. Production commands were validated against real entry points but not executed.

| Script | Purpose / method | J0 | Runtime | Expected input | Generated output | Configuration to edit | Working directory and command |
|---|---|---:|---|---|---|---|---|
| `HPC/Single_OIM/single_EA.py` | One-disorder OIM+ followed by short MH refinement | 0 | MPI/HPC | none | `AllToAll_Hybrid_EA.npz`; diagnostic PDFs | `GLOBAL_SEED`, `N`, `TOTAL_REPLICAS`, `J0_VAL`, schedule, MH settings | `HPC/Single_OIM`; `mpirun -np 8 python single_EA.py` |
| `HPC/Seed_Avg_OIM/avg.py` | 30-disorder OIM+/DH-OIM+ production run | 0 | MPI/HPC | none | `AllToAll_Hybrid_EA_SeedAvg_AveragedData.npz`; PDFs | root seed, seeds, replicas, schedule, MH settings | `HPC/Seed_Avg_OIM`; `mpirun -np 8 python avg.py` |
| `HPC/Seed_Avg_OIM/AF/AF_avg.py` | AF 30-disorder OIM+/DH-OIM+ production run | -4 | MPI/HPC | none | `AllToAll_Hybrid_EA_SeedAvg_AveragedData.npz` in the AF run directory | same plus `J0_VAL` | `HPC/Seed_Avg_OIM/AF`; `mpirun -np 8 python AF_avg.py` |
| `HPC/Seed_Avg_MH/MH.py` | 30-disorder pure-MH baseline | 0 | MPI/HPC | none | `AllToAll_PureMH_MPI_SeedAvg_AveragedData.npz`; PDFs | seed count, replicas, `J0`, update counts, temperatures | `HPC/Seed_Avg_MH`; `mpirun -np 8 python MH.py` |
| `HPC/Seed_Avg_MH/AF/AF_MH.py` | AF 30-disorder pure-MH baseline | -4 | MPI/HPC | none | `AllToAll_PureMH_MPI_SeedAvg_AveragedData.npz` in the AF run directory | same plus `J0` | `HPC/Seed_Avg_MH/AF`; `mpirun -np 8 python AF_MH.py` |
| `OIM/OIM_Averaged_P(q).py` | OIM+/DH-OIM+ energy, overlap, clustering, and weak-field summaries | 0 | local CPU analysis | unprefixed OIM archive | five PDFs and printed statistics | `TARGET_NPZ` | `raw_data`; `python "../OIM/OIM_Averaged_P(q).py"` |
| `OIM/AF_OIM_Averaged_P(q).py` | AF version of preceding analysis | -4 | local CPU analysis | `AF_AllToAll_Hybrid_EA_SeedAvg_AveragedData.npz` | `AF_`-prefixed PDF basenames | `TARGET_NPZ` | `raw_data`; `python "../OIM/AF_OIM_Averaged_P(q).py"` |
| `MH/Pure_MH_Averaged_P(q).py` | Pure-MH energy, overlap, clustering, and fields | 0 | local CPU analysis | unprefixed MH archive | four PDFs and printed statistics | `TARGET_NPZ` | `raw_data`; `python "../MH/Pure_MH_Averaged_P(q).py"` |
| `MH/AF_Pure_MH_Averaged_P(q).py` | AF pure-MH analysis | -4 | local CPU analysis | AF MH archive | `AF_`-prefixed PDF basenames | `TARGET_NPZ` | `raw_data`; `python "../MH/AF_Pure_MH_Averaged_P(q).py"` |
| `energy_prog/energy_prog.py` | Recreates pure MH for seed 52 and plots representative OIM+/MH histories | 0 | local CPU; expensive | `AllToAll_Hybrid_EA.npz` | two comparison PDFs | seed, target archive, MH schedule | `raw_data`; `python ../energy_prog/energy_prog.py` |
| `dendrogram/dendograms.py` | Recreates seed-52 pure MH and plots three Z2 dendrograms | 0 | local CPU; expensive | `AllToAll_Hybrid_EA.npz` | three manuscript-named PDFs | seed, target archive, MH schedule | `raw_data`; `python ../dendrogram/dendograms.py` |
| `schedule/Schedule.py` | Plots four-epoch schedule | schedule | local CPU | none | `OIM+_Schedule_Normalized.pdf`; `OIM_Schedule.pdf` | hard-coded epoch list | `schedule`; `python Schedule.py` |
| `Gset/Gset_OIM_HPC.py` | Four-epoch MPI oscillator search and optional local MH tuning | instance | MPI/HPC; expensive | `G*.txt`; tuning mode reads `checkpoint_G*.txt.npz` | search mode writes 448-replica `checkpoint_G*.txt.npz`; tuning mode prints cut values | `RUN_MODE`, `TARGET_FILES`, replica count, oscillator schedule, MH settings | graph-data directory; `mpirun -np 32 python ../../Gset/Gset_OIM_HPC.py` |
| `Gset/Gset_OIM.py` | Applies 10k-proposal MH refinement to an archived 448-replica OIM checkpoint | instance | local CPU; expensive | `checkpoint_G*.txt.npz` | printed cut values only | `TARGET_NPZ`, refinement temperatures | `raw_data/Gset`; `python ../../Gset/Gset_OIM.py` |
| `Gset/Gset_MH.py` | Full 448-replica pure-MH G-set baseline | instance | local CPU; expensive | `G*.txt` | printed cut values only | `TARGET_TXT`, replica count, update schedule | `raw_data/Gset`; `python ../../Gset/Gset_MH.py` |
| `PP/local_field_analysis.py` | Palmer-Pond local fields, endpoint corrections, empirical seed-bootstrap kernel, and frozen-field reference | 0 | local CPU post-processing | 30-seed OIM archive | two manuscript-named figures, combined figure, PNGs, and four CSV tables | NPZ positional argument, `--outdir`, kernel bins/bootstrap constants | repository root; `python PP/local_field_analysis.py raw_data/AllToAll_Hybrid_EA_SeedAvg_AveragedData.npz --outdir PP/outputs` |

Most legacy production and analysis scripts use source-level configuration
constants. `PP/local_field_analysis.py` is an exception and accepts the input
NPZ and optional output directory through `argparse`.

The active G-set comparison uses a 448-replica oscillator population and a 448-replica pure-MH baseline. The recommended reproduction path uses the supplied checkpoints: `Gset_OIM.py` reads their population from `replica_states.shape[0]`, while `Gset_MH.py` exposes 448 as its comparison default. `Gset_OIM_HPC.py` restores the end-to-end option: search mode distributes 448 replicas over MPI ranks, uses Numba threading within each rank, executes the documented four epochs, and writes the checkpoint archives. Its tuning mode is an alternate batch interface to the short MH refinement logic.

## HPC / Slurm audit

| Slurm file | Partition | Nodes | MPI ranks | CPUs/task | Time | Python entry point |
|---|---|---:|---:|---:|---:|---|
| `HPC/Single_OIM/selective.slurm` | `hamsi` | 2 | 8 | 14 | 01:00:00 | `single_EA.py` |
| `HPC/Seed_Avg_OIM/selective.slurm` | `hamsi` | 8 | 8 | 56 | 06:00:00 | `avg.py` |
| `HPC/Seed_Avg_OIM/AF/selective.slurm` | `hamsi` | 8 | 8 | 56 | 06:00:00 | `AF_avg.py` |
| `HPC/Seed_Avg_MH/MH.slurm` | `hamsi` | 2 | 8 | 14 | 01:00:00 | `MH.py` |
| `HPC/Seed_Avg_MH/AF/MH.slurm` | `hamsi` | 2 | 8 | 14 | 01:00:00 | `AF_MH.py` |
| `Gset/Gset.slurm` | `hamsi` | 8 | 32 | 14 | 24:00:00 | `Gset_OIM_HPC.py` |

All examples load `miniconda3`, activate `spinglass`, load `openmpi`, and export `NUMBA_NUM_THREADS` and `OMP_NUM_THREADS` from `SLURM_CPUS_PER_TASK`. The five SK examples invoke eight MPI ranks; the G-set generator invokes 32 MPI ranks and distributes its 448 replicas across those ranks, with Numba threading inside each rank. Original private account and scratch paths were removed from the public examples. Cluster-specific partition, modules, conda environment, and resources remain examples that require adaptation. They were not production-tested during this repository audit. Submit with `sbatch <file>.slurm` from its directory.

## Dependency audit

AST import scanning found only these third-party packages:

- `numpy`
- `scipy`
- `numba`
- `matplotlib`
- `mpi4py` (HPC/MPI drivers only)

Standard-library imports are `concurrent.futures`, `os`, `sys`, and `time`. No imports of pandas, scikit-learn, networkx, or seaborn were found. `requirements.txt` therefore contains only the five observed third-party packages. No versions or original Python version were reliably available, so packages were not arbitrarily pinned and no `environment.yml` was created.

Local availability check on the audit host found NumPy, SciPy, Numba, and Matplotlib, but not `mpi4py`. All 16 Python files passed a no-write `compile(...)` syntax check.

## Manuscript figure / data mapping

The mapping uses manuscript path, scientific branch, target archive, script output, and code parameters—not filename alone.

| Manuscript item | Method | J0 | Script | Input data | Output from script | Repo status | Confidence |
|---|---|---:|---|---|---|---|---|
| `fig:annealing_schedule` / `OIM_Schedule.pdf` | OIM+ schedule | schedule | `schedule/Schedule.py` | none | exact manuscript basename plus legacy copy | PASS | HIGH |
| `fig:energy_evolution` panel (a) / `Osc_energy_prog.pdf` | OIM+ population trajectory | 0 | `energy_prog/energy_prog.py` | `raw_data/AllToAll_Hybrid_EA.npz` | exact manuscript basename plus legacy copy | PASS | HIGH |
| `fig:energy_evolution` panel (b) / `MH_energy_prog.pdf` | Pure MH plus DH-OIM+ inset | 0 | `energy_prog/energy_prog.py` | same archive plus regenerated pure MH | exact manuscript basename plus legacy copy | PASS | HIGH |
| `fig:energy_histogram` panel (a) / `Avg_Plot_3_Comparative_Energy.pdf` | OIM+ vs DH-OIM+ energy | 0 | `OIM/OIM_Averaged_P(q).py` | unprefixed OIM archive | exact manuscript basename | PASS | HIGH |
| `fig:energy_histogram` panel (b) / `Avg_Plot_PureMH_EnergyDist.pdf` | Pure-MH energy | 0 | `MH/Pure_MH_Averaged_P(q).py` | unprefixed MH archive | exact manuscript basename | PASS | HIGH |
| `fig:qmatrix` panel (a) / `Avg_Plot_2_Comparative_Overlap.pdf` | OIM+ vs DH-OIM+ `P(q)` | 0 | `OIM/OIM_Averaged_P(q).py` | unprefixed OIM archive | exact manuscript basename | PASS | HIGH |
| `fig:qmatrix` panel (b) / `Avg_Plot_PureMH_Overlap.pdf` | Pure-MH `P(q)` | 0 | `MH/Pure_MH_Averaged_P(q).py` | unprefixed MH archive | exact manuscript basename | PASS | HIGH |
| `fig:dendrograms` / `Dendrogram_1_PureMH.pdf` | Pure-MH Z2 clustering | 0 | `dendrogram/dendograms.py` | representative archive; regenerated pure MH | exact manuscript basename | PASS | HIGH |
| `fig:dendrograms` / `Dendrogram_2_PostOscillator.pdf` | OIM+ Z2 clustering | 0 | `dendrogram/dendograms.py` | representative archive | exact manuscript basename | PASS | HIGH |
| `fig:dendrograms` / `Dendrogram_3_PostMH.pdf` | DH-OIM+ Z2 clustering | 0 | `dendrogram/dendograms.py` | representative archive | exact manuscript basename | PASS | HIGH |
| `fig:field_distribution` panel (a) / `Main_LocalFields_Palmer_J0_0.pdf` | Local fields and Palmer-Pond comparison | 0 | `PP/local_field_analysis.py` | `raw_data/AllToAll_Hybrid_EA_SeedAvg_AveragedData.npz` | exact manuscript basename | PASS | HIGH; expected statistics reproduced and plot visually compared |
| `fig:field_distribution` panel (b) / `Main_Kemp_vs_Kfrz_J0_0.pdf` | Empirical vs frozen-field correction kernel with bootstrap intervals | 0 | `PP/local_field_analysis.py` | same 30-seed archive | exact manuscript basename | PASS | HIGH; current script is the canonical reproducible implementation |
| `fig:pq_J0m4` panel (a) / `AF_Avg_Plot_2_Comparative_Overlap.pdf` | OIM+ vs DH-OIM+ `P(q)` | -4 | `OIM/AF_OIM_Averaged_P(q).py` | AF OIM archive | exact manuscript basename | PASS | HIGH |
| `fig:pq_J0m4` panel (b) / `AF_Avg_Plot_PureMH_Overlap.pdf` | Pure-MH `P(q)` | -4 | `MH/AF_Pure_MH_Averaged_P(q).py` | AF MH archive | exact manuscript basename | PASS | HIGH |
| `fig:afm_energies` panel (a) / `AF_Avg_Plot_3_Comparative_Energy.pdf` | OIM+ vs DH-OIM+ energy | -4 | `OIM/AF_OIM_Averaged_P(q).py` | AF OIM archive | exact manuscript basename | PASS | HIGH |
| `fig:afm_energies` panel (b) / `AF_Avg_Plot_PureMH_EnergyDist.pdf` | Pure-MH energy | -4 | `MH/AF_Pure_MH_Averaged_P(q).py` | AF MH archive | exact manuscript basename | PASS | HIGH |

AF verification from stored data: the AF matrices have the intended mean shift `-4/200=-0.02` relative to the corresponding unbiased matrices to numerical precision. Paired OIM and pure-MH `master_J` arrays are identical in each branch.

## Manuscript table mapping

| Manuscript table | Source relationship | Status | Notes |
|---|---|---|---|
| `tab:annealing_schedule` | Hard-coded epoch lists in single- and multi-seed OIM drivers | PASS | Steps, `dt`, temperature endpoints, and SHIL endpoints agree |
| `tab:simulation_parameters` | OIM driver control panels and selection loop | PASS | `N=200`, 402 replicas, four epochs, two-thirds cull |
| `tab:mh_protocols` | OIM refinement and pure-MH drivers | PASS | 10k refinement and 100k/10k/10k pure-MH stages agree |
| `tab:distance_metrics` | OIM/MH analysis scripts and archived state arrays | PASS | Z2 distances, average linkage, and largest-gap threshold are represented |
| `tab:weak_field_metrics` | `calculate_weak_field_metrics` in OIM scripts plus `PP/local_field_analysis.py` | PASS | Both panels and all reported scalar metrics are reproducible from the archived 30-seed data |
| `tab:gset_results` | `Gset_OIM_HPC.py`, `Gset_OIM.py`, `Gset_MH.py`, graph texts, and checkpoints | PASS for protocol mapping | Generator and active comparison use 448 replicas; comparison results are printed rather than archived as a separate table artifact |

## Scientific wording audit

README language was made conservative:

- “Ultrametricity and RSB quality via cophenetic correlation” was replaced with hierarchical clustering and dendrogram fidelity. Cophenetic correlation measures how well a hierarchy represents pairwise distances; it is not by itself proof of replica-symmetry breaking.
- “Basins” and “valleys” were not presented as established dynamical or thermodynamic objects.
- “Oscillator force targeting” was replaced with directly implemented local-field, incident-strength, cancellation-ratio, and corrected-spin diagnostics.
- DH-OIM+ is used consistently for the OIM+ plus short MH workflow.
- No scientific manuscript claims were edited.

## Hard-coded path and personal-information audit

Findings at audit start:

- all five Slurm files contained a personal cluster account name;
- all five Slurm files contained a personal `<SCRATCH_DIR>/...` path;
- three Slurm files invoked filenames absent from their curated directories (`EA.py`, AF `avg.py`, AF `MH.py`);
- README exposed a personal email address;
- local analysis scripts depend on the current working directory and hard-coded target basenames;
- G-set scripts require source edits to select an instance;
- no absolute Windows path was embedded in active Python source.

The subsequently restored `Gset/Gset.slurm` also contained a private account,
scratch path, citation placeholders, and the stale `Gset.py` entry point.

Public-facing remediation:

- account and scratch paths were removed from Slurm examples;
- Slurm examples now change to their own file directory and invoke the actual curated filename;
- the public README omits personal contact details and documents relative working-directory commands;
- scientific Python files were not rewritten because changing data-location behavior was unnecessary and could complicate provenance.

## Raw-data audit

- **Total:** 23 files, 436,948,963 bytes (approximately 416.7 MiB).
- **Types:** 14 `.npz` files (approximately 415.5 MiB) and nine G-set `.txt` files (approximately 1.2 MiB).
- **Largest:** two OIM seed-averaged archives at approximately 64.6 MiB each; two pure-MH archives at approximately 46.1 MiB each; four 2,000-node G-set checkpoints at approximately 37.4 MiB each.
- **GitHub:** no individual file exceeds 100 MiB. The two approximately 64.6 MiB OIM archives exceed GitHub's 50 MiB warning threshold; the approximately 46.1 MiB pure-MH archives do not. The repository is large for a normal source clone.
- **Relevance:** the five SK archives are directly used by manuscript analysis/representative figures. G-set graph texts and the archived 448-replica checkpoints support the table workflow; `Gset/Gset_OIM_HPC.py` provides the checkpoint-regeneration path.

Recommendation: use Git LFS or place full archives in Zenodo/an institutional repository and keep a reduced representative dataset in Git. Because these files are already tracked, LFS migration or history rewriting must be a deliberate manual action; it was not performed automatically.

## Portability and run-command validation

- All documented script paths exist.
- All 16 Python files passed an in-memory syntax compilation without writing `__pycache__`.
- NumPy, SciPy, Numba, and Matplotlib were import-discoverable on the audit host; `mpi4py` was not installed locally.
- Production, energy-progression, dendrogram, G-set, and full analysis programs were not executed because they are expensive or write scientific outputs.
- `PP/local_field_analysis.py` was executed against the tracked 30-seed archive. It reproduced the documented scalar values and generated all declared CSV/PDF/PNG outputs.
- Slurm commands were checked against actual filenames and script main blocks.
- Analysis commands intentionally run from `raw_data/` or `raw_data/Gset/` to satisfy existing current-working-directory assumptions.
- Windows Git operations succeeded with `git -c core.longpaths=true`; ordinary Git commands from the very long parent path produced filename-too-long object errors. A shorter clone path is recommended.

## `.gitignore` audit

The existing Python-oriented ignore file already covered bytecode, virtual environments, caches, test artifacts, and `*.log`. It was extended for `.DS_Store`, Slurm/stdout-stderr artifacts (`*.out`, `*.err`), common editor swap/backup files, and IDE metadata. Scientific `.npz`, `.txt`, `.pdf`, `.png`, and `.svg` files were deliberately not ignored globally.

## License

The repository contains the MIT License, copyright 2026 Ömer Önder. The previous README incorrectly said a license still needed to be added; it now identifies MIT consistently. No license replacement was made.

## Cleanup candidates (not deleted)

| Item | Classification | Reason |
|---|---|---|
| `*/info.txt` files | NEEDS MANUAL DECISION | Short provenance notes, but several contain typos or stale resource descriptions |
| Generated Slurm `*.out` / `*.err` files if later created | LIKELY TEMPORARY | Now ignored; not present in curated tree at audit time |
| AF and unbiased analysis outputs | RETAIN SEPARATELY | Maintained AF scripts now emit `AF_`-prefixed names, preventing accidental overwrite of `J0=0` outputs |
| Large tracked `.npz` archives | NEEDS MANUAL DECISION | Reproduction-relevant, but candidates for LFS/external archival storage |
| Adjacent `clutter/` and `.rar` outside the Git repository | SAFE TO EXCLUDE FROM PUBLIC REPO | Legacy/working material not in the curated repository |

No file was permanently deleted.

## Remaining manual actions before public release

1. Confirm the original Python, NumPy, SciPy, Numba, Matplotlib, mpi4py, MPI, compiler, and BLAS versions; optionally add a locked environment after confirmation.
2. Decide whether to use Git LFS or an external DOI-bearing data archive; update README links accordingly.
3. Decide whether public contact information should be added back intentionally.
4. Run the remaining documented analysis commands in a clean environment and visually compare regenerated PDFs with the manuscript versions.
5. Test all six Slurm examples on the intended cluster after setting account/environment details and staging the G-set graph inputs for the generator job.
6. Use a shorter Windows checkout path or enable Git long-path support before normal Git operations.
7. Review the working-tree diff, then commit and push manually if satisfied. This audit did neither.
