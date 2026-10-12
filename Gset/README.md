# G-set data and executable workflows

The nine G-set rows in the manuscript use graph inputs and saved oscillator phase populations retained under `raw_data/Gset/`. This page describes the retained inputs and the available analysis and simulation programs.

## A. Retained data and reported results

The tracked graph files are `G1.txt`, `G6.txt`, `G11.txt`, `G14.txt`, `G15.txt`, `G22.txt`, `G23.txt`, `G36.txt`, and `G39.txt`. Each has a matching `checkpoint_G*.txt.npz` file. Each checkpoint contains the graph matrix `J`, node count `N`, and `epoch_1_phis` through `epoch_4_phis`: four saved populations of **448 continuous-phase configurations**. `replica_states` equals the retained E4 population. These are pre-refinement inputs; post-DH final configurations are not stored in the checkpoint NPZs.

The intermediate table's maximum cut, mean cut, and top-third threshold (the 149th-highest cut among 448 configurations) are computed directly from each saved population. Run `python Gset/reproduce_reported_gset_results.py` from the repository root to recompute them and the reported DH diagnostic from committed inputs; use `--ref HEAD` to select the local commit explicitly. The script requires NumPy, Numba, and SciPy. The diagnostic binarizes E4 using `sign(cos(phi))`, then applies 10,000 single-spin Metropolis proposals per replica with a linear 0.1-to-0.01 temperature schedule and the seeded replica streams used by `Gset_OIM.py`. The script prints the best-known reference values and calculated percentages.

[`raw_data/Gset/gset_reported_results.csv`](../raw_data/Gset/gset_reported_results.csv) is a machine-readable transcription of the nine **reported manuscript table rows**, not raw simulation output. Post-DH configurations and historical scheduler/job metadata are not part of the released checkpoint artifact set.

## B. Current local analysis and refinement code

`Gset_OIM.py` reads one supplied checkpoint, reports the pre-MH maximum cut, and applies 10,000 MH proposals per replica with a 0.1-to-0.01 temperature schedule. `Gset_MH.py` independently runs a 448-replica pure-MH comparison from one graph text. Both print summaries; neither writes a final-result archive.

From the **repository root**, edit the `TARGET_NPZ` and `TARGET_TXT` constants to choose the matching instance, then run:

```bash
(cd raw_data/Gset && python ../../Gset/Gset_OIM.py)
(cd raw_data/Gset && python ../../Gset/Gset_MH.py)
```

These commands perform local calculations and do not modify the retained checkpoints.

## C. Current MPI oscillator program

`Gset_OIM_HPC.py` has a `RUN_MODE = "search"` path that reads the nine graph texts, evolves 448 phase replicas across four epochs, and writes `checkpoint_G*.txt.npz` files. Its current search path does not perform energy-based selection and repopulation between epochs.

The program also has an optional `RUN_MODE = "tuning"` path. Its current 0.2-to-0.1 MH settings differ from `Gset_OIM.py` and the manuscript's 0.1-to-0.01 G-set refinement settings. It imports `mpi4py` in either mode. Inspect source settings before using either path.

To run the search without risking the retained checkpoints, from the **repository root**, create a separate directory containing copies of the tracked graph texts:

```bash
repo_dir="$(pwd)"
run_dir="$(mktemp -d)"
cp raw_data/Gset/G*.txt "$run_dir/"
(cd "$run_dir" && mpirun -np 32 python "$repo_dir/Gset/Gset_OIM_HPC.py")
```

This requires a compatible system MPI and suitable resources. For a multinode cluster job, choose a run directory on a filesystem shared by all nodes instead of a node-local temporary directory. `Gset.slurm` is a TRUBA-oriented example: it changes into `/arf/scratch/omonder/Gset/phase_record/`, so graph texts must be staged there for that exact example. If adapting the Slurm script, use a separate run directory and update its paths for that site. Do not overwrite the released `raw_data/Gset/checkpoint_G*.txt.npz` files with output from a new search.
