# G-set data and executable workflows

The nine G-set rows in the manuscript use graph inputs and historical OIM+ checkpoint populations retained under `raw_data/Gset/`. The repository also contains programs that can be executed **now**. These are distinct kinds of evidence: the current MPI oscillator program is not established as the source of the historical checkpoints.

## A. Retained historical material

The tracked graph files are `G1.txt`, `G6.txt`, `G11.txt`, `G14.txt`, `G15.txt`, `G22.txt`, `G23.txt`, `G36.txt`, and `G39.txt`. Each has a matching `checkpoint_G*.txt.npz` file. All nine checkpoints contain the graph matrix `J`, node count `N`, and `replica_states`: **448 continuous-phase replicas**. They are pre-refinement inputs, not final DH-OIM+ solutions.

[`raw_data/Gset/gset_reported_results.csv`](../raw_data/Gset/gset_reported_results.csv) is a machine-readable transcription of the nine **reported manuscript table rows**. It is **not reconstructed archived run output**. Final configurations and run logs underlying those values were not retained.

## B. Current local analysis and refinement code

`Gset_OIM.py` reads one supplied checkpoint, reports the pre-MH maximum cut, and applies 10,000 MH proposals per replica with a 0.1-to-0.01 temperature schedule. `Gset_MH.py` independently runs a 448-replica pure-MH comparison from one graph text. Both print summaries; neither writes a final-result archive.

From the **repository root**, edit the `TARGET_NPZ` and `TARGET_TXT` constants to choose the matching instance, then run:

```bash
(cd raw_data/Gset && python ../../Gset/Gset_OIM.py)
(cd raw_data/Gset && python ../../Gset/Gset_MH.py)
```

These are new executions. A matching instance name does not prove identity with the historical manuscript run.

## C. Current MPI oscillator program

`Gset_OIM_HPC.py` has a `RUN_MODE = "search"` path that reads the nine graph texts, evolves 448 phase replicas across four epochs, and writes `checkpoint_G*.txt.npz` files. **Its current search path does not perform the manuscript's energy-based selection and repopulation between epochs.** It is therefore **not a reproduction of the manuscript's selective OIM+ checkpoint history**, and it is **not established as the historical generator of the archived September checkpoints**.

The program also has an optional `RUN_MODE = "tuning"` path. Its current 0.2-to-0.1 MH settings differ from `Gset_OIM.py` and the manuscript's 0.1-to-0.01 G-set refinement settings. It imports `mpi4py` in either mode. Inspect source settings before using either path.

To run the search without risking the retained checkpoints, from the **repository root**, create a separate directory containing copies of the tracked graph texts:

```bash
repo_dir="$(pwd)"
run_dir="$(mktemp -d)"
cp raw_data/Gset/G*.txt "$run_dir/"
(cd "$run_dir" && mpirun -np 32 python "$repo_dir/Gset/Gset_OIM_HPC.py")
```

This requires a compatible system MPI and suitable resources. For a multinode cluster job, choose a run directory on a filesystem shared by all nodes instead of a node-local temporary directory. `Gset.slurm` is a TRUBA-oriented example: it changes into its own `Gset/` script directory, so graph texts must be staged there for that exact example. If adapting the Slurm script, use a separate run directory and update its paths for that site. **Do not overwrite the retained historical `raw_data/Gset/checkpoint_G*.txt.npz` files.** Any files produced by the current search are new, nonselective-run checkpoints and must be labeled as such.

## D. Missing historical provenance

The retained data do not establish the original checkpoint generator's identity or revision, the selection/repopulation history, run logs, final refined DH-OIM+ configurations, pure-MH configurations, per-replica result arrays, or the exact original Python/package/MPI/compiler/BLAS environment. Current executions can produce new results; they cannot reconstruct the historical September checkpoint-producing run from the retained evidence alone.
