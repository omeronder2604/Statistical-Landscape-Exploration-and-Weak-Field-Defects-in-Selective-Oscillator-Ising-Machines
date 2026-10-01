# G-set comparison workflow

The active G-set comparison uses 448 oscillator replicas and a 448-replica
pure-MH baseline. The supplied checkpoints can be used directly as archived
inputs or regenerated with the MPI oscillator program.

## A. HPC checkpoint generation

`Gset_OIM_HPC.py` is the active checkpoint generator. With
`RUN_MODE = "search"`, it distributes `TOTAL_REPLICAS = 448` across the MPI
ranks, uses Numba threading within each rank, executes the four oscillator
epochs, and writes `checkpoint_G*.txt.npz`. Deterministic initialization and
per-replica seeds are derived with NumPy `SeedSequence`.

`Gset.slurm` requests 8 nodes, 32 MPI ranks, and 14 CPUs per task for Numba
threading. This resource layout is not a claim that the replica count is
defined by the number of allocated CPU cores. Because the generator uses
working-directory-relative graph names, stage the listed `G*.txt` inputs next
to the script before submitting the public Slurm example:

```bash
cd Gset
sbatch Gset.slurm
```

Alternatively, launch the generator from `raw_data/Gset` so it reads the
tracked graph files and writes regenerated checkpoints there:

```bash
cd raw_data/Gset
mpirun -np 32 python ../../Gset/Gset_OIM_HPC.py
```

Do not overwrite the supplied archived checkpoints unless replacement is
intentional and separately validated.

## B. Local DH-OIM+ refinement

The recommended repository reproduction path uses the supplied archived
checkpoints with `Gset_OIM.py`. Select one file through `TARGET_NPZ`; the
script reads the actual population from `replica_states.shape[0]` and applies
the short MH refinement:

```bash
cd raw_data/Gset
python ../../Gset/Gset_OIM.py
```

Setting `RUN_MODE = "tuning"` in `Gset_OIM_HPC.py` provides an optional batch
interface to the same MH transition logic for every listed checkpoint. It is
an alternate interface to the refinement stage, not a different optimization
algorithm; its exposed tuning parameters should be checked for the intended
run before use.

## C. Pure-MH comparison

`Gset_MH.py` runs the independent pure-MH baseline directly from a selected
`G*.txt` graph and defaults to 448 replicas:

```bash
cd raw_data/Gset
python ../../Gset/Gset_MH.py
```

Select an instance by editing `TARGET_NPZ` in `Gset_OIM.py` or `TARGET_TXT`
in `Gset_MH.py`. These workflows are computationally expensive. Refinement
and pure-MH programs print Max-Cut summaries rather than writing a separate
results archive.
