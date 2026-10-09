#!/bin/bash
#SBATCH -p hamsi
#SBATCH -A omonder
#SBATCH -J Gset_phase
#SBATCH -N 8
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

cd /arf/scratch/omonder/Gset/phase_record/

module purge
module load miniconda3
conda activate spinglass

export NUMBA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONIOENCODING=utf8

# Work around the MPICH/Hydra failure in MPIDU_Init_shm_init.
# This solver uses MPI only for the final gather, so bypassing local
# shared-memory MPI transport has negligible performance cost.
export MPIR_CVAR_NOLOCAL=1

PYTHON_BIN="$(command -v python)"
CONDA_BIN="$(dirname "$PYTHON_BIN")"
MPIEXEC="$CONDA_BIN/mpiexec"

if [ ! -x "$MPIEXEC" ]; then
    echo "ERROR: No mpiexec found next to active conda Python: $MPIEXEC" >&2
    echo "Python: $PYTHON_BIN" >&2
    exit 2
fi

echo "=== Runtime ==="
echo "Python : $PYTHON_BIN"
echo "MPIExec: $MPIEXEC"
"$MPIEXEC" --version | head -n 5 || true
python - <<'PY'
from mpi4py import MPI
print("mpi4py MPI library:")
print(MPI.Get_library_version())
PY

echo "=== Launch ==="
"$MPIEXEC" -np "$SLURM_NTASKS" python Gset_save_epoch_phases.py
