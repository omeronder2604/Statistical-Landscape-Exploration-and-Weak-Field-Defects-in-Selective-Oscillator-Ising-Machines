#!/bin/bash
#SBATCH -p hamsi
#SBATCH -J Gset  
#SBATCH -N 8                   # 8 nodes
#SBATCH --ntasks=32            # 32 MPI processes total
#SBATCH --cpus-per-task=14     # 14 CPUs per MPI task for Numba threading
#SBATCH --time=24:00:00        
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Load the required modules
module load miniconda3 
conda activate spinglass
module load openmpi

# 2. Tell Numba to use the CPUs allocated to each MPI rank
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONIOENCODING=utf8

# 3. Launch the Python script using mpirun
mpirun -np 32 python Gset_OIM_HPC.py
