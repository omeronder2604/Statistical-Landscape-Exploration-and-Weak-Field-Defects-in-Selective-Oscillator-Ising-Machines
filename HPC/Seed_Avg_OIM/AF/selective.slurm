#!/bin/bash
#SBATCH -p hamsi
#SBATCH -A omonder
#SBATCH -J SK
#SBATCH -N 8                    # Requesting 8 distinct nodes
#SBATCH --ntasks=8              # 8 MPI processes total
#SBATCH --ntasks-per-node=1     # 1 MPI task per node prevents memory/network bottlenecks
#SBATCH --cpus-per-task=56      # Dedicating all 56 cores on a hamsi node to Numba's prange
#SBATCH --time=06:00:00        
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

cd /arf/scratch/omonder/Remedy/linear/Seed_Avg_Sel/AF/

# 1. Load the required modules in the correct order
module load miniconda3 
conda activate spinglass
module load openmpi

# 2. Tell Numba to use the cores we allocated for each task
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 3. Force UTF-8 encoding to prevent Python UnicodeDecodeErrors in SLURM output
export PYTHONIOENCODING=utf8

# 4. Launch the Python script using mpirun
mpirun -np 8 python avg.py