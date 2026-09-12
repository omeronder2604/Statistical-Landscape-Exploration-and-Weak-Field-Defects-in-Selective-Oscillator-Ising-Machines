#!/bin/bash
#SBATCH -p hamsi
#SBATCH -A omonder
#SBATCH -J MH
#SBATCH -N 2                   # Requesting 2 nodes to spread out the heavy computation
#SBATCH --ntasks=8             # 8 MPI processes total (1 process for each of our 8 lambda values)
#SBATCH --cpus-per-task=14     # 14 cores per MPI task (8 tasks * 14 cores = 112 cores across 2 nodes)
#SBATCH --time=01:00:00        
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

cd /arf/scratch/omonder/Remedy/linear/MH/AF/

# 1. Load the required modules in the correct order
module load miniconda3 
conda activate spinglass
module load openmpi

# 2. Tell Numba to use the cores we allocated for each task
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 3. Launch the Python script using mpirun
mpirun -np 8 python MH.py