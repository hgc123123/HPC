#!/bin/bash
#SBATCH --job-name=osu_bibw
#SBATCH --partition=parallel
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --exclusive
module load oneapi
module load osu-micro-benchmarks/7.1-1-oneapi-2023.1.0
ulimit -l unlimited
mpirun osu_bibw
