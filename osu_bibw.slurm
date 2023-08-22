#!/bin/bash
#SBATCH --job-name=osu
#SBATCH --partition=parallel
#SBATCH -n 2                              # There are two processes
#SBATCH --ntasks-per-node=1               # one process per node
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --exclusive
module load gcc openmpi \
  osu-micro-benchmarks/7.1-1-gcc-13.1.0
ulimit -l unlimited
mpirun osu_bibw
