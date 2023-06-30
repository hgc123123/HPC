#!/bin/bash
#SBATCH --job-name=osu
#SBATCH --partition=parallel
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --exclusive
#SBATCH --nodelist=compute[141,142]
module load osu-micro-benchmarks/7.1-1-gcc-13.1.0 gcc openmpi
ulimit -l unlimited
mpirun osu_bibw
