#!/bin/tcsh
#BSUB -nnodes 9
#SBATCH -N 10
#SBATCH -p pbatch
#SBATCH -t 2:00:00

srun -n 360 python input0.py
srun -n 360 python input10.py
