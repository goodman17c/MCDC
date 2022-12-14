#!/bin/tcsh
#SBATCH -N 40
#SBATCH -t 8:00:00

srun -n 1440 python inputraww.py --mode=numba
