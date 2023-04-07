#!/bin/tcsh
#SBATCH -N 10
#SBATCH -p pbatch
#SBATCH -t 24:00:00

srun -n 360 python input1.py
srun -n 360 python input2.py
srun -n 360 python input4.py
srun -n 360 python input5.py
srun -n 360 python input6.py
srun -n 360 python input7.py
srun -n 360 python input9.py
