#!/bin/tcsh
#BSUB -nnodes 9
#SBATCH -N 10
#SBATCH -p pbatch
#SBATCH -t 6:00:00

srun -n 360 python input2.py
srun -n 360 python input4.py
srun -n 360 python input5.py
srun -n 360 python input6.py
srun -n 360 python input7.py
srun -n 360 python input8.py
srun -n 360 python input9.py
