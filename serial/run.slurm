#!/bin/bash

#SBATCH --job-name=serial_nn
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28
#SBATCH --cluster=mpi
#SBATCH --partition=opa
#SBATCH --time=0-01:00:00
#SBATCH --account=ece2166_2022s

module purge
module load python/3.7.0
module load gcc/5.4.0
module load openmpi/3.0.0

pip install --user medmnist


NUM=4

python nn.py -f model.pt -i 229760 -g --num $NUM