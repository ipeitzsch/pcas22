#!/bin/bash

#SBATCH --job-name=DataParallel
#SBATCH --output=output
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=gtx1080
#SBATCH --gres=gpu:4
#SBATCH --account=ece2166_2022s

module purge
module load python/3.7.0
module load cuda/11.6.1
module load gcc/5.4.0
module load openmpi/3.0.0

pip install --user medmnist
pip install --user numba
pip install --user mpi4py

NUM=4

mpirun -n $NUM python dataPara.py -f model.pt -i 229760 -g --num $NUM
