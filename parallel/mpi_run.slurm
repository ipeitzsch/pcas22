#!/bin/bash

#SBATCH --job-name=DataParallel
#SBATCH --output=output
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=gtx1080
#SBATCH --gres=gpu:1
#SBATCH --account=ece2166_2022s

module purge
module load python/3.7.0
module load cuda/11.6.1
module load gcc/5.4.0
module load openmpi/3.0.0

pip install --user medmnist
pip install --user numba
pip install --user mpi4py

NUM=1

mpirun -n $NUM python dataPara_mpi.py -i 229760 -g --num $NUM
