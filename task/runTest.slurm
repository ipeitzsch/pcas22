#!/bin/bash

#SBATCH --job-name=torchTest
#SBATCH --output=test.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=gtx1080
#SBATCH --gres=gpu:2

module purge
module load python/3.7.0
module load cuda/11.6.1
module load gcc/5.4.0

pip install --user medmnist
pip install --user numba

python test.py
