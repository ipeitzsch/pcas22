#!/bin/bash

#SBATCH --job-name=DataParallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=gtx1080
#SBATCH --gres=gpu:2
#SBATCH --account=ece2166_2022s
#SBATCH --output=run2
module purge
module load python/3.7.0
module load cuda/11.6.1
module load gcc/5.4.0

pip install --user medmnist
pip install --user numba

python nn.py -i 229760 -g 2 -o twoGPULarge.csv -r 50
