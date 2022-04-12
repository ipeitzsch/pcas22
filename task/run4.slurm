#!/bin/bash

#SBATCH --job-name=DataParallel
#SBATCH --output=run4.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=gtx1080
#SBATCH --gres=gpu:4
#SBATCH --account=ece2166_2022s

module purge
module load python/3.7.0
module load cuda/11.6.1
module load gcc/5.4.0

pip install --user medmnist
#pip install --user numba

python nn.py -i 229760 -g 4 -o fourGPULarge.csv -r 50
