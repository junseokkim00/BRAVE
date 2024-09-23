#!/bin/sh

#SBATCH -J bart.sh
#SBATCH -p titanxp
#SBATCH --gres=gpu:2
#SBATCH -o bart.out

python bart-base.py