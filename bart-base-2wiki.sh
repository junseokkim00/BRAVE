#!/bin/sh

#SBATCH -J bart_2wiki.sh
#SBATCH -p titanxp
#SBATCH --gres=gpu:1
#SBATCH -o bart_2wiki.out

python bart-base-2wiki.py