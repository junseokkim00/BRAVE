#!/bin/sh

#SBATCH -J main.sh
#SBATCH -p titanxp
#SBATCH --gres=gpu:2
#SBATCH -o eval.out

model_name=None
dataset=2wikimh
split=dev
generation_model_name=bart



python3 main.py --run_type oracle \
    --dataset ${dataset} \
    --model_name ${model_name} \
    --split ${split} \
    --generation_model_name ${generation_model_name} \
    --seed 2024 \
    --num_data 100 \
    --metadata _bart

python3 main.py --run_type no_context \
    --dataset ${dataset} \
    --model_name ${model_name} \
    --split ${split} \
    --generation_model_name ${generation_model_name} \
    --seed 2024 \
    --num_data 100 \
    --metadata _bart

python3 main.py --run_type rag \
    --dataset ${dataset} \
    --model_name ${model_name} \
    --split ${split} \
    --generation_model_name ${generation_model_name} \
    --seed 2024 \
    --num_data 100 \
    --metadata _bart

python3 main.py --run_type brave_bert \
    --dataset ${dataset} \
    --model_name ${model_name} \
    --split ${split} \
    --generation_model_name ${generation_model_name} \
    --seed 2024 \
    --num_data 100 \
    --metadata _bart

# python3 main.py --run_type brave_wo_QD \
#     --dataset ${dataset} \
#     --model_name ${model_name} \
#     --split ${split} \
#     --generation_model_name ${generation_model_name} \
#     --seed 2024 \
#     --num_data 100 \
#     --metadata _bart

# python3 main.py --run_type brave_me \
#     --dataset ${dataset} \
#     --model_name ${model_name} \
#     --split ${split} \
#     --generation_model_name ${generation_model_name} \
#     --seed 2024 \
#     --num_data 100 \
#     --metadata _bart


# python3 main.py --run_type oracle \
#     --dataset ${dataset} \
#     --model_name ${model_name} \
#     --split ${split} \
#     --generation_model_name ${generation_model_name} \
#     --seed 2024 \
#     --num_data 100 \
#     --metadata _bart


# python3 main.py --run_type brave_bert \
#     --model_name gpt-3.5-turbo \
#     --generation_model_name bart \
#     --split validation \
#     --seed 2024 \
#     --num_data 100 \
#     --metadata _bart

python3 main.py --run_type rag \
    --model_name gpt-4o-mini \
    --generation_model_name bart \
    --split validation \
    --seed 2024 \
    --num_data 100 \
    --metadata _bart


# python3 main.py --run_type brave_wo_QD \
#     --model_name gpt-4o-mini \
#     --generation_model_name bart \
#     --split validation \
#     --seed 2024 \
#     --num_data 100 \
#     --metadata _bart

# python3 main.py --run_type brave_me \
#     --model_name gpt-3.5-turbo \
#     --generation_model_name bart \
#     --split validation \
#     --seed 2024 \
#     --num_data 100 \
#     --metadata _bart