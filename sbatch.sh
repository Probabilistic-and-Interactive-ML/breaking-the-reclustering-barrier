#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=p200289
#SBATCH --cpus-per-task=128
#SBATCH --qos=default
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --output=logs/dcn_cifar.log
source ~/.bashrc
mamba activate brb
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0,1,2,3
python runner.py dcn_cifar --workers 4 --num_workers_dataloader 16 --gpu 0 1 2 3