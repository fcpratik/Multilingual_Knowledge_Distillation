#!/bin/bash
#PBS -N kd_datagen
#PBS -P col772.mcs252107.course
#PBS -q standard
#PBS -l select=1:ncpus=8:ngpus=1:mem=32G:centos=skylake
#PBS -l walltime=4:00:00
#PBS -o datagen.out
#PBS -e datagen.err

cd $PBS_O_WORKDIR
module load compiler/cuda/12.1
module load apps/anaconda/3

# Activate your environment (change name if needed)
# conda activate kd_env

pip install -r requirements.txt --quiet

python dataset_generation.py \
    --teacher_model Qwen/Qwen2.5-7B-Instruct \
    --num_samples 4000,2500,1500,1200,800 \
    --output_file data/train.jsonl \
    --dataset_path data/dataset.jsonl \
    --gpu_memory_utilization 0.90 \
    --time_limit 230