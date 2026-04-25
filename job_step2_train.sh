#!/bin/bash
#PBS -N kd_train
#PBS -P col772.mcs252107.course
#PBS -q standard
#PBS -l select=1:ncpus=8:ngpus=1:mem=32G:centos=skylake
#PBS -l walltime=4:00:00
#PBS -o train.out
#PBS -e train.err

cd $PBS_O_WORKDIR
module load compiler/cuda/12.1
module load apps/anaconda/3

# Train Qwen student (in-family)
echo "Training Qwen2.5-1.5B..."
python train_distill.py \
    --student_model Qwen/Qwen2.5-1.5B-Instruct \
    --train_data data/train.jsonl \
    --output_dir output/distilled_qwen \
    --epochs 3 --lr 2e-4 --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --time_limit 110

# Train LLaMA student (cross-family)
echo "Training LLaMA-3.2-1B..."
python train_distill.py \
    --student_model meta-llama/Llama-3.2-1B-Instruct \
    --train_data data/train.jsonl \
    --output_dir output/distilled_llama \
    --epochs 3 --lr 2e-4 --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --time_limit 110