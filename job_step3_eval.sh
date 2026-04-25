#!/bin/bash
#PBS -N kd_eval
#PBS -P col772.mcs252107.course
#PBS -q standard
#PBS -l select=1:ncpus=8:ngpus=1:mem=32G:centos=skylake
#PBS -l walltime=1:00:00
#PBS -o eval.out
#PBS -e eval.err

cd $PBS_O_WORKDIR
module load compiler/cuda/12.1
module load apps/anaconda/3

# Eval Qwen
echo "Evaluating Qwen..."
python inference_eval.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter_path output/distilled_qwen \
    --test_data data/test.jsonl \
    --output_predictions predictions_qwen.jsonl \
    --report_file metrics_qwen.txt \
    --time_limit 25

# Eval LLaMA
echo "Evaluating LLaMA..."
python inference_eval.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --adapter_path output/distilled_llama \
    --test_data data/test.jsonl \
    --output_predictions predictions_llama.jsonl \
    --report_file metrics_llama.txt \
    --time_limit 25

echo "=== RESULTS ==="
echo "--- Qwen ---"
cat metrics_qwen.txt
echo ""
echo "--- LLaMA ---"
cat metrics_llama.txt