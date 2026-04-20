#!/bin/bash
set -e

TEACHER_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"
STUDENT_QWEN="Qwen/Qwen2.5-1.5B-Instruct"
STUDENT_LLAMA="meta-llama/Llama-3.2-1B-Instruct"
NUM_SAMPLES="4000,2500,1500,1200,800"
DATASET_PATH="data/dataset.jsonl"
TRAIN_DATA="data/train.jsonl"
TEST_DATA="data/test.jsonl"
OUTPUT_DIR_QWEN="output/distilled_qwen"
OUTPUT_DIR_LLAMA="output/distilled_llama"

echo "============================================================"
echo " Step 1: Dataset Generation (limit: 230 min)"
echo "============================================================"
python dataset_generation.py \
    --teacher_model ${TEACHER_MODEL} \
    --num_samples ${NUM_SAMPLES} \
    --output_file ${TRAIN_DATA} \
    --dataset_path ${DATASET_PATH} \
    --gpu_memory_utilization 0.85 \
    --time_limit 230

echo ""
echo "============================================================"
echo " Step 2a: Distill Qwen (limit: 110 min)"
echo "============================================================"
python train_distill.py \
    --student_model ${STUDENT_QWEN} \
    --train_data ${TRAIN_DATA} \
    --output_dir ${OUTPUT_DIR_QWEN} \
    --epochs 3 --lr 2e-4 --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --time_limit 110

echo ""
echo "============================================================"
echo " Step 2b: Distill LLaMA (limit: 110 min)"
echo "============================================================"
python train_distill.py \
    --student_model ${STUDENT_LLAMA} \
    --train_data ${TRAIN_DATA} \
    --output_dir ${OUTPUT_DIR_LLAMA} \
    --epochs 3 --lr 2e-4 --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --time_limit 110

echo ""
echo "============================================================"
echo " Step 3a: Eval Qwen (limit: 25 min)"
echo "============================================================"
python inference_eval.py \
    --base_model ${STUDENT_QWEN} \
    --adapter_path ${OUTPUT_DIR_QWEN} \
    --test_data ${TEST_DATA} \
    --output_predictions predictions_qwen.jsonl \
    --report_file metrics_qwen.txt \
    --time_limit 25

echo ""
echo "============================================================"
echo " Step 3b: Eval LLaMA (limit: 25 min)"
echo "============================================================"
python inference_eval.py \
    --base_model ${STUDENT_LLAMA} \
    --adapter_path ${OUTPUT_DIR_LLAMA} \
    --test_data ${TEST_DATA} \
    --output_predictions predictions_llama.jsonl \
    --report_file metrics_llama.txt \
    --time_limit 25

echo ""
echo "============================================================"
echo " DONE"
echo "============================================================"
echo "Qwen:"; cat metrics_qwen.txt
echo ""; echo "LLaMA:"; cat metrics_llama.txt