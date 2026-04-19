#!/bin/bash
# ============================================================
# run_distillation.sh
# Full pipeline: data generation → distillation → evaluation
# ============================================================

set -e  # Exit on error

# ── Configuration ──
TEACHER_MODEL="Qwen/Qwen2.5-7B-Instruct"
STUDENT_QWEN="Qwen/Qwen2.5-1.5B-Instruct"
STUDENT_LLAMA="meta-llama/Llama-3.2-1B-Instruct"

# Sampling: en, hi, bn, kn, ta (total <= 10K)
NUM_SAMPLES="4000,2500,1500,1200,800"

DATASET_PATH="data/dataset.jsonl"
TRAIN_DATA="data/train.jsonl"
TEST_DATA="data/test.jsonl"

OUTPUT_DIR_QWEN="output/distilled_qwen"
OUTPUT_DIR_LLAMA="output/distilled_llama"

echo "============================================================"
echo " Step 1: Dataset Generation (Teacher Prompting)"
echo "============================================================"
python dataset_generation.py \
    --teacher_model ${TEACHER_MODEL} \
    --num_samples ${NUM_SAMPLES} \
    --output_file ${TRAIN_DATA} \
    --dataset_path ${DATASET_PATH} \
    --max_new_tokens 2048 \
    --temperature 0.3 \
    --seed 42

echo ""
echo "============================================================"
echo " Step 2a: Distillation - Qwen2.5-1.5B (In-Family)"
echo "============================================================"
python train_distill.py \
    --student_model ${STUDENT_QWEN} \
    --teacher_model ${TEACHER_MODEL} \
    --train_data ${TRAIN_DATA} \
    --output_dir ${OUTPUT_DIR_QWEN} \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --epochs 3 \
    --lr 2e-4 \
    --max_length 2048 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --mask_prompt_tokens

echo ""
echo "============================================================"
echo " Step 2b: Distillation - LLaMA-3.2-1B (Cross-Family)"
echo "============================================================"
python train_distill.py \
    --student_model ${STUDENT_LLAMA} \
    --teacher_model ${TEACHER_MODEL} \
    --train_data ${TRAIN_DATA} \
    --output_dir ${OUTPUT_DIR_LLAMA} \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --epochs 3 \
    --lr 2e-4 \
    --max_length 2048 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --mask_prompt_tokens

echo ""
echo "============================================================"
echo " Step 3a: Evaluation - Qwen2.5-1.5B"
echo "============================================================"
python inference_eval.py \
    --base_model ${STUDENT_QWEN} \
    --adapter_path ${OUTPUT_DIR_QWEN} \
    --test_data ${TEST_DATA} \
    --output_predictions predictions_qwen.jsonl \
    --report_file metrics_qwen.txt \
    --max_new_tokens 2048

echo ""
echo "============================================================"
echo " Step 3b: Evaluation - LLaMA-3.2-1B"
echo "============================================================"
python inference_eval.py \
    --base_model ${STUDENT_LLAMA} \
    --adapter_path ${OUTPUT_DIR_LLAMA} \
    --test_data ${TEST_DATA} \
    --output_predictions predictions_llama.jsonl \
    --report_file metrics_llama.txt \
    --max_new_tokens 2048

echo ""
echo "============================================================"
echo " Pipeline Complete!"
echo "============================================================"
echo "Results:"
echo "  Qwen metrics  : metrics_qwen.txt"
echo "  LLaMA metrics : metrics_llama.txt"
echo "  Qwen preds    : predictions_qwen.jsonl"
echo "  LLaMA preds   : predictions_llama.jsonl"
echo ""
cat metrics_qwen.txt
echo ""
cat metrics_llama.txt