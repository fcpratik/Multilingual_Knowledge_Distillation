#!/bin/bash
set -e

# ── Models ──
TEACHER="Qwen/Qwen2.5-7B-Instruct"
STUDENT_QWEN="Qwen/Qwen2.5-1.5B-Instruct"
STUDENT_LLAMA="meta-llama/Llama-3.2-1B-Instruct"

# ── Paths ──
DATASET="data/dataset.jsonl"
DATASET_TRAIN="data/dataset_train.jsonl"
DATASET_VAL="data/dataset_val.jsonl"
TRAIN="data/train.jsonl"
OUT_QWEN="output/distilled_qwen"
OUT_LLAMA="output/distilled_llama"

# ── Sampling: en,hi,bn,kn,ta = 10K total (oversample non-English) ──
SAMPLES="2000,2500,2000,2000,1500"

echo "============================================================"
echo " Step 0: Split dataset into train/val (90/10)"
echo "============================================================"
python split_data.py \
    --input ${DATASET} \
    --train_out ${DATASET_TRAIN} \
    --val_out ${DATASET_VAL} \
    --val_ratio 0.10

echo ""
echo "============================================================"
echo " Step 1/5: Dataset Generation (limit: 230 min)"
echo "============================================================"
python dataset_generation.py \
    --teacher_model ${TEACHER} \
    --num_samples ${SAMPLES} \
    --output_file ${TRAIN} \
    --dataset_path ${DATASET_TRAIN} \
    --gpu_memory_utilization 0.90 \
    --time_limit 230

echo ""
echo "============================================================"
echo " Step 2/5: Train Qwen student (limit: 110 min)"
echo "============================================================"
python train_distill.py \
    --student_model ${STUDENT_QWEN} \
    --train_data ${TRAIN} \
    --output_dir ${OUT_QWEN} \
    --original_data ${DATASET_TRAIN} \
    --original_data_ratio 0.5 \
    --epochs 3 --lr 2e-4 --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --time_limit 110

echo ""
echo "============================================================"
echo " Step 3/5: Train LLaMA student (limit: 110 min)"
echo "============================================================"
python train_distill.py \
    --student_model ${STUDENT_LLAMA} \
    --train_data ${TRAIN} \
    --output_dir ${OUT_LLAMA} \
    --original_data ${DATASET_TRAIN} \
    --original_data_ratio 0.5 \
    --epochs 3 --lr 2e-4 --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --time_limit 110

echo ""
echo "============================================================"
echo " Step 4/5: Eval Qwen (limit: 25 min)"
echo "============================================================"
python inference_eval.py \
    --base_model ${STUDENT_QWEN} \
    --adapter_path ${OUT_QWEN} \
    --test_data ${DATASET_VAL} \
    --output_predictions predictions_qwen.jsonl \
    --report_file metrics_qwen.txt \
    --time_limit 25

echo ""
echo "============================================================"
echo " Step 5/5: Eval LLaMA (limit: 25 min)"
echo "============================================================"
python inference_eval.py \
    --base_model ${STUDENT_LLAMA} \
    --adapter_path ${OUT_LLAMA} \
    --test_data ${DATASET_VAL} \
    --output_predictions predictions_llama.jsonl \
    --report_file metrics_llama.txt \
    --time_limit 25

echo ""
echo "============================================================"
echo " RESULTS"
echo "============================================================"
echo "--- Qwen2.5-1.5B ---"
cat metrics_qwen.txt
echo ""
echo "--- LLaMA-3.2-1B ---"
cat metrics_llama.txt