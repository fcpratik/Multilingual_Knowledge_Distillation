from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from utils import TimeGuard


LOGGER = logging.getLogger(__name__)


def setup_logger(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s", force=True)


# ─────────────────────────────────────────────
# Time-safe callback: saves model before timeout
# ─────────────────────────────────────────────

class TimeSafeCallback(TrainerCallback):
    """Saves model and stops training if approaching time limit."""

    def __init__(self, timer: TimeGuard, output_dir: str):
        self.timer = timer
        self.output_dir = output_dir
        self.saved_emergency = False

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self.timer.should_stop() and not self.saved_emergency:
            LOGGER.warning("⚠ TIME LIMIT approaching at step %d! Emergency save...", state.global_step)
            if model is not None:
                model.save_pretrained(self.output_dir)
                if tokenizer is not None:
                    tokenizer.save_pretrained(self.output_dir)
                LOGGER.info("Emergency save complete: %s", self.output_dir)
            self.saved_emergency = True
            control.should_training_stop = True
        return control

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        self.timer.log_status(LOGGER)
        # Save after every epoch as checkpoint
        if model is not None:
            model.save_pretrained(self.output_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(self.output_dir)
            LOGGER.info("Epoch %d checkpoint saved to %s", int(state.epoch), self.output_dir)
        return control


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class CoTDistillDataset(TorchDataset):
    def __init__(self, data_path, tokenizer, max_length=2048, mask_prompt_tokens=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt_tokens = mask_prompt_tokens
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line.strip()))
        LOGGER.info("Loaded %d training samples from %s", len(self.samples), data_path)

    def __len__(self):
        return len(self.samples)

    def _build_chat_text(self, row):
        question = row["question"]
        teacher_gen = row.get("teacher_generation", "")
        user_content = (
            f"You are an expert problem solver. "
            f"Think step by step and solve the following question.\n\n"
            f"Follow this format strictly:\n"
            f"1. First, provide your detailed step-by-step reasoning inside "
            f"<reasoning> and </reasoning> tags.\n"
            f"2. After </reasoning>, write your final answer on a new line in "
            f"exactly this format: #### ANSWER: (X)\n"
            f"   where X is the letter of the correct option.\n\n"
            f"{question}\n\nNow solve this step by step."
        )
        msgs_prompt = [{"role": "user", "content": user_content}]
        msgs_full = [{"role": "user", "content": user_content},
                     {"role": "assistant", "content": teacher_gen}]

        prompt_text = self.tokenizer.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)
        full_text = self.tokenizer.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)
        return prompt_text, full_text

    def __getitem__(self, idx):
        row = self.samples[idx]
        prompt_text, full_text = self._build_chat_text(row)

        full_enc = self.tokenizer(full_text, truncation=True, max_length=self.max_length,
                                  padding=False, return_tensors=None)
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        if self.mask_prompt_tokens:
            prompt_enc = self.tokenizer(prompt_text, truncation=True, max_length=self.max_length,
                                        padding=False, return_tensors=None)
            prompt_len = len(prompt_enc["input_ids"])
            labels = [-100] * prompt_len + input_ids[prompt_len:]
        else:
            labels = input_ids.copy()

        labels = labels[:len(input_ids)]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_student_qlora(model_path, lora_rank=32, lora_alpha=64, lora_dropout=0.05):
    LOGGER.info("Loading student with QLoRA: %s", model_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Knowledge distillation training")
    p.add_argument("--student_model", required=True)
    p.add_argument("--teacher_model", default=None)
    p.add_argument("--train_data", default="data/train.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--mask_prompt_tokens", action="store_true", default=True)
    p.add_argument("--time_limit", type=int, default=230,
                   help="Time limit in minutes for training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger(args.log_level)
    timer = TimeGuard(limit_minutes=args.time_limit, safety_margin_minutes=15)

    LOGGER.info("=" * 60)
    LOGGER.info("Part B: Knowledge Distillation Training")
    LOGGER.info("=" * 60)
    LOGGER.info("Student: %s | Time limit: %d min", args.student_model, args.time_limit)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ──
    model, tokenizer = load_student_qlora(
        args.student_model, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
    )
    timer.log_status(LOGGER)

    # ── 2. Dataset ──
    train_dataset = CoTDistillDataset(
        args.train_data, tokenizer, max_length=args.max_length,
        mask_prompt_tokens=args.mask_prompt_tokens,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True,
                                           pad_to_multiple_of=8, return_tensors="pt")

    # ── 3. Training ──
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, data_collator=data_collator,
        callbacks=[TimeSafeCallback(timer, str(output_dir))],
    )

    LOGGER.info("Starting training (%d epochs, %d samples)...", args.epochs, len(train_dataset))
    trainer.train()

    # ── 4. Final save ──
    LOGGER.info("Saving final model to %s", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save config for reproducibility
    with open(output_dir / "distill_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    timer.log_status(LOGGER)
    LOGGER.info("Training complete!")


if __name__ == "__main__":
    main()