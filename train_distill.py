from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset as HFDataset


LOGGER = logging.getLogger(__name__)


def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class CoTDistillDataset(TorchDataset):
    """
    Dataset for text-based CoT distillation.

    Each sample is tokenized as:
        [PROMPT TOKENS] [TEACHER GENERATION TOKENS]

    Loss is computed ONLY on the teacher generation tokens (prompt is masked with -100).
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        mask_prompt_tokens: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt_tokens = mask_prompt_tokens
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.samples.append(row)

        LOGGER.info("Loaded %d training samples from %s", len(self.samples), data_path)

    def __len__(self):
        return len(self.samples)

    def _build_chat_text(self, row: dict[str, Any]) -> tuple[str, str]:
        """
        Build the full conversation text and return (prompt_text, completion_text).
        The prompt is the user message; the completion is the teacher's generation.
        """
        question = row["question"]
        teacher_gen = row.get("teacher_generation", "")

        # Build user message (same format as teacher prompting)
        user_content = (
            f"You are an expert problem solver. "
            f"Think step by step and solve the following question.\n\n"
            f"Follow this format strictly:\n"
            f"1. First, provide your detailed step-by-step reasoning inside "
            f"<reasoning> and </reasoning> tags.\n"
            f"2. After </reasoning>, write your final answer on a new line in "
            f"exactly this format: #### ANSWER: (X)\n"
            f"   where X is the letter of the correct option.\n\n"
            f"{question}\n\n"
            f"Now solve this step by step."
        )

        messages_prompt = [{"role": "user", "content": user_content}]
        messages_full = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": teacher_gen},
        ]

        # Get the prompt-only text (to determine where to start the loss)
        prompt_text = self.tokenizer.apply_chat_template(
            messages_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
        )

        return prompt_text, full_text

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.samples[idx]
        prompt_text, full_text = self._build_chat_text(row)

        # Tokenize the full text
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = full_encoding["input_ids"]
        attention_mask = full_encoding["attention_mask"]

        # Build labels: mask prompt tokens with -100
        if self.mask_prompt_tokens:
            prompt_encoding = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )
            prompt_len = len(prompt_encoding["input_ids"])

            labels = [-100] * prompt_len + input_ids[prompt_len:]
        else:
            labels = input_ids.copy()

        # Ensure labels has same length as input_ids
        labels = labels[:len(input_ids)]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_student_model_qlora(
    model_name_or_path: str,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
):
    """Load student model with QLoRA (4-bit quantization + LoRA adapters)."""

    LOGGER.info("Loading student model with QLoRA: %s", model_name_or_path)

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Determine target modules based on model architecture
    if target_modules is None:
        # Auto-detect: works for both Qwen2.5 and LLaMA
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    # LoRA config
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Knowledge distillation training with QLoRA")
    parser.add_argument("--student_model", required=True,
                        help="Base student model HF path or local path")
    parser.add_argument("--teacher_model", required=False, default=None,
                        help="Teacher model path (for online distillation, optional)")
    parser.add_argument("--train_data", default="data/train.jsonl",
                        help="Path to train JSONL with teacher generations")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save trained LoRA adapter + config")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup ratio")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max sequence length for tokenization")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")

    # LoRA config
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")

    # Prompt masking
    parser.add_argument(
        "--mask_prompt_tokens",
        action="store_true",
        default=True,
        help="Mask prompt tokens so loss is only on reasoning + final answer",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    LOGGER.info("=" * 60)
    LOGGER.info("Part B: Knowledge Distillation Training")
    LOGGER.info("=" * 60)
    LOGGER.info("Student model: %s", args.student_model)
    LOGGER.info("Train data   : %s", args.train_data)
    LOGGER.info("Output dir   : %s", args.output_dir)
    LOGGER.info("Epochs: %d, LR: %.2e, Batch: %d x %d accum",
                args.epochs, args.lr, args.batch_size, args.gradient_accumulation_steps)
    LOGGER.info("LoRA rank=%d, alpha=%d, dropout=%.2f",
                args.lora_rank, args.lora_alpha, args.lora_dropout)

    # ── 1. Load student model with QLoRA ──
    model, tokenizer = load_student_model_qlora(
        model_name_or_path=args.student_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # ── 2. Prepare dataset ──
    LOGGER.info("Preparing training dataset...")
    train_dataset = CoTDistillDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mask_prompt_tokens=args.mask_prompt_tokens,
    )

    # Data collator that handles padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    # ── 3. Training arguments ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=False,
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
        dataloader_num_workers=2,
    )

    # ── 4. Train ──
    LOGGER.info("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # ── 5. Save ──
    LOGGER.info("Saving LoRA adapter to %s", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Also save training config for reproducibility
    config_path = output_dir / "distill_config.json"
    config = {
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "train_data": args.train_data,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "mask_prompt_tokens": args.mask_prompt_tokens,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    LOGGER.info("=" * 60)
    LOGGER.info("Training complete! Adapter saved to %s", output_dir)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()