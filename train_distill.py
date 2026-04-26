from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

from utils import TimeGuard


LOGGER = logging.getLogger(__name__)


def setup_logger(level):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s", force=True)


class TimeSafeCallback(TrainerCallback):
    def __init__(self, timer, output_dir):
        self.timer = timer
        self.output_dir = output_dir
        self.saved = False

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self.timer.should_stop() and not self.saved:
            LOGGER.warning("TIME LIMIT at step %d! Emergency save...", state.global_step)
            if model:
                model.save_pretrained(self.output_dir)
                if tokenizer:
                    tokenizer.save_pretrained(self.output_dir)
            self.saved = True
            control.should_training_stop = True
        return control

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        self.timer.log_status(LOGGER)
        if model:
            model.save_pretrained(self.output_dir)
            if tokenizer:
                tokenizer.save_pretrained(self.output_dir)
            LOGGER.info("Epoch %d saved to %s", int(state.epoch), self.output_dir)
        return control


class CoTDistillDataset(TorchDataset):
    def __init__(self, data_path, tokenizer, max_length=2048, mask_prompt=True,
                 original_data_path=None, original_data_ratio=0.3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt = mask_prompt
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line.strip()))
        LOGGER.info("Loaded %d teacher-generated samples from %s", len(self.samples), data_path)

        # Mix in original dataset examples (answer-only, no CoT) if provided
        if original_data_path and Path(original_data_path).exists():
            original = []
            with open(original_data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line.strip())
                        # Convert to training format: question + short answer
                        opts = row["options"] if isinstance(row["options"], list) else list(row["options"])
                        question = f"{row['question']}\n\n" + "\n".join(
                            f"({chr(65+i)}) {c}" for i, c in enumerate(opts))
                        answer = str(row.get("answer", "")).upper().strip()
                        answer_text = opts[ord(answer) - 65] if answer and ord(answer) - 65 < len(opts) else ""
                        original.append({
                            "question": question,
                            "teacher_generation": f"<reasoning>\nThe answer is ({answer}) {answer_text}.\n</reasoning>\n#### ANSWER: ({answer})",
                            "final_answer": answer,
                            "gold_answer": answer,
                            "language": row.get("language", "en"),
                        })
            # Sample a fraction to mix in
            import random
            rng = random.Random(42)
            n_original = min(int(len(self.samples) * original_data_ratio), len(original))
            if n_original > 0:
                sampled_original = rng.sample(original, n_original)
                self.samples.extend(sampled_original)
                rng.shuffle(self.samples)
                LOGGER.info("Mixed in %d original dataset samples (ratio=%.2f), total=%d",
                            n_original, original_data_ratio, len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        question = row["question"]
        teacher_gen = row.get("teacher_generation", "")
        user = (
            f"You are an expert problem solver. Think step by step.\n\n"
            f"Format:\n1. Reasoning inside <reasoning></reasoning> tags.\n"
            f"2. Final answer: #### ANSWER: (X)\n\n{question}\n\nSolve step by step."
        )
        prompt_msgs = [{"role": "user", "content": user}]
        full_msgs = [{"role": "user", "content": user},
                     {"role": "assistant", "content": teacher_gen}]

        prompt_text = self.tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_text = self.tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

        full_enc = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding=False)
        ids = full_enc["input_ids"]
        attn = full_enc["attention_mask"]

        if self.mask_prompt:
            p_enc = self.tokenizer(prompt_text, truncation=True, max_length=self.max_length, padding=False)
            pl = len(p_enc["input_ids"])
            labels = [-100] * pl + ids[pl:]
        else:
            labels = ids.copy()

        return {"input_ids": ids, "attention_mask": attn, "labels": labels[:len(ids)]}


def load_student_lora(model_path, lora_rank=32, lora_alpha=64, lora_dropout=0.05):
    LOGGER.info("Loading %s with LoRA (fp16)...", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True,
    ).cuda()
    model.enable_input_require_grads()  # needed for LoRA + gradient checkpointing
    lora = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                      bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return model, tok


def parse_args():
    p = argparse.ArgumentParser()
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
    p.add_argument("--original_data", default=None, help="Path to original dataset.jsonl for supplementary training")
    p.add_argument("--original_data_ratio", type=float, default=0.3, help="Ratio of original data to mix in")
    p.add_argument("--time_limit", type=int, default=230)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger(args.log_level)
    timer = TimeGuard(args.time_limit, safety_margin_minutes=15)

    LOGGER.info("=" * 60)
    LOGGER.info("Part B: Training | Student: %s", args.student_model)
    LOGGER.info("=" * 60)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load_student_lora(args.student_model, args.lora_rank, args.lora_alpha)
    timer.log_status(LOGGER)

    ds = CoTDistillDataset(args.train_data, tok, args.max_length, args.mask_prompt_tokens,
                           original_data_path=args.original_data,
                           original_data_ratio=args.original_data_ratio)
    collator = DataCollatorForSeq2Seq(tok, padding=True, pad_to_multiple_of=8, return_tensors="pt")

    t_args = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr, weight_decay=0.01, warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine", fp16=True, logging_steps=10,
        save_strategy="epoch", save_total_limit=2, seed=args.seed,
        report_to="none", remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
    )

    trainer = Trainer(model=model, args=t_args, train_dataset=ds, data_collator=collator,
                      callbacks=[TimeSafeCallback(timer, str(out))])

    LOGGER.info("Training %d epochs, %d samples...", args.epochs, len(ds))
    trainer.train()

    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    with open(out / "distill_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    timer.log_status(LOGGER)
    LOGGER.info("Training complete!")


if __name__ == "__main__":
    main()