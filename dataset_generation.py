from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

from datasets import Dataset

from utils import load_vllm_llm, prompt_vllm, TimeGuard


LOGGER = logging.getLogger(__name__)
LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]
LANG_CODE_TO_FULL = {"en": "english", "hi": "hindi", "bn": "bengali", "kn": "kannada", "ta": "tamil"}
ANSWER_RE = re.compile(r"####\s*ANSWER\s*:\s*\(?([A-J])\)?", re.IGNORECASE)
REASONING_BLOCK_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL)

LANG_REASONING_INSTRUCTION = {
    "english": "You must reason and explain your thought process in English.",
    "hindi": "आपको हिंदी में अपनी विचार प्रक्रिया का तर्क और व्याख्या करनी चाहिए। You must think and reason in Hindi.",
    "bengali": "আপনাকে বাংলায় আপনার চিন্তা প্রক্রিয়া যুক্তি এবং ব্যাখ্যা করতে হবে। You must think and reason in Bengali.",
    "kannada": "ನೀವು ಕನ್ನಡದಲ್ಲಿ ನಿಮ್ಮ ಆಲೋಚನಾ ಪ್ರಕ್ರಿಯೆಯನ್ನು ತರ್ಕ ಮತ್ತು ವಿವರಿಸಬೇಕು. You must think and reason in Kannada.",
    "tamil": "நீங்கள் தமிழில் உங்கள் சிந்தனை செயல்முறையை நியாயப்படுத்தி விளக்க வேண்டும். You must think and reason in Tamil.",
}


def setup_logger(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s", force=True)


def _options_to_text(options: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(options))


def _normalize_language(raw: str) -> str:
    return LANG_CODE_TO_FULL.get(raw.lower().strip(), raw.lower().strip())


def _build_instruction(row: dict[str, Any]) -> str:
    options = row["options"] if isinstance(row["options"], list) else list(row["options"])
    return f"{row['question']}\n\n{_options_to_text(options)}"


def sample_datasets(samples_per_language, dataset_path="data/dataset.jsonl", seed=42):
    full_data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line.strip())
                row["language"] = _normalize_language(row.get("language", "en"))
                full_data.append(row)

    LOGGER.info("Loaded %d total instances from %s", len(full_data), dataset_path)
    by_lang = {lang: [] for lang in LANGUAGES}
    for row in full_data:
        if row["language"] in by_lang:
            by_lang[row["language"]].append(row)

    for lang in LANGUAGES:
        LOGGER.info("  %s: %d available", lang, len(by_lang[lang]))

    rng = random.Random(seed)
    sampled = []
    for lang, count in zip(LANGUAGES, samples_per_language):
        avail = by_lang[lang]
        n = min(count, len(avail))
        sampled.extend(rng.sample(avail, n))
        LOGGER.info("  Sampled %d for %s", n, lang)

    rng.shuffle(sampled)
    LOGGER.info("Total sampled: %d (limit: 10000)", len(sampled))
    return Dataset.from_list(sampled)


def format_teacher_prompt(instruction: str, language: str) -> str:
    lang_inst = LANG_REASONING_INSTRUCTION.get(language, LANG_REASONING_INSTRUCTION["english"])
    return (
        f"You are an expert problem solver. {lang_inst}\n\n"
        f"Follow this format strictly:\n"
        f"1. First, provide your detailed step-by-step reasoning inside "
        f"<reasoning> and </reasoning> tags.\n"
        f"2. After </reasoning>, write your final answer on a new line in "
        f"exactly this format: #### ANSWER: (X)\n"
        f"   where X is the letter of the correct option.\n\n"
        f"{instruction}\n\nNow solve this step by step."
    )


def parse_generation(gen: str) -> dict:
    reasoning_match = REASONING_BLOCK_RE.search(gen)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else gen.strip()
    answer_match = ANSWER_RE.search(gen)
    final_answer = answer_match.group(1).upper() if answer_match else ""
    return {"reasoning": reasoning, "final_answer": final_answer, "raw_generation": gen.strip()}


def generate_batch(llm, tokenizer, prompts, max_new_tokens=2048, temperature=0.3):
    batch_messages = [[{"role": "user", "content": p}] for p in prompts]
    generations = prompt_vllm(llm, tokenizer, batch_messages,
                              max_new_tokens=max_new_tokens, temperature=temperature,
                              top_p=0.95, use_tqdm=True)
    return [parse_generation(g) for g in generations]


def save_records(records, output_path):
    """Save current records to disk (checkpoint)."""
    with open(output_path, "w", encoding="utf-8") as fp:
        for r in records:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    LOGGER.info("Checkpoint: saved %d records to %s", len(records), output_path)


def parse_args():
    p = argparse.ArgumentParser(description="Query teacher and build train corpus")
    p.add_argument("--teacher_model", required=True)
    p.add_argument("--num_samples", type=str, required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--dataset_path", default="data/dataset.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=256,
                   help="Number of prompts per vLLM batch (vLLM handles internal batching)")
    p.add_argument("--time_limit", type=int, default=230,
                   help="Time limit in minutes (saves before this)")
    p.add_argument("--filter_incorrect", action="store_true", default=True)
    p.add_argument("--retry_incorrect", action="store_true", default=True)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger(args.log_level)
    timer = TimeGuard(limit_minutes=args.time_limit, safety_margin_minutes=15)

    # Parse sample counts
    counts = list(map(int, args.num_samples.split(",")))
    assert len(counts) == 5 and sum(counts) <= 10000

    LOGGER.info("=" * 60)
    LOGGER.info("Part A: Distillation Data Creation")
    LOGGER.info("=" * 60)
    LOGGER.info("Teacher: %s | Budget: %d | Time limit: %d min",
                args.teacher_model, sum(counts), args.time_limit)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Sample ──
    sampled = sample_datasets(counts, args.dataset_path, args.seed)
    LOGGER.info("Collected %d samples", len(sampled))

    # ── 2. Load teacher ──
    LOGGER.info("Loading teacher model...")
    teacher, tokenizer = load_vllm_llm(
        model_id=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    timer.log_status(LOGGER)

    # ── 3. Build all prompts ──
    all_prompts = []
    all_rows = []
    for i in range(len(sampled)):
        row = sampled[i]
        all_prompts.append(format_teacher_prompt(_build_instruction(row), row["language"]))
        all_rows.append(row)

    # ── 4. Generate in batches with checkpointing ──
    correct_records = []
    incorrect_indices = []
    batch_size = args.batch_size
    total = len(all_prompts)

    LOGGER.info("Generating CoT in batches of %d...", batch_size)

    for start in range(0, total, batch_size):
        if timer.should_stop():
            LOGGER.warning("TIME LIMIT approaching! Saving what we have...")
            break

        end = min(start + batch_size, total)
        batch_prompts = all_prompts[start:end]

        LOGGER.info("Batch %d-%d / %d", start, end, total)
        parsed_batch = generate_batch(teacher, tokenizer, batch_prompts,
                                      max_new_tokens=args.max_new_tokens,
                                      temperature=args.temperature)

        for i, parsed in enumerate(parsed_batch):
            global_idx = start + i
            row = all_rows[global_idx]
            gold = str(row.get("answer", "")).upper().strip()
            pred = parsed["final_answer"]

            if not pred:
                incorrect_indices.append(global_idx)
                continue
            if args.filter_incorrect and pred != gold:
                incorrect_indices.append(global_idx)
                continue

            correct_records.append({
                "question": _build_instruction(row),
                "reasoning": parsed["reasoning"],
                "final_answer": pred,
                "gold_answer": gold,
                "language": row["language"],
                "subject": row.get("subject", ""),
                "teacher_generation": parsed["raw_generation"],
            })

        # Checkpoint after each batch
        save_records(correct_records, output_path)
        timer.log_status(LOGGER)

    LOGGER.info("First pass: %d correct, %d to retry", len(correct_records), len(incorrect_indices))

    # ── 5. Retry incorrect (only if time allows) ──
    if args.retry_incorrect and incorrect_indices and not timer.should_stop():
        LOGGER.info("Retrying %d incorrect (temp=0.7)...", len(incorrect_indices))
        retry_prompts = [all_prompts[i] for i in incorrect_indices]

        for start in range(0, len(retry_prompts), batch_size):
            if timer.should_stop():
                LOGGER.warning("TIME LIMIT approaching during retry! Stopping.")
                break

            end = min(start + batch_size, len(retry_prompts))
            batch = retry_prompts[start:end]
            parsed_batch = generate_batch(teacher, tokenizer, batch,
                                          max_new_tokens=args.max_new_tokens,
                                          temperature=0.7)

            for i, parsed in enumerate(parsed_batch):
                idx = incorrect_indices[start + i]
                row = all_rows[idx]
                gold = str(row.get("answer", "")).upper().strip()
                if parsed["final_answer"] == gold:
                    correct_records.append({
                        "question": _build_instruction(row),
                        "reasoning": parsed["reasoning"],
                        "final_answer": parsed["final_answer"],
                        "gold_answer": gold,
                        "language": row["language"],
                        "subject": row.get("subject", ""),
                        "teacher_generation": parsed["raw_generation"],
                    })

            save_records(correct_records, output_path)

    # ── 6. Final save ──
    save_records(correct_records, output_path)

    lang_counts = {lang: 0 for lang in LANGUAGES}
    for r in correct_records:
        lang_counts[r["language"]] = lang_counts.get(r["language"], 0) + 1

    LOGGER.info("=" * 60)
    LOGGER.info("FINAL: %d training instances", len(correct_records))
    for lang in LANGUAGES:
        LOGGER.info("  %s: %d", lang, lang_counts.get(lang, 0))
    LOGGER.info("Saved to: %s", output_path)
    timer.log_status(LOGGER)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()