from __future__ import annotations

import argparse
import json
import logging
import random
import re
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


def _options_to_text(options):
    return "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(options))


def _normalize_language(raw):
    return LANG_CODE_TO_FULL.get(raw.lower().strip(), raw.lower().strip())


def _build_instruction(row):
    opts = row["options"] if isinstance(row["options"], list) else list(row["options"])
    return f"{row['question']}\n\n{_options_to_text(opts)}"


def sample_datasets(samples_per_language, dataset_path="data/dataset.jsonl", seed=42):
    full_data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line.strip())
                row["language"] = _normalize_language(row.get("language", "en"))
                full_data.append(row)

    LOGGER.info("Loaded %d instances from %s", len(full_data), dataset_path)
    by_lang = {lang: [] for lang in LANGUAGES}
    for row in full_data:
        if row["language"] in by_lang:
            by_lang[row["language"]].append(row)

    rng = random.Random(seed)
    sampled = []
    for lang, count in zip(LANGUAGES, samples_per_language):
        n = min(count, len(by_lang[lang]))
        sampled.extend(rng.sample(by_lang[lang], n))
        LOGGER.info("  Sampled %d / %d for %s", n, len(by_lang[lang]), lang)

    rng.shuffle(sampled)
    LOGGER.info("Total sampled: %d", len(sampled))
    return Dataset.from_list(sampled)


def format_teacher_prompt(instruction, language):
    lang_inst = LANG_REASONING_INSTRUCTION.get(language, LANG_REASONING_INSTRUCTION["english"])
    return (
        f"You are an expert problem solver. {lang_inst}\n\n"
        f"Follow this format strictly:\n"
        f"1. Provide detailed step-by-step reasoning inside <reasoning> and </reasoning> tags.\n"
        f"2. After </reasoning>, write: #### ANSWER: (X)\n\n"
        f"{instruction}\n\nNow solve this step by step."
    )


def parse_generation(gen):
    rm = REASONING_BLOCK_RE.search(gen)
    reasoning = rm.group(1).strip() if rm else gen.strip()
    am = ANSWER_RE.search(gen)
    answer = am.group(1).upper() if am else ""
    return {"reasoning": reasoning, "final_answer": answer, "raw_generation": gen.strip()}


def generate_batch(llm, tokenizer, prompts, max_new_tokens=2048, temperature=0.3):
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    gens = prompt_vllm(llm, tokenizer, msgs, max_new_tokens=max_new_tokens,
                       temperature=temperature, top_p=0.95, use_tqdm=True)
    return [parse_generation(g) for g in gens]


def save_records(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    LOGGER.info("Saved %d records to %s", len(records), path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_model", required=True)
    p.add_argument("--num_samples", type=str, required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--dataset_path", default="data/dataset.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--time_limit", type=int, default=230)
    p.add_argument("--filter_incorrect", action="store_true", default=True)
    p.add_argument("--retry_incorrect", action="store_true", default=True)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger(args.log_level)
    timer = TimeGuard(limit_minutes=args.time_limit, safety_margin_minutes=15)

    counts = list(map(int, args.num_samples.split(",")))
    assert len(counts) == 5 and sum(counts) <= 10000

    LOGGER.info("=" * 60)
    LOGGER.info("Part A: Dataset Generation | Teacher: %s", args.teacher_model)
    LOGGER.info("=" * 60)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sampled = sample_datasets(counts, args.dataset_path, args.seed)

    LOGGER.info("Loading teacher model...")
    teacher, tokenizer = load_vllm_llm(
        args.teacher_model, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    timer.log_status(LOGGER)

    # Filter out prompts that exceed context length
    max_prompt_tokens = 4096 - args.max_new_tokens - 64  # safety margin
    all_prompts, all_rows = [], []
    skipped = 0
    for i in range(len(sampled)):
        row = sampled[i]
        prompt = format_teacher_prompt(_build_instruction(row), row["language"])
        prompt_tokens = len(tokenizer.encode(prompt))
        if prompt_tokens > max_prompt_tokens:
            skipped += 1
            continue
        all_prompts.append(prompt)
        all_rows.append(row)
    LOGGER.info("Kept %d prompts, skipped %d (too long)", len(all_prompts), skipped)

    correct_records, incorrect_indices = [], []
    bs = args.batch_size

    for start in range(0, len(all_prompts), bs):
        if timer.should_stop():
            LOGGER.warning("TIME LIMIT! Saving current progress...")
            break
        end = min(start + bs, len(all_prompts))
        LOGGER.info("Batch %d-%d / %d", start, end, len(all_prompts))

        try:
            parsed = generate_batch(teacher, tokenizer, all_prompts[start:end],
                                    args.max_new_tokens, args.temperature)
        except Exception as e:
            LOGGER.warning("Batch %d-%d failed: %s. Skipping...", start, end, e)
            continue
        for i, p in enumerate(parsed):
            idx = start + i
            row = all_rows[idx]
            gold = str(row.get("answer", "")).upper().strip()
            if not p["final_answer"] or (args.filter_incorrect and p["final_answer"] != gold):
                incorrect_indices.append(idx)
                continue
            correct_records.append({
                "question": _build_instruction(row), "reasoning": p["reasoning"],
                "final_answer": p["final_answer"], "gold_answer": gold,
                "language": row["language"], "subject": row.get("subject", ""),
                "teacher_generation": p["raw_generation"],
            })
        save_records(correct_records, output_path)
        timer.log_status(LOGGER)

    LOGGER.info("First pass: %d correct, %d incorrect", len(correct_records), len(incorrect_indices))

    # Retry
    if args.retry_incorrect and incorrect_indices and not timer.should_stop():
        LOGGER.info("Retrying %d incorrect...", len(incorrect_indices))
        retry_prompts = [all_prompts[i] for i in incorrect_indices]
        for start in range(0, len(retry_prompts), bs):
            if timer.should_stop():
                break
            end = min(start + bs, len(retry_prompts))
            parsed = generate_batch(teacher, tokenizer, retry_prompts[start:end],
                                    args.max_new_tokens, 0.7)
            for i, p in enumerate(parsed):
                row = all_rows[incorrect_indices[start + i]]
                gold = str(row.get("answer", "")).upper().strip()
                if p["final_answer"] == gold:
                    correct_records.append({
                        "question": _build_instruction(row), "reasoning": p["reasoning"],
                        "final_answer": p["final_answer"], "gold_answer": gold,
                        "language": row["language"], "subject": row.get("subject", ""),
                        "teacher_generation": p["raw_generation"],
                    })
            save_records(correct_records, output_path)

    save_records(correct_records, output_path)
    lc = {}
    for r in correct_records:
        lc[r["language"]] = lc.get(r["language"], 0) + 1
    LOGGER.info("FINAL: %d instances | %s", len(correct_records),
                " | ".join(f"{l}: {lc.get(l,0)}" for l in LANGUAGES))
    timer.log_status(LOGGER)


if __name__ == "__main__":
    main()