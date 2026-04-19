from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets

from utils import load_vllm_llm, prompt_vllm


LOGGER = logging.getLogger(__name__)
LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]
LANG_CODE_TO_FULL = {"en": "english", "hi": "hindi", "bn": "bengali", "kn": "kannada", "ta": "tamil"}
ANSWER_RE = re.compile(r"####\s*ANSWER\s*:\s*\(?([A-J])\)?", re.IGNORECASE)
REASONING_BLOCK_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>",
    re.IGNORECASE | re.DOTALL,
)

# ── Language-aware reasoning instructions ──
LANG_REASONING_INSTRUCTION = {
    "english": "You must reason and explain your thought process in English.",
    "hindi": "आपको हिंदी में अपनी विचार प्रक्रिया का तर्क और व्याख्या करनी चाहिए। You must think and reason in Hindi.",
    "bengali": "আপনাকে বাংলায় আপনার চিন্তা প্রক্রিয়া যুক্তি এবং ব্যাখ্যা করতে হবে। You must think and reason in Bengali.",
    "kannada": "ನೀವು ಕನ್ನಡದಲ್ಲಿ ನಿಮ್ಮ ಆಲೋಚನಾ ಪ್ರಕ್ರಿಯೆಯನ್ನು ತರ್ಕ ಮತ್ತು ವಿವರಿಸಬೇಕು. You must think and reason in Kannada.",
    "tamil": "நீங்கள் தமிழில் உங்கள் சிந்தனை செயல்முறையை நியாயப்படுத்தி விளக்க வேண்டும். You must think and reason in Tamil.",
}


def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def _options_to_text(options: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(
        f"({letters[idx]}) {choice}" for idx, choice in enumerate(options)
    )


def _normalize_language(raw_lang: str) -> str:
    """Normalize language codes like 'en' -> 'english', or pass through if already full."""
    raw = raw_lang.lower().strip()
    return LANG_CODE_TO_FULL.get(raw, raw)


def sample_datasets(
        samples_per_language: list[int],
        dataset_path: str = "data/dataset.jsonl",
        seed: int = 42,
) -> Dataset:
    """Load dataset.jsonl and sample the requested number of rows per language."""
    if len(samples_per_language) != len(LANGUAGES):
        raise ValueError(
            "--num_samples must contain 5 comma-separated integers for "
            "english,hindi,bengali,kannada,tamil"
        )
    if any(count < 0 for count in samples_per_language):
        raise ValueError("--num_samples values must be >= 0")
    if sum(samples_per_language) <= 0:
        raise ValueError("--num_samples must request at least one sample")

    # Load all rows from JSONL
    full_data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Normalize the language field
            row["language"] = _normalize_language(row.get("language", "en"))
            full_data.append(row)

    LOGGER.info("Loaded %d total instances from %s", len(full_data), dataset_path)

    # Group by language
    by_lang: dict[str, list[dict]] = {lang: [] for lang in LANGUAGES}
    for row in full_data:
        lang = row["language"]
        if lang in by_lang:
            by_lang[lang].append(row)

    for lang in LANGUAGES:
        LOGGER.info("  %s: %d available", lang, len(by_lang[lang]))

    # Sample per language
    rng = random.Random(seed)
    sampled_rows = []
    for lang, count in zip(LANGUAGES, samples_per_language):
        available = by_lang[lang]
        if count == 0:
            continue
        if count > len(available):
            LOGGER.warning(
                "Requested %d for %s but only %d available; using all.",
                count, lang, len(available),
            )
            count = len(available)
        selected = rng.sample(available, count)
        sampled_rows.extend(selected)
        LOGGER.info("  Sampled %d for %s", len(selected), lang)

    LOGGER.info("Total sampled: %d (limit: 10000)", len(sampled_rows))

    # Shuffle
    rng.shuffle(sampled_rows)
    return Dataset.from_list(sampled_rows)


def format_teacher_prompt(instruction: str, language: str) -> str:
    """
    Build a language-aware prompt that enforces CoT reasoning and a final answer.

    Instructs the teacher to:
    1. Reason step-by-step in the question's language
    2. Wrap reasoning inside <reasoning>...</reasoning> tags
    3. End with #### ANSWER: (X)
    """
    lang_instruction = LANG_REASONING_INSTRUCTION.get(
        language, LANG_REASONING_INSTRUCTION["english"]
    )

    prompt = (
        f"You are an expert problem solver. {lang_instruction}\n\n"
        f"Follow this format strictly:\n"
        f"1. First, provide your detailed step-by-step reasoning inside "
        f"<reasoning> and </reasoning> tags.\n"
        f"2. After </reasoning>, write your final answer on a new line in "
        f"exactly this format: #### ANSWER: (X)\n"
        f"   where X is the letter of the correct option.\n\n"
        f"{instruction}\n\n"
        f"Now solve this step by step."
    )
    return prompt


def generate_and_parse(
    llm,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 2048,
    temperature: float = 0.3,
) -> list[dict[str, str]]:
    """
    Query teacher model in batch and extract reasoning + final answer.
    Returns a list of dicts: {reasoning, final_answer, raw_generation}
    """
    # Build chat messages for each prompt
    batch_messages = [
        [{"role": "user", "content": p}] for p in prompts
    ]

    # Generate using the provided vLLM utility
    generations = prompt_vllm(
        llm,
        tokenizer,
        batch_messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        use_tqdm=True,
    )

    results = []
    for gen in generations:
        # Extract reasoning block
        reasoning_match = REASONING_BLOCK_RE.search(gen)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else gen.strip()

        # Extract final answer
        answer_match = ANSWER_RE.search(gen)
        final_answer = answer_match.group(1).upper() if answer_match else ""

        results.append({
            "reasoning": reasoning,
            "final_answer": final_answer,
            "raw_generation": gen.strip(),
        })

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query teacher and build train corpus JSONL"
    )
    parser.add_argument(
        "--teacher_model",
        required=True,
        help="Hugging Face path to the teacher model",
    )
    parser.add_argument(
        "--num_samples",
        type=str,
        required=True,
        help=(
            "Comma-separated sample counts for english,hindi,bengali,"
            "kannada,tamil"
        ),
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output JSONL path for train corpus",
    )
    parser.add_argument(
        "--dataset_path",
        default="data/dataset.jsonl",
        help="Path to the source dataset.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.6,
        help="Target fraction of GPU memory for vLLM; lower if startup fails",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max tokens for teacher generation per question",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for teacher (first pass)",
    )
    parser.add_argument(
        "--filter_incorrect",
        action="store_true",
        default=True,
        help="Filter out instances where teacher answer != gold answer",
    )
    parser.add_argument(
        "--retry_incorrect",
        action="store_true",
        default=True,
        help="Retry incorrect answers with temperature=0.7",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def _parse_num_samples(raw_value: str) -> list[int]:
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if len(parts) != len(LANGUAGES):
        raise ValueError(
            "--num_samples must contain exactly 5 comma-separated integers "
            "for english,hindi,bengali,kannada,tamil"
        )

    try:
        counts = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(
            "--num_samples must contain only integers"
        ) from exc

    if any(count < 0 for count in counts):
        raise ValueError("--num_samples values must be >= 0")

    if sum(counts) > 10000:
        raise ValueError(
            f"Total samples ({sum(counts)}) exceeds the 10K limit!"
        )

    return counts


def _build_instruction(row: dict[str, Any]) -> str:
    options = row["options"]
    if not isinstance(options, list):
        options = list(options)
    return f"{row['question']}\n\n{_options_to_text(options)}"


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    samples_per_language = _parse_num_samples(args.num_samples)
    total_requested = sum(samples_per_language)

    LOGGER.info("=" * 60)
    LOGGER.info("Part A: Distillation Data Creation")
    LOGGER.info("=" * 60)
    LOGGER.info("Teacher model : %s", args.teacher_model)
    LOGGER.info("Sampling budget: %d / 10000", total_requested)
    for lang, cnt in zip(LANGUAGES, samples_per_language):
        LOGGER.info("  %s: %d", lang, cnt)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Sample dataset ──
    sampled = sample_datasets(
        samples_per_language=samples_per_language,
        dataset_path=args.dataset_path,
        seed=args.seed,
    )
    LOGGER.info("Collected %d samples", len(sampled))

    # ── 2. Load teacher ──
    LOGGER.info("Loading teacher model via vLLM...")
    teacher, tokenizer = load_vllm_llm(
        model_id=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # ── 3. Build prompts and generate ──
    LOGGER.info("Building teacher prompts...")
    all_prompts = []
    all_rows = []
    for i in range(len(sampled)):
        row = sampled[i]
        question_with_choices = _build_instruction(row)
        prompt = format_teacher_prompt(question_with_choices, row["language"])
        all_prompts.append(prompt)
        all_rows.append(row)

    LOGGER.info(
        "Generating CoT for %d prompts (max_tokens=%d, temp=%.2f)...",
        len(all_prompts), args.max_new_tokens, args.temperature,
    )
    all_parsed = generate_and_parse(
        teacher,
        tokenizer,
        all_prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # ── 4. Filter: keep only correct teacher answers ──
    correct_records = []
    incorrect_indices = []
    stats = {"correct": 0, "incorrect": 0, "unparseable": 0}

    for i, (row, parsed) in enumerate(zip(all_rows, all_parsed)):
        gold = str(row.get("answer", "")).upper().strip()
        pred = parsed["final_answer"]

        if not pred:
            stats["unparseable"] += 1
            incorrect_indices.append(i)
            continue

        if args.filter_incorrect and pred != gold:
            stats["incorrect"] += 1
            incorrect_indices.append(i)
            continue

        stats["correct"] += 1
        record = {
            "question": _build_instruction(row),
            "reasoning": parsed["reasoning"],
            "final_answer": parsed["final_answer"],
            "gold_answer": gold,
            "language": row["language"],
            "subject": row.get("subject", ""),
            "teacher_generation": parsed["raw_generation"],
        }
        correct_records.append(record)

    LOGGER.info(
        "First pass: %d correct, %d incorrect, %d unparseable (total %d)",
        stats["correct"], stats["incorrect"], stats["unparseable"], len(all_rows),
    )

    # ── 5. Retry incorrect with higher temperature ──
    if args.retry_incorrect and incorrect_indices:
        LOGGER.info(
            "Retrying %d incorrect/unparseable with temperature=0.7...",
            len(incorrect_indices),
        )
        retry_prompts = [all_prompts[i] for i in incorrect_indices]

        retry_parsed = generate_and_parse(
            teacher,
            tokenizer,
            retry_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
        )

        recovered = 0
        for idx, parsed in zip(incorrect_indices, retry_parsed):
            row = all_rows[idx]
            gold = str(row.get("answer", "")).upper().strip()
            pred = parsed["final_answer"]
            if pred and pred == gold:
                record = {
                    "question": _build_instruction(row),
                    "reasoning": parsed["reasoning"],
                    "final_answer": parsed["final_answer"],
                    "gold_answer": gold,
                    "language": row["language"],
                    "subject": row.get("subject", ""),
                    "teacher_generation": parsed["raw_generation"],
                }
                correct_records.append(record)
                recovered += 1

        LOGGER.info("Recovered %d additional correct instances on retry", recovered)

    # ── 6. Save training JSONL ──
    with output_path.open("w", encoding="utf-8") as fp:
        for record in correct_records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── 7. Summary ──
    lang_counts = {lang: 0 for lang in LANGUAGES}
    for r in correct_records:
        lang_counts[r["language"]] = lang_counts.get(r["language"], 0) + 1

    LOGGER.info("=" * 60)
    LOGGER.info("Final training set: %d instances", len(correct_records))
    for lang in LANGUAGES:
        LOGGER.info("  %s: %d", lang, lang_counts.get(lang, 0))
    LOGGER.info("Saved to: %s", output_path)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()