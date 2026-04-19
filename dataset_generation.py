from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets

from data.mmlupro import MMLUPro
from utils import load_vllm_llm, prompt_vllm


LOGGER = logging.getLogger(__name__)
LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]
ANSWER_RE = re.compile(r"####\s*ANSWER\s*:\s*([A-J])", re.IGNORECASE)
REASONING_BLOCK_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>",
    re.IGNORECASE | re.DOTALL,
)


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


def sample_datasets(
        samples_per_language: list[int],
) -> Dataset:
    """Fetch and sample the requested number of rows for each language."""
    if len(samples_per_language) != len(LANGUAGES):
        raise ValueError(
            "--num_samples must contain 5 comma-separated integers for "
            "english,hindi,bengali,kannada,tamil"
        )
    if any(count < 0 for count in samples_per_language):
        raise ValueError("--num_samples values must be >= 0")
    if sum(samples_per_language) <= 0:
        raise ValueError("--num_samples must request at least one sample")

    pass


def format_teacher_prompt(instruction: str, language: str) -> str:
    """Build a language-aware prompt that enforces reasoning and final answer."""
    pass


def generate_and_parse(model, tokenizer, prompt: str) -> dict[str, str]:
    """Query teacher model once and extract instruction/reasoning/final answer."""
    pass


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
    parser.add_argument("--split", default="test", help="Dataset split")
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

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sampled = sample_datasets(
        samples_per_language=samples_per_language,
        split=args.split,
        seed=args.seed,
    )
    LOGGER.info("Collected %d samples", len(sampled))

    teacher, tokenizer = load_vllm_llm(
        model_id=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    written = 0
    with output_path.open("w", encoding="utf-8") as fp:
        for row in sampled:
            question_with_choices = _build_instruction(row)
            prompt = format_teacher_prompt(
                question_with_choices, row["language"])
            parsed = generate_and_parse(teacher, tokenizer, prompt)

            record = {
                "question": question_with_choices,
                "reasoning": parsed["reasoning"],
                "final_answer": parsed["final_answer"],
                "gold_answer": str(row.get("answer", "")).upper()[:1],
                "language": row["language"],
                "subject": row.get("subject"),
                "prompt": prompt,
                "teacher_generation": parsed["raw_generation"],
            }
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    LOGGER.info("Saved %d rows to %s", written, output_path)


if __name__ == "__main__":
    main()
