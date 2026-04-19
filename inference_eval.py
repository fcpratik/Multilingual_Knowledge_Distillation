from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from utils import load_vllm_llm, prompt_vllm


LOGGER = logging.getLogger(__name__)
LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]
LANGUAGE_LABELS = {
    "english": "English",
    "hindi": "Hindi",
    "bengali": "Bengali",
    "kannada": "Kannada",
    "tamil": "Tamil",
}
LANG_CODE_TO_FULL = {"en": "english", "hi": "hindi", "bn": "bengali", "kn": "kannada", "ta": "tamil"}
ANSWER_TAG_RE = re.compile(r"####\s*ANSWER\s*:\s*\(?([A-J])\)?", re.IGNORECASE)
LAST_LINE_LETTER_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)


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
    raw = raw_lang.lower().strip()
    return LANG_CODE_TO_FULL.get(raw, raw)


def build_inference_prompt(question: str, options: list[str]) -> str:
    """Build the inference prompt (same format the student was trained on)."""
    options_text = _options_to_text(options)
    instruction = f"{question}\n\n{options_text}"

    prompt = (
        f"You are an expert problem solver. "
        f"Think step by step and solve the following question.\n\n"
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


def parse_predicted_answer(generation: str) -> str:
    """
    Extract the predicted answer letter from a model generation.
    Tries multiple fallback strategies:
    1. #### ANSWER: (X) pattern
    2. Last single capital letter A-J on its own line
    3. Last mentioned letter in the text
    """
    # Strategy 1: Standard answer tag
    match = ANSWER_TAG_RE.search(generation)
    if match:
        return match.group(1).upper()

    # Strategy 2: Look at the last few lines for a standalone letter
    lines = generation.strip().split("\n")
    for line in reversed(lines[-5:]):
        line = line.strip()
        # Check if line is just a letter or "(X)" pattern
        m = re.match(r"^\(?([A-J])\)?\.?$", line.strip(), re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Strategy 3: Last letter A-J mentioned
    all_letters = re.findall(r"\b([A-J])\b", generation)
    if all_letters:
        return all_letters[-1].upper()

    return ""


def load_test_data(test_path: str) -> list[dict]:
    """Load test JSONL file."""
    data = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["language"] = _normalize_language(row.get("language", "en"))
            data.append(row)
    return data


def run_inference_vllm(
    base_model: str,
    adapter_path: str | None,
    test_data: list[dict],
    max_new_tokens: int = 2048,
    gpu_memory_utilization: float = 0.6,
    tensor_parallel_size: int = 1,
) -> list[dict]:
    """
    Run inference using vLLM. If adapter_path is provided, merge LoRA first.
    """

    # If adapter_path is given, merge the adapter into the base model first
    if adapter_path and adapter_path.strip():
        LOGGER.info("Merging LoRA adapter from %s into %s...", adapter_path, base_model)
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import tempfile

        # Load base + adapter
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cpu",
        )
        model_with_adapter = PeftModel.from_pretrained(base, adapter_path)
        merged = model_with_adapter.merge_and_unload()

        # Save merged model to temp dir for vLLM
        merge_dir = Path(adapter_path) / "merged"
        merge_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(merge_dir))

        tok = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        tok.save_pretrained(str(merge_dir))

        model_path = str(merge_dir)
        LOGGER.info("Merged model saved to %s", model_path)

        # Free memory
        del base, model_with_adapter, merged
        torch.cuda.empty_cache()
    else:
        model_path = base_model

    # Load with vLLM
    LOGGER.info("Loading model with vLLM: %s", model_path)
    llm, tokenizer = load_vllm_llm(
        model_id=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # Build prompts
    LOGGER.info("Building inference prompts for %d test instances...", len(test_data))
    prompts = []
    for row in test_data:
        options = row["options"]
        if not isinstance(options, list):
            options = list(options)
        prompt = build_inference_prompt(row["question"], options)
        prompts.append(prompt)

    # Build chat messages
    batch_messages = [
        [{"role": "user", "content": p}] for p in prompts
    ]

    # Generate
    LOGGER.info("Running inference...")
    generations = prompt_vllm(
        llm,
        tokenizer,
        batch_messages,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        use_tqdm=True,
    )

    # Parse predictions
    results = []
    for row, gen in zip(test_data, generations):
        predicted = parse_predicted_answer(gen)
        gold = str(row.get("answer", "")).upper().strip()

        results.append({
            "language": row["language"],
            "subject": row.get("subject", ""),
            "question": row["question"],
            "gold_answer": gold,
            "predicted_answer": predicted,
            "generation": gen.strip(),
        })

    return results


def compute_metrics(predictions: list[dict]) -> dict[str, float]:
    """Compute accuracy per language and overall."""
    lang_correct = defaultdict(int)
    lang_total = defaultdict(int)

    for pred in predictions:
        lang = pred["language"]
        lang_total[lang] += 1
        if pred["predicted_answer"] == pred["gold_answer"]:
            lang_correct[lang] += 1

    metrics = {}
    total_correct = 0
    total_count = 0

    for lang in LANGUAGES:
        count = lang_total.get(lang, 0)
        correct = lang_correct.get(lang, 0)
        acc = (correct / count * 100) if count > 0 else 0.0
        metrics[lang] = acc
        total_correct += correct
        total_count += count

    metrics["overall"] = (total_correct / total_count * 100) if total_count > 0 else 0.0
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference + eval on test JSONL")
    parser.add_argument("--base_model", required=True,
                        help="Base student model path (HF ID or local)")
    parser.add_argument("--adapter_path", default="",
                        help="Optional PEFT/LoRA adapter path")
    parser.add_argument("--test_data", required=True,
                        help="Test JSONL path")
    parser.add_argument("--output_predictions", required=True,
                        help="Predictions JSONL output path")
    parser.add_argument("--report_file", required=True,
                        help="Metrics report text file output path")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max new tokens for generation")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.6,
        help="vLLM GPU memory utilization",
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


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    LOGGER.info("=" * 60)
    LOGGER.info("Part C: Student Inference and Evaluation")
    LOGGER.info("=" * 60)
    LOGGER.info("Base model   : %s", args.base_model)
    LOGGER.info("Adapter path : %s", args.adapter_path or "(none)")
    LOGGER.info("Test data    : %s", args.test_data)

    # ── 1. Load test data ──
    test_data = load_test_data(args.test_data)
    LOGGER.info("Loaded %d test instances", len(test_data))

    # ── 2. Run inference ──
    predictions = run_inference_vllm(
        base_model=args.base_model,
        adapter_path=args.adapter_path if args.adapter_path else None,
        test_data=test_data,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # ── 3. Save predictions ──
    pred_path = Path(args.output_predictions)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    LOGGER.info("Saved %d predictions to %s", len(predictions), pred_path)

    # ── 4. Compute and save metrics ──
    metrics = compute_metrics(predictions)

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        for lang in LANGUAGES:
            label = LANGUAGE_LABELS.get(lang, lang)
            line = f"{label.upper()} ACCURACY: {metrics.get(lang, 0.0):.2f}"
            f.write(line + "\n")
            LOGGER.info(line)
        overall_line = f"OVERALL ACCURACY: {metrics.get('overall', 0.0):.2f}"
        f.write(overall_line + "\n")
        LOGGER.info(overall_line)

    LOGGER.info("Saved metrics to %s", report_path)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()