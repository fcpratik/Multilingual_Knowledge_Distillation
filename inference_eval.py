from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch

from utils import load_vllm_llm, prompt_vllm, TimeGuard


LOGGER = logging.getLogger(__name__)
LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]
LANGUAGE_LABELS = {"english": "English", "hindi": "Hindi", "bengali": "Bengali",
                   "kannada": "Kannada", "tamil": "Tamil"}
LANG_CODE_TO_FULL = {"en": "english", "hi": "hindi", "bn": "bengali", "kn": "kannada", "ta": "tamil"}
ANSWER_TAG_RE = re.compile(r"####\s*ANSWER\s*:\s*\(?([A-J])\)?", re.IGNORECASE)


def setup_logger(level):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s", force=True)


def _options_to_text(options):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(options))


def build_inference_prompt(question, options):
    options_text = _options_to_text(options)
    return (
        f"You are an expert problem solver. "
        f"Think step by step and solve the following question.\n\n"
        f"Follow this format strictly:\n"
        f"1. First, provide your detailed step-by-step reasoning inside "
        f"<reasoning> and </reasoning> tags.\n"
        f"2. After </reasoning>, write your final answer on a new line in "
        f"exactly this format: #### ANSWER: (X)\n"
        f"   where X is the letter of the correct option.\n\n"
        f"{question}\n\n{options_text}\n\nNow solve this step by step."
    )


def parse_answer(gen):
    m = ANSWER_TAG_RE.search(gen)
    if m:
        return m.group(1).upper()
    lines = gen.strip().split("\n")
    for line in reversed(lines[-5:]):
        m2 = re.match(r"^\(?([A-J])\)?\.?$", line.strip(), re.IGNORECASE)
        if m2:
            return m2.group(1).upper()
    all_letters = re.findall(r"\b([A-J])\b", gen)
    return all_letters[-1].upper() if all_letters else ""


def load_test_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line.strip())
                row["language"] = LANG_CODE_TO_FULL.get(
                    row.get("language", "en").lower().strip(),
                    row.get("language", "english").lower().strip())
                data.append(row)
    return data


def run_inference(base_model, adapter_path, test_data, max_new_tokens=2048,
                  gpu_mem=0.85, tp=1, timer=None):
    # Merge adapter if provided
    if adapter_path and adapter_path.strip():
        LOGGER.info("Merging LoRA adapter...")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import tempfile

        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu")
        merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()

        merge_dir = Path(adapter_path) / "merged"
        merge_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(merge_dir))
        AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True).save_pretrained(str(merge_dir))

        model_path = str(merge_dir)
        del base, merged
        torch.cuda.empty_cache()
    else:
        model_path = base_model

    llm, tokenizer = load_vllm_llm(model_path, gpu_memory_utilization=gpu_mem,
                                    tensor_parallel_size=tp)

    # Build prompts
    prompts = []
    for row in test_data:
        opts = row["options"] if isinstance(row["options"], list) else list(row["options"])
        prompts.append(build_inference_prompt(row["question"], opts))

    batch_msgs = [[{"role": "user", "content": p}] for p in prompts]

    LOGGER.info("Running inference on %d samples...", len(test_data))
    generations = prompt_vllm(llm, tokenizer, batch_msgs,
                              max_new_tokens=max_new_tokens, temperature=0.0, use_tqdm=True)

    results = []
    for row, gen in zip(test_data, generations):
        results.append({
            "language": row["language"],
            "subject": row.get("subject", ""),
            "question": row["question"],
            "gold_answer": str(row.get("answer", "")).upper().strip(),
            "predicted_answer": parse_answer(gen),
            "generation": gen.strip(),
        })
    return results


def compute_metrics(predictions):
    lang_correct, lang_total = defaultdict(int), defaultdict(int)
    for p in predictions:
        lang_total[p["language"]] += 1
        if p["predicted_answer"] == p["gold_answer"]:
            lang_correct[p["language"]] += 1

    metrics = {}
    tc, tt = 0, 0
    for lang in LANGUAGES:
        n = lang_total.get(lang, 0)
        c = lang_correct.get(lang, 0)
        metrics[lang] = (c / n * 100) if n > 0 else 0.0
        tc += c
        tt += n
    metrics["overall"] = (tc / tt * 100) if tt > 0 else 0.0
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", default="")
    p.add_argument("--test_data", required=True)
    p.add_argument("--output_predictions", required=True)
    p.add_argument("--report_file", required=True)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--time_limit", type=int, default=55)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger(args.log_level)
    timer = TimeGuard(limit_minutes=args.time_limit, safety_margin_minutes=5)

    LOGGER.info("=" * 60)
    LOGGER.info("Part C: Inference & Evaluation")
    LOGGER.info("=" * 60)

    test_data = load_test_data(args.test_data)
    LOGGER.info("Loaded %d test instances", len(test_data))

    predictions = run_inference(
        args.base_model, args.adapter_path or None, test_data,
        max_new_tokens=args.max_new_tokens,
        gpu_mem=args.gpu_memory_utilization,
        tp=args.tensor_parallel_size, timer=timer,
    )

    # Save predictions
    pred_path = Path(args.output_predictions)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Save metrics
    metrics = compute_metrics(predictions)
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        for lang in LANGUAGES:
            line = f"{LANGUAGE_LABELS[lang].upper()} ACCURACY: {metrics[lang]:.2f}"
            f.write(line + "\n")
            LOGGER.info(line)
        f.write(f"OVERALL ACCURACY: {metrics['overall']:.2f}\n")
        LOGGER.info("OVERALL ACCURACY: %.2f", metrics["overall"])

    timer.log_status(LOGGER)
    LOGGER.info("Done!")


if __name__ == "__main__":
    main()