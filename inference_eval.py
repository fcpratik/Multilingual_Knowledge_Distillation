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
LANG_LABELS = {"english": "English", "hindi": "Hindi", "bengali": "Bengali",
               "kannada": "Kannada", "tamil": "Tamil"}
LANG_CODE_TO_FULL = {"en": "english", "hi": "hindi", "bn": "bengali", "kn": "kannada", "ta": "tamil"}
ANSWER_RE = re.compile(r"####\s*ANSWER\s*:\s*\(?([A-J])\)?", re.IGNORECASE)


def setup_logger(level):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s", force=True)


def _options_to_text(opts):
    return "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(opts))


def build_prompt(question, options):
    return (
        f"You are an expert problem solver. Think step by step.\n\n"
        f"Format:\n1. Reasoning inside <reasoning></reasoning> tags.\n"
        f"2. Final answer: #### ANSWER: (X)\n\n"
        f"{question}\n\n{_options_to_text(options)}\n\nSolve step by step."
    )


def parse_answer(gen):
    m = ANSWER_RE.search(gen)
    if m: return m.group(1).upper()
    for line in reversed(gen.strip().split("\n")[-5:]):
        m2 = re.match(r"^\(?([A-J])\)?\.?$", line.strip(), re.IGNORECASE)
        if m2: return m2.group(1).upper()
    letters = re.findall(r"\b([A-J])\b", gen)
    return letters[-1].upper() if letters else ""


def load_test(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line.strip())
                row["language"] = LANG_CODE_TO_FULL.get(
                    row.get("language","en").lower().strip(),
                    row.get("language","english").lower().strip())
                data.append(row)
    return data


def run_inference(base_model, adapter_path, test_data, max_tokens=2048, gpu_mem=0.90, tp=1):
    if adapter_path and adapter_path.strip():
        LOGGER.info("Merging LoRA adapter...")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16,
                                                     trust_remote_code=True)
        merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
        merge_dir = Path(adapter_path) / "merged"
        merge_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(merge_dir))
        AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True).save_pretrained(str(merge_dir))
        model_path = str(merge_dir)
        del base, merged; torch.cuda.empty_cache()
    else:
        model_path = base_model

    llm, tok = load_vllm_llm(model_path, gpu_memory_utilization=gpu_mem, tensor_parallel_size=tp)

    msgs = []
    for row in test_data:
        opts = row["options"] if isinstance(row["options"], list) else list(row["options"])
        msgs.append([{"role": "user", "content": build_prompt(row["question"], opts)}])

    LOGGER.info("Inference on %d samples...", len(test_data))
    gens = prompt_vllm(llm, tok, msgs, max_new_tokens=max_tokens, temperature=0.0, use_tqdm=True)

    results = []
    for row, gen in zip(test_data, gens):
        results.append({
            "language": row["language"], "subject": row.get("subject",""),
            "question": row["question"],
            "gold_answer": str(row.get("answer","")).upper().strip(),
            "predicted_answer": parse_answer(gen), "generation": gen.strip(),
        })
    return results


def compute_metrics(preds):
    lc, lt = defaultdict(int), defaultdict(int)
    for p in preds:
        lt[p["language"]] += 1
        if p["predicted_answer"] == p["gold_answer"]:
            lc[p["language"]] += 1
    metrics = {}
    tc = tt = 0
    for lang in LANGUAGES:
        n, c = lt.get(lang,0), lc.get(lang,0)
        metrics[lang] = (c/n*100) if n else 0.0
        tc += c; tt += n
    metrics["overall"] = (tc/tt*100) if tt else 0.0
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", default="")
    p.add_argument("--test_data", required=True)
    p.add_argument("--output_predictions", required=True)
    p.add_argument("--report_file", required=True)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--time_limit", type=int, default=55)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger(args.log_level)
    timer = TimeGuard(args.time_limit, safety_margin_minutes=5)

    LOGGER.info("=" * 60)
    LOGGER.info("Part C: Inference & Evaluation")
    LOGGER.info("=" * 60)

    test_data = load_test(args.test_data)
    LOGGER.info("Loaded %d test instances", len(test_data))

    preds = run_inference(args.base_model, args.adapter_path or None, test_data,
                          args.max_new_tokens, args.gpu_memory_utilization,
                          args.tensor_parallel_size)

    Path(args.output_predictions).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_predictions, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    metrics = compute_metrics(preds)
    Path(args.report_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_file, "w") as f:
        for lang in LANGUAGES:
            line = f"{LANG_LABELS[lang].upper()} ACCURACY: {metrics[lang]:.2f}"
            f.write(line + "\n"); LOGGER.info(line)
        f.write(f"OVERALL ACCURACY: {metrics['overall']:.2f}\n")
    LOGGER.info("OVERALL: %.2f", metrics["overall"])
    timer.log_status(LOGGER)


if __name__ == "__main__":
    main()