"""
Split dataset.jsonl into train and validation sets.
Ensures balanced language representation in both splits.
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]
LANG_CODE_TO_FULL = {"en": "english", "hi": "hindi", "bn": "bengali", "kn": "kannada", "ta": "tamil"}


def main():
    p = argparse.ArgumentParser(description="Split dataset into train/val with balanced languages")
    p.add_argument("--input", default="data/dataset.jsonl", help="Path to full dataset")
    p.add_argument("--train_out", default="data/dataset_train.jsonl", help="Output train split")
    p.add_argument("--val_out", default="data/dataset_val.jsonl", help="Output val split")
    p.add_argument("--val_ratio", type=float, default=0.10, help="Fraction for validation")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Load all data
    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line.strip())
                row["language"] = LANG_CODE_TO_FULL.get(
                    row.get("language", "en").lower().strip(),
                    row.get("language", "english").lower().strip()
                )
                data.append(row)

    print(f"Loaded {len(data)} total instances")

    # Group by language for stratified split
    by_lang = defaultdict(list)
    for row in data:
        by_lang[row["language"]].append(row)

    rng = random.Random(args.seed)
    train_set, val_set = [], []

    for lang in LANGUAGES:
        items = by_lang[lang]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * args.val_ratio))
        val_set.extend(items[:n_val])
        train_set.extend(items[n_val:])
        print(f"  {lang}: {len(items)} total → {len(items) - n_val} train, {n_val} val")

    # Include any languages not in LANGUAGES list
    for lang, items in by_lang.items():
        if lang not in LANGUAGES:
            rng.shuffle(items)
            n_val = max(1, int(len(items) * args.val_ratio))
            val_set.extend(items[:n_val])
            train_set.extend(items[n_val:])

    rng.shuffle(train_set)
    rng.shuffle(val_set)

    # Save
    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.train_out, "w", encoding="utf-8") as f:
        for r in train_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(args.val_out, "w", encoding="utf-8") as f:
        for r in val_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved: {len(train_set)} train → {args.train_out}")
    print(f"Saved: {len(val_set)} val   → {args.val_out}")


if __name__ == "__main__":
    main()
