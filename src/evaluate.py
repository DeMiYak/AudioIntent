from __future__ import annotations

"""
Evaluate intent predictions against gold annotations.

Works on both:
  - gold_dialogues.jsonl format (has 'annotations' field)
  - predictions JSONL format (has 'intent_type' field, one prediction per line)

Usage (evaluate ML predictions against gold):
    python -m src.evaluate \
        --gold data/processed/gold_dialogues.jsonl \
        --predictions artifacts/ml_predictions.jsonl \
        --output artifacts/eval_ml_metrics.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


INTENT_TYPES = ["contact_open", "contact_close"]


# ---------------------------------------------------------------------------
# Вспомогательные функции загрузки
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Индекс gold: utterance_id -> множество intent_types
# ---------------------------------------------------------------------------

def build_gold_index(
    gold_records: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """
    Returns {utterance_id: {intent_type, ...}} from gold records.
    A record with no annotations gets an empty set (= "none").
    """
    index: dict[str, set[str]] = {}
    for rec in gold_records:
        uid = str(rec.get("utterance_id", ""))
        annotations = rec.get("annotations") or []
        intent_types = {
            str(ann["intent_type"])
            for ann in annotations
            if ann.get("intent_type") in INTENT_TYPES
        }
        index[uid] = intent_types
    return index


# ---------------------------------------------------------------------------
# Индекс предсказаний: utterance_id -> множество intent_types
# ---------------------------------------------------------------------------

def build_pred_index(
    pred_records: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """
    Returns {utterance_id: {intent_type, ...}} from prediction records.
    Each prediction record has 'utterance_id' and 'intent_type'.
    """
    index: dict[str, set[str]] = defaultdict(set)
    for rec in pred_records:
        uid = str(rec.get("utterance_id", ""))
        intent = str(rec.get("intent_type", ""))
        if intent in INTENT_TYPES:
            index[uid].add(intent)
    return dict(index)


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def compute_metrics(
    gold_index: dict[str, set[str]],
    pred_index: dict[str, set[str]],
    intent_type: str | None = None,
) -> dict[str, Any]:
    """
    Computes precision, recall, F1 at utterance level.

    If intent_type is None, counts any non-empty gold/pred as positive.
    If intent_type is specified, counts only that type.
    """
    tp = fp = fn = 0

    all_uids = set(gold_index) | set(pred_index)

    for uid in all_uids:
        gold_intents = gold_index.get(uid, set())
        pred_intents = pred_index.get(uid, set())

        if intent_type is not None:
            gold_pos = intent_type in gold_intents
            pred_pos = intent_type in pred_intents
        else:
            gold_pos = bool(gold_intents)
            pred_pos = bool(pred_intents)

        if gold_pos and pred_pos:
            tp += 1
        elif pred_pos and not gold_pos:
            fp += 1
        elif gold_pos and not pred_pos:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "gold_positive": tp + fn,
        "pred_positive": tp + fp,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate intent predictions vs gold")
    parser.add_argument(
        "--gold",
        required=True,
        help="Path to gold_dialogues.jsonl",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions JSONL (one prediction per line with utterance_id + intent_type)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to save evaluation metrics JSON (optional)",
    )
    args = parser.parse_args(argv)

    gold_records = load_jsonl(args.gold)
    pred_records = load_jsonl(args.predictions)
    print(f"Gold: {len(gold_records)} records, Predictions: {len(pred_records)} records")

    gold_index = build_gold_index(gold_records)
    pred_index = build_pred_index(pred_records)

    metrics: dict[str, Any] = {}
    for intent in INTENT_TYPES + [None]:
        label = intent if intent is not None else "overall"
        metrics[label] = compute_metrics(gold_index, pred_index, intent_type=intent)

    print("\n--- Evaluation Results ---")
    header = f"{'Intent':20s}  {'P':>6}  {'R':>6}  {'F1':>6}  {'Gold':>6}  {'Pred':>6}"
    print(header)
    print("-" * len(header))
    for label, m in metrics.items():
        print(
            f"{label:20s}  {m['precision']:6.3f}  {m['recall']:6.3f}"
            f"  {m['f1']:6.3f}  {m['gold_positive']:6d}  {m['pred_positive']:6d}"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"gold": args.gold, "predictions": args.predictions, "metrics": metrics}, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    main()
