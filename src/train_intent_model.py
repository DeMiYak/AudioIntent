from __future__ import annotations

"""
Train the ML intent classifier from gold dialogues.

Usage:
    python -m src.train_intent_model \
        --fit-input data/processed/gold_dialogues.jsonl \
        --model-output data/models/intent_classifier.joblib \
        --stats-output data/models/train_stats.json
"""

import argparse
import json
from pathlib import Path

from .ml_intent import (
    IntentClassifier,
    compute_train_stats,
    get_label,
    load_jsonl,
    save_model,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train ML intent classifier")
    parser.add_argument(
        "--fit-input",
        required=True,
        help="Path to gold_dialogues.jsonl used for training",
    )
    parser.add_argument(
        "--model-output",
        default="data/models/intent_classifier.joblib",
        help="Where to save the fitted model (joblib)",
    )
    parser.add_argument(
        "--stats-output",
        default=None,
        help="Where to save training statistics JSON (optional)",
    )
    args = parser.parse_args(argv)

    print(f"Loading training data from: {args.fit_input}")
    records = load_jsonl(args.fit_input)
    print(f"  {len(records)} records loaded")

    from collections import Counter
    label_counts = Counter(get_label(r) for r in records)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    print("Fitting IntentClassifier...")
    model = IntentClassifier()
    model.fit(records)
    print(f"  Classes: {model.classes_}")

    model_path = save_model(model, args.model_output)
    print(f"Model saved to: {model_path}")

    if args.stats_output:
        stats = compute_train_stats(records, model)
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Training stats saved to: {stats_path}")

        report = stats.get("train_classification_report", {})
        for cls in ["contact_open", "contact_close", "none"]:
            row = report.get(cls, {})
            print(
                f"  {cls:20s}  P={row.get('precision', 0):.3f}"
                f"  R={row.get('recall', 0):.3f}"
                f"  F1={row.get('f1-score', 0):.3f}"
                f"  support={int(row.get('support', 0))}"
            )


if __name__ == "__main__":
    main()
