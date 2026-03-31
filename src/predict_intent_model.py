from __future__ import annotations

"""
Run the ML intent classifier on utterances and output predictions.

Usage:
    python -m src.predict_intent_model \
        --model data/models/intent_classifier.joblib \
        --input utterances.jsonl \
        --output predictions.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any

from .ml_intent import load_jsonl, load_model, predict_for_records


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict intent with ML classifier")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to fitted IntentClassifier (.joblib)",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to utterances JSONL file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where predictions JSONL will be written",
    )
    args = parser.parse_args(argv)

    print(f"Loading model from: {args.model}")
    model = load_model(args.model)

    print(f"Loading utterances from: {args.input}")
    records = load_jsonl(args.input)
    print(f"  {len(records)} utterances loaded")

    predictions = predict_for_records(records, model)
    print(f"  {len(predictions)} predictions (contact_open / contact_close only)")

    out_path = save_jsonl(predictions, args.output)
    print(f"Predictions saved to: {out_path}")


if __name__ == "__main__":
    main()
