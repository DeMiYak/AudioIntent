"""
Missed-gold diagnostic for the validation pipeline.

Usage:
    python -m src.error_analysis \
        --gold-jsonl data/processed/gold_dialogues.jsonl \
        --predictions-jsonl artifacts/val_run/windows/*/predictions.jsonl \
        --miss-analysis-output artifacts/miss_analysis.json

Each missed gold event is categorised into one of:
    not_candidate       – the utterance produced zero rule/ML candidates
    filtered_by_ml      – a candidate was generated but dropped by the ML filter
    wrong_type          – the utterance has a prediction but with the opposite intent
    wrong_speaker       – phrase matched but attributed to a different speaker
    asr_mismatch        – utterance_id not found at all in predictions (ASR/alignment gap)
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Span matching helpers
# ---------------------------------------------------------------------------

def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def _spans_match(gold_span: tuple[int, int, str], pred: dict[str, Any]) -> bool:
    gs, ge, gi = gold_span
    return (
        str(pred.get("intent_type", "")) == gi
        and _overlap(gs, ge, int(pred.get("char_start", 0)), int(pred.get("char_end", 0)))
    )


# ---------------------------------------------------------------------------
# Category logic
# ---------------------------------------------------------------------------

_CATEGORY_ORDER = [
    "asr_mismatch",
    "not_candidate",
    "filtered_by_ml",
    "wrong_type",
    "wrong_speaker",
    "matched",   # TP — included for completeness
]


def categorise_missed_events(
    gold_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    rule_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Returns a dict with per-event categorisation and aggregate counts.

    Parameters
    ----------
    gold_records:
        The gold JSONL rows, each with ``utterance_id`` and ``annotations``.
    predictions:
        Final pipeline predictions (after ML filter / conflict resolution).
    rule_candidates:
        Optional raw rule-based candidates *before* ML filter.  When provided,
        enables the ``filtered_by_ml`` category.
    """
    # Build lookup structures
    preds_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        preds_by_uid[str(p.get("utterance_id", ""))].append(p)

    cands_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if rule_candidates:
        for c in rule_candidates:
            cands_by_uid[str(c.get("utterance_id", ""))].append(c)

    events: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    # Index of all utterance_ids that appear in predictions (any intent)
    predicted_uids: set[str] = set(preds_by_uid.keys())

    for record in gold_records:
        uid = str(record.get("utterance_id", ""))
        text = str(record.get("text", ""))
        speaker = str(record.get("speaker_name", ""))
        for ann in record.get("annotations", []):
            intent = str(ann.get("intent_type", ""))
            cs = int(ann.get("char_start", 0))
            ce = int(ann.get("char_end", len(text)))
            phrase = text[cs:ce]

            gold_span = (cs, ce, intent)
            uid_preds = preds_by_uid.get(uid, [])

            # Check TP first
            exact_match = any(_spans_match(gold_span, p) for p in uid_preds)
            if exact_match:
                category = "matched"
            elif uid not in predicted_uids and not uid_preds:
                # utterance_id never appeared in any prediction record
                category = "asr_mismatch"
            else:
                # Check if any candidate was generated
                uid_cands = cands_by_uid.get(uid, [])
                any_candidate = any(
                    str(c.get("intent_type", "")) == intent
                    and _overlap(cs, ce, int(c.get("char_start", 0)), int(c.get("char_end", 0)))
                    for c in uid_cands
                )

                any_pred_same_type = any(
                    str(p.get("intent_type", "")) == intent for p in uid_preds
                )
                any_pred_other_type = any(
                    str(p.get("intent_type", "")) != intent for p in uid_preds
                )
                any_pred_wrong_speaker = any(
                    str(p.get("intent_type", "")) == intent
                    and str(p.get("speaker_name", "")) != speaker
                    for p in uid_preds
                )

                if any_candidate and rule_candidates is not None and not any_pred_same_type:
                    category = "filtered_by_ml"
                elif any_pred_wrong_speaker:
                    category = "wrong_speaker"
                elif any_pred_other_type and not any_pred_same_type:
                    category = "wrong_type"
                elif rule_candidates is None and not uid_preds:
                    category = "not_candidate"
                else:
                    category = "not_candidate"

            counts[category] += 1
            events.append({
                "utterance_id": uid,
                "speaker_name": speaker,
                "intent_type": intent,
                "phrase": phrase,
                "char_start": cs,
                "char_end": ce,
                "category": category,
            })

    total_gold = len(events)
    total_missed = total_gold - counts.get("matched", 0)

    return {
        "total_gold": total_gold,
        "total_matched": counts.get("matched", 0),
        "total_missed": total_missed,
        "counts": {k: counts.get(k, 0) for k in _CATEGORY_ORDER},
        "events": events,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Missed-gold diagnostic: categorise why gold events were not predicted."
    )
    parser.add_argument(
        "--gold-jsonl",
        type=str,
        default="data/processed/gold_dialogues.jsonl",
        help="Gold JSONL file with utterance_id + annotations.",
    )
    parser.add_argument(
        "--predictions-jsonl",
        type=str,
        nargs="+",
        default=None,
        help="Prediction JSONL files (supports glob patterns).",
    )
    parser.add_argument(
        "--rule-candidates-jsonl",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Raw rule-based candidates before ML filter (enables filtered_by_ml category). "
            "Supports glob patterns."
        ),
    )
    parser.add_argument(
        "--miss-analysis-output",
        type=str,
        default="artifacts/miss_analysis.json",
        help="Output path for the miss analysis JSON.",
    )
    return parser.parse_args()


def _expand_globs(patterns: list[str] | None) -> list[Path]:
    if not patterns:
        return []
    paths: list[Path] = []
    for pat in patterns:
        expanded = glob.glob(pat, recursive=True)
        if expanded:
            paths.extend(Path(p) for p in sorted(expanded))
        else:
            p = Path(pat)
            if p.exists():
                paths.append(p)
    return paths


def main() -> None:
    args = parse_args()

    gold_records = load_jsonl(args.gold_jsonl)

    pred_paths = _expand_globs(args.predictions_jsonl)
    predictions: list[dict[str, Any]] = []
    for p in pred_paths:
        predictions.extend(load_jsonl(p))

    cand_paths = _expand_globs(args.rule_candidates_jsonl)
    rule_candidates: list[dict[str, Any]] | None = None
    if cand_paths:
        rule_candidates = []
        for p in cand_paths:
            rule_candidates.extend(load_jsonl(p))

    report = categorise_missed_events(
        gold_records=gold_records,
        predictions=predictions,
        rule_candidates=rule_candidates,
    )

    save_json(report, args.miss_analysis_output)

    print(f"\n=== MISS ANALYSIS ===")
    print(f"Total gold events : {report['total_gold']}")
    print(f"Matched (TP)      : {report['total_matched']}")
    print(f"Missed            : {report['total_missed']}")
    print("\nMiss categories:")
    for cat, cnt in report["counts"].items():
        if cat != "matched":
            pct = f"{100*cnt/report['total_gold']:.1f}%" if report["total_gold"] else "-"
            print(f"  {cat:<20} {cnt:>4}  ({pct})")
    print(f"\nSaved to: {args.miss_analysis_output}")


if __name__ == "__main__":
    main()
