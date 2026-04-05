"""
Сравнение ML-классификаторов на validation-выборке.

Для каждого классификатора:
  1. Обучает модель на gold_dialogues.jsonl
  2. Запускает pipeline в combined-режиме на validation-окнах
  3. Считает метрики по схеме evaluation.ipynb
  4. Сохраняет сводную таблицу в JSON и печатает в консоль

Использование:
    python -m src.compare_classifiers \
        --fit-input data/processed/gold_dialogues.jsonl \
        --gold-excel data/raw/gold/data_val.xlsx \
        --transcript-dir artifacts/validation_status_svoboden_asr_diarization_colab/windows \
        --samples-dir data/raw/validation/audio_profiles \
        --output artifacts/classifier_comparison.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from .ml_intent import CLASSIFIER_CHOICES, IntentClassifier, load_jsonl, save_model


def _parse_pairs(df: pd.DataFrame, column: str, category: str) -> set[tuple[str, str, str]]:
    pairs: set[tuple[str, str, str]] = set()
    for val in df[column]:
        if pd.isna(val):
            continue
        val = str(val).strip()
        if val in ("", "0"):
            continue
        for piece in val.split(";"):
            piece = piece.strip().lower()
            if " - " in piece:
                speaker, phrase = piece.split(" - ", 1)
                pairs.add((speaker.strip(), phrase.strip(), category))
    return pairs


def _greedy_eval(
    gold: set[tuple[str, str, str]], pred: set[tuple[str, str, str]]
) -> tuple[int, int, float]:
    gold_l, pred_l = sorted(gold), sorted(pred)
    used: set[int] = set()
    matched = exact = 0
    sims: list[float] = []
    for g_s, g_p, g_c in gold_l:
        best_j, best_s = None, -1.0
        for j, (p_s, p_p, p_c) in enumerate(pred_l):
            if j in used or g_c != p_c or g_s != p_s:
                continue
            s = SequenceMatcher(None, g_p, p_p).ratio()
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None:
            used.add(best_j)
            matched += 1
            if g_p == pred_l[best_j][1]:
                exact += 1
            sims.append(best_s)
    return matched, exact, float(np.mean(sims)) if sims else 0.0


def _evaluate(
    gold_df: pd.DataFrame, pred_df: pd.DataFrame
) -> dict[str, dict]:
    go = _parse_pairs(gold_df, "opening", "opening")
    gc = _parse_pairs(gold_df, "closing", "closing")
    po = _parse_pairs(pred_df, "opening", "opening")
    pc = _parse_pairs(pred_df, "closing", "closing")

    results = {}
    for name, g, p in [("all", go | gc, po | pc), ("opening", go, po), ("closing", gc, pc)]:
        j = len(g & p) / len(g | p) if (g | p) else 1.0
        pr = len(g & p) / len(p) if p else 0.0
        r = len(g & p) / len(g) if g else 0.0
        f1 = 2 * pr * r / (pr + r) if pr + r else 0.0
        matched, exact, avg_sim = _greedy_eval(g, p)
        results[name] = {
            "gold": len(g),
            "pred": len(p),
            "jaccard": round(j, 4),
            "precision": round(pr, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "matched": matched,
            "exact": exact,
            "avg_sim": round(avg_sim, 4),
        }
    return results


def _run_pipeline(
    classifier_type: str,
    model_path: Path,
    gold_excel: Path,
    transcript_dir: Path,
    samples_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Запускает pipeline и возвращает (pred_df, gold_df)."""
    pairs_path = output_dir / "extracted_pairs.xlsx"
    gold_path = output_dir / "gold.xlsx"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.pipeline",
        "--gold-excel", str(gold_excel),
        "--transcript-input-dir", str(transcript_dir),
        "--diarization-input-dir", str(transcript_dir),
        "--samples-dir", str(samples_dir),
        "--output-dir", str(output_dir),
        "--extracted-pairs-output", str(pairs_path),
        "--gold-output", str(gold_path),
        "--diarization-segment-mode", "regular",
        "--intent-mode", "combined",
        "--ml-model", str(model_path),
        "--ml-confidence-threshold", "0.35",
        "--similarity-threshold", "0.48",
        "--skip-asr", "--skip-diarization",
        "--fit-input", "data/processed/gold_dialogues.jsonl",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] pipeline exited {result.returncode}")
        print(result.stderr[-500:] if result.stderr else "")

    pred_df = pd.read_excel(pairs_path)
    gold_df = pd.read_excel(gold_path)
    return pred_df, gold_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Сравнение ML-классификаторов")
    parser.add_argument("--fit-input", required=True)
    parser.add_argument("--gold-excel", required=True)
    parser.add_argument("--transcript-dir", required=True)
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument(
        "--output", default="artifacts/classifier_comparison.json",
        help="Путь для сохранения сводной таблицы JSON",
    )
    parser.add_argument(
        "--classifiers", nargs="+", default=CLASSIFIER_CHOICES,
        choices=CLASSIFIER_CHOICES,
        help="Какие классификаторы сравнивать (по умолчанию все)",
    )
    args = parser.parse_args(argv)

    records = load_jsonl(args.fit_input)
    print(f"Загружено {len(records)} обучающих записей\n")

    comparison: list[dict] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for clf_type in args.classifiers:
            print(f"=== {clf_type.upper()} ===")

            # Обучение
            print(f"  Обучение...")
            model = IntentClassifier(classifier_type=clf_type)
            model.fit(records)
            model_path = tmp / f"model_{clf_type}.joblib"
            save_model(model, model_path)

            # Статистика обучения
            from .ml_intent import compute_train_stats
            stats = compute_train_stats(records, model)
            train_report = stats.get("train_classification_report", {})
            for cls in ["contact_open", "contact_close", "none"]:
                row = train_report.get(cls, {})
                print(
                    f"    {cls:20s}  P={row.get('precision', 0):.3f}"
                    f"  R={row.get('recall', 0):.3f}"
                    f"  F1={row.get('f1-score', 0):.3f}"
                )

            # Validation
            print(f"  Validation pipeline...")
            out_dir = tmp / f"val_{clf_type}"
            pred_df, gold_df = _run_pipeline(
                classifier_type=clf_type,
                model_path=model_path,
                gold_excel=Path(args.gold_excel),
                transcript_dir=Path(args.transcript_dir),
                samples_dir=Path(args.samples_dir),
                output_dir=out_dir,
            )

            metrics = _evaluate(gold_df, pred_df)
            m = metrics["all"]
            print(
                f"  ALL: pred={m['pred']} J={m['jaccard']:.4f} "
                f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
                f"matched={m['matched']} exact={m['exact']} avg_sim={m['avg_sim']:.4f}"
            )
            print()

            comparison.append({
                "classifier": clf_type,
                "train_report": train_report,
                "validation_metrics": metrics,
            })

    # Сводная таблица в консоль
    print("\n" + "=" * 90)
    print(f"{'Classifier':<8} {'Pred':>5} {'Jaccard':>8} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'matched':>8} {'exact':>6} {'avg_sim':>8}")
    print("-" * 90)
    for entry in comparison:
        clf = entry["classifier"]
        m = entry["validation_metrics"]["all"]
        print(
            f"{clf:<8} {m['pred']:>5} {m['jaccard']:>8.4f} {m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['matched']:>8} "
            f"{m['exact']:>6} {m['avg_sim']:>8.4f}"
        )

    # Сохранение
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
