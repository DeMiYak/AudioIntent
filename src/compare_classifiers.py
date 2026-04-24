"""
Сравнение ML-классификаторов.

Два режима оценки:
  1. Кросс-валидация на gold_dialogues.jsonl (5-fold, threshold-independent)
  2. Validation pipeline: обучение на полных данных, запуск на validation-окнах
     с индивидуально подобранным порогом (threshold sweep по PR-кривой)

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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_curve

from .ml_intent import (
    CLASSIFIER_CHOICES,
    IntentClassifier,
    _build_classifier,
    _records_to_numerical,
    _records_to_texts,
    compute_dialogue_positions,
    compute_neighbour_texts,
    get_label,
    load_jsonl,
    save_model,
)
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Кросс-валидация
# ---------------------------------------------------------------------------

def _cv_report(records: list[dict], classifier_type: str, n_splits: int = 5) -> dict:
    """
    Стратифицированная k-fold CV на gold_dialogues.jsonl.
    Возвращает усреднённые precision/recall/F1 по трём классам.
    """
    labels = [get_label(r) for r in records]
    all_preds: list[str] = []
    all_true: list[str] = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(records, labels)):
        train_recs = [records[i] for i in train_idx]
        val_recs   = [records[i] for i in val_idx]

        # Обучение на fold-train
        tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5),
            max_features=50_000, sublinear_tf=True,
        )
        clf = _build_classifier(classifier_type)

        train_texts = _records_to_texts(train_recs)
        train_pos   = compute_dialogue_positions(train_recs)
        train_nbrs  = compute_neighbour_texts(train_recs)
        train_labels = [get_label(r) for r in train_recs]

        X_tr = hstack([
            tfidf.fit_transform(train_texts),
            csr_matrix(_records_to_numerical(train_recs, train_pos, train_nbrs)),
        ])
        clf.fit(X_tr, train_labels)

        # Предсказание на fold-val
        val_texts = _records_to_texts(val_recs)
        val_pos   = compute_dialogue_positions(val_recs)
        val_nbrs  = compute_neighbour_texts(val_recs)

        X_val = hstack([
            tfidf.transform(val_texts),
            csr_matrix(_records_to_numerical(val_recs, val_pos, val_nbrs)),
        ])
        proba = clf.predict_proba(X_val)
        pred_indices = np.argmax(proba, axis=1)
        classes = list(clf.classes_)
        preds = [classes[i] for i in pred_indices]

        all_preds.extend(preds)
        all_true.extend([get_label(r) for r in val_recs])

    report = classification_report(
        all_true, all_preds,
        labels=["contact_open", "contact_close", "none"],
        output_dict=True,
        zero_division=0,
    )
    return report


# ---------------------------------------------------------------------------
# Подбор порога (threshold sweep)
# ---------------------------------------------------------------------------

def _find_best_threshold(
    records: list[dict],
    model: IntentClassifier,
    target_class: str = "contact_open",
    beta: float = 1.0,
) -> float:
    """
    Ищет порог уверенности, максимизирующий F-beta для target_class на обучающих данных.
    Используется отдельно для contact_open и contact_close и берётся минимум.
    """
    labels = np.array([get_label(r) for r in records])
    _, proba = model.predict_proba(records)
    class_idx = model.classes_.index(target_class)
    scores = proba[:, class_idx]
    binary_labels = (labels == target_class).astype(int)

    precision, recall, thresholds = precision_recall_curve(binary_labels, scores)
    # F-beta: beta=1 → F1, beta<1 → precision-weighted
    beta2 = beta ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        f_beta = np.where(
            precision + recall > 0,
            (1 + beta2) * precision * recall / (beta2 * precision + recall),
            0.0,
        )
    best_idx = np.argmax(f_beta[:-1])  # thresholds has len = len(precision)-1
    return float(thresholds[best_idx])


# ---------------------------------------------------------------------------
# Validation pipeline (переиспользуем из предыдущей версии)
# ---------------------------------------------------------------------------

def _parse_pairs_phrase_only(df: pd.DataFrame, column: str, category: str) -> set[tuple[str, str]]:
    """Как _parse_pairs, но без speaker — только (phrase, category)."""
    pairs: set[tuple[str, str]] = set()
    for val in df[column]:
        if pd.isna(val):
            continue
        val = str(val).strip()
        if val in ("", "0"):
            continue
        for piece in val.split(";"):
            piece = piece.strip().lower()
            if " - " in piece:
                _, phrase = piece.split(" - ", 1)
                pairs.add((phrase.strip(), category))
    return pairs


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


def _compute_diagnostics(gold_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """
    Дополнительные диагностики поверх базовой оценки:
    - phrase_only: метрики без учёта speaker (выявляет, сколько ошибок — speaker attribution)
    - speaker_mismatch_count: кол-во пар, где фраза почти совпадает, но speaker разный
    """
    diag: dict = {}

    # Phrase-only evaluation
    phrase_results: dict[str, dict] = {}
    for name, col, cat in [("opening", "opening", "opening"), ("closing", "closing", "closing")]:
        go = _parse_pairs_phrase_only(gold_df, col, cat)
        po = _parse_pairs_phrase_only(pred_df, col, cat)
        j = len(go & po) / len(go | po) if (go | po) else 1.0
        pr = len(go & po) / len(po) if po else 0.0
        r = len(go & po) / len(go) if go else 0.0
        f1 = 2 * pr * r / (pr + r) if pr + r else 0.0
        phrase_results[name] = {
            "gold": len(go), "pred": len(po),
            "jaccard": round(j, 4), "precision": round(pr, 4),
            "recall": round(r, 4), "f1": round(f1, 4),
        }
    go_all = _parse_pairs_phrase_only(gold_df, "opening", "opening") | _parse_pairs_phrase_only(gold_df, "closing", "closing")
    po_all = _parse_pairs_phrase_only(pred_df, "opening", "opening") | _parse_pairs_phrase_only(pred_df, "closing", "closing")
    j = len(go_all & po_all) / len(go_all | po_all) if (go_all | po_all) else 1.0
    pr = len(go_all & po_all) / len(po_all) if po_all else 0.0
    r = len(go_all & po_all) / len(go_all) if go_all else 0.0
    f1 = 2 * pr * r / (pr + r) if pr + r else 0.0
    phrase_results["all"] = {
        "gold": len(go_all), "pred": len(po_all),
        "jaccard": round(j, 4), "precision": round(pr, 4),
        "recall": round(r, 4), "f1": round(f1, 4),
    }
    diag["phrase_only"] = phrase_results

    # Speaker mismatch: phrase near-matches between gold and pred with wrong speaker
    gold_full = _parse_pairs(gold_df, "opening", "opening") | _parse_pairs(gold_df, "closing", "closing")
    pred_full = _parse_pairs(pred_df, "opening", "opening") | _parse_pairs(pred_df, "closing", "closing")
    speaker_mismatches = 0
    for g_s, g_p, g_c in gold_full:
        for p_s, p_p, p_c in pred_full:
            if g_c == p_c and g_s != p_s and SequenceMatcher(None, g_p, p_p).ratio() >= 0.8:
                speaker_mismatches += 1
                break
    diag["speaker_mismatch_count"] = speaker_mismatches

    return diag


def _evaluate(gold_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict[str, dict]:
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
            "gold": len(g), "pred": len(p),
            "jaccard": round(j, 4), "precision": round(pr, 4),
            "recall": round(r, 4), "f1": round(f1, 4),
            "matched": matched, "exact": exact, "avg_sim": round(avg_sim, 4),
        }
    return results


def _run_pipeline(
    model_path: Path,
    threshold: float,
    gold_excel: Path,
    transcript_dir: Path,
    samples_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs_path = output_dir / "extracted_pairs.xlsx"
    gold_path  = output_dir / "gold.xlsx"
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
        "--ml-confidence-threshold", str(round(threshold, 4)),
        "--similarity-threshold", "0.48",
        "--skip-asr", "--skip-diarization",
        "--fit-input", "data/processed/gold_dialogues.jsonl",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] pipeline exited {result.returncode}")
        print(result.stderr[-300:] if result.stderr else "")

    return pd.read_excel(pairs_path), pd.read_excel(gold_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Сравнение ML-классификаторов")
    parser.add_argument("--fit-input", required=True)
    parser.add_argument("--gold-excel", required=True)
    parser.add_argument("--transcript-dir", required=True)
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument(
        "--output", default="artifacts/classifier_comparison.json",
    )
    parser.add_argument(
        "--classifiers", nargs="+", default=CLASSIFIER_CHOICES,
        choices=CLASSIFIER_CHOICES,
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Число fold для кросс-валидации",
    )
    parser.add_argument(
        "--fixed-threshold", type=float, default=None,
        help="Если указан — использовать фиксированный порог вместо sweep",
    )
    parser.add_argument(
        "--diagnostics-output", type=str, default=None,
        help=(
            "Если указан — сохранить расширенные диагностики (phrase-only метрики, "
            "speaker mismatch count) в отдельный JSON файл."
        ),
    )
    args = parser.parse_args(argv)

    records = load_jsonl(args.fit_input)
    print(f"Загружено {len(records)} обучающих записей\n")

    comparison: list[dict] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for clf_type in args.classifiers:
            print(f"{'=' * 60}")
            print(f"  Классификатор: {clf_type.upper()}")
            print(f"{'=' * 60}")

            # --- Кросс-валидация ---
            print(f"  Кросс-валидация ({args.cv_folds}-fold)...")
            cv_rep = _cv_report(records, clf_type, n_splits=args.cv_folds)
            for cls in ["contact_open", "contact_close", "none"]:
                row = cv_rep.get(cls, {})
                print(
                    f"    {cls:20s}  P={row.get('precision', 0):.3f}"
                    f"  R={row.get('recall', 0):.3f}"
                    f"  F1={row.get('f1-score', 0):.3f}"
                )

            # --- Обучение на полных данных ---
            print("  Обучение на полных данных...")
            model = IntentClassifier(classifier_type=clf_type)
            model.fit(records)
            model_path = tmp / f"model_{clf_type}.joblib"
            save_model(model, model_path)

            # --- Подбор порога ---
            if args.fixed_threshold is not None:
                threshold = args.fixed_threshold
                print(f"  Порог (фиксированный): {threshold:.4f}")
            else:
                t_open  = _find_best_threshold(records, model, "contact_open",  beta=1.0)
                t_close = _find_best_threshold(records, model, "contact_close", beta=1.0)
                threshold = min(t_open, t_close)
                print(f"  Порог (sweep): open={t_open:.4f}  close={t_close:.4f}  used={threshold:.4f}")

            # --- Validation pipeline ---
            print(f"  Validation pipeline (threshold={threshold:.4f})...")
            out_dir = tmp / f"val_{clf_type}"
            pred_df, gold_df = _run_pipeline(
                model_path=model_path,
                threshold=threshold,
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

            diag = None
            if args.diagnostics_output:
                diag = _compute_diagnostics(gold_df, pred_df)
                po = diag["phrase_only"]["all"]
                print(
                    f"  phrase-only (no speaker): pred={po['pred']} "
                    f"P={po['precision']:.4f} R={po['recall']:.4f} F1={po['f1']:.4f} "
                    f"speaker_mismatches={diag['speaker_mismatch_count']}"
                )
            print()

            entry: dict = {
                "classifier": clf_type,
                "threshold_used": round(threshold, 4),
                "cv_report": cv_rep,
                "validation_metrics": metrics,
            }
            if diag is not None:
                entry["diagnostics"] = diag
            comparison.append(entry)

    # --- Сводная таблица ---
    print("\n" + "=" * 100)
    print(f"{'Clf':<6} {'Thr':>5}  "
          f"CV open F1  CV close F1  "
          f"{'Pred':>5} {'J':>7} {'P':>7} {'R':>7} {'F1':>7} {'matched':>7} {'exact':>5} {'avg_sim':>8}")
    print("-" * 100)
    for entry in comparison:
        clf = entry["classifier"]
        thr = entry["threshold_used"]
        cv  = entry["cv_report"]
        m   = entry["validation_metrics"]["all"]
        cv_open  = cv.get("contact_open",  {}).get("f1-score", 0)
        cv_close = cv.get("contact_close", {}).get("f1-score", 0)
        print(
            f"{clf:<6} {thr:>5.3f}  "
            f"{cv_open:>10.3f}  {cv_close:>11.3f}  "
            f"{m['pred']:>5} {m['jaccard']:>7.4f} {m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['matched']:>7} "
            f"{m['exact']:>5} {m['avg_sim']:>8.4f}"
        )

    # --- Сохранение ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {out_path}")

    if args.diagnostics_output:
        diag_all = [
            {"classifier": e["classifier"], "diagnostics": e.get("diagnostics", {})}
            for e in comparison
        ]
        diag_path = Path(args.diagnostics_output)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with diag_path.open("w", encoding="utf-8") as f:
            json.dump(diag_all, f, ensure_ascii=False, indent=2)
        print(f"Диагностики сохранены: {diag_path}")


if __name__ == "__main__":
    main()
