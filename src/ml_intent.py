from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


LABELS = ["none", "contact_open", "contact_close"]


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def get_label(record: dict[str, Any]) -> str:
    """
    Extracts a single classification label from a gold record.

    Priority: contact_open > contact_close > none.
    If annotations contain both types, contact_open wins.
    """
    annotations = record.get("annotations") or []
    intent_types = {str(ann["intent_type"]) for ann in annotations if ann.get("intent_type")}
    if "contact_open" in intent_types:
        return "contact_open"
    if "contact_close" in intent_types:
        return "contact_close"
    return "none"


# ---------------------------------------------------------------------------
# Dialogue position features
# ---------------------------------------------------------------------------

def compute_dialogue_positions(records: list[dict[str, Any]]) -> dict[str, float]:
    """
    Returns {utterance_id: relative_position} where relative_position in [0, 1].

    Groups by dialogue_id. Position 0.0 = first utterance, 1.0 = last.
    Single-utterance dialogues get 0.0.
    """
    by_dialogue: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        did = str(rec.get("dialogue_id", ""))
        by_dialogue.setdefault(did, []).append(rec)

    positions: dict[str, float] = {}
    for group in by_dialogue.values():
        n = len(group)
        for i, rec in enumerate(group):
            uid = str(rec.get("utterance_id", ""))
            positions[uid] = float(i) / max(n - 1, 1)
    return positions


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def _records_to_texts(records: list[dict[str, Any]]) -> list[str]:
    return [str(rec.get("text") or "") for rec in records]


def _records_to_numerical(
    records: list[dict[str, Any]],
    positions: dict[str, float],
) -> np.ndarray:
    rows = []
    for rec in records:
        uid = str(rec.get("utterance_id", ""))
        text = str(rec.get("text") or "")
        rel = positions.get(uid, 0.5)
        token_count = len(text.split())
        rows.append([
            rel,
            token_count,
            float(rel <= 0.30),
            float(rel >= 0.70),
        ])
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    TF-IDF char n-gram + LogisticRegression classifier.

    Stores:
      - tfidf: fitted TfidfVectorizer
      - clf: fitted LogisticRegression
      - label_encoder: maps int index -> label string
    """

    def __init__(self) -> None:
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=50_000,
            sublinear_tf=True,
        )
        self.clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            multi_class="multinomial",
            solver="lbfgs",
        )
        self.classes_: list[str] = []

    def fit(self, records: list[dict[str, Any]]) -> "IntentClassifier":
        labels = [get_label(r) for r in records]
        positions = compute_dialogue_positions(records)
        texts = _records_to_texts(records)
        X_tfidf = self.tfidf.fit_transform(texts)
        X_num = _records_to_numerical(records, positions)
        X = hstack([X_tfidf, csr_matrix(X_num)])
        self.clf.fit(X, labels)
        self.classes_ = list(self.clf.classes_)
        return self

    def predict_proba(
        self,
        records: list[dict[str, Any]],
        positions: dict[str, float] | None = None,
    ) -> tuple[list[str], np.ndarray]:
        if positions is None:
            positions = compute_dialogue_positions(records)
        texts = _records_to_texts(records)
        X_tfidf = self.tfidf.transform(texts)
        X_num = _records_to_numerical(records, positions)
        X = hstack([X_tfidf, csr_matrix(X_num)])
        proba = self.clf.predict_proba(X)
        pred_indices = np.argmax(proba, axis=1)
        pred_labels = [self.classes_[i] for i in pred_indices]
        return pred_labels, proba


# ---------------------------------------------------------------------------
# Pipeline-compatible prediction
# ---------------------------------------------------------------------------

def predict_for_records(
    records: list[dict[str, Any]],
    model: IntentClassifier,
    positions: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Runs the ML classifier on a list of utterance records and returns
    predictions in the same format as rule_based_intent.predict_for_records.

    Only records classified as contact_open or contact_close are included.
    Each prediction dict contains the fields expected by pair_formatter:
      intent_type, speaker_name, expression, start_time, char_start, char_end,
      utterance_id, dialogue_id, confidence.
    """
    if not records:
        return []

    if positions is None:
        positions = compute_dialogue_positions(records)

    pred_labels, proba = model.predict_proba(records, positions)

    predictions: list[dict[str, Any]] = []
    for rec, label, prob_row in zip(records, pred_labels, proba):
        if label == "none":
            continue

        label_idx = model.classes_.index(label)
        confidence = float(prob_row[label_idx])
        text = str(rec.get("text") or "")

        predictions.append({
            "dialogue_id": rec.get("dialogue_id", ""),
            "utterance_id": rec.get("utterance_id", ""),
            "speaker_name": rec.get("speaker_name", "unknown"),
            "source_text": text,
            "expression": text,
            "intent_type": label,
            "char_start": 0,
            "char_end": len(text),
            "start_time": rec.get("start_time"),
            "end_time": rec.get("end_time"),
            "confidence": confidence,
            "rule_expression": "ml_classifier",
            "rule_frequency": 0,
        })

    return predictions


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_model(model: IntentClassifier, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path) -> IntentClassifier:
    return joblib.load(str(path))


# ---------------------------------------------------------------------------
# Training statistics
# ---------------------------------------------------------------------------

def compute_train_stats(
    records: list[dict[str, Any]],
    model: IntentClassifier,
) -> dict[str, Any]:
    from collections import Counter
    from sklearn.metrics import classification_report

    labels = [get_label(r) for r in records]
    label_counts = dict(Counter(labels))

    positions = compute_dialogue_positions(records)
    pred_labels, _ = model.predict_proba(records, positions)

    report = classification_report(
        labels,
        pred_labels,
        labels=["contact_open", "contact_close", "none"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "num_records": len(records),
        "label_counts": label_counts,
        "train_classification_report": report,
    }


# ---------------------------------------------------------------------------
# Utility: load JSONL
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
