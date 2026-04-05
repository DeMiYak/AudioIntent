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

_TRIM_CHARS = " \t\n,;:-—–"
_TRIM_RIGHT_CHARS = _TRIM_CHARS + ".!?"


def _extract_representative_phrase(text: str, intent_type: str) -> str:
    """
    Returns a short representative phrase from the utterance text.

    Strategy (in priority order):
    1. If a MANUAL_PATTERN matches anywhere in the text, return that matched span
       (same short phrase the rule-based system would produce).
    2. If an OPEN_STRONG / CLOSE_STRONG marker appears, return the matched token.
    3. Fall back to first sentence (opening) or last sentence (closing).

    This ensures ML-only predictions produce short phrases comparable to gold,
    rather than outputting the full utterance text.
    """
    from .rule_based_intent import (
        CLOSE_STRONG,
        MANUAL_PATTERNS,
        OPEN_STRONG,
        TOKEN_PATTERN,
        normalize_for_matching,
        split_into_sentence_spans,
    )

    # 1. Сначала проверить MANUAL_PATTERNS (наиболее специфичные)
    for pattern_info in MANUAL_PATTERNS:
        if pattern_info["intent_type"] != intent_type:
            continue
        m = pattern_info["pattern"].search(text)
        if m:
            raw = text[m.start():m.end()].strip(_TRIM_CHARS).rstrip(_TRIM_RIGHT_CHARS).strip()
            if raw:
                return raw

    # 2. Проверить токены сильных маркеров
    strong_set = OPEN_STRONG if intent_type == "contact_open" else CLOSE_STRONG
    normalized = normalize_for_matching(text)
    for token in TOKEN_PATTERN.findall(normalized):
        if token in strong_set:
            return token

    # 3. Откат к первому / последнему предложению
    spans = split_into_sentence_spans(text)
    if not spans:
        raw = text
    elif intent_type == "contact_open":
        start, end = spans[0]
        raw = text[start:end]
    else:
        start, end = spans[-1]
        raw = text[start:end]

    raw = raw.strip(_TRIM_CHARS).rstrip(_TRIM_RIGHT_CHARS).strip()
    return raw or text.strip()


# ---------------------------------------------------------------------------
# Извлечение меток
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
# Признаки позиции в диалоге
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


def compute_neighbour_texts(records: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """
    Returns {utterance_id: (prev_text, next_text)} within the same dialogue.
    Boundary utterances get empty strings for the missing neighbour.
    """
    by_dialogue: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        did = str(rec.get("dialogue_id", ""))
        by_dialogue.setdefault(did, []).append(rec)

    neighbours: dict[str, tuple[str, str]] = {}
    for group in by_dialogue.values():
        texts = [str(r.get("text") or "") for r in group]
        for i, rec in enumerate(group):
            uid = str(rec.get("utterance_id", ""))
            prev_text = texts[i - 1] if i > 0 else ""
            next_text = texts[i + 1] if i < len(texts) - 1 else ""
            neighbours[uid] = (prev_text, next_text)
    return neighbours


# ---------------------------------------------------------------------------
# Построение признаков
# ---------------------------------------------------------------------------

# Лексика приветствий и прощаний для быстрой проверки контекста соседних реплик
_OPEN_TOKENS = frozenset([
    "привет", "здравствуй", "здравствуйте", "добрый", "алло", "хай",
    "hello", "hi", "hey", "приветствую", "добрейший",
])
_CLOSE_TOKENS = frozenset([
    "пока", "до", "свидания", "прощай", "прощайте", "всего", "удачи",
    "чао", "bye", "goodbye", "бывай", "бывайте",
])


def _context_score(text: str) -> tuple[float, float]:
    """Возвращает (open_score, close_score) для текста соседней реплики."""
    tokens = set(text.lower().split())
    open_score = float(bool(tokens & _OPEN_TOKENS))
    close_score = float(bool(tokens & _CLOSE_TOKENS))
    return open_score, close_score


def _records_to_texts(records: list[dict[str, Any]]) -> list[str]:
    return [str(rec.get("text") or "") for rec in records]


def _records_to_numerical(
    records: list[dict[str, Any]],
    positions: dict[str, float],
    neighbours: dict[str, tuple[str, str]] | None = None,
) -> np.ndarray:
    rows = []
    for rec in records:
        uid = str(rec.get("utterance_id", ""))
        text = str(rec.get("text") or "")
        rel = positions.get(uid, 0.5)
        token_count = len(text.split())

        # Признаки контекста: наличие маркеров открытия/закрытия у соседних реплик
        prev_text, next_text = neighbours.get(uid, ("", "")) if neighbours else ("", "")
        prev_open, prev_close = _context_score(prev_text)
        next_open, next_close = _context_score(next_text)
        prev_is_empty = float(not prev_text)
        next_is_empty = float(not next_text)

        rows.append([
            rel,
            token_count,
            float(rel <= 0.30),
            float(rel >= 0.70),
            prev_open,
            prev_close,
            next_open,
            next_close,
            prev_is_empty,   # первая реплика в диалоге — сильный признак открытия
            next_is_empty,   # последняя реплика в диалоге — сильный признак закрытия
        ])
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Контейнер модели
# ---------------------------------------------------------------------------

CLASSIFIER_CHOICES = ["lr", "svm", "nb", "sgd", "ridge"]


def _build_classifier(classifier_type: str):
    """Создаёт sklearn-классификатор по строковому ключу."""
    from sklearn.naive_bayes import ComplementNB
    from sklearn.linear_model import SGDClassifier, RidgeClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    if classifier_type == "lr":
        return LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs",
        )
    if classifier_type == "svm":
        # LinearSVC не поддерживает predict_proba — оборачиваем в CalibratedClassifierCV
        return CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=5000, class_weight="balanced"), cv=3,
        )
    if classifier_type == "nb":
        # ComplementNB лучше работает с несбалансированными классами чем MultinomialNB
        return ComplementNB(alpha=0.1)
    if classifier_type == "sgd":
        return SGDClassifier(
            loss="modified_huber", max_iter=1000, class_weight="balanced",
            random_state=42,
        )
    if classifier_type == "ridge":
        # RidgeClassifier не поддерживает predict_proba — оборачиваем
        return CalibratedClassifierCV(RidgeClassifier(class_weight="balanced"), cv=3)
    raise ValueError(f"Неизвестный тип классификатора: {classifier_type!r}. "
                     f"Допустимые: {CLASSIFIER_CHOICES}")


class IntentClassifier:
    """
    TF-IDF char n-gram + sklearn-классификатор.

    Поддерживаемые типы (параметр classifier_type):
      'lr'    — LogisticRegression (по умолчанию)
      'svm'   — LinearSVC + CalibratedClassifierCV
      'nb'    — ComplementNB
      'sgd'   — SGDClassifier (modified_huber loss)
      'ridge' — RidgeClassifier + CalibratedClassifierCV
    """

    def __init__(self, classifier_type: str = "lr") -> None:
        self.classifier_type = classifier_type
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=50_000,
            sublinear_tf=True,
        )
        self.clf = _build_classifier(classifier_type)
        self.classes_: list[str] = []

    def fit(self, records: list[dict[str, Any]]) -> "IntentClassifier":
        labels = [get_label(r) for r in records]
        positions = compute_dialogue_positions(records)
        neighbours = compute_neighbour_texts(records)
        texts = _records_to_texts(records)
        X_tfidf = self.tfidf.fit_transform(texts)
        X_num = _records_to_numerical(records, positions, neighbours)
        X = hstack([X_tfidf, csr_matrix(X_num)])
        self.clf.fit(X, labels)
        self.classes_ = list(self.clf.classes_)
        return self

    def predict_proba(
        self,
        records: list[dict[str, Any]],
        positions: dict[str, float] | None = None,
        neighbours: dict[str, tuple[str, str]] | None = None,
    ) -> tuple[list[str], np.ndarray]:
        if positions is None:
            positions = compute_dialogue_positions(records)
        if neighbours is None:
            neighbours = compute_neighbour_texts(records)
        texts = _records_to_texts(records)
        X_tfidf = self.tfidf.transform(texts)
        X_num = _records_to_numerical(records, positions, neighbours)
        X = hstack([X_tfidf, csr_matrix(X_num)])
        proba = self.clf.predict_proba(X)
        pred_indices = np.argmax(proba, axis=1)
        pred_labels = [self.classes_[i] for i in pred_indices]
        return pred_labels, proba


# ---------------------------------------------------------------------------
# Предсказание, совместимое с пайплайном
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
    neighbours = compute_neighbour_texts(records)

    pred_labels, proba = model.predict_proba(records, positions, neighbours)

    predictions: list[dict[str, Any]] = []
    for rec, label, prob_row in zip(records, pred_labels, proba):
        if label == "none":
            continue

        label_idx = model.classes_.index(label)
        confidence = float(prob_row[label_idx])
        text = str(rec.get("text") or "")
        expression = _extract_representative_phrase(text, label)

        predictions.append({
            "dialogue_id": rec.get("dialogue_id", ""),
            "utterance_id": rec.get("utterance_id", ""),
            "speaker_name": rec.get("speaker_name", "unknown"),
            "source_text": text,
            "expression": expression,
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
# Сериализация
# ---------------------------------------------------------------------------

def save_model(model: IntentClassifier, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path) -> IntentClassifier:
    return joblib.load(str(path))


# ---------------------------------------------------------------------------
# Статистика обучения
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
    neighbours = compute_neighbour_texts(records)
    pred_labels, _ = model.predict_proba(records, positions, neighbours)

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
# Утилита: загрузка JSONL
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
