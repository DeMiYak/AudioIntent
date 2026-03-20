from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Загружает JSONL-файл в список словарей.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """
    Сохраняет список словарей в JSONL.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(data: dict[str, Any] | list[dict[str, Any]], path: str | Path) -> None:
    """
    Сохраняет объект в JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_for_matching(text: str) -> str:
    """
    Нормализация текста для сопоставления:
    - lowercase
    - ё -> е

    Важно: длина строки не меняется, поэтому char-индексы остаются валидными.
    """
    return text.lower().replace("ё", "е")


def tokenize_text(text: str) -> list[str]:
    """
    Простая токенизация:
    - слова
    - знаки препинания отдельно

    Args:
        text (str): Входной текст для токенизации.

    Returns:
        list[str]: Список токенов, может быть пустым для пустого входа.

    Edge cases:
        - Возвращает пустой список для пустого входа.
        - Корректно обрабатывает Unicode-символы.
    """
    return TOKEN_PATTERN.findall(text)


def char_to_token_spans(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """
    Для каждого токена вычисляет span (char_start, char_end) в исходном тексте.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0

    for token in tokens:
        start = text.find(token, cursor)
        if start == -1:
            raise ValueError(f"Не удалось найти токен '{token}' в тексте: {text}")
        end = start + len(token)
        spans.append((start, end))
        cursor = end

    return spans


def char_span_to_token_span(
    text: str,
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    """
    Переводит символьный span в span по индексам токенов.
    """
    tokens = tokenize_text(text)
    token_spans = char_to_token_spans(text, tokens)

    covered_tokens: list[int] = []

    for token_idx, (tok_start, tok_end) in enumerate(token_spans):
        overlap = not (tok_end <= char_start or tok_start >= char_end)
        if overlap:
            covered_tokens.append(token_idx)

    if not covered_tokens:
        return -1, -1

    return covered_tokens[0], covered_tokens[-1]


def build_expression_regex(expression_text: str) -> re.Pattern[str]:
    """
    Строит regex для поиска выражения в тексте.

    Особенности:
    - матчинг идёт по нормализованному тексту
    - пробелы в выражении допускают вариативность через \\s+
    - по краям добавляются word-boundary-подобные ограничения, если это нужно
    """
    expr = normalize_for_matching(expression_text).strip()
    if not expr:
        raise ValueError("Пустое выражение нельзя превратить в regex.")

    parts = [re.escape(part) for part in re.split(r"\s+", expr) if part]
    body = r"\s+".join(parts)

    non_space_chars = [ch for ch in expr if not ch.isspace()]
    first_char = non_space_chars[0] if non_space_chars else ""
    last_char = non_space_chars[-1] if non_space_chars else ""

    if first_char and (first_char.isalnum() or first_char == "_"):
        body = rf"(?<!\w){body}"

    if last_char and (last_char.isalnum() or last_char == "_"):
        body = rf"{body}(?!\w)"

    return re.compile(body, re.UNICODE)


def extract_lexicon_from_gold(
    gold_records: list[dict[str, Any]],
    min_freq: int = 1,
) -> dict[str, Any]:
    """
    Из gold-разметки строит словарь выражений для rule-based baseline.

    Итоговая структура:
    {
      "contact_open": [
         {"expression_text": "...", "frequency": 10},
         ...
      ],
      "contact_close": [
         {"expression_text": "...", "frequency": 4},
         ...
      ]
    }
    """
    counter_by_intent: dict[str, Counter[str]] = {
        "contact_open": Counter(),
        "contact_close": Counter(),
    }

    for record in gold_records:
        annotations = record.get("annotations", [])
        for ann in annotations:
            expression_text = str(ann["expression_text"]).strip()
            intent_type = ann["intent_type"]

            if not expression_text:
                continue
            if intent_type not in counter_by_intent:
                continue

            normalized_expr = normalize_for_matching(expression_text)
            counter_by_intent[intent_type][normalized_expr] += 1

    lexicon: dict[str, Any] = {
        "contact_open": [],
        "contact_close": [],
    }

    for intent_type, counter in counter_by_intent.items():
        items = [
            {"expression_text": expr, "frequency": freq}
            for expr, freq in counter.items()
            if freq >= min_freq
        ]
        items.sort(key=lambda x: (-len(x["expression_text"]), -x["frequency"], x["expression_text"]))
        lexicon[intent_type] = items

    return lexicon


def compile_lexicon(lexicon: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Компилирует словарь выражений в regex-паттерны для быстрого поиска.
    """
    compiled: dict[str, list[dict[str, Any]]] = {
        "contact_open": [],
        "contact_close": [],
    }

    for intent_type, items in lexicon.items():
        compiled_items: list[dict[str, Any]] = []

        for item in items:
            expr = item["expression_text"]
            freq = int(item["frequency"])
            pattern = build_expression_regex(expr)

            compiled_items.append(
                {
                    "expression_text": expr,
                    "frequency": freq,
                    "pattern": pattern,
                }
            )

        compiled[intent_type] = compiled_items

    return compiled


def collect_candidates_for_text(
    dialogue_id: str,
    utterance_id: str,
    speaker_name: str,
    text: str,
    compiled_lexicon: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Ищет все возможные rule-based совпадения в реплике.
    """
    normalized_text = normalize_for_matching(text)
    candidates: list[dict[str, Any]] = []

    for intent_type, items in compiled_lexicon.items():
        for item in items:
            expr = item["expression_text"]
            freq = item["frequency"]
            pattern = item["pattern"]

            for match in pattern.finditer(normalized_text):
                char_start, char_end = match.span()
                matched_text = text[char_start:char_end]
                token_start, token_end = char_span_to_token_span(text, char_start, char_end)

                candidates.append(
                    {
                        "dialogue_id": dialogue_id,
                        "utterance_id": utterance_id,
                        "speaker_name": speaker_name,
                        "source_text": text,
                        "expression": matched_text,
                        "intent_type": intent_type,
                        "char_start": char_start,
                        "char_end": char_end,
                        "token_start": token_start,
                        "token_end": token_end,
                        "confidence": 1.0,
                        "rule_expression": expr,
                        "rule_frequency": freq,
                    }
                )

    return candidates


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """
    Проверяет пересечение двух span'ов.
    """
    return not (a_end <= b_start or a_start >= b_end)


def resolve_overlaps(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Удаляет конфликтующие пересекающиеся совпадения.

    Приоритет:
    1. более длинный span
    2. более частое выражение в gold
    3. меньший char_start
    """
    if not candidates:
        return []

    sorted_candidates = sorted(
        candidates,
        key=lambda x: (
            -(x["char_end"] - x["char_start"]),
            -x["rule_frequency"],
            x["char_start"],
            x["intent_type"],
        ),
    )

    selected: list[dict[str, Any]] = []

    for cand in sorted_candidates:
        has_overlap = any(
            spans_overlap(
                cand["char_start"],
                cand["char_end"],
                chosen["char_start"],
                chosen["char_end"],
            )
            for chosen in selected
        )

        if not has_overlap:
            selected.append(cand)

    selected.sort(key=lambda x: (x["char_start"], x["char_end"]))
    return selected


def predict_for_record(
    record: dict[str, Any],
    compiled_lexicon: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Запускает rule-based baseline на одной реплике.

    Дополнительно протаскивает полезный контекст из record в итоговые предсказания:
    - speaker_label
    - start_time / end_time
    - source_window_id / source_window_row_id
    """
    dialogue_id = record.get("dialogue_id", "")
    utterance_id = record.get("utterance_id", "")
    speaker_name = record.get("speaker_name", "unknown")
    text = record.get("text", "")

    candidates = collect_candidates_for_text(
        dialogue_id=dialogue_id,
        utterance_id=utterance_id,
        speaker_name=speaker_name,
        text=text,
        compiled_lexicon=compiled_lexicon,
    )

    predictions = resolve_overlaps(candidates)

    passthrough_fields = [
        "speaker_label",
        "start_time",
        "end_time",
        "source_window_id",
        "source_window_row_id",
        "source_film",
        "source_start_sec",
        "source_end_sec",
    ]
    for pred in predictions:
        for field in passthrough_fields:
            if field in record:
                pred[field] = record[field]

    return predictions


def predict_for_records(
    records: list[dict[str, Any]],
    compiled_lexicon: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Запускает baseline на наборе реплик.
    """
    all_predictions: list[dict[str, Any]] = []

    for record in records:
        preds = predict_for_record(record, compiled_lexicon)
        all_predictions.extend(preds)

    return all_predictions


def build_gold_span_index(records: list[dict[str, Any]]) -> dict[str, set[tuple[int, int, str]]]:
    """
    Строит индекс gold-span'ов по utterance_id:
        utterance_id -> {(char_start, char_end, intent_type), ...}
    """
    gold_index: dict[str, set[tuple[int, int, str]]] = defaultdict(set)

    for record in records:
        utterance_id = record["utterance_id"]
        annotations = record.get("annotations", [])

        for ann in annotations:
            gold_index[utterance_id].add(
                (
                    int(ann["char_start"]),
                    int(ann["char_end"]),
                    ann["intent_type"],
                )
            )

    return gold_index


def build_pred_span_index(predictions: list[dict[str, Any]]) -> dict[str, set[tuple[int, int, str]]]:
    """
    Строит индекс предсказанных span'ов по utterance_id.
    """
    pred_index: dict[str, set[tuple[int, int, str]]] = defaultdict(set)

    for pred in predictions:
        utterance_id = pred["utterance_id"]
        pred_index[utterance_id].add(
            (
                int(pred["char_start"]),
                int(pred["char_end"]),
                pred["intent_type"],
            )
        )

    return pred_index


def compute_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """
    Считает precision / recall / f1.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_predictions(
    gold_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Оценивает baseline по exact-match на span + label.
    """
    gold_index = build_gold_span_index(gold_records)
    pred_index = build_pred_span_index(predictions)

    utterance_ids = sorted(set(gold_index.keys()) | set(pred_index.keys()))

    overall_tp = overall_fp = overall_fn = 0
    per_label_counts = {
        "contact_open": {"tp": 0, "fp": 0, "fn": 0},
        "contact_close": {"tp": 0, "fp": 0, "fn": 0},
    }

    for utterance_id in utterance_ids:
        gold_spans = gold_index.get(utterance_id, set())
        pred_spans = pred_index.get(utterance_id, set())

        tp_set = gold_spans & pred_spans
        fp_set = pred_spans - gold_spans
        fn_set = gold_spans - pred_spans

        overall_tp += len(tp_set)
        overall_fp += len(fp_set)
        overall_fn += len(fn_set)

        for _, _, label in tp_set:
            per_label_counts[label]["tp"] += 1
        for _, _, label in fp_set:
            per_label_counts[label]["fp"] += 1
        for _, _, label in fn_set:
            per_label_counts[label]["fn"] += 1

    metrics = {
        "overall": {
            "tp": overall_tp,
            "fp": overall_fp,
            "fn": overall_fn,
            **compute_prf(overall_tp, overall_fp, overall_fn),
        },
        "per_label": {},
    }

    for label, counts in per_label_counts.items():
        metrics["per_label"][label] = {
            **counts,
            **compute_prf(counts["tp"], counts["fp"], counts["fn"]),
        }

    return metrics


def print_lexicon_stats(lexicon: dict[str, Any]) -> None:
    """
    Печатает краткую статистику по словарю baseline.
    """
    print("\n=== RULE LEXICON STATS ===")
    for intent_type, items in lexicon.items():
        print(f"{intent_type}: {len(items)} выражений")

        top_items = items[:10]
        if top_items:
            print("  Топ-10:")
            for item in top_items:
                print(f"    {item['expression_text']} ({item['frequency']})")


def print_metrics(metrics: dict[str, Any]) -> None:
    """
    Печатает метрики baseline.
    """
    print("\n=== RULE-BASED BASELINE METRICS ===")
    overall = metrics["overall"]
    print(
        f"overall -> TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} "
        f"P={overall['precision']} R={overall['recall']} F1={overall['f1']}"
    )

    print("\nПо классам:")
    for label, values in metrics["per_label"].items():
        print(
            f"{label} -> TP={values['tp']} FP={values['fp']} FN={values['fn']} "
            f"P={values['precision']} R={values['recall']} F1={values['f1']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rule-based baseline для извлечения контактоустанавливающих и контактозавершающих выражений."
    )
    parser.add_argument(
        "--fit-input",
        type=str,
        default="data/processed/gold_dialogues.jsonl",
        help="JSONL-файл, из которого строится словарь baseline.",
    )
    parser.add_argument(
        "--predict-input",
        type=str,
        default="data/processed/gold_dialogues.jsonl",
        help="JSONL-файл, на котором выполняется rule-based inference.",
    )
    parser.add_argument(
        "--lexicon-output",
        type=str,
        default="data/processed/rule_lexicon.json",
        help="Куда сохранить словарь baseline.",
    )
    parser.add_argument(
        "--predictions-output",
        type=str,
        default="artifacts/rule_based_predictions.jsonl",
        help="Куда сохранить предсказания baseline.",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="artifacts/rule_based_metrics.json",
        help="Куда сохранить метрики baseline.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Минимальная частота выражения в gold для включения в словарь.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fit_input = Path(args.fit_input)
    predict_input = Path(args.predict_input)
    lexicon_output = Path(args.lexicon_output)
    predictions_output = Path(args.predictions_output)
    metrics_output = Path(args.metrics_output)

    if not fit_input.exists():
        raise FileNotFoundError(f"Не найден fit-input: {fit_input}")
    if not predict_input.exists():
        raise FileNotFoundError(f"Не найден predict-input: {predict_input}")

    fit_records = load_jsonl(fit_input)
    predict_records = load_jsonl(predict_input)

    lexicon = extract_lexicon_from_gold(fit_records, min_freq=args.min_freq)
    save_json(lexicon, lexicon_output)
    print_lexicon_stats(lexicon)

    compiled_lexicon = compile_lexicon(lexicon)
    predictions = predict_for_records(predict_records, compiled_lexicon)
    save_jsonl(predictions, predictions_output)

    metrics = evaluate_predictions(predict_records, predictions)
    save_json(metrics, metrics_output)
    print_metrics(metrics)

    print("\n=== OUTPUT PATHS ===")
    print(f"Словарь сохранён в: {lexicon_output}")
    print(f"Предсказания сохранены в: {predictions_output}")
    print(f"Метрики сохранены в: {metrics_output}")


if __name__ == "__main__":
    main()