from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


TAG_PATTERN = re.compile(r"<(opening|closing)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)


def normalize_multiline_text(value: Any) -> str:
    """
    Приводит текст из Excel-ячейки к единому виду:
    - заменяет разные варианты перевода строки на '\n'
    - убирает внешние пробелы
    """
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def split_dialogue_into_lines(marked_text: str) -> list[str]:
    """
    Делит диалог на отдельные непустые строки-реплики.
    """
    text = normalize_multiline_text(marked_text)
    return [line.strip() for line in text.split("\n") if line.strip()]


def parse_annotated_text(text: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Удаляет теги <opening> / <closing> из текста реплики
    и возвращает:
    - clean_text: текст без тегов
    - annotations: список аннотаций со span-границами в clean_text

    Пример:
    Вход:
        "<opening>Здравия желаю</opening>, товарищ подполковник."

    Выход:
        clean_text = "Здравия желаю, товарищ подполковник."
        annotations = [
            {
                "expression_text": "Здравия желаю",
                "intent_type": "contact_open",
                "char_start": 0,
                "char_end": 14
            }
        ]
    """
    annotations: list[dict[str, Any]] = []
    clean_parts: list[str] = []

    source_cursor = 0
    clean_cursor = 0

    for match in TAG_PATTERN.finditer(text):
        match_start, match_end = match.span()
        tag_type = match.group(1).lower()
        span_text = match.group(2)

        # Добавляем обычный текст перед тегом
        prefix = text[source_cursor:match_start]
        clean_parts.append(prefix)
        clean_cursor += len(prefix)

        # Добавляем содержимое тега уже без самих тегов
        clean_parts.append(span_text)
        ann_start = clean_cursor
        ann_end = clean_cursor + len(span_text)

        annotations.append(
            {
                "expression_text": span_text,
                "intent_type": "contact_open" if tag_type == "opening" else "contact_close",
                "char_start": ann_start,
                "char_end": ann_end,
            }
        )

        clean_cursor = ann_end
        source_cursor = match_end

    # Добавляем хвост текста после последнего тега
    suffix = text[source_cursor:]
    clean_parts.append(suffix)

    clean_text = "".join(clean_parts)
    return clean_text, annotations


def parse_utterance_line(line: str) -> dict[str, Any] | None:
    """
    Парсит одну строку вида:
        "Имя персонажа: текст реплики"

    Возвращает словарь:
        {
            "speaker_name": ...,
            "text": ...,
            "annotations": ...
        }

    Если строка не содержит ':', возвращает None.
    """
    if ":" not in line:
        return None

    speaker_name, replica = line.split(":", 1)
    speaker_name = speaker_name.strip()
    replica = replica.strip()

    if not speaker_name or not replica:
        return None

    clean_text, annotations = parse_annotated_text(replica)

    return {
        "speaker_name": speaker_name,
        "text": clean_text,
        "annotations": annotations,
    }


def build_gold_records(
    df: pd.DataFrame,
    id_col: str,
    text_col: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Преобразует DataFrame с колонками:
    - id_col
    - text_col

    в список канонических gold-records формата JSONL.

    Возвращает:
    - records
    - skipped_lines_info
    """
    records: list[dict[str, Any]] = []
    skipped_lines_info: list[dict[str, Any]] = []

    for row_idx, row in df.iterrows():
        raw_dialogue_id = row.get(id_col)
        dialogue_id = str(raw_dialogue_id).strip() if not pd.isna(raw_dialogue_id) else f"dlg_{row_idx:04d}"

        marked_text = normalize_multiline_text(row.get(text_col, ""))
        if not marked_text:
            continue

        lines = split_dialogue_into_lines(marked_text)

        utterance_counter = 0
        for line_number, line in enumerate(lines, start=1):
            parsed = parse_utterance_line(line)
            if parsed is None:
                skipped_lines_info.append(
                    {
                        "dialogue_id": dialogue_id,
                        "line_number": line_number,
                        "line_text": line,
                        "reason": "line_has_no_valid_speaker_separator",
                    }
                )
                continue

            utterance_counter += 1
            utterance_id = f"{dialogue_id}_utt_{utterance_counter:03d}"

            record = {
                "dialogue_id": dialogue_id,
                "utterance_id": utterance_id,
                "speaker_name": parsed["speaker_name"],
                "text": parsed["text"],
                "annotations": parsed["annotations"],
            }
            records.append(record)

    return records, skipped_lines_info


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """
    Сохраняет список словарей в JSONL.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_stats(records: list[dict[str, Any]], skipped_lines_info: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Собирает базовую статистику по подготовленной gold-разметке.
    """
    dialogue_ids = {record["dialogue_id"] for record in records}
    utterances_count = len(records)

    annotations_count = 0
    annotated_utterances_count = 0
    intent_counter: Counter[str] = Counter()
    expressions_counter: Counter[str] = Counter()

    for record in records:
        anns = record.get("annotations", [])
        if anns:
            annotated_utterances_count += 1

        for ann in anns:
            annotations_count += 1
            intent_counter[ann["intent_type"]] += 1
            expressions_counter[ann["expression_text"].strip().lower()] += 1

    top_expressions = [
        {"expression_text": expr, "count": count}
        for expr, count in expressions_counter.most_common(20)
    ]

    stats = {
        "dialogues_count": len(dialogue_ids),
        "utterances_count": utterances_count,
        "annotated_utterances_count": annotated_utterances_count,
        "annotations_count": annotations_count,
        "intent_distribution": dict(intent_counter),
        "top_expressions": top_expressions,
        "skipped_lines_count": len(skipped_lines_info),
    }
    return stats


def print_stats(stats: dict[str, Any]) -> None:
    """
    Печатает статистику в консоль.
    """
    print("\n=== GOLD PREPROCESSING STATS ===")
    print(f"Диалогов: {stats['dialogues_count']}")
    print(f"Реплик: {stats['utterances_count']}")
    print(f"Реплик с разметкой: {stats['annotated_utterances_count']}")
    print(f"Всего размеченных выражений: {stats['annotations_count']}")
    print(f"Пропущенных строк: {stats['skipped_lines_count']}")

    print("\nРаспределение по типам:")
    for intent_type, count in stats["intent_distribution"].items():
        print(f"  {intent_type}: {count}")

    print("\nТоп-20 выражений:")
    for item in stats["top_expressions"]:
        print(f"  {item['expression_text']}: {item['count']}")


def save_stats(stats: dict[str, Any], path: Path) -> None:
    """
    Сохраняет статистику в JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def save_skipped_lines(skipped_lines_info: list[dict[str, Any]], path: Path) -> None:
    """
    Сохраняет информацию о строках, которые не удалось распарсить.
    Это полезно для отладки формата Excel-файла.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(skipped_lines_info, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подготовка золотой разметки из Excel-файла с тегами <opening>/<closing>."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/gold/gold_dialogues.xlsx",
        help="Путь к Excel-файлу с gold-разметкой.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/gold_dialogues.jsonl",
        help="Путь к итоговому JSONL-файлу.",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default="data/processed/gold_stats.json",
        help="Путь к JSON-файлу со статистикой.",
    )
    parser.add_argument(
        "--skipped-output",
        type=str,
        default="data/processed/gold_skipped_lines.json",
        help="Путь к JSON-файлу с пропущенными строками.",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        default="0",
        help="Имя листа Excel или индекс листа. По умолчанию: 0.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="id",
        help="Название колонки с ID диалога.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="marked_text",
        help="Название колонки с размеченным текстом.",
    )
    return parser.parse_args()


def resolve_sheet_name(sheet_name_arg: str) -> int | str:
    """
    Если пользователь передал '0', '1' и т.д. — превращаем в int.
    Иначе трактуем как строковое имя листа.
    """
    return int(sheet_name_arg) if sheet_name_arg.isdigit() else sheet_name_arg


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_output_path = Path(args.stats_output)
    skipped_output_path = Path(args.skipped_output)
    sheet_name = resolve_sheet_name(args.sheet_name)

    if not input_path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {input_path}")

    df = pd.read_excel(input_path, sheet_name=sheet_name)

    missing_cols = [col for col in [args.id_col, args.text_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"В Excel-файле отсутствуют колонки: {missing_cols}. "
            f"Найденные колонки: {list(df.columns)}"
        )

    records, skipped_lines_info = build_gold_records(
        df=df,
        id_col=args.id_col,
        text_col=args.text_col,
    )

    save_jsonl(records, output_path)

    stats = build_stats(records, skipped_lines_info)
    save_stats(stats, stats_output_path)
    save_skipped_lines(skipped_lines_info, skipped_output_path)
    print_stats(stats)

    print(f"\nГотово. JSONL сохранён в: {output_path}")
    print(f"Статистика сохранена в: {stats_output_path}")
    print(f"Пропущенные строки сохранены в: {skipped_output_path}")


if __name__ == "__main__":
    main()