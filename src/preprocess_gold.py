from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import time
from pathlib import Path
from typing import Any

import pandas as pd


TAG_PATTERN = re.compile(r"<(opening|closing)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)


def resolve_input_path(path_arg: str | Path) -> Path:
    """
    Пытается найти Excel-файл с gold-разметкой.

    Поддерживаем:
    - явный путь от пользователя;
    - ожидаемые имена `data_val.xlsx` или `Данные_.xlsx`;
    - zip-имя, в котором кириллица была превращена в #U0414...
    """
    path = Path(path_arg)
    if path.exists():
        return path

    gold_dir = path.parent if path.parent != Path(".") else Path("data/raw/gold")
    if gold_dir.exists():
        fallback_candidates = [
            gold_dir / "data_val.xlsx",
            gold_dir / "Данные_.xlsx",
            gold_dir / "gold_dialogues.xlsx",
        ]
        fallback_candidates.extend(sorted(gold_dir.glob("#U*.xlsx")))
        fallback_candidates.extend(
            p for p in sorted(gold_dir.glob("*.xlsx"))
            if "sample" not in p.name.lower()
        )

        for candidate in fallback_candidates:
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Не найден входной Excel-файл: {path_arg}")


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


def normalize_time_value(value: Any) -> str | None:
    if pd.isna(value):
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    text = str(value).strip()
    return text or None


def time_to_seconds(value: Any) -> float | None:
    """
    Превращает time / datetime / строку HH:MM:SS в секунды.
    """
    if pd.isna(value):
        return None

    if isinstance(value, time):
        return float(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000)

    if hasattr(value, "hour") and hasattr(value, "minute") and hasattr(value, "second"):
        return float(value.hour * 3600 + value.minute * 60 + value.second + getattr(value, "microsecond", 0) / 1_000_000)

    text = str(value).strip()
    if not text:
        return None

    parts = text.split(":")
    if len(parts) != 3:
        return None

    hours, minutes, seconds = parts
    return float(int(hours) * 3600 + int(minutes) * 60 + float(seconds))


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
    """
    annotations: list[dict[str, Any]] = []
    clean_parts: list[str] = []

    source_cursor = 0
    clean_cursor = 0

    for match in TAG_PATTERN.finditer(text):
        match_start, match_end = match.span()
        tag_type = match.group(1).lower()
        span_text = match.group(2)

        prefix = text[source_cursor:match_start]
        clean_parts.append(prefix)
        clean_cursor += len(prefix)

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

    suffix = text[source_cursor:]
    clean_parts.append(suffix)

    clean_text = "".join(clean_parts)
    return clean_text, annotations


def parse_utterance_line(line: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Парсит одну строку вида:
        "Имя персонажа: текст реплики"

    Возвращает:
    - словарь реплики, если строка корректна;
    - причину пропуска, если строку нельзя интерпретировать как реплику.

    Отдельно отлавливаем строки-вставки вида:
        "Незнакомец, позвонивший Вадиму:"
    которые обозначают спикера, но не содержат самой реплики.
    """
    if ":" not in line:
        return None, "line_has_no_valid_speaker_separator"

    speaker_name, replica = line.split(":", 1)
    speaker_name = speaker_name.strip()
    replica = replica.strip()

    if speaker_name and not replica:
        return None, "speaker_header_without_text"

    if not speaker_name or not replica:
        return None, "line_has_no_valid_speaker_separator"

    clean_text, annotations = parse_annotated_text(replica)

    return {
        "speaker_name": speaker_name,
        "text": clean_text,
        "annotations": annotations,
    }, None


def build_gold_records(
    df: pd.DataFrame,
    id_col: str,
    text_col: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Преобразует DataFrame листа "Диалоги" в список канонических gold-records.
    """
    records: list[dict[str, Any]] = []
    skipped_lines_info: list[dict[str, Any]] = []

    for row_idx, row in df.iterrows():
        raw_dialogue_id = row.get(id_col)
        if pd.isna(raw_dialogue_id):
            continue

        dialogue_id = str(raw_dialogue_id).strip()
        marked_text = normalize_multiline_text(row.get(text_col, ""))
        if not marked_text:
            continue

        source_type = normalize_multiline_text(row.get("Тип"))
        source_film = normalize_multiline_text(row.get("Фильм"))
        source_context = normalize_multiline_text(row.get("Реплики - полный контекст"))
        start_time_raw = row.get("Время начала")
        end_time_raw = row.get("Время окончания")
        start_sec = time_to_seconds(start_time_raw)
        end_sec = time_to_seconds(end_time_raw)

        if start_sec is not None and end_sec is not None and end_sec < start_sec:
            end_sec += 24 * 3600

        lines = split_dialogue_into_lines(marked_text)

        utterance_counter = 0
        for line_number, line in enumerate(lines, start=1):
            parsed, skip_reason = parse_utterance_line(line)
            if parsed is None:
                skipped_lines_info.append(
                    {
                        "dialogue_id": dialogue_id,
                        "line_number": line_number,
                        "line_text": line,
                        "reason": skip_reason,
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
                "source_film": source_film,
                "source_type": source_type,
                "source_context": source_context,
                "source_start_time": normalize_time_value(start_time_raw),
                "source_end_time": normalize_time_value(end_time_raw),
                "source_start_sec": start_sec,
                "source_end_sec": end_sec,
                "source_line_number": line_number,
                "source_sheet": "Диалоги",
            }
            records.append(record)

    return records, skipped_lines_info


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_stats(records: list[dict[str, Any]], skipped_lines_info: list[dict[str, Any]]) -> dict[str, Any]:
    dialogue_ids = {record["dialogue_id"] for record in records}
    utterances_count = len(records)

    annotations_count = 0
    annotated_utterances_count = 0
    intent_counter: Counter[str] = Counter()
    expressions_counter: Counter[str] = Counter()
    films_counter: Counter[str] = Counter()
    skip_reasons_counter: Counter[str] = Counter()

    for record in records:
        films_counter[str(record.get("source_film", ""))] += 1

        anns = record.get("annotations", [])
        if anns:
            annotated_utterances_count += 1

        for ann in anns:
            annotations_count += 1
            intent_counter[ann["intent_type"]] += 1
            expressions_counter[ann["expression_text"].strip().lower()] += 1

    for item in skipped_lines_info:
        skip_reasons_counter[str(item["reason"])] += 1

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
        "films_distribution": dict(films_counter),
        "skipped_lines_count": len(skipped_lines_info),
        "skip_reasons": dict(skip_reasons_counter),
    }
    return stats


def print_stats(stats: dict[str, Any]) -> None:
    print("\n=== GOLD PREPROCESSING STATS ===")
    print(f"Диалогов: {stats['dialogues_count']}")
    print(f"Реплик: {stats['utterances_count']}")
    print(f"Реплик с разметкой: {stats['annotated_utterances_count']}")
    print(f"Всего размеченных выражений: {stats['annotations_count']}")
    print(f"Пропущенных строк: {stats['skipped_lines_count']}")

    print("\nРаспределение по типам:")
    for intent_type, count in stats["intent_distribution"].items():
        print(f"  {intent_type}: {count}")

    print("\nПричины пропусков:")
    for reason, count in stats["skip_reasons"].items():
        print(f"  {reason}: {count}")

    print("\nТоп-20 выражений:")
    for item in stats["top_expressions"]:
        print(f"  {item['expression_text']}: {item['count']}")


def save_stats(stats: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def save_skipped_lines(skipped_lines_info: list[dict[str, Any]], path: Path) -> None:
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
        default="data/raw/gold/data_val.xlsx",
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
        default="Диалоги",
        help="Имя листа Excel или индекс листа.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="ID",
        help="Название колонки с ID диалога.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="Аннотация",
        help="Название колонки с размеченным текстом.",
    )
    return parser.parse_args()


def resolve_sheet_name(sheet_name_arg: str) -> int | str:
    return int(sheet_name_arg) if sheet_name_arg.isdigit() else sheet_name_arg


def main() -> None:
    args = parse_args()

    input_path = resolve_input_path(args.input)
    output_path = Path(args.output)
    stats_output_path = Path(args.stats_output)
    skipped_output_path = Path(args.skipped_output)
    sheet_name = resolve_sheet_name(args.sheet_name)

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

    print(f"\nВходной Excel: {input_path}")
    print(f"Готово. JSONL сохранён в: {output_path}")
    print(f"Статистика сохранена в: {stats_output_path}")
    print(f"Пропущенные строки сохранены в: {skipped_output_path}")


if __name__ == "__main__":
    main()
