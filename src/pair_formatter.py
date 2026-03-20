from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


INTENT_TO_COLUMN = {
    "contact_open": "opening",
    "contact_close": "closing",
}


def normalize_piece(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def pair_to_string(speaker: str, phrase: str) -> str:
    return f"{speaker} - {phrase}"


def deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def aggregate_window_predictions(
    window: dict[str, Any],
    predictions: list[dict[str, Any]],
    unknown_speaker_name: str = "unknown",
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[tuple[float, int], str]]] = {
        "opening": [],
        "closing": [],
    }

    sorted_predictions = sorted(
        predictions,
        key=lambda item: (
            float(item.get("start_time", 0.0) or 0.0),
            int(item.get("char_start", 0) or 0),
            normalize_piece(item.get("speaker_name"), unknown_speaker_name),
            normalize_piece(item.get("expression")),
        ),
    )

    for pred in sorted_predictions:
        column = INTENT_TO_COLUMN.get(str(pred.get("intent_type")))
        if column is None:
            continue

        speaker = normalize_piece(pred.get("speaker_name"), unknown_speaker_name)
        phrase = normalize_piece(pred.get("expression"))
        if not phrase:
            continue

        pair_text = pair_to_string(speaker, phrase)
        sort_key = (
            float(pred.get("start_time", 0.0) or 0.0),
            int(pred.get("char_start", 0) or 0),
        )
        grouped[column].append((sort_key, pair_text))

    opening_pairs = deduplicate_preserve_order([pair for _, pair in sorted(grouped["opening"])])
    closing_pairs = deduplicate_preserve_order([pair for _, pair in sorted(grouped["closing"])])

    return {
        "ID": window["row_id"],
        "Фильм": window.get("film"),
        "Время начала": window.get("start_time"),
        "Время окончания": window.get("end_time"),
        "opening": "; ".join(opening_pairs) if opening_pairs else None,
        "closing": "; ".join(closing_pairs) if closing_pairs else None,
    }


def build_extracted_pairs_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = ["ID", "Фильм", "Время начала", "Время окончания", "opening", "closing"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=columns)
    return df[columns]


def save_extracted_pairs(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    return output_path
