from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .preprocess_gold import resolve_input_path, time_to_seconds


def normalize_cell_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text or None


def load_validation_windows(
    excel_path: str | Path,
    sheet_name: str = "Вал - Статус свободен",
) -> list[dict[str, Any]]:
    """
    Загружает validation-окна из Excel.
    """
    resolved_path = resolve_input_path(excel_path)
    df = pd.read_excel(resolved_path, sheet_name=sheet_name)

    windows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        raw_id = row.get("ID")
        if pd.isna(raw_id):
            continue

        start_sec = time_to_seconds(row.get("Время начала"))
        end_sec = time_to_seconds(row.get("Время окончания"))

        if start_sec is None or end_sec is None:
            continue

        if end_sec < start_sec:
            end_sec += 24 * 3600

        row_id_text = str(int(raw_id)) if isinstance(raw_id, float) and raw_id.is_integer() else str(raw_id).strip()
        windows.append(
            {
                "row_id": row_id_text,
                "window_id": f"val_{row_id_text}",
                "film": normalize_cell_text(row.get("Фильм")),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "duration_sec": round(float(end_sec - start_sec), 3),
                "start_time": normalize_cell_text(row.get("Время начала")),
                "end_time": normalize_cell_text(row.get("Время окончания")),
                "context": normalize_cell_text(row.get("Реплики - полный контекст")),
                "dialogue_type": normalize_cell_text(row.get("Тип")),
                "annotation": normalize_cell_text(row.get("Аннотация")),
                "gold_opening": normalize_cell_text(row.get("opening")),
                "gold_closing": normalize_cell_text(row.get("closing")),
            }
        )

    windows.sort(key=lambda x: (x["start_sec"], x["end_sec"], x["row_id"]))
    return windows


def load_character_names(
    excel_path: str | Path,
    sheet_name: str = "Статус свободен - персонажи",
) -> list[str]:
    resolved_path = resolve_input_path(excel_path)
    df = pd.read_excel(resolved_path, sheet_name=sheet_name)

    if "Имя" not in df.columns:
        return []

    names = [
        str(value).strip()
        for value in df["Имя"].tolist()
        if not pd.isna(value) and str(value).strip()
    ]
    return names


def build_gold_dataframe_for_evaluation(windows: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window in windows:
        rows.append(
            {
                "ID": window["row_id"],
                "Фильм": window.get("film"),
                "Время начала": window.get("start_time"),
                "Время окончания": window.get("end_time"),
                "opening": window.get("gold_opening"),
                "closing": window.get("gold_closing"),
            }
        )
    return pd.DataFrame(rows)


def save_dataframe_to_excel(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    return output_path
