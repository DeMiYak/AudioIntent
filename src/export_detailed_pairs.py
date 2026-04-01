from __future__ import annotations

"""Exports a flat row-per-event table from pipeline predictions.

Reads all predictions.jsonl from output-dir/windows/*/predictions.jsonl,
computes absolute film timestamps, and writes a single Excel file with columns
matching the gold annotation format (Диалоги sheet):

    ID | Фильм | Время начала | Время окончания | Тип | Аннотация

Usage:
  python -m src.export_detailed_pairs \
    --output-dir artifacts/test_piter_fm \
    --excel artifacts/test_piter_fm/detailed_pairs.xlsx
"""

import argparse
import json
from pathlib import Path


def seconds_to_hms(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def load_predictions(output_dir: Path, source_dir: Path | None = None) -> list[dict]:
    """Load predictions, patching source_start_sec from chunk_info.json when missing."""
    windows_dir = output_dir / "windows"

    # Build chunk_start_sec lookup from chunk_info.json files
    # chunk_info lives in source_dir (transcript-input-dir) or output_dir/windows
    chunk_start_lookup: dict[str, float] = {}
    search_roots = [d for d in [source_dir, windows_dir] if d is not None]
    for root in search_roots:
        if not root.exists():
            continue
        for chunk_dir in root.iterdir():
            info_path = chunk_dir / "chunk_info.json"
            if info_path.exists():
                with info_path.open(encoding="utf-8") as f:
                    info = json.load(f)
                chunk_start_lookup[info["window_id"]] = float(info.get("start_sec", 0.0))

    rows = []
    for chunk_dir in sorted(windows_dir.iterdir()):
        pred_path = chunk_dir / "predictions.jsonl"
        if not pred_path.exists():
            continue
        chunk_id = chunk_dir.name
        fallback_start = chunk_start_lookup.get(chunk_id, 0.0)
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            pred = json.loads(line)
            # Patch missing fields from chunk directory context
            if pred.get("source_start_sec") is None:
                pred["source_start_sec"] = fallback_start
            if pred.get("source_window_id") is None:
                pred["source_window_id"] = chunk_id
            rows.append(pred)
    return rows


def build_dataframe(predictions: list[dict], film_name: str = "") -> "pd.DataFrame":
    import pandas as pd

    # Sort by absolute start time across all chunks
    def abs_start(p: dict) -> float:
        return float(p.get("source_start_sec", 0.0)) + float(p.get("start_time", 0.0))

    predictions = sorted(predictions, key=abs_start)

    # Derive film name fallback from any non-empty source_film
    if not film_name:
        for p in predictions:
            film_name = p.get("source_film") or ""
            if film_name:
                break

    records = []
    for i, p in enumerate(predictions, start=1):
        chunk_start = float(p.get("source_start_sec", 0.0))
        t_start = abs_start(p)
        t_end = chunk_start + float(p.get("end_time", float(p.get("start_time", 0.0))))
        intent = p.get("intent_type", "")
        type_ru = "установление контакта" if intent == "contact_open" else "прекращение контакта"
        tag = "opening" if intent == "contact_open" else "closing"
        speaker = p.get("speaker_name", "")
        phrase = p.get("expression", "")
        annotation = f"{speaker}: <{tag}>{phrase}</{tag}>"
        speaker_phrase = f"{speaker} - {phrase}"
        records.append({
            "ID": i,
            "Фильм": film_name or p.get("source_film") or "",
            "Время начала": seconds_to_hms(t_start),
            "Время окончания": seconds_to_hms(t_end),
            "Тип": type_ru,
            "Аннотация": annotation,
            "opening": speaker_phrase if intent == "contact_open" else None,
            "closing": speaker_phrase if intent == "contact_close" else None,
        })

    return pd.DataFrame(records)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export flat row-per-event table from pipeline predictions."
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Pipeline output directory (contains windows/*/predictions.jsonl)",
    )
    parser.add_argument(
        "--excel", default=None,
        help="Output Excel path (default: <output-dir>/detailed_pairs.xlsx)",
    )
    parser.add_argument(
        "--exclude-unknown", action="store_true",
        help="Drop rows where speaker is 'unknown' or 'unknown_speaker'",
    )
    parser.add_argument(
        "--exclude-chunk", type=int, default=None,
        help="Exclude a specific chunk by index (e.g. 9 to drop credits chunk)",
    )
    parser.add_argument(
        "--source-dir", type=str, default=None,
        help="Directory containing chunk_info.json files (e.g. artifacts/test_piter_fm_asr_colab/windows)",
    )
    parser.add_argument(
        "--film-name", type=str, default="",
        help="Override film name in the Фильм column (e.g. 'Питер FM')",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    excel_path = Path(args.excel) if args.excel else output_dir / "detailed_pairs.xlsx"

    source_dir = Path(args.source_dir) if args.source_dir else None
    predictions = load_predictions(output_dir, source_dir=source_dir)
    print(f"Loaded {len(predictions)} predictions from {output_dir / 'windows'}")

    if args.exclude_unknown:
        before = len(predictions)
        predictions = [
            p for p in predictions
            if p.get("speaker_name", "") not in ("unknown", "unknown_speaker")
        ]
        print(f"Dropped {before - len(predictions)} unknown-speaker rows")

    if args.exclude_chunk is not None:
        chunk_id = f"chunk_{args.exclude_chunk:03d}"
        before = len(predictions)
        predictions = [p for p in predictions if p.get("source_window_id") != chunk_id]
        print(f"Dropped {before - len(predictions)} rows from {chunk_id}")

    df = build_dataframe(predictions, film_name=args.film_name)

    excel_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(excel_path, index=False)
    print(f"Saved {len(df)} rows to {excel_path}")

    # Quick summary
    print()
    print("By type:")
    print(df["Тип"].value_counts().to_string())


if __name__ == "__main__":
    main()
