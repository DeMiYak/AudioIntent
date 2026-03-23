from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_diarization_segments(diarization_result: dict[str, Any]) -> list[dict[str, Any]]:
    segments = []
    for seg in diarization_result.get("segments", []):
        start = seg.get("start_time")
        end = seg.get("end_time")
        if start is None or end is None:
            continue
        segments.append(
            {
                "speaker_label": str(seg.get("speaker_label", "unknown_speaker")),
                "start_time": float(start),
                "end_time": float(end),
            }
        )
    segments.sort(key=lambda x: (x["start_time"], x["end_time"], x["speaker_label"]))
    return segments


def _best_speaker_for_span(
    start_time: float,
    end_time: float,
    diarization_segments: list[dict[str, Any]],
    unknown_speaker_label: str,
    max_nonoverlap_assign_distance_sec: float,
) -> str:
    best_speaker = unknown_speaker_label
    best_overlap = 0.0
    midpoint = (start_time + end_time) / 2.0
    best_distance = None

    for seg in diarization_segments:
        seg_start = float(seg["start_time"])
        seg_end = float(seg["end_time"])
        overlap = max(0.0, min(end_time, seg_end) - max(start_time, seg_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = str(seg["speaker_label"])

        if overlap <= 0.0:
            if midpoint < seg_start:
                distance = seg_start - midpoint
            elif midpoint > seg_end:
                distance = midpoint - seg_end
            else:
                distance = 0.0
            if best_distance is None or distance < best_distance:
                best_distance = distance
                if distance <= max_nonoverlap_assign_distance_sec and best_overlap == 0.0:
                    best_speaker = str(seg["speaker_label"])

    return best_speaker


def _word_units_from_segments(asr_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for seg in asr_segments:
        for word in seg.get("words", []) or []:
            start = word.get("start")
            end = word.get("end")
            text = str(word.get("word", "")).strip()
            if start is None or end is None or not text:
                continue
            units.append({"start_time": float(start), "end_time": float(end), "text": text})
    return units


def _segment_units_from_segments(asr_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for seg in asr_segments:
        start = seg.get("start")
        end = seg.get("end")
        text = str(seg.get("text", "")).strip()
        if start is None or end is None or not text:
            continue
        units.append({"start_time": float(start), "end_time": float(end), "text": text})
    return units


def extract_timed_units_from_asr(
    asr_data: dict[str, Any],
    diarization_segments: list[dict[str, Any]],
    unknown_speaker_label: str = "unknown_speaker",
    max_nonoverlap_assign_distance_sec: float = 1.0,
) -> tuple[list[dict[str, Any]], bool]:
    asr_segments = asr_data.get("segments", [])
    word_units = _word_units_from_segments(asr_segments)
    used_word_level = bool(word_units)
    units = word_units if word_units else _segment_units_from_segments(asr_segments)

    timed_units: list[dict[str, Any]] = []
    for idx, unit in enumerate(units, start=1):
        speaker_label = _best_speaker_for_span(
            start_time=float(unit["start_time"]),
            end_time=float(unit["end_time"]),
            diarization_segments=diarization_segments,
            unknown_speaker_label=unknown_speaker_label,
            max_nonoverlap_assign_distance_sec=max_nonoverlap_assign_distance_sec,
        )
        timed_units.append(
            {
                "unit_id": idx,
                "start_time": float(unit["start_time"]),
                "end_time": float(unit["end_time"]),
                "text": str(unit["text"]).strip(),
                "speaker_label": speaker_label,
            }
        )

    return timed_units, used_word_level


def build_utterances_from_units(
    timed_units: list[dict[str, Any]],
    max_pause_sec: float = 0.8,
    max_total_duration_sec: float = 20.0,
) -> list[dict[str, Any]]:
    if not timed_units:
        return []

    timed_units = sorted(timed_units, key=lambda x: (x["start_time"], x["end_time"], x["unit_id"]))
    utterances: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def finalize_current() -> None:
        nonlocal current
        if current is None:
            return
        text = " ".join(part for part in current["parts"] if part).strip()
        utterances.append(
            {
                "utterance_id": f"utt_{len(utterances) + 1:03d}",
                "speaker_label": current["speaker_label"],
                "speaker_name": current["speaker_label"],
                "start_time": current["start_time"],
                "end_time": current["end_time"],
                "text": text,
            }
        )
        current = None

    for unit in timed_units:
        if current is None:
            current = {
                "speaker_label": unit["speaker_label"],
                "start_time": unit["start_time"],
                "end_time": unit["end_time"],
                "parts": [unit["text"]],
            }
            continue

        pause = float(unit["start_time"]) - float(current["end_time"])
        new_total = float(unit["end_time"]) - float(current["start_time"])
        same_speaker = unit["speaker_label"] == current["speaker_label"]

        if same_speaker and pause <= max_pause_sec and new_total <= max_total_duration_sec:
            current["parts"].append(unit["text"])
            current["end_time"] = unit["end_time"]
        else:
            finalize_current()
            current = {
                "speaker_label": unit["speaker_label"],
                "start_time": unit["start_time"],
                "end_time": unit["end_time"],
                "parts": [unit["text"]],
            }

    finalize_current()
    return utterances


def build_stats(
    asr_data: dict[str, Any],
    diarization_segments: list[dict[str, Any]],
    timed_units: list[dict[str, Any]],
    utterances: list[dict[str, Any]],
    used_word_level_mode: bool,
    unknown_speaker_label: str = "unknown_speaker",
    max_pause_sec: float = 0.8,
) -> dict[str, Any]:
    return {
        "num_asr_segments": len(asr_data.get("segments", [])),
        "num_diarization_segments": len(diarization_segments),
        "num_timed_units": len(timed_units),
        "num_utterances": len(utterances),
        "used_word_level_mode": bool(used_word_level_mode),
        "unknown_speaker_units": sum(
            1 for unit in timed_units if unit.get("speaker_label") == unknown_speaker_label
        ),
        "max_pause_sec": float(max_pause_sec),
    }
