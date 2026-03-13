from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PUNCT_NO_LEADING_SPACE = {
    ".", ",", "!", "?", ";", ":", ")", "]", "}", "»", "…", "%", "—"
}
PUNCT_NO_TRAILING_SPACE = {
    "(", "[", "{", "«"
}


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def overlap_duration(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def interval_midpoint(start_time: float, end_time: float) -> float:
    return (start_time + end_time) / 2.0


def normalize_diarization_segments(diarization_data: dict[str, Any]) -> list[dict[str, Any]]:
    segments = diarization_data.get("segments", [])
    normalized: list[dict[str, Any]] = []

    for idx, seg in enumerate(segments, start=1):
        start_time = seg.get("start_time")
        end_time = seg.get("end_time")
        speaker_label = seg.get("speaker_label")

        if start_time is None or end_time is None or speaker_label is None:
            continue

        normalized.append(
            {
                "segment_id": seg.get("segment_id", f"spkseg_{idx:04d}"),
                "start_time": float(start_time),
                "end_time": float(end_time),
                "speaker_label": str(speaker_label),
            }
        )

    normalized.sort(key=lambda x: (x["start_time"], x["end_time"], x["speaker_label"]))
    return normalized


def assign_speaker_label(
    start_time: float,
    end_time: float,
    diarization_segments: list[dict[str, Any]],
    unknown_speaker_label: str = "unknown_speaker",
    max_nonoverlap_assign_distance_sec: float = 1.0,
) -> str:
    """
    Назначает speaker_label временному интервалу.

    Логика:
    1. Сначала ищется diarization-сегмент с максимальным пересечением.
    2. Если пересечений нет, берётся ближайший сегмент по midpoint,
       но только если он достаточно близко.
    3. Иначе возвращается unknown_speaker.
    """
    best_overlap = 0.0
    best_speaker = None

    for seg in diarization_segments:
        ov = overlap_duration(start_time, end_time, seg["start_time"], seg["end_time"])
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = seg["speaker_label"]

    if best_speaker is not None and best_overlap > 0.0:
        return best_speaker

    current_mid = interval_midpoint(start_time, end_time)
    best_distance = None
    nearest_speaker = None

    for seg in diarization_segments:
        seg_mid = interval_midpoint(seg["start_time"], seg["end_time"])
        dist = abs(current_mid - seg_mid)

        if best_distance is None or dist < best_distance:
            best_distance = dist
            nearest_speaker = seg["speaker_label"]

    if best_distance is not None and best_distance <= max_nonoverlap_assign_distance_sec:
        return nearest_speaker or unknown_speaker_label

    return unknown_speaker_label


def extract_timed_units_from_asr(
    asr_data: dict[str, Any],
    diarization_segments: list[dict[str, Any]],
    unknown_speaker_label: str = "unknown_speaker",
    max_nonoverlap_assign_distance_sec: float = 1.0,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Превращает ASR-результат в последовательность timed units.

    Предпочтительный режим:
    - word-level units, если у сегментов есть words с таймкодами

    Fallback:
    - segment-level units, если words отсутствуют или не выровнены

    Возвращает:
    - список units
    - флаг, использовался ли word-level режим
    """
    segments = asr_data.get("segments", [])
    timed_units: list[dict[str, Any]] = []

    has_any_aligned_words = False
    for seg in segments:
        words = seg.get("words") or []
        for word_info in words:
            if word_info.get("start_time") is not None and word_info.get("end_time") is not None:
                has_any_aligned_words = True
                break
        if has_any_aligned_words:
            break

    if has_any_aligned_words:
        for seg in segments:
            words = seg.get("words") or []

            valid_words = [
                word_info for word_info in words
                if word_info.get("word")
                and word_info.get("start_time") is not None
                and word_info.get("end_time") is not None
            ]

            if valid_words:
                for word_info in valid_words:
                    start_time = float(word_info["start_time"])
                    end_time = float(word_info["end_time"])
                    speaker_label = assign_speaker_label(
                        start_time=start_time,
                        end_time=end_time,
                        diarization_segments=diarization_segments,
                        unknown_speaker_label=unknown_speaker_label,
                        max_nonoverlap_assign_distance_sec=max_nonoverlap_assign_distance_sec,
                    )
                    timed_units.append(
                        {
                            "unit_type": "word",
                            "text": str(word_info["word"]).strip(),
                            "start_time": round(start_time, 3),
                            "end_time": round(end_time, 3),
                            "speaker_label": speaker_label,
                            "source_segment_id": seg.get("segment_id"),
                        }
                    )
            else:
                # fallback для конкретного сегмента без word-level alignment
                seg_text = str(seg.get("text", "")).strip()
                seg_start = seg.get("start_time")
                seg_end = seg.get("end_time")

                if seg_text and seg_start is not None and seg_end is not None:
                    start_time = float(seg_start)
                    end_time = float(seg_end)
                    speaker_label = assign_speaker_label(
                        start_time=start_time,
                        end_time=end_time,
                        diarization_segments=diarization_segments,
                        unknown_speaker_label=unknown_speaker_label,
                        max_nonoverlap_assign_distance_sec=max_nonoverlap_assign_distance_sec,
                    )
                    timed_units.append(
                        {
                            "unit_type": "segment_fallback",
                            "text": seg_text,
                            "start_time": round(start_time, 3),
                            "end_time": round(end_time, 3),
                            "speaker_label": speaker_label,
                            "source_segment_id": seg.get("segment_id"),
                        }
                    )

        timed_units.sort(key=lambda x: (x["start_time"], x["end_time"]))
        return timed_units, True

    # Глобальный fallback: сегментный режим
    for seg in segments:
        seg_text = str(seg.get("text", "")).strip()
        seg_start = seg.get("start_time")
        seg_end = seg.get("end_time")

        if not seg_text or seg_start is None or seg_end is None:
            continue

        start_time = float(seg_start)
        end_time = float(seg_end)
        speaker_label = assign_speaker_label(
            start_time=start_time,
            end_time=end_time,
            diarization_segments=diarization_segments,
            unknown_speaker_label=unknown_speaker_label,
            max_nonoverlap_assign_distance_sec=max_nonoverlap_assign_distance_sec,
        )

        timed_units.append(
            {
                "unit_type": "segment",
                "text": seg_text,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "speaker_label": speaker_label,
                "source_segment_id": seg.get("segment_id"),
            }
        )

    timed_units.sort(key=lambda x: (x["start_time"], x["end_time"]))
    return timed_units, False


def append_token_to_text(current_text: str, token: str) -> str:
    token = token.strip()
    if not token:
        return current_text

    if not current_text:
        return token

    if token in PUNCT_NO_LEADING_SPACE:
        return current_text + token

    if current_text[-1] in PUNCT_NO_TRAILING_SPACE:
        return current_text + token

    return current_text + " " + token


def build_text_from_units(units: list[dict[str, Any]]) -> str:
    text = ""
    for unit in units:
        text = append_token_to_text(text, unit["text"])
    return text.strip()


def convert_units_to_words(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Приводит timed units к полю words в формате проекта.
    """
    words: list[dict[str, Any]] = []
    for unit in units:
        words.append(
            {
                "word": unit["text"],
                "start_time": unit["start_time"],
                "end_time": unit["end_time"],
            }
        )
    return words


def should_start_new_utterance(
    previous_unit: dict[str, Any],
    current_unit: dict[str, Any],
    max_pause_sec: float,
) -> bool:
    """
    Решает, нужно ли начинать новую utterance.
    """
    if previous_unit["speaker_label"] != current_unit["speaker_label"]:
        return True

    gap = float(current_unit["start_time"]) - float(previous_unit["end_time"])
    if gap > max_pause_sec:
        return True

    return False


def build_utterances_from_units(
    timed_units: list[dict[str, Any]],
    max_pause_sec: float = 1.0,
    speaker_name_default: Any = None,
) -> list[dict[str, Any]]:
    """
    Склеивает последовательность timed units в utterances.

    Правила:
    - новая utterance начинается при смене speaker_label
    - новая utterance начинается при паузе > max_pause_sec
    """
    if not timed_units:
        return []

    utterances: list[dict[str, Any]] = []
    current_group: list[dict[str, Any]] = [timed_units[0]]

    for unit in timed_units[1:]:
        previous_unit = current_group[-1]
        if should_start_new_utterance(previous_unit, unit, max_pause_sec=max_pause_sec):
            utterances.append(current_group)
            current_group = [unit]
        else:
            current_group.append(unit)

    utterances.append(current_group)

    result: list[dict[str, Any]] = []

    for idx, group in enumerate(utterances, start=1):
        start_time = round(float(group[0]["start_time"]), 3)
        end_time = round(float(group[-1]["end_time"]), 3)
        speaker_label = str(group[0]["speaker_label"])
        text = build_text_from_units(group)
        words = convert_units_to_words(group)

        result.append(
            {
                "utterance_id": f"utt_{idx:04d}",
                "start_time": start_time,
                "end_time": end_time,
                "speaker_label": speaker_label,
                "speaker_name": speaker_name_default,
                "text": text,
                "words": words,
            }
        )

    return result


def build_stats(
    asr_data: dict[str, Any],
    diarization_segments: list[dict[str, Any]],
    timed_units: list[dict[str, Any]],
    utterances: list[dict[str, Any]],
    used_word_level_mode: bool,
    unknown_speaker_label: str,
    max_pause_sec: float,
) -> dict[str, Any]:
    speaker_counter: dict[str, int] = {}
    unknown_units = 0

    for unit in timed_units:
        label = unit["speaker_label"]
        speaker_counter[label] = speaker_counter.get(label, 0) + 1
        if label == unknown_speaker_label:
            unknown_units += 1

    utterance_counter_by_speaker: dict[str, int] = {}
    for utt in utterances:
        label = utt["speaker_label"]
        utterance_counter_by_speaker[label] = utterance_counter_by_speaker.get(label, 0) + 1

    return {
        "audio_path": asr_data.get("audio_path"),
        "language": asr_data.get("language"),
        "alignment_used": asr_data.get("alignment_used"),
        "num_asr_segments": len(asr_data.get("segments", [])),
        "num_diarization_segments": len(diarization_segments),
        "num_timed_units": len(timed_units),
        "used_word_level_mode": used_word_level_mode,
        "num_utterances": len(utterances),
        "max_pause_sec": max_pause_sec,
        "unknown_speaker_label": unknown_speaker_label,
        "unknown_timed_units": unknown_units,
        "timed_units_per_speaker": speaker_counter,
        "utterances_per_speaker": utterance_counter_by_speaker,
    }


def print_stats(stats: dict[str, Any]) -> None:
    print("\n=== UTTERANCE BUILDER STATS ===")
    print(f"ASR-сегментов: {stats['num_asr_segments']}")
    print(f"Diarization-сегментов: {stats['num_diarization_segments']}")
    print(f"Timed units: {stats['num_timed_units']}")
    print(f"Режим: {'word-level' if stats['used_word_level_mode'] else 'segment-level fallback'}")
    print(f"Utterances: {stats['num_utterances']}")
    print(f"Timed units с unknown speaker: {stats['unknown_timed_units']}")

    print("\nUtterances по speaker_label:")
    for speaker_label, count in sorted(stats["utterances_per_speaker"].items()):
        print(f"  {speaker_label}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Этап 5: сбор utterances из ASR и diarization."
    )
    parser.add_argument(
        "--asr-input",
        type=str,
        required=True,
        help="Путь к JSON-файлу ASR из этапа 4.",
    )
    parser.add_argument(
        "--diarization-input",
        type=str,
        required=True,
        help="Путь к JSON-файлу diarization из этапа 4.",
    )
    parser.add_argument(
        "--utterances-output",
        type=str,
        required=True,
        help="Куда сохранить JSONL с utterances.",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        required=True,
        help="Куда сохранить JSON со статистикой.",
    )
    parser.add_argument(
        "--max-pause-sec",
        type=float,
        default=1.0,
        help="Максимальная пауза между соседними units внутри одной utterance.",
    )
    parser.add_argument(
        "--unknown-speaker-label",
        type=str,
        default="unknown_speaker",
        help="Метка для случаев, когда speaker не удалось назначить.",
    )
    parser.add_argument(
        "--max-nonoverlap-assign-distance-sec",
        type=float,
        default=1.0,
        help="Максимальная дистанция до ближайшего diarization-сегмента, если нет пересечения.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    asr_data = load_json(args.asr_input)
    diarization_data = load_json(args.diarization_input)

    diarization_segments = normalize_diarization_segments(diarization_data)

    if not diarization_segments:
        raise ValueError("В diarization-файле не найдено ни одного корректного speaker segment.")

    timed_units, used_word_level_mode = extract_timed_units_from_asr(
        asr_data=asr_data,
        diarization_segments=diarization_segments,
        unknown_speaker_label=args.unknown_speaker_label,
        max_nonoverlap_assign_distance_sec=args.max_nonoverlap_assign_distance_sec,
    )

    if not timed_units:
        raise ValueError("Не удалось построить ни одного timed unit из ASR-результата.")

    utterances = build_utterances_from_units(
        timed_units=timed_units,
        max_pause_sec=args.max_pause_sec,
        speaker_name_default=None,
    )

    save_jsonl(utterances, args.utterances_output)

    stats = build_stats(
        asr_data=asr_data,
        diarization_segments=diarization_segments,
        timed_units=timed_units,
        utterances=utterances,
        used_word_level_mode=used_word_level_mode,
        unknown_speaker_label=args.unknown_speaker_label,
        max_pause_sec=args.max_pause_sec,
    )
    save_json(stats, args.stats_output)
    print_stats(stats)

    print(f"\nUtterances сохранены в: {args.utterances_output}")
    print(f"Статистика сохранена в: {args.stats_output}")


if __name__ == "__main__":
    main()