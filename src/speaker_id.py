from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav


SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a", ".ogg",
    ".aac", ".ac3", ".dts", ".mp4", ".mkv", ".mov", ".avi",
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(data: dict[str, Any] | list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Ошибка при выполнении внешней команды:\n" + " ".join(command)
        ) from exc


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def _finalize_wav(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32)


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Читает аудиофайл и приводит его к float32 mono waveform.

    Сначала пробует direct read через soundfile.
    Если формат не поддерживается (например, ac3 / dts / mkv),
    используется ffmpeg-decoding во временный WAV.
    """
    path = Path(path)

    try:
        wav, sr = sf.read(path)
        return _finalize_wav(wav), sr
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            command = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(tmp_path),
            ]
            run_command(command)
            wav, sr = sf.read(tmp_path)
            return _finalize_wav(wav), sr
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def safe_round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def infer_character_name_from_flat_file(path: Path) -> str:
    """
    Если samples лежат в одной папке, пытаемся извлечь имя персонажа
    из имени файла.

    Поддерживаемые форматы:
    - "Анна.wav"
    - "Анна__01.wav"
    - "Анна__sample2.wav"
    """
    stem = path.stem
    if "__" in stem:
        return stem.split("__", 1)[0].strip()
    return stem.strip()


def discover_sample_groups(samples_dir: str | Path) -> dict[str, list[Path]]:
    """
    Поддерживает несколько способов расположения audio_profiles:

    Вариант A:
    data/raw/validation/audio_profiles/
      Никита/
        1.wav
        2.ac3
      Афина/
        sample1.wav

    Вариант B:
    data/raw/validation/audio_profiles/
      Никита__1.wav
      Никита__2.ac3
      Афина__1.wav

    Вариант C:
    рекурсивная вложенность, где имя персонажа = имя ближайшей папки с файлами.
    """
    samples_dir = Path(samples_dir)
    if not samples_dir.exists():
        raise FileNotFoundError(f"Не найдена папка с сэмплами: {samples_dir}")

    groups: dict[str, list[Path]] = defaultdict(list)

    audio_files = sorted(fp for fp in samples_dir.rglob("*") if is_audio_file(fp))
    if not audio_files:
        return {}

    for audio_file in audio_files:
        relative_parts = audio_file.relative_to(samples_dir).parts

        if len(relative_parts) >= 2:
            character_name = relative_parts[0].strip()
        else:
            character_name = infer_character_name_from_flat_file(audio_file)

        if not character_name:
            continue

        groups[character_name].append(audio_file)

    return {name: sorted(files) for name, files in sorted(groups.items())}


def build_character_embeddings(
    sample_groups: dict[str, list[Path]],
    encoder: VoiceEncoder,
    min_sample_duration_sec: float = 0.5,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """
    Строит усреднённый voice embedding для каждого персонажа
    по его голосовым сэмплам.
    """
    character_embeddings: dict[str, np.ndarray] = {}
    character_profiles: list[dict[str, Any]] = []

    for character_name, sample_paths in sorted(sample_groups.items()):
        sample_embeddings: list[np.ndarray] = []
        valid_sample_count = 0
        skipped_sample_count = 0
        total_duration_sec = 0.0

        for sample_path in sample_paths:
            wav, sr = read_audio(sample_path)
            duration_sec = len(wav) / sr if sr > 0 else 0.0

            if duration_sec < min_sample_duration_sec:
                skipped_sample_count += 1
                continue

            wav_pp = preprocess_wav(wav, source_sr=sr)
            if len(wav_pp) == 0:
                skipped_sample_count += 1
                continue

            embedding = encoder.embed_utterance(wav_pp)
            sample_embeddings.append(embedding)
            valid_sample_count += 1
            total_duration_sec += duration_sec

        if sample_embeddings:
            character_embeddings[character_name] = np.mean(sample_embeddings, axis=0)

        character_profiles.append(
            {
                "character_name": character_name,
                "num_discovered_samples": len(sample_paths),
                "num_valid_samples": valid_sample_count,
                "num_skipped_samples": skipped_sample_count,
                "total_valid_duration_sec": safe_round(total_duration_sec, 3),
                "embedding_built": bool(sample_embeddings),
            }
        )

    return character_embeddings, character_profiles


def extract_audio_segment(
    wav: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    """
    Вырезает аудиосегмент по таймкодам.
    """
    start_idx = max(0, int(start_time * sr))
    end_idx = min(len(wav), int(end_time * sr))

    if end_idx <= start_idx:
        return np.array([], dtype=np.float32)

    return wav[start_idx:end_idx].astype(np.float32)


def collect_segments_by_speaker(
    utterances: list[dict[str, Any]],
    min_utterance_duration_sec: float = 0.7,
    unknown_speaker_label: str = "unknown_speaker",
) -> dict[str, list[dict[str, float]]]:
    """
    Собирает интервалы utterances по speaker_label.
    """
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)

    for utt in utterances:
        speaker_label = str(utt.get("speaker_label", unknown_speaker_label))
        if speaker_label == unknown_speaker_label:
            continue

        start_time = utt.get("start_time")
        end_time = utt.get("end_time")

        if start_time is None or end_time is None:
            continue

        start_time = float(start_time)
        end_time = float(end_time)
        duration_sec = end_time - start_time

        if duration_sec < min_utterance_duration_sec:
            continue

        grouped[speaker_label].append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": duration_sec,
            }
        )

    for speaker_label in grouped:
        grouped[speaker_label].sort(key=lambda x: (-x["duration_sec"], x["start_time"]))

    return dict(grouped)


def build_speaker_embeddings(
    audio_input: str | Path,
    utterances: list[dict[str, Any]],
    encoder: VoiceEncoder,
    min_utterance_duration_sec: float = 0.7,
    min_total_duration_sec: float = 1.5,
    max_total_duration_sec: float = 45.0,
    unknown_speaker_label: str = "unknown_speaker",
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """
    Для каждого diarized speaker_label строит embedding
    по набору utterance-сегментов из исходного аудио.
    """
    wav, sr = read_audio(audio_input)
    grouped_segments = collect_segments_by_speaker(
        utterances=utterances,
        min_utterance_duration_sec=min_utterance_duration_sec,
        unknown_speaker_label=unknown_speaker_label,
    )

    speaker_embeddings: dict[str, np.ndarray] = {}
    speaker_profiles: list[dict[str, Any]] = []

    for speaker_label, segments in sorted(grouped_segments.items()):
        collected_pieces: list[np.ndarray] = []
        used_segments = 0
        total_duration_sec = 0.0

        for seg in segments:
            if total_duration_sec >= max_total_duration_sec:
                break

            piece = extract_audio_segment(
                wav=wav,
                sr=sr,
                start_time=seg["start_time"],
                end_time=seg["end_time"],
            )
            if len(piece) == 0:
                continue

            collected_pieces.append(piece)
            total_duration_sec += seg["duration_sec"]
            used_segments += 1

        if collected_pieces:
            concatenated = np.concatenate(collected_pieces)
            wav_pp = preprocess_wav(concatenated, source_sr=sr)

            if len(wav_pp) > 0 and total_duration_sec >= min_total_duration_sec:
                embedding = encoder.embed_utterance(wav_pp)
                speaker_embeddings[speaker_label] = embedding
                embedding_built = True
            else:
                embedding_built = False
        else:
            embedding_built = False

        speaker_profiles.append(
            {
                "speaker_label": speaker_label,
                "num_candidate_segments": len(segments),
                "num_used_segments": used_segments,
                "total_used_duration_sec": safe_round(total_duration_sec, 3),
                "embedding_built": embedding_built,
            }
        )

    return speaker_embeddings, speaker_profiles


def assign_speakers_to_characters(
    speaker_embeddings: dict[str, np.ndarray],
    character_embeddings: dict[str, np.ndarray],
    similarity_threshold: float = 0.65,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Для каждого speaker_label ищет наиболее похожего персонажа.
    """
    assignments: list[dict[str, Any]] = []

    for speaker_label, speaker_emb in sorted(speaker_embeddings.items()):
        scored_candidates: list[dict[str, Any]] = []

        for character_name, character_emb in character_embeddings.items():
            sim = cosine_similarity(speaker_emb, character_emb)
            scored_candidates.append(
                {
                    "character_name": character_name,
                    "similarity": round(sim, 4),
                }
            )

        scored_candidates.sort(key=lambda x: (-x["similarity"], x["character_name"]))

        if scored_candidates:
            best = scored_candidates[0]
            if best["similarity"] >= similarity_threshold:
                speaker_name = best["character_name"]
                status = "matched"
            else:
                speaker_name = "unknown"
                status = "below_threshold"
        else:
            best = None
            speaker_name = "unknown"
            status = "no_character_embeddings"

        assignments.append(
            {
                "speaker_label": speaker_label,
                "speaker_name": speaker_name,
                "status": status,
                "best_similarity": best["similarity"] if best else None,
                "top_candidates": scored_candidates[:top_k],
            }
        )

    return assignments


def apply_assignments_to_utterances(
    utterances: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    unknown_speaker_label: str = "unknown_speaker",
) -> list[dict[str, Any]]:
    """
    Обновляет utterances, добавляя speaker_name и speaker_similarity.
    """
    assignment_map = {item["speaker_label"]: item for item in assignments}
    updated: list[dict[str, Any]] = []

    for utt in utterances:
        speaker_label = str(utt.get("speaker_label", unknown_speaker_label))
        assignment = assignment_map.get(speaker_label)

        utt_copy = dict(utt)

        if assignment is None:
            if speaker_label == unknown_speaker_label:
                utt_copy["speaker_name"] = "unknown"
                utt_copy["speaker_similarity"] = None
            else:
                utt_copy["speaker_name"] = "unknown"
                utt_copy["speaker_similarity"] = None
        else:
            utt_copy["speaker_name"] = assignment["speaker_name"]
            utt_copy["speaker_similarity"] = assignment["best_similarity"]

        updated.append(utt_copy)

    return updated


def build_stats(
    character_profiles: list[dict[str, Any]],
    speaker_profiles: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    updated_utterances: list[dict[str, Any]],
) -> dict[str, Any]:
    matched_assignments = [a for a in assignments if a["status"] == "matched"]
    unknown_assignments = [a for a in assignments if a["speaker_name"] == "unknown"]

    utterances_per_name: dict[str, int] = defaultdict(int)
    for utt in updated_utterances:
        utterances_per_name[str(utt.get("speaker_name", "unknown"))] += 1

    return {
        "num_character_profiles": len(character_profiles),
        "num_valid_character_profiles": sum(1 for p in character_profiles if p["embedding_built"]),
        "num_speaker_profiles": len(speaker_profiles),
        "num_valid_speaker_profiles": sum(1 for p in speaker_profiles if p["embedding_built"]),
        "num_assignments": len(assignments),
        "num_matched_assignments": len(matched_assignments),
        "num_unknown_assignments": len(unknown_assignments),
        "utterances_per_speaker_name": dict(sorted(utterances_per_name.items())),
    }


def print_stats(stats: dict[str, Any]) -> None:
    print("\n=== SPEAKER IDENTIFICATION STATS ===")
    print(f"Character profiles: {stats['num_character_profiles']}")
    print(f"Valid character profiles: {stats['num_valid_character_profiles']}")
    print(f"Speaker profiles: {stats['num_speaker_profiles']}")
    print(f"Valid speaker profiles: {stats['num_valid_speaker_profiles']}")
    print(f"Assignments: {stats['num_assignments']}")
    print(f"Matched assignments: {stats['num_matched_assignments']}")
    print(f"Unknown assignments: {stats['num_unknown_assignments']}")

    print("\nUtterances per speaker_name:")
    for speaker_name, count in stats["utterances_per_speaker_name"].items():
        print(f"  {speaker_name}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Этап 6: speaker identification по голосовым сэмплам."
    )
    parser.add_argument(
        "--audio-input",
        type=str,
        required=True,
        help="Путь к WAV-файлу, на котором строились utterances.",
    )
    parser.add_argument(
        "--utterances-input",
        type=str,
        required=True,
        help="JSONL-файл с utterances из этапа 5.",
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        required=True,
        help="Папка с голосовыми сэмплами персонажей.",
    )
    parser.add_argument(
        "--utterances-output",
        type=str,
        required=True,
        help="Куда сохранить utterances с полями speaker_name и speaker_similarity.",
    )
    parser.add_argument(
        "--mapping-output",
        type=str,
        required=True,
        help="Куда сохранить JSON с mapping speaker_label -> speaker_name.",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        required=True,
        help="Куда сохранить JSON со статистикой этапа.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.65,
        help="Порог cosine similarity для принятия решения о сопоставлении.",
    )
    parser.add_argument(
        "--min-sample-duration-sec",
        type=float,
        default=0.5,
        help="Минимальная длительность одного голосового sample-файла персонажа.",
    )
    parser.add_argument(
        "--min-utterance-duration-sec",
        type=float,
        default=0.7,
        help="Минимальная длительность utterance для построения speaker embedding.",
    )
    parser.add_argument(
        "--min-total-duration-sec",
        type=float,
        default=1.5,
        help="Минимальная суммарная длительность речи speaker_label для построения embedding.",
    )
    parser.add_argument(
        "--max-total-duration-sec",
        type=float,
        default=45.0,
        help="Максимальная суммарная длительность речи speaker_label, используемая для embedding.",
    )
    parser.add_argument(
        "--unknown-speaker-label",
        type=str,
        default="unknown_speaker",
        help="Метка неизвестного speaker_label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    utterances = load_jsonl(args.utterances_input)
    sample_groups = discover_sample_groups(args.samples_dir)

    if not sample_groups:
        raise ValueError(
            "Не удалось найти ни одного голосового сэмпла. "
            "Проверьте структуру папки samples."
        )

    encoder = VoiceEncoder()

    character_embeddings, character_profiles = build_character_embeddings(
        sample_groups=sample_groups,
        encoder=encoder,
        min_sample_duration_sec=args.min_sample_duration_sec,
    )

    if not character_embeddings:
        raise ValueError(
            "Не удалось построить ни одного character embedding. "
            "Проверьте качество и длительность sample-файлов."
        )

    speaker_embeddings, speaker_profiles = build_speaker_embeddings(
        audio_input=args.audio_input,
        utterances=utterances,
        encoder=encoder,
        min_utterance_duration_sec=args.min_utterance_duration_sec,
        min_total_duration_sec=args.min_total_duration_sec,
        max_total_duration_sec=args.max_total_duration_sec,
        unknown_speaker_label=args.unknown_speaker_label,
    )

    assignments = assign_speakers_to_characters(
        speaker_embeddings=speaker_embeddings,
        character_embeddings=character_embeddings,
        similarity_threshold=args.similarity_threshold,
        top_k=3,
    )

    updated_utterances = apply_assignments_to_utterances(
        utterances=utterances,
        assignments=assignments,
        unknown_speaker_label=args.unknown_speaker_label,
    )

    mapping_payload = {
        "similarity_threshold": args.similarity_threshold,
        "character_profiles": character_profiles,
        "speaker_profiles": speaker_profiles,
        "assignments": assignments,
    }

    save_jsonl(updated_utterances, args.utterances_output)
    save_json(mapping_payload, args.mapping_output)

    stats = build_stats(
        character_profiles=character_profiles,
        speaker_profiles=speaker_profiles,
        assignments=assignments,
        updated_utterances=updated_utterances,
    )
    save_json(stats, args.stats_output)
    print_stats(stats)

    print(f"\nUtterances with speaker names saved to: {args.utterances_output}")
    print(f"Speaker mapping saved to: {args.mapping_output}")
    print(f"Stats saved to: {args.stats_output}")


if __name__ == "__main__":
    main()