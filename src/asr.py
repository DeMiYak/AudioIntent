from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from faster_whisper import WhisperModel

try:
    from faster_whisper import BatchedInferencePipeline
except Exception:  # pragma: no cover
    BatchedInferencePipeline = None


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Ошибка при выполнении внешней команды:\n" + " ".join(command)
        ) from exc


def prepare_audio(
    media_input: str | Path,
    audio_output: str | Path,
    start_sec: float | None = None,
    duration_sec: float | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    media_input = Path(media_input)
    audio_output = Path(audio_output)
    audio_output.parent.mkdir(parents=True, exist_ok=True)

    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if start_sec is not None:
        command.extend(["-ss", str(float(start_sec))])
    command.extend(["-i", str(media_input)])
    if duration_sec is not None:
        command.extend(["-t", str(float(duration_sec))])
    command.extend(
        [
            "-vn",
            "-ac",
            str(int(channels)),
            "-ar",
            str(int(sample_rate)),
            "-c:a",
            "pcm_s16le",
            str(audio_output),
        ]
    )
    run_command(command)
    return audio_output


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_compute_type(device: str, compute_type: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def _segment_to_dict(segment: Any) -> dict[str, Any]:
    words = []
    for word in getattr(segment, "words", []) or []:
        words.append(
            {
                "start": float(word.start) if word.start is not None else None,
                "end": float(word.end) if word.end is not None else None,
                "word": str(word.word),
                "probability": float(word.probability)
                if getattr(word, "probability", None) is not None
                else None,
            }
        )

    return {
        "id": int(getattr(segment, "id", 0) or 0),
        "start": float(segment.start),
        "end": float(segment.end),
        "text": str(segment.text).strip(),
        "words": words,
    }


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = "large-v3",
    device: str = "auto",
    compute_type: str = "auto",
    batch_size: int = 8,
    language: str = "ru",
    vad_filter: bool = True,
    beam_size: int = 5,
    word_timestamps: bool = True,
    perform_alignment: bool = False,
) -> dict[str, Any]:
    del perform_alignment  # совместимость со старым контрактом без WhisperX.

    audio_path = Path(audio_path)
    resolved_device = resolve_device(device)
    resolved_compute_type = resolve_compute_type(resolved_device, compute_type)

    model = WhisperModel(
        str(model_name),
        device=resolved_device,
        compute_type=resolved_compute_type,
    )

    transcribe_kwargs = {
        "language": language,
        "vad_filter": vad_filter,
        "beam_size": beam_size,
        "word_timestamps": word_timestamps,
        "condition_on_previous_text": False,
    }

    if BatchedInferencePipeline is not None and batch_size and batch_size > 1:
        pipeline = BatchedInferencePipeline(model=model)
        segments, info = pipeline.transcribe(
            str(audio_path),
            batch_size=batch_size,
            **transcribe_kwargs,
        )
        batched_mode = True
    else:
        segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)
        batched_mode = False

    segment_dicts = [_segment_to_dict(segment) for segment in segments]

    return {
        "source": "faster-whisper",
        "model_name": str(model_name),
        "device": resolved_device,
        "compute_type": resolved_compute_type,
        "batched_mode": batched_mode,
        "batch_size": int(batch_size),
        "language": getattr(info, "language", language),
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
        "duration_sec": float(getattr(info, "duration", 0.0) or 0.0),
        "segments": segment_dicts,
    }
