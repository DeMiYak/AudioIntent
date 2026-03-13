from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import whisperx


def run_command(command: list[str]) -> None:
    """
    Запускает внешнюю команду и выбрасывает исключение при ошибке.
    """
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Ошибка при выполнении внешней команды:\n"
            + " ".join(command)
        ) from exc


def resolve_device(device_arg: str) -> str:
    """
    Определяет устройство для инференса.
    """
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def resolve_compute_type(device: str, compute_type_arg: str) -> str:
    """
    Выбирает compute_type в зависимости от устройства.
    """
    if compute_type_arg != "auto":
        return compute_type_arg

    if device == "cuda":
        return "float16"
    return "int8"


def prepare_audio(
    media_input: str | Path,
    audio_output: str | Path,
    start_sec: float | None = None,
    duration_sec: float | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """
    Вырезает фрагмент из видео/аудио и приводит его к WAV mono 16kHz.
    """
    media_input = Path(media_input)
    audio_output = Path(audio_output)
    audio_output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(media_input),
    ]

    if start_sec is not None:
        command.extend(["-ss", str(start_sec)])
    if duration_sec is not None:
        command.extend(["-t", str(duration_sec)])

    command.extend(
        [
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(audio_output),
        ]
    )

    run_command(command)
    return audio_output


def _normalize_word(word_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "word": str(word_info.get("word", "")).strip(),
        "start_time": round(float(word_info["start"]), 3) if word_info.get("start") is not None else None,
        "end_time": round(float(word_info["end"]), 3) if word_info.get("end") is not None else None,
        "score": round(float(word_info["score"]), 4) if word_info.get("score") is not None else None,
    }


def normalize_transcription_result(
    result: dict[str, Any],
    audio_path: str | Path,
    model_name: str,
    device: str,
    compute_type: str,
    alignment_used: bool,
) -> dict[str, Any]:
    """
    Приводит результат WhisperX к стабильному JSON-формату проекта.
    """
    normalized_segments: list[dict[str, Any]] = []

    for idx, segment in enumerate(result.get("segments", []), start=1):
        words = segment.get("words", []) or []
        normalized_words = [_normalize_word(word) for word in words]

        normalized_segments.append(
            {
                "segment_id": f"seg_{idx:04d}",
                "start_time": round(float(segment["start"]), 3) if segment.get("start") is not None else None,
                "end_time": round(float(segment["end"]), 3) if segment.get("end") is not None else None,
                "text": str(segment.get("text", "")).strip(),
                "words": normalized_words,
            }
        )

    return {
        "audio_path": str(audio_path),
        "model_name": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": result.get("language"),
        "alignment_used": alignment_used,
        "segments": normalized_segments,
    }


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = "medium",
    device: str = "auto",
    compute_type: str = "auto",
    batch_size: int = 8,
    perform_alignment: bool = True,
) -> dict[str, Any]:
    """
    Транскрибирует аудио через WhisperX и, по возможности, выполняет forced alignment.
    """
    audio_path = Path(audio_path)
    resolved_device = resolve_device(device)
    resolved_compute_type = resolve_compute_type(resolved_device, compute_type)

    model = whisperx.load_model(
        model_name,
        resolved_device,
        compute_type=resolved_compute_type,
    )

    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size)

    alignment_used = False

    # Освобождаем память от основной ASR-модели перед alignment
    del model
    gc.collect()
    if resolved_device == "cuda":
        torch.cuda.empty_cache()

    if perform_alignment:
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=resolved_device,
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                resolved_device,
                return_char_alignments=False,
            )
            alignment_used = True

            del model_a
            gc.collect()
            if resolved_device == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            print(
                "[WARN] Alignment не выполнен. Будут сохранены только сегментные таймкоды.\n"
                f"Причина: {exc}"
            )

    return normalize_transcription_result(
        result=result,
        audio_path=audio_path,
        model_name=model_name,
        device=resolved_device,
        compute_type=resolved_compute_type,
        alignment_used=alignment_used,
    )


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Этап 4: подготовка аудио и ASR через WhisperX."
    )
    parser.add_argument(
        "--media-input",
        type=str,
        required=True,
        help="Путь к видеофайлу или аудиофайлу.",
    )
    parser.add_argument(
        "--prepared-audio-output",
        type=str,
        required=True,
        help="Куда сохранить подготовленный WAV-файл.",
    )
    parser.add_argument(
        "--transcript-output",
        type=str,
        required=True,
        help="Куда сохранить JSON с результатом ASR.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help="Начало фрагмента в секундах.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Длительность фрагмента в секундах.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Частота дискретизации итогового WAV.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Количество аудиоканалов итогового WAV.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="medium",
        help="WhisperX model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Устройство для инференса.",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="auto",
        choices=["auto", "float16", "int8", "float32"],
        help="compute_type для WhisperX.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Размер батча для ASR.",
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="Если указан флаг, forced alignment не выполняется.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepared_audio_path = prepare_audio(
        media_input=args.media_input,
        audio_output=args.prepared_audio_output,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        sample_rate=args.sample_rate,
        channels=args.channels,
    )

    transcription = transcribe_audio(
        audio_path=prepared_audio_path,
        model_name=args.model_name,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        perform_alignment=not args.skip_alignment,
    )

    save_json(transcription, args.transcript_output)

    print(f"Подготовленное аудио сохранено в: {prepared_audio_path}")
    print(f"ASR-результат сохранён в: {args.transcript_output}")


if __name__ == "__main__":
    main()