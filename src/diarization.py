from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from pyannote.audio import Pipeline


def resolve_device(device_arg: str) -> str:
    """
    Определяет устройство для инференса.
    """
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_diarization_pipeline(model_name: str, hf_token: str, device: str) -> Pipeline:
    last_error: Exception | None = None
    load_auth_mode: str | None = None
    load_auth_fallback_used = False

    for kwargs, mode in (({"token": hf_token}, "token"), ({"use_auth_token": hf_token}, "use_auth_token")):
        try:
            if mode == "token":
                print(f"[INFO] Loading pyannote pipeline with token=... | model={model_name}")
            else:
                print(
                    "[FALLBACK] Pipeline.from_pretrained(..., token=...) is not supported by the current stack. "
                    "Switching to use_auth_token=..."
                )
                load_auth_fallback_used = True

            pipeline = Pipeline.from_pretrained(model_name, **kwargs)
            load_auth_mode = mode
            break
        except TypeError as e:
            last_error = e
            continue
    else:
        raise last_error or RuntimeError("Не удалось загрузить pyannote pipeline.")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Запрошен CUDA, но GPU недоступен.")
        pipeline.to(torch.device("cuda"))
        print("[INFO] Pipeline moved to CUDA.")

    # техническая мета-информация, чтобы потом вернуть её в JSON
    pipeline._audiointent_load_meta = {
        "load_auth_mode": load_auth_mode,
        "load_auth_fallback_used": load_auth_fallback_used,
    }

    return pipeline


def _annotation_to_segments(annotation: Any) -> tuple[list[dict[str, Any]], int]:
    """
    Преобразует pyannote Annotation в JSON-friendly список сегментов.
    Возвращает (segments, num_speakers_detected).
    """
    segments: list[dict[str, Any]] = []
    speakers: set[str] = set()

    for idx, (turn, _, speaker_label) in enumerate(annotation.itertracks(yield_label=True), start=1):
        start_time = round(float(turn.start), 3)
        end_time = round(float(turn.end), 3)
        speaker_label = str(speaker_label)

        segments.append(
            {
                "segment_id": f"spkseg_{idx:04d}",
                "start_time": start_time,
                "end_time": end_time,
                "speaker_label": speaker_label,
            }
        )
        speakers.add(speaker_label)

    segments.sort(key=lambda x: (x["start_time"], x["end_time"], x["speaker_label"]))
    return segments, len(speakers)


def run_diarization(
    audio_path: str | Path,
    model_name: str,
    hf_token: str,
    device: str = "auto",
    use_exclusive: bool = True,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict[str, Any]:
    """
    Выполняет speaker diarization и возвращает результат в JSON-формате проекта.

    Теперь сохраняются ОБА варианта сегментов:
    - regular_segments
    - exclusive_segments

    Для обратной совместимости ключ `segments` остаётся и содержит тот вариант,
    который выбран параметром `use_exclusive`.
    """
    audio_path = Path(audio_path)
    resolved_device = resolve_device(device)

    pipeline = load_diarization_pipeline(
        model_name=model_name,
        hf_token=hf_token,
        device=resolved_device,
    )

    inference_kwargs: dict[str, Any] = {}
    if num_speakers is not None:
        inference_kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            inference_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            inference_kwargs["max_speakers"] = max_speakers

    output = pipeline(str(audio_path), **inference_kwargs)

    # 1) regular diarization — основной raw-output
    regular_annotation = getattr(output, "speaker_diarization", output)
    regular_segments, regular_num_speakers = _annotation_to_segments(regular_annotation)

    # 2) exclusive diarization — вспомогательный output community-1 для удобной стыковки с ASR
    if hasattr(output, "exclusive_speaker_diarization"):
        exclusive_annotation = output.exclusive_speaker_diarization
        exclusive_segments, exclusive_num_speakers = _annotation_to_segments(exclusive_annotation)
        exclusive_fallback_used = False
    else:
        print(
            "[FALLBACK] output.exclusive_speaker_diarization is unavailable. "
            "Using regular speaker diarization instead."
        )
        exclusive_segments = list(regular_segments)
        exclusive_num_speakers = regular_num_speakers
        exclusive_fallback_used = True

    # 3) segments — backward-compatible поле для текущего downstream
    if use_exclusive:
        selected_segments = exclusive_segments
        selected_num_speakers = exclusive_num_speakers
        selected_segments_mode = "exclusive"
        used_exclusive = True
    else:
        selected_segments = regular_segments
        selected_num_speakers = regular_num_speakers
        selected_segments_mode = "regular"
        used_exclusive = False

    load_meta = getattr(pipeline, "_audiointent_load_meta", {})

    return {
        "audio_path": str(audio_path),
        "model_name": model_name,
        "device": resolved_device,
        "load_auth_mode": load_meta.get("load_auth_mode"),
        "load_auth_fallback_used": load_meta.get("load_auth_fallback_used", False),
        "used_exclusive_diarization": used_exclusive,
        "exclusive_fallback_used": exclusive_fallback_used,
        "selected_segments_mode": selected_segments_mode,
        "num_speakers_detected": selected_num_speakers,
        "num_speakers_detected_regular": regular_num_speakers,
        "num_speakers_detected_exclusive": exclusive_num_speakers,
        # backward-compatible
        "segments": selected_segments,
        # debug / analysis
        "regular_segments": regular_segments,
        "exclusive_segments": exclusive_segments,
    }


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Этап 4: speaker diarization через pyannote.audio."
    )
    parser.add_argument(
        "--audio-input",
        type=str,
        required=True,
        help="Путь к подготовленному WAV-файлу.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Куда сохранить JSON с diarization-сегментами.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token. Если не передан, будет взят из HF_TOKEN.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="pyannote/speaker-diarization-community-1",
        help="Название pyannote-пайплайна.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Устройство для инференса.",
    )
    parser.add_argument(
        "--disable-exclusive",
        action="store_true",
        help="Если указан флаг, поле segments будет заполнено regular diarization, а не exclusive.",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Точное число спикеров, если оно известно.",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Минимальное число спикеров.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Максимальное число спикеров.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "Не найден Hugging Face token. Передайте его через --hf-token "
            "или задайте переменную окружения HF_TOKEN."
        )

    diarization_result = run_diarization(
        audio_path=args.audio_input,
        model_name=args.model_name,
        hf_token=hf_token,
        device=args.device,
        use_exclusive=not args.disable_exclusive,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    save_json(diarization_result, args.output)
    print(f"Diarization-результат сохранён в: {args.output}")


if __name__ == "__main__":
    main()
