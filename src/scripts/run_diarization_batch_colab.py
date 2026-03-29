from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Must be set before importing pyannote/lightning/torchmetrics/matplotlib chain.
_current_backend = os.environ.get("MPLBACKEND", "")
if _current_backend.startswith("module://matplotlib_inline"):
    print(
        "[FALLBACK] Detected notebook matplotlib backend "
        f"'{_current_backend}'. Switching to 'Agg' for standalone runner."
    )
    os.environ["MPLBACKEND"] = "Agg"
else:
    os.environ.setdefault("MPLBACKEND", "Agg")

import soundfile as sf
import torch
import torchaudio.functional as F
from pyannote.audio import Pipeline

try:
    from pyannote.audio.pipelines.utils.hook import ProgressHook
except Exception:
    ProgressHook = None


def _print_info(message: str) -> None:
    print(f"[INFO] {message}")


def _print_warn(message: str) -> None:
    print(f"[WARN] {message}")


def _print_fallback(message: str) -> None:
    print(f"[FALLBACK] {message}")


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_window_dirs(windows_dir: str | Path) -> list[Path]:
    windows_dir = Path(windows_dir)
    return sorted([p for p in windows_dir.iterdir() if p.is_dir()])


def get_output_window_dir(window_dir: str | Path, *, write_in_place: bool, output_root: str | Path) -> Path:
    window_dir = Path(window_dir)
    if write_in_place:
        return window_dir
    return Path(output_root) / "windows" / window_dir.name


def load_diarization_pipeline_with_fallback(
    model_name: str,
    hf_token: str,
    device: str,
) -> tuple[Pipeline, dict[str, Any]]:
    resolved_device = resolve_device(device)
    metadata = {
        "resolved_device": resolved_device,
        "auth_mode": None,
        "auth_fallback_used": False,
        "gpu_requested": resolved_device == "cuda",
        "gpu_used": False,
    }

    try:
        _print_info(f"Loading pyannote pipeline with token=... | model={model_name}")
        pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
        metadata["auth_mode"] = "token"
    except TypeError as e:
        message = str(e)
        if "token" not in message and "unexpected keyword" not in message:
            raise
        metadata["auth_fallback_used"] = True
        metadata["auth_mode"] = "use_auth_token"
        _print_fallback(
            "Pipeline.from_pretrained(..., token=...) не поддерживается текущей версией стека. "
            "Переключаемся на use_auth_token=..."
        )
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)

    if resolved_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Запрошен device='cuda', но GPU недоступен в runtime.")
        pipeline.to(torch.device("cuda"))
        metadata["gpu_used"] = True
        _print_info("Pipeline moved to CUDA.")
    else:
        _print_warn("Pipeline will run on CPU.")

    return pipeline, metadata


def _annotation_to_segments(annotation: Any) -> tuple[list[dict[str, Any]], int]:
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


def load_audio_for_pyannote(audio_path: str | Path) -> dict[str, Any]:
    audio_path = str(audio_path)
    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T)  # [channels, time]

    if waveform.shape[0] > 1:
        _print_fallback("Multi-channel audio detected. Downmixing to mono.")
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        _print_fallback(f"Resampling audio from {sample_rate} Hz to 16000 Hz.")
        waveform = F.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    _print_fallback("Using in-memory audio input to bypass pyannote/torchcodec file decoding.")
    return {"waveform": waveform, "sample_rate": sample_rate}


def run_diarization_for_audio(
    audio_path: str | Path,
    model_name: str,
    hf_token: str,
    device: str = "auto",
    use_exclusive: bool = True,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict[str, Any]:
    audio_path = Path(audio_path)

    pipeline, load_meta = load_diarization_pipeline_with_fallback(
        model_name=model_name,
        hf_token=hf_token,
        device=device,
    )

    inference_kwargs: dict[str, Any] = {}
    if num_speakers is not None:
        inference_kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            inference_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            inference_kwargs["max_speakers"] = max_speakers

    _print_info(f"Running diarization for: {audio_path.name}")
    _print_info(f"inference kwargs: {inference_kwargs}")

    audio_input = load_audio_for_pyannote(audio_path)

    if ProgressHook is not None:
        with ProgressHook() as hook:
            output = pipeline(audio_input, hook=hook, **inference_kwargs)
    else:
        output = pipeline(audio_input, **inference_kwargs)

    regular_annotation = getattr(output, "speaker_diarization", output)
    regular_segments, regular_num_speakers = _annotation_to_segments(regular_annotation)

    if hasattr(output, "exclusive_speaker_diarization"):
        exclusive_annotation = output.exclusive_speaker_diarization
        exclusive_segments, exclusive_num_speakers = _annotation_to_segments(exclusive_annotation)
        exclusive_fallback_used = False
    else:
        _print_fallback(
            "exclusive_speaker_diarization недоступен в результате. "
            "Using regular diarization as exclusive fallback."
        )
        exclusive_segments = list(regular_segments)
        exclusive_num_speakers = regular_num_speakers
        exclusive_fallback_used = True

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

    return {
        "audio_path": str(audio_path),
        "model_name": model_name,
        "device": load_meta["resolved_device"],
        "load_auth_mode": load_meta["auth_mode"],
        "load_auth_fallback_used": load_meta["auth_fallback_used"],
        "used_exclusive_diarization": used_exclusive,
        "exclusive_fallback_used": exclusive_fallback_used,
        "selected_segments_mode": selected_segments_mode,
        "num_speakers_detected": selected_num_speakers,
        "num_speakers_detected_regular": regular_num_speakers,
        "num_speakers_detected_exclusive": exclusive_num_speakers,
        # backward-compatible selected output
        "segments": selected_segments,
        # full debug output
        "regular_segments": regular_segments,
        "exclusive_segments": exclusive_segments,
    }


def _window_result_row(
    *,
    window_id: str,
    status: str,
    audio_path: Path,
    transcript_path: Path,
    diarization_path: Path,
    payload: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    row = {
        "window_id": window_id,
        "status": status,
        "audio_path": str(audio_path),
        "transcript_exists": transcript_path.exists(),
        "diarization_output": str(diarization_path),
    }

    if payload is not None:
        regular_segments = payload.get("regular_segments", [])
        exclusive_segments = payload.get("exclusive_segments", [])
        selected_segments = payload.get("segments", [])
        row.update(
            {
                "num_segments": len(selected_segments),
                "num_regular_segments": len(regular_segments),
                "num_exclusive_segments": len(exclusive_segments),
                "regular_exclusive_segment_count_differs": len(regular_segments) != len(exclusive_segments),
                "selected_segments_mode": payload.get("selected_segments_mode"),
                "num_speakers_detected": payload.get("num_speakers_detected"),
                "num_speakers_detected_regular": payload.get("num_speakers_detected_regular"),
                "num_speakers_detected_exclusive": payload.get("num_speakers_detected_exclusive"),
                "regular_exclusive_speaker_count_differs": payload.get("num_speakers_detected_regular") != payload.get("num_speakers_detected_exclusive"),
                "load_auth_mode": payload.get("load_auth_mode"),
                "load_auth_fallback_used": payload.get("load_auth_fallback_used"),
                "used_exclusive_diarization": payload.get("used_exclusive_diarization"),
                "exclusive_fallback_used": payload.get("exclusive_fallback_used"),
            }
        )

    if error is not None:
        row.update(
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )

    return row


def run_batch_diarization(
    windows_dir: str | Path,
    *,
    output_root: str | Path,
    write_in_place: bool,
    model_name: str,
    hf_token: str,
    device: str = "auto",
    use_exclusive: bool = True,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    force_rerun: bool = False,
    limit: int | None = None,
    allowed_window_ids: set[str] | None = None,
) -> dict[str, Any]:
    windows = iter_window_dirs(windows_dir)
    if allowed_window_ids:
        windows = [p for p in windows if p.name in allowed_window_ids]
    if limit is not None:
        windows = windows[:limit]

    if not windows:
        raise RuntimeError("Не найдено ни одного окна для diarization.")

    results: list[dict[str, Any]] = []

    for index, window_dir in enumerate(windows, start=1):
        audio_path = window_dir / "audio.wav"
        transcript_path = window_dir / "transcript.json"
        output_window_dir = get_output_window_dir(
            window_dir,
            write_in_place=write_in_place,
            output_root=output_root,
        )
        diarization_path = output_window_dir / "diarization.json"

        _print_info(f"[{index}/{len(windows)}] window_id={window_dir.name}")

        if not audio_path.exists():
            _print_warn(f"audio.wav not found: {audio_path}")
            results.append(
                _window_result_row(
                    window_id=window_dir.name,
                    status="missing_audio",
                    audio_path=audio_path,
                    transcript_path=transcript_path,
                    diarization_path=diarization_path,
                )
            )
            continue

        if diarization_path.exists() and not force_rerun:
            _print_info(f"Skipping existing diarization: {diarization_path}")
            payload = load_json(diarization_path)
            results.append(
                _window_result_row(
                    window_id=window_dir.name,
                    status="skipped_existing",
                    audio_path=audio_path,
                    transcript_path=transcript_path,
                    diarization_path=diarization_path,
                    payload=payload,
                )
            )
            continue

        try:
            diarization_result = run_diarization_for_audio(
                audio_path=audio_path,
                model_name=model_name,
                hf_token=hf_token,
                device=device,
                use_exclusive=use_exclusive,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            save_json(diarization_result, diarization_path)
            results.append(
                _window_result_row(
                    window_id=window_dir.name,
                    status="ok",
                    audio_path=audio_path,
                    transcript_path=transcript_path,
                    diarization_path=diarization_path,
                    payload=diarization_result,
                )
            )
        except Exception as e:
            _print_warn(f"Diarization failed for {window_dir.name}: {e}")
            traceback.print_exc()
            results.append(
                _window_result_row(
                    window_id=window_dir.name,
                    status="error",
                    audio_path=audio_path,
                    transcript_path=transcript_path,
                    diarization_path=diarization_path,
                    error=e,
                )
            )

    summary = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "windows_dir": str(Path(windows_dir)),
        "write_in_place": bool(write_in_place),
        "output_root": str(Path(output_root)),
        "model_name": model_name,
        "device": device,
        "use_exclusive": bool(use_exclusive),
        "num_windows_requested": len(windows),
        "num_ok": sum(1 for row in results if row["status"] == "ok"),
        "num_skipped_existing": sum(1 for row in results if row["status"] == "skipped_existing"),
        "num_errors": sum(1 for row in results if row["status"] == "error"),
        "num_missing_audio": sum(1 for row in results if row["status"] == "missing_audio"),
        "windows": results,
    }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--write-in-place", action="store_true")
    parser.add_argument("--model-name", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-exclusive", action="store_true")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--single-window-id", default=None)
    parser.add_argument("--summary-path", default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN не найден в окружении.")

    allowed_window_ids = {args.single_window_id} if args.single_window_id else None

    summary = run_batch_diarization(
        windows_dir=args.windows_dir,
        output_root=args.output_root,
        write_in_place=args.write_in_place,
        model_name=args.model_name,
        hf_token=hf_token,
        device=args.device,
        use_exclusive=args.use_exclusive,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        force_rerun=args.force_rerun,
        limit=args.limit,
        allowed_window_ids=allowed_window_ids,
    )

    if args.summary_path:
        summary_path = Path(args.summary_path)
    else:
        # More intuitive default: summary goes to output_root,
        # even when diarization.json is written in-place into windows.
        summary_path = Path(args.output_root) / "diarization_run_summary.json"

    save_json(summary, summary_path)
    _print_info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
