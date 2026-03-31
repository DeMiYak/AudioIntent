from __future__ import annotations

"""Splits a film into fixed-duration audio chunks for the test pipeline.

Creates a windows/ directory structure compatible with the ASR and diarization
Colab notebooks, then with the post-processing pipeline (--scan-windows flag).

Usage:
  python -m src.chunk_film \\
    --input data/raw/test/Peter_FM_2006.mkv \\
    --output-dir artifacts/test_piter_fm_asr_colab/windows \\
    --chunk-duration 600

Output:
  artifacts/test_piter_fm_asr_colab/windows/
    chunk_001/
      audio.wav        (16 kHz mono WAV)
      chunk_info.json  (start_sec, end_sec, window_id, ...)
    chunk_002/
      ...
"""

import argparse
import json
import subprocess
from pathlib import Path


def get_duration_sec(input_path: Path) -> float:
    """Returns media file duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")
    return float(result.stdout.strip())


def extract_chunk(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float,
    sample_rate: int = 16000,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_sec),
        "-i", str(input_path),
        "-t", str(duration_sec),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Split a film into fixed-duration audio chunks for the test pipeline."
    )
    parser.add_argument("--input", required=True, help="Path to video or audio file")
    parser.add_argument(
        "--output-dir", required=True,
        help="Root directory for chunk windows (e.g. artifacts/test_piter_fm_asr_colab/windows)",
    )
    parser.add_argument(
        "--chunk-duration", type=float, default=600.0,
        help="Duration of each chunk in seconds (default: 600 = 10 min)",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.0,
        help="Overlap between consecutive chunks in seconds (default: 0)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Sample rate for output audio.wav (default: 16000)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract audio.wav even if it already exists",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_duration = get_duration_sec(input_path)
    chunk_dur = args.chunk_duration
    step = chunk_dur - args.overlap

    starts: list[float] = []
    t = 0.0
    while t < total_duration:
        starts.append(t)
        t += step

    print(f"Film    : {input_path.name}")
    print(f"Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Chunk   : {chunk_dur}s, overlap: {args.overlap}s")
    print(f"Chunks  : {len(starts)}")
    print()

    for idx, start_sec in enumerate(starts, start=1):
        end_sec = min(start_sec + chunk_dur, total_duration)
        actual_dur = end_sec - start_sec
        window_id = f"chunk_{idx:03d}"
        window_dir = output_dir / window_id
        audio_path = window_dir / "audio.wav"
        info_path = window_dir / "chunk_info.json"

        chunk_info = {
            "window_id": window_id,
            "chunk_idx": idx,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "duration_sec": round(actual_dur, 3),
            "film": input_path.name,
        }

        if audio_path.exists() and not args.force:
            print(f"  {window_id}: {start_sec:.1f}s – {end_sec:.1f}s  SKIP (audio.wav exists)")
            # Still update chunk_info.json in case it's missing
            info_path.write_text(json.dumps(chunk_info, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        print(f"  {window_id}: {start_sec:.1f}s – {end_sec:.1f}s ({actual_dur:.1f}s) ...", end=" ", flush=True)
        extract_chunk(input_path, audio_path, start_sec, actual_dur, sample_rate=args.sample_rate)
        info_path.write_text(json.dumps(chunk_info, ensure_ascii=False, indent=2), encoding="utf-8")
        print("OK")

    print(f"\nDone. {len(starts)} chunks written to {output_dir}")


if __name__ == "__main__":
    main()
