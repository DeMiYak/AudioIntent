from __future__ import annotations

"""
Helper for manually creating voice profile samples for a character.

For each character you want to identify in the test film:
  1. Watch the film, note timestamps where the character speaks alone and clearly.
  2. Run this script once per segment (or pass multiple --segment entries).

Usage examples:

  # Single segment
  python -m src.extract_audio_profile \
    --input data/raw/test/piter_fm.mkv \
    --character "Маша" \
    --segment 00:04:12 00:04:20 \
    --output-dir data/raw/test/audio_profiles

  # Multiple segments at once
  python -m src.extract_audio_profile \
    --input data/raw/test/piter_fm.mkv \
    --character "Маша" \
    --segment 00:04:12 00:04:20 \
    --segment 00:12:05 00:12:15 \
    --segment 01:02:33 01:02:45 \
    --output-dir data/raw/test/audio_profiles

Output structure (Variant A — subfolder per character):
  data/raw/test/audio_profiles/
    Маша/
      01.wav
      02.wav
    Максим/
      01.wav

Tips for good profiles:
  - Pick segments where the character speaks alone (no overlapping voices).
  - Aim for 5–15 seconds of clean speech per segment.
  - 3–5 segments per character is usually enough.
  - Avoid music, noise, or very short utterances (< 2 sec).
"""

import argparse
import subprocess
from pathlib import Path


def hms_to_seconds(ts: str) -> float:
    """Converts HH:MM:SS or MM:SS or plain seconds string to float seconds."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(ts)


def extract_segment(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError(f"end must be after start: {start_sec} -> {end_sec}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", str(start_sec),
        "-i", str(input_path),
        "-t", str(duration),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed:\n{result.stderr}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract voice profile samples for a character from a video/audio file."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the video or audio file (mkv, mp4, ac3, wav, etc.)",
    )
    parser.add_argument(
        "--character", required=True,
        help="Character name (used as subfolder name, e.g. 'Маша')",
    )
    parser.add_argument(
        "--segment", nargs=2, metavar=("START", "END"),
        action="append", dest="segments", default=[],
        help="Start and end timestamps (HH:MM:SS or seconds). Repeat for multiple segments.",
    )
    parser.add_argument(
        "--output-dir", default="data/raw/test/audio_profiles",
        help="Root output directory for audio profiles (default: data/raw/test/audio_profiles)",
    )
    args = parser.parse_args(argv)

    if not args.segments:
        parser.error("Provide at least one --segment START END")

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    char_dir = Path(args.output_dir) / args.character
    char_dir.mkdir(parents=True, exist_ok=True)

    # Find next available index to avoid overwriting existing samples
    existing = sorted(char_dir.glob("*.wav"))
    next_idx = len(existing) + 1

    print(f"Character : {args.character}")
    print(f"Output dir: {char_dir}")
    print()

    for i, (start_str, end_str) in enumerate(args.segments):
        start_sec = hms_to_seconds(start_str)
        end_sec = hms_to_seconds(end_str)
        duration = end_sec - start_sec
        out_file = char_dir / f"{next_idx + i:02d}.wav"

        print(f"  Segment {i+1}: {start_str} -> {end_str} ({duration:.1f}s) -> {out_file.name}")
        extract_segment(input_path, out_file, start_sec, end_sec)
        print(f"    OK")

    total = len(list(char_dir.glob("*.wav")))
    print(f"\nDone. {args.character} now has {total} sample(s) in {char_dir}")


if __name__ == "__main__":
    main()
