from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from resemblyzer import VoiceEncoder

from .asr import prepare_audio, save_json as save_json_dict, transcribe_audio
from .diarization import run_diarization
from .pair_formatter import aggregate_window_predictions, build_extracted_pairs_dataframe, save_extracted_pairs
from .rule_based_intent import compile_lexicon, extract_lexicon_from_gold, load_jsonl, predict_for_records, save_json as save_json_generic
from .speaker_id import (
    apply_assignments_to_utterances,
    assign_speakers_to_characters,
    build_character_embeddings,
    build_speaker_embeddings,
    discover_sample_groups,
    save_json as save_json_speaker,
    save_jsonl as save_jsonl_speaker,
)
from .utterance_builder import (
    build_stats as build_utterance_stats,
    build_utterances_from_units,
    extract_timed_units_from_asr,
    normalize_diarization_segments,
    save_jsonl as save_jsonl_utterances,
)
from .validation_io import build_gold_dataframe_for_evaluation, load_validation_windows, save_dataframe_to_excel


MEDIA_EXTENSIONS = [".mkv", ".mp4", ".mov", ".avi", ".ac3", ".wav", ".flac", ".mp3", ".m4a", ".aac", ".dts"]


def dump_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def auto_detect_media_input(validation_dir: str | Path) -> Path | None:
    validation_dir = Path(validation_dir)
    if not validation_dir.exists():
        return None

    all_candidates: list[Path] = []
    for ext in MEDIA_EXTENSIONS:
        all_candidates.extend(sorted(validation_dir.glob(f"*{ext}")))

    if not all_candidates:
        return None

    prioritized = sorted(
        all_candidates,
        key=lambda p: (
            0 if "статус" in p.name.lower() else 1,
            0 if p.suffix.lower() in {".ac3", ".wav", ".flac", ".m4a"} else 1,
            p.name.lower(),
        ),
    )
    return prioritized[0]


def auto_detect_samples_dir(validation_dir: str | Path) -> Path | None:
    validation_dir = Path(validation_dir)
    if not validation_dir.exists():
        return None

    candidate_names = ["audio_profiles", "samples", "profiles"]
    for name in candidate_names:
        candidate = validation_dir / name
        if candidate.exists() and candidate.is_dir():
            return candidate

    subdirs = sorted(p for p in validation_dir.iterdir() if p.is_dir())
    for subdir in subdirs:
        if discoverable_audio_count(subdir) > 0:
            return subdir

    return None


def discoverable_audio_count(path: str | Path) -> int:
    path = Path(path)
    return sum(1 for fp in path.rglob("*") if fp.is_file() and fp.suffix.lower() in MEDIA_EXTENSIONS)


def enrich_utterances_with_window_metadata(
    utterances: list[dict[str, Any]],
    window: dict[str, Any],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for utt in utterances:
        item = dict(utt)
        item["source_window_id"] = window["window_id"]
        item["source_window_row_id"] = window["row_id"]
        item["source_film"] = window.get("film")
        item["source_start_sec"] = window.get("start_sec")
        item["source_end_sec"] = window.get("end_sec")
        item["dialogue_id"] = window["window_id"]
        enriched.append(item)
    return enriched


def run_validation_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    windows_dir = output_dir / "windows"
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_dir.mkdir(parents=True, exist_ok=True)

    windows = load_validation_windows(args.gold_excel, sheet_name=args.validation_sheet)
    if args.limit is not None:
        windows = windows[: args.limit]

    if not windows:
        raise RuntimeError("Не найдено ни одного validation-окна.")

    media_input = Path(args.media_input) if args.media_input else auto_detect_media_input(args.validation_dir)
    if media_input is None or not media_input.exists():
        raise FileNotFoundError(
            "Не найден media input для validation-фильма. Передайте --media-input "
            "или положите файл в data/raw/validation."
        )

    samples_dir = Path(args.samples_dir) if args.samples_dir else auto_detect_samples_dir(args.validation_dir)
    if samples_dir is None or not samples_dir.exists():
        raise FileNotFoundError(
            "Не найдена папка audio_profiles. Передайте --samples-dir "
            "или положите её в data/raw/validation/audio_profiles."
        )

    fit_records = load_jsonl(args.fit_input)
    lexicon = extract_lexicon_from_gold(fit_records, min_freq=args.min_freq)
    compiled_lexicon = compile_lexicon(lexicon)
    save_json_generic(lexicon, output_dir / "rule_lexicon.json")

    sample_groups = discover_sample_groups(samples_dir)
    encoder = VoiceEncoder()
    character_embeddings, character_profiles = build_character_embeddings(
        sample_groups=sample_groups,
        encoder=encoder,
        min_sample_duration_sec=args.min_sample_duration_sec,
    )
    save_json_speaker(character_profiles, output_dir / "character_profiles.json")

    all_output_rows: list[dict[str, Any]] = []
    all_window_summaries: list[dict[str, Any]] = []

    for index, window in enumerate(windows, start=1):
        window_dir = windows_dir / window["window_id"]
        window_dir.mkdir(parents=True, exist_ok=True)

        effective_start = max(0.0, float(window["start_sec"]) - float(args.window_padding_sec))
        effective_end = float(window["end_sec"]) + float(args.window_padding_sec)
        duration_sec = effective_end - effective_start

        prepared_audio_path = window_dir / "audio.wav"
        transcript_path = window_dir / "transcript.json"
        diarization_path = window_dir / "diarization.json"
        utterances_path = window_dir / "utterances.jsonl"
        utterances_named_path = window_dir / "utterances_named.jsonl"
        assignments_path = window_dir / "speaker_assignments.json"
        predictions_path = window_dir / "predictions.jsonl"
        summary_path = window_dir / "summary.json"

        prepare_audio(
            media_input=media_input,
            audio_output=prepared_audio_path,
            start_sec=effective_start,
            duration_sec=duration_sec,
            sample_rate=args.sample_rate,
            channels=args.channels,
        )

        transcript = transcribe_audio(
            audio_path=prepared_audio_path,
            model_name=args.asr_model_name,
            device=args.device,
            compute_type=args.compute_type,
            batch_size=args.batch_size,
            perform_alignment=not args.skip_alignment,
        )
        save_json_dict(transcript, transcript_path)

        diarization_result = run_diarization(
            audio_path=prepared_audio_path,
            model_name=args.diarization_model_name,
            hf_token=args.hf_token,
            device=args.device,
            use_exclusive=not args.disable_exclusive_diarization,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        save_json_dict(diarization_result, diarization_path)

        diarization_segments = normalize_diarization_segments(diarization_result)
        timed_units, used_word_level = extract_timed_units_from_asr(
            asr_data=transcript,
            diarization_segments=diarization_segments,
            unknown_speaker_label=args.unknown_speaker_label,
            max_nonoverlap_assign_distance_sec=args.max_nonoverlap_assign_distance_sec,
        )

        utterances = build_utterances_from_units(
            timed_units=timed_units,
            max_pause_sec=args.max_pause_within_utterance_sec,
        )
        utterances = enrich_utterances_with_window_metadata(utterances, window)
        save_jsonl_utterances(utterances, utterances_path)

        speaker_embeddings, speaker_profiles = build_speaker_embeddings(
            audio_input=prepared_audio_path,
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
            top_k=args.top_k_candidates,
        )
        utterances_named = apply_assignments_to_utterances(
            utterances=utterances,
            assignments=assignments,
            unknown_speaker_label=args.unknown_speaker_label,
        )
        save_jsonl_speaker(utterances_named, utterances_named_path)
        save_json_speaker(assignments, assignments_path)

        predictions = predict_for_records(
            records=utterances_named,
            compiled_lexicon=compiled_lexicon,
        )
        save_jsonl_speaker(predictions, predictions_path)

        row = aggregate_window_predictions(window, predictions, unknown_speaker_name=args.unknown_speaker_name)
        all_output_rows.append(row)

        utterance_stats = build_utterance_stats(
            asr_data=transcript,
            diarization_segments=diarization_segments,
            timed_units=timed_units,
            utterances=utterances,
            used_word_level_mode=used_word_level,
            unknown_speaker_label=args.unknown_speaker_label,
            max_pause_sec=args.max_pause_within_utterance_sec,
        )

        window_summary = {
            "index": index,
            "window_id": window["window_id"],
            "row_id": window["row_id"],
            "film": window.get("film"),
            "start_sec": window["start_sec"],
            "end_sec": window["end_sec"],
            "effective_start_sec": effective_start,
            "effective_end_sec": effective_end,
            "duration_sec": window["duration_sec"],
            "num_timed_units": len(timed_units),
            "num_utterances": len(utterances),
            "num_predictions": len(predictions),
            "num_assignments": len(assignments),
            "used_word_level_alignment": used_word_level,
            "predicted_opening": row.get("opening"),
            "predicted_closing": row.get("closing"),
            "utterance_stats": utterance_stats,
            "speaker_profiles": speaker_profiles,
        }
        dump_json(summary_path, window_summary)
        all_window_summaries.append(window_summary)

    extracted_pairs_df = build_extracted_pairs_dataframe(all_output_rows)
    extracted_pairs_path = save_extracted_pairs(extracted_pairs_df, args.extracted_pairs_output)

    gold_df = build_gold_dataframe_for_evaluation(windows)
    gold_output_path = save_dataframe_to_excel(gold_df, args.gold_output)

    overall_summary = {
        "num_windows": len(windows),
        "media_input": str(media_input),
        "samples_dir": str(samples_dir),
        "fit_input": str(Path(args.fit_input)),
        "character_profiles_count": len(character_profiles),
        "output_dir": str(output_dir),
        "extracted_pairs_output": str(extracted_pairs_path),
        "gold_output": str(gold_output_path),
        "window_summaries": all_window_summaries,
    }
    dump_json(output_dir / "run_summary.json", overall_summary)
    return overall_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation pipeline под новый формат оценки (opening/closing speaker-phrase pairs)."
    )
    parser.add_argument("--gold-excel", type=str, default="data/raw/gold/Данные_.xlsx")
    parser.add_argument("--validation-sheet", type=str, default="Вал - Статус свободен")
    parser.add_argument("--fit-input", type=str, default="data/processed/gold_dialogues.jsonl")
    parser.add_argument("--validation-dir", type=str, default="data/raw/validation")
    parser.add_argument("--media-input", type=str, default=None)
    parser.add_argument("--samples-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="artifacts/validation_status_svoboden")
    parser.add_argument("--extracted-pairs-output", type=str, default="artifacts/validation_status_svoboden/extracted_pairs.xlsx")
    parser.add_argument("--gold-output", type=str, default="artifacts/validation_status_svoboden/gold.xlsx")

    parser.add_argument("--asr-model-name", type=str, default="medium")
    parser.add_argument("--diarization-model-name", type=str, default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute-type", type=str, default="auto", choices=["auto", "float16", "int8", "float32"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-alignment", action="store_true")
    parser.add_argument("--disable-exclusive-diarization", action="store_true")

    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--window-padding-sec", type=float, default=0.0)
    parser.add_argument("--max-pause-within-utterance-sec", type=float, default=0.8)
    parser.add_argument("--max-total-utterance-duration-sec", type=float, default=20.0)
    parser.add_argument("--max-nonoverlap-assign-distance-sec", type=float, default=1.0)

    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=4)

    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--min-sample-duration-sec", type=float, default=0.5)
    parser.add_argument("--min-utterance-duration-sec", type=float, default=0.7)
    parser.add_argument("--min-total-duration-sec", type=float, default=1.5)
    parser.add_argument("--max-total-duration-sec", type=float, default=45.0)
    parser.add_argument("--similarity-threshold", type=float, default=0.65)
    parser.add_argument("--top-k-candidates", type=int, default=3)
    parser.add_argument("--unknown-speaker-label", type=str, default="unknown_speaker")
    parser.add_argument("--unknown-speaker-name", type=str, default="unknown")
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_token is None:
        args.hf_token = os.getenv("HF_TOKEN")

    if not args.hf_token:
        raise ValueError(
            "Для diarization нужен Hugging Face token. Передайте --hf-token "
            "или задайте переменную окружения HF_TOKEN."
        )

    summary = run_validation_pipeline(args)
    print(f"Validation pipeline завершён. Окон с обработкой: {summary['num_windows']}")
    print(f"Gold сохранён в: {summary['gold_output']}")
    print(f"Predictions сохранены в: {summary['extracted_pairs_output']}")


if __name__ == "__main__":
    main()
