from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .asr import load_json as load_json_dict
from .asr import prepare_audio, save_json as save_json_dict, transcribe_audio
from .pair_formatter import build_extracted_pairs_dataframe, save_extracted_pairs, aggregate_window_predictions
from .rule_based_intent import (
    compile_lexicon,
    extract_lexicon_from_gold,
    load_jsonl,
    predict_for_records as rbi_predict_for_records,
    save_json as save_json_generic,
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


def select_diarization_segments(
    diarization_result: dict[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    """
    Выбирает нужный вид сегментов из diarization.json.

    mode='auto'      -> diarization_result["segments"]  (обычно == exclusive)
    mode='regular'   -> diarization_result["regular_segments"] (fallback: segments)
    mode='exclusive' -> diarization_result["exclusive_segments"] (fallback: segments)
    """
    fallback = diarization_result.get("segments", [])
    if mode == "regular":
        return diarization_result.get("regular_segments", fallback)
    if mode == "exclusive":
        return diarization_result.get("exclusive_segments", fallback)
    return fallback


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
        item["utterance_id"] = f"{window['window_id']}_utt_{len(enriched)+1:03d}"
        enriched.append(item)
    return enriched


def find_existing_stage_file(
    *,
    current_window_dir: Path,
    filename: str,
    input_dir: str | Path | None,
    window_id: str,
) -> Path | None:
    default_candidate = current_window_dir / filename
    if default_candidate.exists():
        return default_candidate

    if input_dir is None:
        return None

    base_dir = Path(input_dir)
    candidates = [
        base_dir / window_id / filename,
        base_dir / filename,
        base_dir / f"{window_id}_{filename}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def stage_mode_flags(args: argparse.Namespace) -> dict[str, bool]:
    if args.only_asr and args.only_diarization:
        raise ValueError("Нельзя одновременно передать --only-asr и --only-diarization.")

    do_asr = not args.skip_asr and not args.only_diarization
    do_diarization = not args.skip_diarization and not args.only_asr
    full_postprocess = not args.only_asr and not args.only_diarization
    return {
        "do_asr": do_asr,
        "do_diarization": do_diarization,
        "full_postprocess": full_postprocess,
    }


_CLOSE_LEXICAL_KW = frozenset(["до свидания", "пока", "прощай", "увидимся", "созвонимся"])
_OPEN_LEXICAL_KW = frozenset(["познакомимся", "знакомиться", "меня зовут", "здравствуйте"])
_CONF_MARGIN = 0.05


def resolve_intent_conflicts(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Гарантирует, что у каждого utterance_id есть не более одного типа (open/close).

    Алгоритм разрешения конфликта:
    1. Сравнить максимальную уверенность каждого типа; выбрать более уверенный,
       если разница > _CONF_MARGIN.
    2. Иначе применить лексические жёсткие переопределения (закрывающие слова
       имеют приоритет над открывающими при равной уверенности).
    3. Если ни одно правило не решает конфликт — отбросить оба.
    """
    by_uid: dict[str, list[dict[str, Any]]] = {}
    for pred in predictions:
        uid = str(pred.get("utterance_id", ""))
        by_uid.setdefault(uid, []).append(pred)

    result: list[dict[str, Any]] = []
    for uid, preds in by_uid.items():
        types = {str(p.get("intent_type", "")) for p in preds}
        if "contact_open" not in types or "contact_close" not in types:
            result.extend(preds)
            continue

        open_preds = [p for p in preds if str(p.get("intent_type", "")) == "contact_open"]
        close_preds = [p for p in preds if str(p.get("intent_type", "")) == "contact_close"]
        open_conf = max(float(p.get("confidence", 0.5)) for p in open_preds)
        close_conf = max(float(p.get("confidence", 0.5)) for p in close_preds)

        if open_conf - close_conf > _CONF_MARGIN:
            keep = "contact_open"
        elif close_conf - open_conf > _CONF_MARGIN:
            keep = "contact_close"
        else:
            text = str((open_preds + close_preds)[0].get("source_text", "")).lower()
            has_close = any(kw in text for kw in _CLOSE_LEXICAL_KW)
            has_open = any(kw in text for kw in _OPEN_LEXICAL_KW)
            if has_close and not has_open:
                keep = "contact_close"
            elif has_open and not has_close:
                keep = "contact_open"
            else:
                keep = None  # неоднозначно — отбросить оба

        if keep is not None:
            for p in preds:
                if str(p.get("intent_type", "")) == keep:
                    p_copy = dict(p)
                    p_copy["conflict_resolution"] = keep
                    result.append(p_copy)

    return result


def _run_intent_extraction(
    utterances: list[dict[str, Any]],
    intent_mode: str,
    compiled_lexicon: dict[str, Any] | None,
    ml_model: Any | None,
    ml_confidence_threshold: float = 0.5,
    open_threshold: float | None = None,
    close_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """
    Dispatches intent extraction to rule-based, ML, or combined mode.

    intent_mode: 'rule' | 'ml' | 'combined'
    Combined = union of both; rule-based takes priority on duplicate utterance+intent.
    ml_confidence_threshold: fallback minimum confidence (both labels).
    open_threshold / close_threshold: per-label overrides (take precedence).
    """
    from .ml_intent import predict_for_records as ml_predict_for_records

    _open_thr = open_threshold if open_threshold is not None else ml_confidence_threshold
    _close_thr = close_threshold if close_threshold is not None else ml_confidence_threshold

    def _passes_threshold(pred: dict[str, Any]) -> bool:
        conf = float(pred.get("confidence", 1.0))
        intent = str(pred.get("intent_type", ""))
        if intent == "contact_open":
            return conf >= _open_thr
        if intent == "contact_close":
            return conf >= _close_thr
        return conf >= ml_confidence_threshold

    if intent_mode == "rule":
        preds = rbi_predict_for_records(
            records=utterances,
            compiled_lexicon=compiled_lexicon or {},
        )
        return resolve_intent_conflicts(preds)

    if intent_mode == "ml":
        if ml_model is None:
            raise RuntimeError("--intent-mode ml requires --ml-model to be provided")
        preds = ml_predict_for_records(records=utterances, model=ml_model)
        preds = [p for p in preds if _passes_threshold(p)]
        return resolve_intent_conflicts(preds)

    # combined: rule-based фильтруется ML-согласием; ML добавляет непокрытые случаи
    rbi_preds = rbi_predict_for_records(
        records=utterances,
        compiled_lexicon=compiled_lexicon or {},
    )
    if ml_model is None:
        return rbi_preds
    ml_preds = ml_predict_for_records(records=utterances, model=ml_model)
    ml_preds = [p for p in ml_preds if _passes_threshold(p)]

    # Множество пар (utterance_id, intent_type), одобренных ML
    ml_agreed: set[tuple[str, str]] = {
        (str(p.get("utterance_id", "")), str(p.get("intent_type", "")))
        for p in ml_preds
    }

    # Rule-based пропускаются только если ML согласен с тем же типом для той же реплики.
    # Исключение: только Tier A освобождён от фильтра — однозначные паттерны без риска FP.
    # Tier B и Tier C обязательно проходят ML-фильтр.
    from .rule_based_intent import MANUAL_RULES_TIER_A
    manual_rule_names = MANUAL_RULES_TIER_A

    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []
    for pred in rbi_preds:
        key = (str(pred.get("utterance_id", "")), str(pred.get("intent_type", "")))
        rule_expr = str(pred.get("rule_expression", ""))
        is_manual = rule_expr in manual_rule_names
        if not is_manual and key not in ml_agreed:
            continue
        seen.add(key)
        merged.append(pred)
    for pred in ml_preds:
        key = (str(pred.get("utterance_id", "")), str(pred.get("intent_type", "")))
        if key not in seen:
            seen.add(key)
            merged.append(pred)

    return resolve_intent_conflicts(merged)


def run_validation_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    modes = stage_mode_flags(args)

    output_dir = Path(args.output_dir)
    windows_dir = output_dir / "windows"
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_dir.mkdir(parents=True, exist_ok=True)

    if args.scan_windows:
        # Режим тестового фильма: сканировать существующие директории чанков вместо чтения gold Excel.
        scan_root = Path(args.transcript_input_dir) if args.transcript_input_dir else windows_dir
        chunk_dirs = sorted(p for p in scan_root.iterdir() if p.is_dir()) if scan_root.exists() else []
        windows = []
        for chunk_dir in chunk_dirs:
            info_path = chunk_dir / "chunk_info.json"
            if info_path.exists():
                with info_path.open(encoding="utf-8") as _f:
                    info = json.load(_f)
                windows.append({
                    "window_id": info["window_id"],
                    "row_id": info["chunk_idx"],
                    "start_sec": info["start_sec"],
                    "end_sec": info["end_sec"],
                    "duration_sec": info["duration_sec"],
                    "film": info.get("film"),
                })
            else:
                windows.append({
                    "window_id": chunk_dir.name,
                    "row_id": len(windows) + 1,
                    "start_sec": 0.0,
                    "end_sec": 0.0,
                    "duration_sec": 0.0,
                    "film": None,
                })
        gold_output_path = None
    else:
        windows = load_validation_windows(args.gold_excel, sheet_name=args.validation_sheet)
        gold_df = build_gold_dataframe_for_evaluation(windows)
        gold_output_path = save_dataframe_to_excel(gold_df, args.gold_output)

    if args.limit is not None:
        windows = windows[: args.limit]

    if not windows:
        raise RuntimeError("Не найдено ни одного validation-окна.")

    media_input = None
    if not args.scan_windows:
        media_input = Path(args.media_input) if args.media_input else auto_detect_media_input(args.validation_dir)
        if media_input is None or not media_input.exists():
            raise FileNotFoundError(
                "Не найден media input для validation-фильма. Передайте --media-input "
                "или положите файл в data/raw/validation."
            )
    elif args.media_input:
        media_input = Path(args.media_input)

    lexicon: dict[str, list[dict[str, Any]]] | None = None
    compiled_lexicon: dict[str, Any] | None = None
    ml_model = None
    encoder = None
    character_embeddings = None
    character_profiles: list[dict[str, Any]] = []
    samples_dir: Path | None = None

    if modes["full_postprocess"]:
        fit_records = load_jsonl(args.fit_input)
        lexicon = extract_lexicon_from_gold(fit_records, min_freq=args.min_freq)
        compiled_lexicon = compile_lexicon(lexicon)
        save_json_generic(lexicon, output_dir / "rule_lexicon.json")

        if args.intent_mode in ("ml", "combined") and args.ml_model:
            from .ml_intent import load_model as load_ml_model
            ml_model = load_ml_model(args.ml_model)

        samples_dir = Path(args.samples_dir) if args.samples_dir else auto_detect_samples_dir(args.validation_dir)
        if samples_dir is None or not samples_dir.exists():
            raise FileNotFoundError(
                "Не найдена папка audio_profiles. Передайте --samples-dir "
                "или положите её в data/raw/validation/audio_profiles."
            )

        from .speaker_id import (
            VoiceEncoder,
            build_character_embeddings,
            discover_sample_groups,
            save_json as save_json_speaker,
        )

        encoder = VoiceEncoder()
        sample_groups = discover_sample_groups(samples_dir)
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

        if args.scan_windows:
            source_audio = find_existing_stage_file(
                current_window_dir=window_dir,
                filename="audio.wav",
                input_dir=args.transcript_input_dir,
                window_id=window["window_id"],
            )
            prepared_audio_path = source_audio if source_audio is not None else window_dir / "audio.wav"
        else:
            prepared_audio_path = window_dir / "audio.wav"
        transcript_path = window_dir / "transcript.json"
        diarization_path = window_dir / "diarization.json"
        utterances_path = window_dir / "utterances.jsonl"
        utterances_named_path = window_dir / "utterances_named.jsonl"
        assignments_path = window_dir / "speaker_assignments.json"
        predictions_path = window_dir / "predictions.jsonl"
        summary_path = window_dir / "summary.json"

        if args.scan_windows:
            pass  # аудио уже создано chunk_film.py, находится в transcript-input-dir
        else:
            prepare_audio(
                media_input=media_input,
                audio_output=prepared_audio_path,
                start_sec=effective_start,
                duration_sec=duration_sec,
                sample_rate=args.sample_rate,
                channels=args.channels,
            )

        transcript: dict[str, Any] | None = None
        diarization_result: dict[str, Any] | None = None

        transcript_source = None
        if modes["do_asr"]:
            transcript = transcribe_audio(
                audio_path=prepared_audio_path,
                model_name=args.asr_model_name,
                device=args.device,
                compute_type=args.compute_type,
                batch_size=args.batch_size,
                perform_alignment=not args.skip_alignment,
            )
            save_json_dict(transcript, transcript_path)
            transcript_source = "generated"
        else:
            existing_transcript = find_existing_stage_file(
                current_window_dir=window_dir,
                filename="transcript.json",
                input_dir=args.transcript_input_dir,
                window_id=window["window_id"],
            )
            if existing_transcript is not None:
                transcript = load_json_dict(existing_transcript)
                transcript_source = str(existing_transcript)

        if args.only_asr:
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
                "stage": "asr_only",
                "transcript_source": transcript_source,
                "transcript_output": str(transcript_path) if transcript_path.exists() else None,
            }
            dump_json(summary_path, window_summary)
            all_window_summaries.append(window_summary)
            continue

        diarization_source = None
        if modes["do_diarization"]:
            from .diarization import run_diarization

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
            diarization_source = "generated"
        else:
            existing_diarization = find_existing_stage_file(
                current_window_dir=window_dir,
                filename="diarization.json",
                input_dir=args.diarization_input_dir,
                window_id=window["window_id"],
            )
            if existing_diarization is not None:
                diarization_result = load_json_dict(existing_diarization)
                diarization_source = str(existing_diarization)

        if args.only_diarization:
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
                "stage": "diarization_only",
                "diarization_source": diarization_source,
                "diarization_output": str(diarization_path) if diarization_path.exists() else None,
            }
            dump_json(summary_path, window_summary)
            all_window_summaries.append(window_summary)
            continue

        if transcript is None:
            raise FileNotFoundError(
                f"Для окна {window['window_id']} не найден transcript.json. "
                "Либо не пропускайте ASR, либо передайте --transcript-input-dir."
            )
        if diarization_result is None:
            raise FileNotFoundError(
                f"Для окна {window['window_id']} не найден diarization.json. "
                "Либо не пропускайте diarization, либо передайте --diarization-input-dir."
            )
        if encoder is None or compiled_lexicon is None or character_embeddings is None:
            raise RuntimeError("Для полного постпроцессинга не инициализированы lexicon/encoder/character embeddings.")

        from .speaker_id import (
            apply_assignments_to_utterances,
            assign_speakers_to_characters,
            build_speaker_embeddings,
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

        selected_raw_segments = select_diarization_segments(
            diarization_result, args.diarization_segment_mode
        )
        diarization_segments = normalize_diarization_segments(
            {"segments": selected_raw_segments}
        )
        timed_units, used_word_level = extract_timed_units_from_asr(
            asr_data=transcript,
            diarization_segments=diarization_segments,
            unknown_speaker_label=args.unknown_speaker_label,
            max_nonoverlap_assign_distance_sec=args.max_nonoverlap_assign_distance_sec,
        )

        utterances = build_utterances_from_units(
            timed_units=timed_units,
            max_pause_sec=args.max_pause_within_utterance_sec,
            max_total_duration_sec=args.max_total_utterance_duration_sec,
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
            similarity_margin_threshold=args.similarity_margin_threshold,
            top_k=args.top_k_candidates,
            score_mode=args.speaker_score_mode,
        )
        utterances_named = apply_assignments_to_utterances(
            utterances=utterances,
            assignments=assignments,
            unknown_speaker_label=args.unknown_speaker_label,
        )
        save_jsonl_speaker(utterances_named, utterances_named_path)
        save_json_speaker(assignments, assignments_path)

        predictions = _run_intent_extraction(
            utterances=utterances_named,
            intent_mode=args.intent_mode,
            compiled_lexicon=compiled_lexicon,
            ml_model=ml_model,
            ml_confidence_threshold=args.ml_confidence_threshold,
            open_threshold=args.open_threshold,
            close_threshold=args.close_threshold,
        )

        if args.use_sequence_decoder and predictions:
            from .sequence_decode import viterbi_decode
            predictions = viterbi_decode(utterances_named, predictions)

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
            "diarization_segment_mode": args.diarization_segment_mode,
            "predicted_opening": row.get("opening"),
            "predicted_closing": row.get("closing"),
            "transcript_source": transcript_source,
            "diarization_source": diarization_source,
            "utterance_stats": utterance_stats,
            "speaker_profiles": speaker_profiles,
        }
        dump_json(summary_path, window_summary)
        all_window_summaries.append(window_summary)

    extracted_pairs_path = None
    if modes["full_postprocess"]:
        extracted_pairs_df = build_extracted_pairs_dataframe(all_output_rows)
        extracted_pairs_path = save_extracted_pairs(extracted_pairs_df, args.extracted_pairs_output)

    overall_summary = {
        "num_windows": len(windows),
        "media_input": str(media_input),
        "samples_dir": str(samples_dir) if samples_dir is not None else None,
        "fit_input": str(Path(args.fit_input)) if modes["full_postprocess"] else None,
        "character_profiles_count": len(character_profiles),
        "output_dir": str(output_dir),
        "stage_mode": (
            "asr_only" if args.only_asr else "diarization_only" if args.only_diarization else "full_pipeline"
        ),
        "extracted_pairs_output": str(extracted_pairs_path) if extracted_pairs_path is not None else None,
        "gold_output": str(gold_output_path) if gold_output_path is not None else None,
        "window_summaries": all_window_summaries,
        "args": vars(args),
    }
    dump_json(output_dir / "run_summary.json", overall_summary)
    return overall_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation pipeline под новый формат оценки (opening/closing speaker-phrase pairs)."
    )
    parser.add_argument("--gold-excel", type=str, default="data/raw/gold/data_val.xlsx")
    parser.add_argument("--validation-sheet", type=str, default="Вал - Статус свободен")
    parser.add_argument("--fit-input", type=str, default="data/processed/gold_dialogues.jsonl")
    parser.add_argument("--validation-dir", type=str, default="data/raw/validation")
    parser.add_argument("--media-input", type=str, default=None)
    parser.add_argument("--samples-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="artifacts/validation_status_svoboden_asr_diarization")
    parser.add_argument(
        "--extracted-pairs-output",
        type=str,
        default="artifacts/validation_status_svoboden_asr_diarization/extracted_pairs.xlsx",
    )
    parser.add_argument("--gold-output", type=str, default="artifacts/validation_status_svoboden_asr_diarization/gold.xlsx")

    parser.add_argument(
        "--scan-windows", action="store_true",
        help=(
            "Test-film mode: scan --transcript-input-dir (or output-dir/windows) for existing "
            "chunk directories instead of loading windows from --gold-excel. "
            "Use after running src.chunk_film."
        ),
    )
    parser.add_argument("--transcript-input-dir", type=str, default=None)
    parser.add_argument("--diarization-input-dir", type=str, default=None)
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-diarization", action="store_true")
    parser.add_argument("--only-asr", action="store_true")
    parser.add_argument("--only-diarization", action="store_true")

    parser.add_argument("--asr-model-name", type=str, default="medium")
    parser.add_argument(
        "--diarization-model-name",
        type=str,
        default="pyannote/speaker-diarization-community-1",
    )
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
    parser.add_argument(
        "--similarity-margin-threshold",
        type=float,
        default=0.05,
        help=(
            "Минимальная разница между лучшим и вторым speaker score для уверенного сопоставления. "
            "Спикеры с margin ниже порога помечаются unknown."
        ),
    )
    parser.add_argument(
        "--speaker-score-mode",
        type=str,
        default="raw",
        choices=["raw", "snorm"],
        help=(
            "Score for speaker-character assignment: raw cosine or cohort-normalized S-Norm. "
            "For v13 ECAPA calibration use raw with --similarity-threshold 0.30 "
            "and --similarity-margin-threshold 0.02; then compare snorm separately."
        ),
    )
    parser.add_argument("--top-k-candidates", type=int, default=3)
    parser.add_argument("--unknown-speaker-label", type=str, default="unknown_speaker")
    parser.add_argument("--unknown-speaker-name", type=str, default="unknown")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--intent-mode",
        type=str,
        default="rule",
        choices=["rule", "ml", "combined"],
        help=(
            "Метод извлечения намерений: "
            "'rule' = только rule-based (default), "
            "'ml' = только ML-классификатор, "
            "'combined' = объединение обоих (rule-based имеет приоритет при совпадении)."
        ),
    )
    parser.add_argument(
        "--ml-model",
        type=str,
        default=None,
        help="Путь к обученной ML-модели (.joblib). Требуется для --intent-mode ml|combined.",
    )
    parser.add_argument(
        "--ml-confidence-threshold",
        type=float,
        default=0.5,
        help=(
            "Минимальный порог уверенности ML-предсказания (0.0–1.0). "
            "Резервный порог для обоих классов; перекрывается --open-threshold / --close-threshold. "
            "Применяется в режимах ml и combined. По умолчанию 0.5."
        ),
    )
    parser.add_argument(
        "--open-threshold",
        type=float,
        default=0.32,
        help=(
            "Порог уверенности ML для contact_open (перекрывает --ml-confidence-threshold). "
            "По умолчанию 0.32."
        ),
    )
    parser.add_argument(
        "--close-threshold",
        type=float,
        default=0.30,
        help=(
            "Порог уверенности ML для contact_close (перекрывает --ml-confidence-threshold). "
            "По умолчанию 0.30."
        ),
    )
    parser.add_argument(
        "--use-sequence-decoder",
        action="store_true",
        default=False,
        help=(
            "Применить Viterbi-декодер поверх предсказаний намерений. "
            "Отбрасывает изолированные слабые события и штрафует close→open без паузы. "
            "По умолчанию отключён."
        ),
    )
    parser.add_argument(
        "--diarization-segment-mode",
        type=str,
        default="auto",
        choices=["auto", "regular", "exclusive"],
        help=(
            "Какой вид сегментов брать из diarization.json для utterance building. "
            "'auto' = поле 'segments' (текущее поведение, обычно == exclusive). "
            "'regular' = поле 'regular_segments'. "
            "'exclusive' = поле 'exclusive_segments'."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_token is None:
        args.hf_token = os.getenv("HF_TOKEN")

    needs_diarization_run = not args.skip_diarization and not args.only_asr
    if needs_diarization_run and not args.hf_token:
        raise ValueError(
            "Для diarization нужен Hugging Face token. Передайте --hf-token "
            "или задайте переменную окружения HF_TOKEN."
        )

    summary = run_validation_pipeline(args)
    print(f"Validation pipeline завершён. Окон с обработкой: {summary['num_windows']}")
    if summary.get("gold_output"):
        print(f"Gold сохранён в: {summary['gold_output']}")
    if summary.get("extracted_pairs_output"):
        print(f"Predictions сохранены в: {summary['extracted_pairs_output']}")
    else:
        print(f"Режим запуска: {summary['stage_mode']}. Итоговый Excel с парами на этом шаге не формировался.")


if __name__ == "__main__":
    main()
