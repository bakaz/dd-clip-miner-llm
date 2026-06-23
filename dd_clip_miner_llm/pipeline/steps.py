from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from ..asr import Transcriber
from ..config import get_asr_fingerprint
from ..ffmpeg import extract_audio, get_duration
from ..models import ContentMatch, ContentResult, TranscriptSegment
from ..paths import safe_path_part
from ..report import write_match_context_reports, write_reports
from ..merger import build_content_results
from ..recognizers import get_recognizer
from ..profile_state import _write_valid_debug_manifest
from .export import _export_results, _export_sus_clips, _write_structured_summary
from .utils import (
    _check_previous_run,
    _get_content_types,
    _is_summary_only,
    _load_previous_matches,
    _load_previous_segments,
    _load_previous_summary,
    _print_summary,
    _save_progress,
    _write_asr_state,
)


def _setup_pipeline_dirs(
    out: Path,
    config: dict[str, Any],
) -> tuple[Path, Path, Path, Path, Path, str, bool]:
    """Create pipeline directories and return key paths."""
    audio_dir = out / "01_audio"
    asr_dir = out / "02_asr"
    profile_enabled = bool(config.get("_profile_enabled", False))
    profile_name = safe_path_part(str(config.get("_profile_name") or "default"))
    llm_base_dir = asr_dir / "llm"
    clips_dir = out / "03_clips"
    reports_dir = out / "04_reports"
    if profile_enabled:
        llm_base_dir = llm_base_dir / profile_name
        clips_dir = clips_dir / profile_name
        reports_dir = reports_dir / profile_name
    for d in [audio_dir, asr_dir, llm_base_dir, clips_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return audio_dir, asr_dir, llm_base_dir, clips_dir, reports_dir, profile_name, profile_enabled


def _extract_audio_step(
    input_path: Path,
    audio_dir: Path,
    config: dict[str, Any],
    out: Path,
    *,
    reuse_audio: bool,
) -> Path:
    """Step 1: Extract audio from video or reuse existing."""
    source_wav = audio_dir / "source.wav"
    if reuse_audio:
        print("[1/3] 音频提取: 复用已有结果")
    else:
        print("[1/3] Extracting audio...", flush=True)
        audio_config = config.get("audio", {})
        extract_audio(
            input_path, source_wav,
            sample_rate=int(audio_config.get("sample_rate", 16000)),
            channels=int(audio_config.get("channels", 1)),
        )
    _save_progress(out, input_path, "audio")
    return source_wav


def _run_asr_step(
    source_wav: Path,
    asr_dir: Path,
    config: dict[str, Any],
    out: Path,
    input_path: Path,
    *,
    reuse_asr: bool,
) -> list[TranscriptSegment]:
    """Step 2: Run ASR transcription or reuse existing.
    Enhanced reuse: only if transcript + asr_state match input audio identity and ASR fingerprint.
    Old dir without asr_state.json triggers one re-run.
    After run, write asr_state.json.
    """
    transcript_path = asr_dir / "transcript.json"
    state_path = asr_dir / "asr_state.json"

    do_reuse = reuse_asr and transcript_path.exists()
    if do_reuse:
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                curr_audio = str(source_wav.resolve())
                curr_size = source_wav.stat().st_size if source_wav.exists() else None
                curr_mtime = source_wav.stat().st_mtime if source_wav.exists() else None
                curr_fp = get_asr_fingerprint(config)
                size_match = state.get("audio_size") == curr_size
                fp_match = state.get("asr_fingerprint") == curr_fp
                mtime_match = state.get("audio_mtime") == curr_mtime

                if not (size_match and fp_match):
                    do_reuse = False
                elif not mtime_match:
                    # For explicit reuse scenarios (user sets last_completed_step='asr'
                    # and wants to reuse existing transcript for new profile / post-processing),
                    # we allow reuse even if mtime drifted (e.g. previous re-extracts touched the wav).
                    # Size + fingerprint match is sufficient to confirm it's the same audio content.
                    print("  [info] audio mtime differs from asr_state (common after re-extracts), "
                          "but size + fingerprint match — reusing transcript (explicit reuse mode)")
                    # keep do_reuse = True

            except Exception as exc:
                logger.debug("Failed to read asr_state.json: %s", exc)
                do_reuse = False
        else:
            print("  [info] 旧 ASR 目录缺少 asr_state.json，重新运行 ASR")
            do_reuse = False

    if do_reuse:
        print("[2/3] ASR 转写: 复用已有结果")
        segments = _load_previous_segments(asr_dir)
        if segments is None:
            print("  [warn] 无法加载之前的 ASR 结果，重新运行...")
            do_reuse = False

    if not do_reuse:
        from ..asr_fallback import (
            is_faster_whisper_fallback_enabled,
            is_qwen3_fallback_enabled,
            transcribe_qwen3_with_fallback,
            transcribe_with_fallback,
        )

        if is_faster_whisper_fallback_enabled(config.get("asr", {})):
            print("[2/3] Running Whisper ASR... (batch + standard fallback)", flush=True)
            segments, fallback_meta = transcribe_with_fallback(source_wav, config["asr"], asr_dir)
            inference_mode = f"{fallback_meta['primary_mode']}+fallback:{fallback_meta['fallback_mode']}"
            print(
                "  ASR fallback ranges: "
                f"{fallback_meta['fallback_range_count']}, "
                f"fallback segments: {fallback_meta['fallback_segment_count']}, "
                f"merged: {fallback_meta['merged_segment_count']}",
                flush=True,
            )
        elif is_qwen3_fallback_enabled(config.get("asr", {})):
            print("[2/3] Running Qwen3 ASR... (primary + suspicious-range fallback)", flush=True)
            segments, fallback_meta = transcribe_qwen3_with_fallback(source_wav, config["asr"], asr_dir)
            inference_mode = f"qwen3+fallback:chunk{fallback_meta['chunk_seconds']}"
            print(
                "  Qwen3 fallback ranges: "
                f"{fallback_meta['fallback_range_count']}, "
                f"fallback segments: {fallback_meta['fallback_segment_count']}, "
                f"merged: {fallback_meta['merged_segment_count']}",
                flush=True,
            )
        else:
            transcriber = Transcriber(config, asr_dir=asr_dir)
            inference_mode = transcriber.inference_mode
            print(f"[2/3] Running ASR... (inference_mode: {transcriber.inference_mode})", flush=True)
            segments = transcriber.transcribe(source_wav)
        transcript_path.write_text(
            json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_asr_state(asr_dir, source_wav, config, inference_mode, segments)
    _save_progress(out, input_path, "asr")
    print(f"  Transcribed {len(segments)} segments", flush=True)
    return segments


def _run_recognition_loop(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    content_types: list[str],
    llm_base_dir: Path,
    clips_dir: Path,
    reports_dir: Path,
    run_dir: Path,
    asr_dir: Path,
    manifest_path: Path,
    input_path: Path,
    total_duration: float,
    naming_profile: Any,
    prev_progress: dict[str, Any] | None,
    profile_enabled: bool,
    profile_reusable: bool,
) -> dict[str, list[ContentResult]]:
    """Step 3: Run LLM recognition for each content type."""
    all_results: dict[str, list[ContentResult]] = {}

    for ct_idx, content_type in enumerate(content_types, 1):
        recognizer = get_recognizer(content_type)
        if recognizer is None:
            print(f"  [warn] 未找到识别器: {content_type}")
            continue

        type_config = config.get(content_type, {})
        if not type_config.get("enabled", True):
            print(f"  {content_type}: 已禁用，跳过")
            continue

        print(f"\n  === {content_type} 识别 ({ct_idx}/{len(content_types)}) ===")
        llm_dir = llm_base_dir / content_type
        llm_dir.mkdir(parents=True, exist_ok=True)

        if _is_summary_only(recognizer, config):
            reuse_summary = False
            summary = None
            if prev_progress and (not profile_enabled or profile_reusable):
                summary = _load_previous_summary(llm_dir)
                reuse_summary = summary is not None
            if reuse_summary:
                print("  LLM 总结: 复用已有结果")
            else:
                from ..llm import identify_structured_content
                summary = identify_structured_content(segments, config, recognizer, debug_dir=llm_dir)
            _write_structured_summary(summary or {}, recognizer, llm_dir, reports_dir, content_type, config, naming_profile)
            _write_valid_debug_manifest(llm_dir)
            print(f"  Wrote {content_type} summary")
            all_results[content_type] = []
            continue

        reuse_llm = False
        if prev_progress and (not profile_enabled or profile_reusable):
            reuse_llm = llm_dir.exists() and (llm_dir / "matches.json").exists()
        if reuse_llm:
            print(f"  LLM 识别: 复用已有结果")
            matches = _load_previous_matches(llm_dir, content_type)
            if matches is None:
                reuse_llm = False

        if not reuse_llm:
            from ..config import is_risk_routed_kv
            if content_type == "song" and is_risk_routed_kv(config):
                from ..song_postprocess.song_kv import run_risk_routed_kv_pipeline
                matches = run_risk_routed_kv_pipeline(
                    segments, config, recognizer, llm_dir,
                )
            else:
                from ..llm import identify_content
                matches = identify_content(
                    segments, config, recognizer,
                    debug_dir=llm_dir, debug_phase="main",
                )
                matches = recognizer.post_process(segments, config, matches, llm_dir)
            (llm_dir / "matches.json").write_text(
                json.dumps([m.to_dict() for m in matches], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            write_match_context_reports(
                matches, segments, llm_dir,
                context_segments=int(config["output"].get("match_context_segments", 10)),
                content_type=content_type,
            )

        _write_valid_debug_manifest(llm_dir)
        print(f"  Found {len(matches)} {content_type} matches")

        results = build_content_results(segments, matches, total_duration, config, content_type)
        _export_results(
            results,
            input_path,
            clips_dir,
            config,
            content_type,
            naming_profile,
            run_dir=run_dir,
            llm_dir=llm_dir,
            reports_dir=reports_dir / content_type,
            transcript_path=asr_dir / "transcript.json",
            manifest_path=manifest_path,
            total_duration=total_duration,
        )

        # 导出 sus 文件夹（被合并的未知歌曲原始片段）
        merge_events_path = llm_dir / "merge_events.json"
        if merge_events_path.exists():
            try:
                merge_events = json.loads(merge_events_path.read_text(encoding="utf-8"))
                if merge_events:
                    _export_sus_clips(
                        merge_events, segments,
                        input_path, clips_dir, config, content_type,
                    )
            except (json.JSONDecodeError, OSError):
                pass

        type_reports_dir = reports_dir / content_type
        type_reports_dir.mkdir(parents=True, exist_ok=True)
        write_reports(results, type_reports_dir, content_type)
        all_results[content_type] = results

    return all_results
