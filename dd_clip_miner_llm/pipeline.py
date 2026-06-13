"""核心流水线

编排完整的内容识别流程：
1. 音频提取
2. ASR 转写
3. LLM 识别（通过识别器架构）
4. 片段导出
5. 报告生成
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from .asr import Transcriber
from .ffmpeg import cut_audio, cut_video, extract_audio, get_duration
from .merger import build_content_results
from .models import ContentMatch, ContentResult, TranscriptSegment
from .asr_backends import resolve_asr_model_name
from .paths import safe_path_part, stage_input_for_ffmpeg
from .recognizers import get_recognizer, list_recognizers
from .clip_naming import ClipNamingProfile, resolve_clip_naming_profile, resolve_export_stem
from .report import write_match_context_reports, write_reports
from .config import get_asr_fingerprint, get_asr_inference_mode
from .config import get_asr_fingerprint, get_asr_inference_mode
from .profile_state import (
    _config_fingerprint,
    _format_usage_summary_console,
    _transcript_fingerprint,
    _profile_state_matches,
    _transcript_fingerprint,
    _write_profile_comparison,
    _write_profile_state,
    _write_usage_summary,
    _write_valid_debug_manifest,
)

def _safe_filename(value: str, fallback: str = "untitled") -> str:
    return safe_path_part(value, fallback=fallback)


def _check_previous_run(out: Path, input_path: Path) -> dict[str, Any] | None:
    progress_path = out / "progress.json"
    if not progress_path.exists():
        return None
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        prev_input = progress.get("input_video", "")
        if Path(prev_input).resolve() == input_path.resolve():
            return progress
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Failed to read progress.json: %s", exc)
    return None


def _save_progress(out: Path, input_path: Path, step: str, data: dict[str, Any] | None = None) -> None:
    progress_path = out / "progress.json"
    try:
        progress = {}
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress["input_video"] = str(input_path)
        progress["last_completed_step"] = step
        if data:
            progress[step] = data
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("Failed to save progress.json: %s", exc)


def _write_asr_state(
    asr_dir: Path,
    source_wav: Path,
    config: dict[str, Any],
    inference_mode: str,
    segments: list[TranscriptSegment],
) -> None:
    """Write asr_state.json with audio identity, ASR fingerprint, mode, model, transcript fp."""
    state_path = asr_dir / "asr_state.json"
    try:
        audio_info: dict[str, Any] = {"input_audio": str(source_wav.resolve())}
        if source_wav.exists():
            stat = source_wav.stat()
            audio_info["audio_size"] = stat.st_size
            audio_info["audio_mtime"] = stat.st_mtime
        state = {
            **audio_info,
            "asr_fingerprint": get_asr_fingerprint(config),
            "inference_mode": inference_mode,
            "model": str(config.get("asr", {}).get("model") or "unknown"),
            "transcript_fingerprint": _transcript_fingerprint(segments),
        }
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.debug("Failed to write asr_state.json: %s", exc)


def _load_previous_segments(asr_dir: Path) -> list[TranscriptSegment] | None:
    transcript_path = asr_dir / "transcript.json"
    if not transcript_path.exists():
        return None
    try:
        return [
            TranscriptSegment(start=s["start"], end=s["end"], text=s["text"])
            for s in json.loads(transcript_path.read_text(encoding="utf-8"))
        ]
    except (json.JSONDecodeError, OSError):
        return None


def _load_previous_matches(llm_dir: Path, content_type: str) -> list[ContentMatch] | None:
    matches_path = llm_dir / "matches.json"
    if not matches_path.exists():
        return None
    try:
        return [
            ContentMatch(
                content_type=m.get("content_type", content_type),
                title=m["title"],
                segment_indices=m.get("segment_indices", []),
                confidence=m.get("confidence", 0.5),
                tags=m.get("tags", []),
                description=m.get("description", ""),
                artist=m.get("artist", ""),
                lyrics_snippet=m.get("lyrics_snippet", ""),
            )
            for m in json.loads(matches_path.read_text(encoding="utf-8"))
        ]
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def _load_previous_summary(llm_dir: Path) -> dict[str, Any] | None:
    summary_path = llm_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or not summary:
            return None
        if summary.get("error"):
            return None
        if not isinstance(summary.get("level_1"), list) and not isinstance(summary.get("overall"), dict):
            return None
        return summary
    except (json.JSONDecodeError, OSError):
        return None


def _is_summary_only(recognizer: Any, config: dict[str, Any]) -> bool:
    type_config = config.get(recognizer.name, {})
    default_config = getattr(recognizer, "default_config", {})
    return bool(type_config.get("summary_only", default_config.get("summary_only", False)))


def _write_structured_summary(
    summary: dict[str, Any],
    recognizer: Any,
    llm_dir: Path,
    reports_dir: Path,
    content_type: str,
    config: dict[str, Any],
    naming_profile: Any = None,
) -> None:
    # 构建文件名：【streamername】summary-YYMMDD 或 summary
    if naming_profile and naming_profile.streamer and naming_profile.date:
        stem = f"【{naming_profile.streamer}】summary-{naming_profile.date}"
    else:
        stem = "summary"

    for target_dir in (llm_dir, reports_dir / content_type):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / f"{stem}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        formatter = getattr(recognizer, "format_summary_markdown", None)
        if callable(formatter):
            markdown = formatter(summary, config)
        else:
            markdown = "```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n"
        (target_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")


def _export_results(
    results: list[ContentResult],
    input_path: Path,
    clips_dir: Path,
    config: dict[str, Any],
    content_type: str,
    naming_profile: ClipNamingProfile | None = None,
) -> None:
    """导出音视频片段（并行执行）"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    audio_ext = str(config["output"].get("audio_extension", "mp3")).lstrip(".")
    audio_bitrate_kbps = int(config["output"].get("audio_bitrate_kbps") or 320)
    video_ext = str(config["output"].get("video_extension", "mp4")).lstrip(".")
    video_codec = str(config["output"].get("video_codec", "copy"))
    max_workers = int(config["output"].get("max_export_workers", 4))

    audio_dir_out = clips_dir / "audio" / content_type
    video_dir_out = clips_dir / "video" / content_type
    do_audio = config["output"].get("audio_segments", True)
    do_video = config["output"].get("video_clips", True)

    tasks: list[tuple[ContentResult, str, str]] = []
    for result in results:
        stem = resolve_export_stem(
            result, config, content_type, naming_profile,
            legacy_safe_filename=_safe_filename,
        )
        tasks.append((result, stem, "audio" if do_audio else None))
        tasks.append((result, stem, "video" if do_video else None))

    def _export_one(result: ContentResult, stem: str, kind: str) -> None:
        try:
            if kind == "audio":
                target = audio_dir_out / f"{stem}.{audio_ext}"
                copy_audio = audio_ext.lower() in {"aac", "m4a"}
                cut_audio(input_path, target, result.start, result.end, copy_codec=copy_audio, bitrate_kbps=audio_bitrate_kbps)
                result.audio_path = target
            elif kind == "video":
                target = video_dir_out / f"{stem}.{video_ext}"
                cut_video(input_path, target, result.start, result.end, video_codec=video_codec)
                result.video_path = target
        except Exception as exc:
            result.errors.append(f"{kind} export failed: {exc}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_export_one, result, stem, kind)
            for result, stem, kind in tasks
            if kind is not None
        ]
        for future in as_completed(futures):
            future.result()  # raise any exceptions


def _get_content_types(config: dict[str, Any]) -> list[str]:
    """获取要处理的内容类型列表"""
    content_types = config.get("content_types", {})
    
    # 新格式：字典 {"song": true, "dialogue": false, ...}
    if isinstance(content_types, dict):
        return [ct for ct, enabled in content_types.items() if enabled]
    
    # 旧格式兼容：列表 ["song", "dialogue", ...]
    if isinstance(content_types, list) and content_types:
        return content_types
    
    # 向后兼容：检查各个类型的 enabled 状态
    available = list_recognizers()
    result = []
    for ct in available:
        type_config = config.get(ct, {})
        if type_config.get("enabled", True):
            result.append(ct)
    return result if result else ["song"]


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
        transcriber = Transcriber(config)
        print(f"[2/3] Running Whisper ASR... (inference_mode: {transcriber.inference_mode})", flush=True)
        segments = transcriber.transcribe(source_wav)
        transcript_path.write_text(
            json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_asr_state(asr_dir, source_wav, config, transcriber.inference_mode, segments)
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
    input_path: Path,
    total_duration: float,
    naming_profile: ClipNamingProfile | None,
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
                from .llm import identify_structured_content
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
            from .config import is_risk_routed_v3
            if content_type == "song" and is_risk_routed_v3(config):
                from .song_postprocess.v3 import run_risk_routed_v3_pipeline
                matches = run_risk_routed_v3_pipeline(
                    segments, config, recognizer, llm_dir,
                )
            else:
                from .llm import identify_content
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
        _export_results(results, input_path, clips_dir, config, content_type, naming_profile)

        type_reports_dir = reports_dir / content_type
        type_reports_dir.mkdir(parents=True, exist_ok=True)
        write_reports(results, type_reports_dir, content_type)
        all_results[content_type] = results

    return all_results


def _write_manifest_and_summary(
    out: Path,
    config: dict[str, Any],
    input_path: Path,
    total_duration: float,
    segments: list[TranscriptSegment],
    all_results: dict[str, list[ContentResult]],
    llm_base_dir: Path,
    asr_dir: Path,
    profile_name: str,
    profile_enabled: bool,
    profile_state_path: Path,
    config_fingerprint: str,
    transcript_fingerprint: str,
    asr_inference_mode: str,
) -> None:
    """Write manifest, usage summary, and profile state."""
    _save_progress(out, input_path, "export")
    _print_summary(all_results)

    usage_summary = _write_usage_summary(llm_base_dir)
    usage_console = _format_usage_summary_console(usage_summary)
    if usage_console:
        print(usage_console)

    manifest = {
        "input_video": str(input_path),
        "profile": config.get("_profile_name"),
        "total_duration": total_duration,
        "segment_count": len(segments),
        "content_types": {ct: len(results) for ct, results in all_results.items()},
        "config": {
            "asr_model": resolve_asr_model_name(config.get("asr", {})),
            "asr_inference_mode": asr_inference_mode,
            "llm_model": config.get("llm", {}).get("model", "unknown"),
        },
        "llm_usage": usage_summary,
    }
    manifest_path = out / (f"manifest.{profile_name}.json" if profile_enabled else "manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if profile_enabled:
        _write_profile_state(
            profile_state_path,
            input_path=input_path,
            config=config,
            config_fingerprint=config_fingerprint,
            transcript_fingerprint=transcript_fingerprint,
            status="complete",
        )
        _write_profile_comparison(asr_dir / "llm")
    _save_progress(out, input_path, "done")


def run_pipeline(
    input_video: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, list[ContentResult]]:
    """
    运行完整流水线，返回按类型分组的结果。
    Returns:
        {"song": [...], "dialogue": [...], "highlight": [...], "funny": [...]}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_path = stage_input_for_ffmpeg(input_video, out / "00_input").resolve()

    naming_profile = resolve_clip_naming_profile(
        input_video, config,
        config_path=Path(config_path).parent if config_path else None,
        extra_texts=[out.name],
    )
    if naming_profile is not None:
        profile_path = out / "clip_naming.json"
        profile_path.write_text(
            json.dumps(naming_profile.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[naming] 【{naming_profile.streamer}】*-{naming_profile.date} "
            f"({naming_profile.source}, score={naming_profile.score:.2f})"
        )

    audio_dir, asr_dir, llm_base_dir, clips_dir, reports_dir, profile_name, profile_enabled = (
        _setup_pipeline_dirs(out, config)
    )
    content_types = _get_content_types(config)

    prev_progress = _check_previous_run(out, input_path)
    reuse_audio = False
    reuse_asr = False
    if prev_progress:
        last_step = prev_progress.get("last_completed_step", "")
        print(f"[info] 检测到上次运行结果（完成到 {last_step}），检查可复用的部分...")
        reuse_audio = last_step in ("audio", "asr", "llm", "done") and (audio_dir / "source.wav").exists()
        reuse_asr = last_step in ("asr", "llm", "done") and (asr_dir / "transcript.json").exists()
        print(f"  音频提取: {'复用' if reuse_audio else '需要重新运行'}")
        print(f"  ASR 转写: {'复用' if reuse_asr else '需要重新运行'}")

    # Robust fallback for "reprocess same audio" scenarios (AB对照 with different profiles,
    # or "set progress last='asr' then re-run" on a video that was previously staged under
    # a different path representation).
    #
    # We do NOT require the input_video path string in progress to exactly match the
    # post-stage_input_for_ffmpeg input_path.
    #
    # Presence of the artifacts + the detailed validation inside _run_asr_step
    # (wav size/mtime + asr_fingerprint match against asr_state) is sufficient and correct.
    # This lets users reliably skip ASR/audio-extract when they want to reuse the
    # existing transcript for new LLM/review/clipping work on the *same content*.
    if not reuse_audio and (audio_dir / "source.wav").exists():
        reuse_audio = True
    if not reuse_asr and (asr_dir / "transcript.json").exists() and (asr_dir / "asr_state.json").exists():
        reuse_asr = True

    if prev_progress is None and (reuse_audio or reuse_asr):
        # Inform the user that we are reusing based on existing artifacts (not exact prior run match)
        print("[info] 检测到已有的音频/ASR 产物，将尝试复用（详细匹配由 _run_asr_step 负责）...")

    source_wav = _extract_audio_step(input_path, audio_dir, config, out, reuse_audio=reuse_audio)
    total_duration = get_duration(input_path)
    segments = _run_asr_step(source_wav, asr_dir, config, out, input_path, reuse_asr=reuse_asr)

    asr_inference_mode = get_asr_inference_mode(config.get("asr", {}))
    config_fingerprint = _config_fingerprint(config)
    transcript_fingerprint = _transcript_fingerprint(segments)
    profile_state_path = llm_base_dir / "profile.json"
    profile_reusable = (
        profile_enabled
        and _profile_state_matches(
            profile_state_path,
            input_path=input_path,
            config_fingerprint=config_fingerprint,
            transcript_fingerprint=transcript_fingerprint,
        )
    )
    if profile_enabled and not profile_reusable:
        _write_profile_state(
            profile_state_path,
            input_path=input_path,
            config=config,
            config_fingerprint=config_fingerprint,
            transcript_fingerprint=transcript_fingerprint,
            status="running",
        )

    print("[3/3] Identifying content with LLM...", flush=True)
    all_results = _run_recognition_loop(
        segments, config, content_types, llm_base_dir, clips_dir, reports_dir,
        input_path, total_duration, naming_profile, prev_progress,
        profile_enabled, profile_reusable,
    )

    _write_manifest_and_summary(
        out, config, input_path, total_duration, segments, all_results,
        llm_base_dir, asr_dir, profile_name, profile_enabled,
        profile_state_path, config_fingerprint, transcript_fingerprint,
        asr_inference_mode,
    )

    return all_results


# 兼容旧项目的函数别名
def run_pipeline_songs(
    input_video: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[ContentResult]:
    """运行流水线，仅返回歌曲结果（兼容旧项目）"""
    from copy import deepcopy
    config_copy = deepcopy(config)
    config_copy["content_types"] = ["song"]
    results = run_pipeline(input_video, output_dir, config_copy)
    return results.get("song", [])


def _print_summary(all_results: dict[str, list[ContentResult]]) -> None:
    """输出识别结果摘要"""
    print(f"\n{'='*60}")
    print(f"识别结果摘要:")
    print(f"{'='*60}")
    
    for content_type, results in all_results.items():
        print(f"\n  {content_type}: {len(results)} 个片段")
        for r in results[:5]:  # 最多显示5个
            tc_start = f"{int(r.start//3600):02d}:{int((r.start%3600)//60):02d}:{int(r.start%60):02d}"
            tc_end = f"{int(r.end//3600):02d}:{int((r.end%3600)//60):02d}:{int(r.end%60):02d}"
            print(f"    [{r.index}] {r.title} ({tc_start}-{tc_end}, {r.duration:.1f}s)")
        if len(results) > 5:
            print(f"    ... 还有 {len(results) - 5} 个")
    
    print(f"\n{'='*60}")
