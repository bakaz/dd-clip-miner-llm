"""核心流水线

编排完整的内容识别流程：
1. 音频提取
2. ASR 转写
3. LLM 识别（通过识别器架构）
4. 片段导出
5. 报告生成
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..clip_naming import resolve_clip_naming_profile
from ..config import get_asr_inference_mode
from ..ffmpeg import cut_audio, cut_video, get_duration
from ..paths import stage_input_for_ffmpeg
from ..profile_state import (
    _config_fingerprint,
    _transcript_fingerprint,
    _profile_state_matches,
    _write_profile_state,
)
from .orchestrator import _write_manifest_and_summary
from .steps import (
    _extract_audio_step,
    _run_asr_step,
    _run_recognition_loop,
    _setup_pipeline_dirs,
)
from .utils import _check_previous_run, _get_content_types


def run_pipeline(
    input_video: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, list]:
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
            __import__("json").dumps(naming_profile.to_dict(), ensure_ascii=False, indent=2),
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
    manifest_path = out / (f"manifest.{profile_name}.json" if profile_enabled else "manifest.json")
    all_results = _run_recognition_loop(
        segments, config, content_types, llm_base_dir, clips_dir, reports_dir,
        out, asr_dir, manifest_path, input_path, total_duration, naming_profile, prev_progress,
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
) -> list:
    """运行流水线，仅返回歌曲结果（兼容旧项目）"""
    from copy import deepcopy
    config_copy = deepcopy(config)
    config_copy["content_types"] = ["song"]
    results = run_pipeline(input_video, output_dir, config_copy)
    return results.get("song", [])


def _export_results(*args: Any, **kwargs: Any) -> None:
    """Backward-compatible shim for callers that patched the old pipeline module."""
    from . import export as export_module

    original_cut_audio = export_module.cut_audio
    original_cut_video = export_module.cut_video
    export_module.cut_audio = cut_audio
    export_module.cut_video = cut_video
    try:
        export_module._export_results(*args, **kwargs)
    finally:
        export_module.cut_audio = original_cut_audio
        export_module.cut_video = original_cut_video


__all__ = [
    "run_pipeline",
    "run_pipeline_songs",
    "_export_results",
]
