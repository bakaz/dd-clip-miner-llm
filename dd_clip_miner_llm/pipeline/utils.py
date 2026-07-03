from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from ..paths import safe_path_part


def _safe_filename(value: str, fallback: str = "untitled") -> str:
    return safe_path_part(value, fallback=fallback)


@dataclass
class DurationInfo:
    """Resolved duration with provenance metadata for export bounds decisions.

    Attributes:
        container_duration: Duration reported by container (ffprobe), or None.
        audio_duration: Duration of the extracted source.wav, or None.
        asr_last_end: End time of the last ASR segment, or None.
        resolved_duration: Best-estimate duration (max of all available).
        trust_level: ``"high"`` when audio or ASR corroboration exists,
            ``"low"`` when only container metadata is available.
        hard_bounds: ``True`` when export should skip/clamp out-of-bounds
            results (trustworthy duration), ``False`` to warn-only.
    """
    container_duration: float | None
    audio_duration: float | None
    asr_last_end: float | None
    resolved_duration: float
    trust_level: str  # "high" | "low"
    hard_bounds: bool


def _llm_batch_debug_complete(llm_dir: Path) -> bool:
    """Check that no LLM batch debug file carries an incomplete marker.

    Scans *llm_dir* for ``llm_batch_*.json`` files.  If any contains:
    - ``scan_incomplete`` true
    - ``finish_reason`` equal to ``"length"``
    - a non-empty ``error``

    returns ``False`` (incomplete results exist — should not reuse).

    Returns ``True`` when the directory does not exist, no batch debug
    files are found, or all files pass the check.
    """
    if not llm_dir.is_dir():
        return True
    try:
        for entry in llm_dir.iterdir():
            if not (entry.name.startswith("llm_batch_") and entry.suffix == ".json"):
                continue
            try:
                payload = json.loads(entry.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("scan_incomplete"):
                return False
            if payload.get("finish_reason") == "length":
                return False
            if payload.get("error"):
                return False
    except OSError:
        pass
    return True


def _llm_progress_complete(progress: dict[str, Any] | None, content_type: str | None = None) -> bool:
    """Return False when progress.json carries an incomplete LLM marker.

    This is a pipeline-level guard for resume/skip decisions.  Debug-batch
    reuse is already protected by ``_llm_batch_debug_complete``; this prevents a
    looser ``last_completed_step in ('llm', 'done')`` style check from treating
    a previous incomplete LLM scan as fully reusable.
    """
    if not isinstance(progress, dict):
        return True

    def _payload_complete(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return True
        if payload.get("scan_incomplete"):
            return False
        if payload.get("finish_reason") == "length":
            return False
        if payload.get("error"):
            return False
        return True

    candidates: list[Any] = [progress, progress.get("llm")]
    if content_type:
        candidates.append(progress.get(content_type))
        llm_data = progress.get("llm")
        if isinstance(llm_data, dict):
            candidates.append(llm_data.get(content_type))

    return all(_payload_complete(candidate) for candidate in candidates)


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
    segments: list,
) -> None:
    """Write asr_state.json with audio identity, ASR fingerprint, mode, model, transcript fp."""
    from ..config import get_asr_fingerprint
    from ..profile_state import _transcript_fingerprint
    from ..asr_backends import resolve_asr_model_name

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
            "model": resolve_asr_model_name(config.get("asr", {})),
            "transcript_fingerprint": _transcript_fingerprint(segments),
        }
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.debug("Failed to write asr_state.json: %s", exc)


def _load_previous_segments(asr_dir: Path) -> list | None:
    from ..models import TranscriptSegment

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


def _load_previous_matches(llm_dir: Path, content_type: str) -> list | None:
    from ..models import ContentMatch

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


def _load_previous_summary(llm_dir: Path, content_type: str = "daily_summary") -> dict[str, Any] | None:
    stem = "summary" if content_type == "daily_summary" else content_type
    candidates = [llm_dir / f"{stem}.json"]
    if stem != "summary":
        candidates.append(llm_dir / "summary.json")
    summary_path = next((path for path in candidates if path.exists()), None)
    if summary_path is None:
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or not summary:
            return None
        if summary.get("error"):
            return None
        summary_content_type = summary.get("content_type")
        if summary_content_type is not None and summary_content_type != content_type:
            return None
        if not isinstance(summary.get("level_1"), list) and not isinstance(summary.get("overall"), dict):
            return None
        return summary
    except (json.JSONDecodeError, OSError):
        return None


def _is_summary_only(recognizer: Any, config: dict[str, Any]) -> bool:
    get_type_config = getattr(recognizer, "get_type_config", None)
    raw_cfg = get_type_config(config) if callable(get_type_config) else config.get(recognizer.name, {})
    type_config = raw_cfg if isinstance(raw_cfg, dict) else {}
    default_config: dict[str, Any] = getattr(recognizer, "default_config", {})
    return bool(type_config.get("summary_only", default_config.get("summary_only", False)))


def resolve_total_duration_info(
    input_path: Path,
    source_wav: Path,
    segments: list,
) -> DurationInfo:
    """Resolve media duration and return provenance metadata.

    Returns a :class:`DurationInfo` with the resolved float (max of container,
    audio, and ASR last-end) as well as trust indicators for export bounds.

    Trust logic (``hard_bounds``):
    * ``True`` (``trust_level="high"``) when:

      - Container and audio both exist and agree within tolerance (|diff| ≤
        max(2s, 0.5% of larger value)), OR
      - Only audio exists (no container metadata available), OR
      - Only ASR segments exist (no container, no audio).
      In all these cases the resolved duration is considered reliable for
      video/audio export bounds.

    * ``False`` (``trust_level="low"``) when:

      - Container and audio both exist but disagree beyond tolerance — one of
        them is unreliable for video clipping bounds, so we warn only and let
        ffmpeg attempt the cut rather than silently discarding results.
      - Only bare container metadata is available (no audio, no ASR).
    """
    from ..ffmpeg import get_duration

    container_duration: float | None = None
    audio_duration: float | None = None
    asr_last_end: float | None = None

    try:
        container_duration = get_duration(input_path)
    except Exception:
        pass

    if source_wav.exists():
        try:
            audio_duration = get_duration(source_wav)
        except Exception:
            pass

    asr_last_end = float(segments[-1].end) if segments else None

    # Resolve: use the maximum of all available sources
    candidates: list[float] = []
    if container_duration is not None:
        candidates.append(container_duration)
    if audio_duration is not None:
        candidates.append(audio_duration)
    if asr_last_end is not None:
        candidates.append(asr_last_end)
    resolved_duration = max(candidates) if candidates else 0.0

    # ── Trust determination ──────────────────────────────────────────
    has_audio = audio_duration is not None
    has_container = container_duration is not None
    has_asr = asr_last_end is not None

    # When both container and audio exist, check whether they agree.
    container_audio_agree = True
    if has_container and has_audio:
        larger = max(container_duration, audio_duration)
        tolerance = max(2.0, larger * 0.005)
        container_audio_agree = (
            abs(container_duration - audio_duration) <= tolerance
        )

    if has_container and has_audio and not container_audio_agree:
        # Container and audio diverge significantly — can't trust the max
        # for video clip bounds. Warn but keep resolved_duration.
        hard_bounds = False
        trust_level = "low"
    elif has_audio or has_asr:
        hard_bounds = True
        trust_level = "high"
    else:
        # Only bare container metadata — no corroboration.
        hard_bounds = False
        trust_level = "low"

    # Print correction info when container under-reports
    if container_duration is not None and resolved_duration > container_duration + 1.0:
        print(
            f"[info] total_duration corrected: {container_duration:.1f}s -> "
            f"{resolved_duration:.1f}s  (trust={trust_level})",
            flush=True,
        )

    return DurationInfo(
        container_duration=container_duration,
        audio_duration=audio_duration,
        asr_last_end=asr_last_end,
        resolved_duration=resolved_duration,
        trust_level=trust_level,
        hard_bounds=hard_bounds,
    )


def resolve_total_duration(
    input_path: Path,
    source_wav: Path,
    segments: list,
) -> float:
    """Resolve media duration (legacy interface, returns bare float).

    Delegates to :func:`resolve_total_duration_info` for implementation.
    """
    return resolve_total_duration_info(input_path, source_wav, segments).resolved_duration


def _get_content_types(config: dict[str, Any]) -> list[str]:
    """获取要处理的内容类型列表"""
    from ..recognizers import list_recognizers

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


def _print_summary(all_results: dict[str, list]) -> None:
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
