from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from ..paths import safe_path_part


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


def resolve_total_duration(
    input_path: Path,
    source_wav: Path,
    segments: list,
) -> float:
    """Resolve media duration when container metadata under-reports (e.g. B站 LiveHime FLV)."""
    from ..ffmpeg import get_duration

    total_duration = get_duration(input_path)
    reported = total_duration
    if segments:
        total_duration = max(total_duration, float(segments[-1].end))
    if source_wav.exists():
        total_duration = max(total_duration, get_duration(source_wav))
    if total_duration > reported + 1.0:
        print(
            f"[info] total_duration corrected: {reported:.1f}s -> {total_duration:.1f}s",
            flush=True,
        )
    return total_duration


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
