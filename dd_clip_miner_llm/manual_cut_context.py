"""Manual cut from context JSON — like post-merge but for arbitrary time ranges."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .ffmpeg import cut_audio, cut_video


class ManualCutContextError(RuntimeError):
    """Raised when a manual-cut-context request cannot be fulfilled."""


def manual_cut_from_context(
    context_path: str | Path,
    start_str: str,
    end_str: str,
    filename: str | None = None,
) -> dict[str, Any]:
    context_file = Path(context_path)
    context = _load_json_object(context_file)

    input_video = _resolve_input_video(context, context_file.parent)
    output_dir = context_file.parent
    output_suffix = _determine_output_suffix(context)

    start = _parse_time(start_str)
    end = _parse_time(end_str)
    if end <= start:
        raise ManualCutContextError(f"End time ({end_str}) must be after start time ({start_str})")

    if filename and filename.strip():
        stem = filename.strip()
    else:
        stem = f"manual_{_sanitize_time(start_str)}-{_sanitize_time(end_str)}"

    output_path = _unique_path(output_dir / f"{stem}{output_suffix}")

    config = context.get("config", {})
    output_config = config.get("output", {})

    if output_suffix == ".mp4":
        video_codec = str(output_config.get("video_codec") or "copy")
        cut_video(input_video, output_path, start, end, video_codec=video_codec)
    else:
        audio_bitrate_kbps = int(output_config.get("audio_bitrate_kbps") or 320)
        copy_audio = output_suffix.lower() in {".aac", ".m4a"}
        cut_audio(input_video, output_path, start, end, copy_codec=copy_audio, bitrate_kbps=audio_bitrate_kbps)

    return {
        "input_video": str(input_video),
        "output_path": str(output_path),
        "start": start_str,
        "end": end_str,
        "start_seconds": start,
        "end_seconds": end,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManualCutContextError(f"Context file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManualCutContextError(f"Invalid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ManualCutContextError(f"Expected JSON object in: {path}")
    return data


def _resolve_input_video(context: dict[str, Any], context_dir: Path) -> Path:
    value = context.get("input_video")
    if not value:
        raise ManualCutContextError("Context is missing 'input_video'")
    path = Path(str(value))
    if path.is_absolute():
        if path.exists():
            return path
        raise ManualCutContextError(f"Input video not found: {path}")
    candidate = context_dir / path
    if candidate.exists():
        return candidate.resolve()
    raise ManualCutContextError(f"Input video not found: {candidate}")


def _determine_output_suffix(context: dict[str, Any]) -> str:
    config = context.get("config", {})
    output_config = config.get("output", {})
    video_codec = output_config.get("video_codec")
    if video_codec:
        return ".mp4"
    return ".mp3"


def _parse_time(value: str) -> float:
    text = str(value).strip()
    if not text:
        raise ManualCutContextError("Time value is empty")
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    raise ManualCutContextError(f"Invalid time format: {value}")


def _sanitize_time(value: str) -> str:
    return re.sub(r"[^\d]", "", value.replace(":", ""))


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise ManualCutContextError(f"Could not find unused output path: {path}")
