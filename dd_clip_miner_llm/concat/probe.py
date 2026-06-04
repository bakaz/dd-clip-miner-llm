from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from .models import VideoMeta


def probe_many(paths: list[str | Path]) -> list[VideoMeta]:
    return [probe_one(path) for path in paths]


def probe_one(path: str | Path) -> VideoMeta:
    source = Path(path)
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return VideoMeta(
            path=source,
            duration=None,
            has_video=True,
            has_audio=True,
            video_codec=None,
            width=None,
            height=None,
            fps=None,
            pix_fmt=None,
            sar=None,
            audio_codec=None,
            audio_sample_rate=None,
            audio_channels=None,
            audio_layout=None,
            probe_ok=False,
            error="ffprobe not found",
        )

    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-show_streams",
            "-of",
            "json",
            str(source),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return _failed_meta(source, completed.stderr.strip())

    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return _failed_meta(source, str(exc))

    streams = data.get("streams", [])
    video = _first_stream(streams, "video")
    audio = _first_stream(streams, "audio")
    duration = _float_or_none(data.get("format", {}).get("duration"))

    return VideoMeta(
        path=source,
        duration=duration,
        has_video=video is not None,
        has_audio=audio is not None,
        video_codec=_stream_value(video, "codec_name"),
        width=_int_or_none(_stream_value(video, "width")),
        height=_int_or_none(_stream_value(video, "height")),
        fps=_fps(video),
        pix_fmt=_stream_value(video, "pix_fmt"),
        sar=_stream_value(video, "sample_aspect_ratio"),
        audio_codec=_stream_value(audio, "codec_name"),
        audio_sample_rate=_int_or_none(_stream_value(audio, "sample_rate")),
        audio_channels=_int_or_none(_stream_value(audio, "channels")),
        audio_layout=_stream_value(audio, "channel_layout"),
    )


def _failed_meta(path: Path, error: str) -> VideoMeta:
    return VideoMeta(
        path=path,
        duration=None,
        has_video=False,
        has_audio=False,
        video_codec=None,
        width=None,
        height=None,
        fps=None,
        pix_fmt=None,
        sar=None,
        audio_codec=None,
        audio_sample_rate=None,
        audio_channels=None,
        audio_layout=None,
        probe_ok=False,
        error=error,
    )


def _first_stream(streams: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for stream in streams:
        if stream.get("codec_type") == kind:
            return stream
    return None


def _stream_value(stream: dict[str, Any] | None, key: str) -> Any:
    if stream is None:
        return None
    return stream.get(key)


def _fps(stream: dict[str, Any] | None) -> float | None:
    if not stream:
        return None
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key)
        if not value or value == "0/0":
            continue
        try:
            rate = Fraction(str(value))
        except (ValueError, ZeroDivisionError):
            continue
        if rate.denominator:
            return float(rate)
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

