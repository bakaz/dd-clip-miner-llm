from __future__ import annotations

import shutil
import subprocess
from os import devnull
from pathlib import Path

from .command import run_command
from .compat import pkg_attr
from .errors import FFmpegError
from .probe import get_video_resolution


def get_min_video_size(videos: list[str | Path]) -> tuple[int, int] | None:
    """获取像素面积最小的视频尺寸。"""
    min_area = float("inf")
    min_size: tuple[int, int] | None = None

    for video in videos:
        try:
            width, height = get_video_resolution(video)
            area = width * height
            if area < min_area:
                min_area = area
                min_size = (width, height)
        except (FFmpegError, ValueError):
            continue

    return min_size


def sum_media_durations(videos: list[str | Path]) -> float | None:
    total = 0.0
    get_duration = pkg_attr("get_duration")
    for video in videos:
        try:
            total += get_duration(video)
        except (FFmpegError, ValueError):
            return None
    return total


def validate_concat_duration(output: Path, expected_duration: float | None) -> None:
    if expected_duration is None:
        return
    get_duration = pkg_attr("get_duration")
    get_stream_duration = pkg_attr("_get_stream_duration")
    actual_duration = get_duration(output)
    tolerance = max(30.0, expected_duration * 0.005)
    if actual_duration + tolerance < expected_duration:
        raise FFmpegError(
            "Concat output duration is too short: "
            f"{actual_duration:.3f}s, expected about {expected_duration:.3f}s"
        )
    if actual_duration > expected_duration + tolerance:
        raise FFmpegError(
            "Concat output duration is too long: "
            f"{actual_duration:.3f}s, expected about {expected_duration:.3f}s"
        )
    video_duration = get_stream_duration(output, "v:0")
    if video_duration is not None and video_duration + tolerance < expected_duration:
        raise FFmpegError(
            "Concat output video stream duration is too short: "
            f"{video_duration:.3f}s, expected about {expected_duration:.3f}s"
        )
    if video_duration is not None and video_duration > expected_duration + tolerance:
        raise FFmpegError(
            "Concat output video stream duration is too long: "
            f"{video_duration:.3f}s, expected about {expected_duration:.3f}s"
        )


def has_audio_stream(input_media: str | Path) -> bool:
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return True
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(input_media),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())


def validate_audio_decodable(output: Path, ffmpeg_bin: str) -> None:
    if not pkg_attr("has_audio_stream")(output):
        return
    try:
        run_command([
            ffmpeg_bin,
            "-v",
            "error",
            "-i",
            str(output),
            "-map",
            "0:a:0",
            "-vn",
            "-f",
            "null",
            devnull,
        ])
    except FFmpegError as exc:
        raise FFmpegError(
            f"Concat output audio is not decodable: {output}\n{exc}"
        ) from exc