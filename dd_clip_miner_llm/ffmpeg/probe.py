from __future__ import annotations

import json
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from .command import require_binary, run_command
from .errors import FFmpegError

def get_video_codec(video_path: str | Path) -> str | None:
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return None
    codec = completed.stdout.strip().lower()
    return codec or None
def get_video_resolution(video_path: str | Path) -> tuple[int, int]:
    """获取视频分辨率（宽, 高）"""
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        completed = subprocess.run(
            [ffprobe_bin, "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            parts = completed.stdout.strip().split(",")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
    
    raise FFmpegError(f"Could not get video resolution: {video_path}")


def parse_fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            den = float(denominator)
            if den == 0:
                return None
            return float(numerator) / den
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def get_video_fps(video_path: str | Path) -> float | None:
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return None
    streams = payload.get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        fps = parse_fraction(stream.get(key))
        if fps and fps > 0:
            return fps
    return None


def _repair_video_filter_args(fps: float | None) -> list[str]:
    if fps and fps > 0:
        fps_text = f"{fps:.6f}".rstrip("0").rstrip(".")
def detect_ffmpeg_hwaccels(ffmpeg_bin: str | None = None) -> set[str]:
    exe = ffmpeg_bin or require_binary("ffmpeg")
    completed = subprocess.run(
        [exe, "-hide_banner", "-hwaccels"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = f"{completed.stdout}\n{completed.stderr}"
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lower().startswith("hardware acceleration")
    }


def detect_gpu_devices() -> list[str]:
    probes = [
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
        ["wmic", "path", "win32_VideoController", "get", "name"],
        ["lspci"],
    ]
    for args in probes:
        output = run_probe(args)
        if not output:
            continue
        if args[0] == "lspci":
            devices = [
                line.strip()
                for line in output.splitlines()
                if re.search(r"\b(vga|3d|display)\b", line, flags=re.IGNORECASE)
            ]
        else:
            devices = [
                line.strip()
                for line in output.splitlines()
                if line.strip() and line.strip().lower() != "name"
            ]
        if devices:
            return devices
    return []


def detect_ffmpeg_environment(ffmpeg_bin: str | None = None) -> dict[str, object]:
    exe = ffmpeg_bin or require_binary("ffmpeg")
    encoders = detect_video_encoders(exe)
    return {
        "ffmpeg": exe,
        "gpus": detect_gpu_devices(),
        "hwaccels": sorted(detect_ffmpeg_hwaccels(exe)),
        "video_encoders": sorted(encoders),
        "auto_reencode_order": auto_reencode_order(encoders),
        "recommended_video_codec": "copy",
    }


def auto_reencode_order(encoders: set[str]) -> list[str]:
    ordered: list[str] = []
    for codec, label in [
        ("h264_nvenc", "nv"),
        ("h264_qsv", "intel"),
        ("h264_amf", "amd"),
        ("libx264", "cpu"),
    ]:
        if codec in encoders:
            ordered.append(label)
    return ordered
def run_probe(args: list[str], timeout: int = 10) -> str:
    executable = shutil.which(args[0])
    if not executable:
        return ""
    try:
        completed = subprocess.run(
            [executable, *args[1:]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip() or completed.stderr.strip()


@lru_cache(maxsize=8)
def detect_video_encoders(ffmpeg_bin: str | None = None) -> set[str]:
    exe = ffmpeg_bin or require_binary("ffmpeg")
    completed = subprocess.run(
        [exe, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = f"{completed.stdout}\n{completed.stderr}"
    return set(re.findall(r"\b(h264_nvenc|h264_qsv|h264_amf|libx264)\b", text))


def get_duration(input_media: str | Path) -> float:
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        completed = subprocess.run(
            [ffprobe_bin, "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(input_media)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            return float(completed.stdout.strip())

    ffmpeg_bin = require_binary("ffmpeg")
    completed = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", str(input_media)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    duration = parse_ffmpeg_duration(completed.stderr)
    if duration is None:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise FFmpegError(f"Could not read media duration: {input_media}\n{detail}")
    return duration


def get_stream_duration(input_media: str | Path, stream_selector: str) -> float | None:
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "quiet",
            "-select_streams",
            stream_selector,
            "-show_entries",
            "stream=duration",
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
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip().splitlines()
    if not text:
        return None
    try:
        return float(text[0])
    except ValueError:
        return None


def parse_ffmpeg_duration(text: str) -> float | None:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds
