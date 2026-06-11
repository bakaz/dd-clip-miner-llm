from __future__ import annotations

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

from .models import VideoMeta

# 缓存文件名
_CACHE_FILENAME = ".probe_cache.json"


def _get_cache_path(paths: list[str | Path]) -> Path | None:
    """获取缓存文件路径（在第一个输入文件的父目录）"""
    if not paths:
        return None
    parent = Path(paths[0]).parent
    if parent.exists():
        return parent / _CACHE_FILENAME
    return None


def _load_cache(cache_path: Path) -> dict[str, Any]:
    """加载缓存"""
    try:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    """保存缓存"""
    try:
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def _cache_key(path: Path) -> str:
    """生成缓存键（文件路径 + 修改时间）"""
    try:
        mtime = os.path.getmtime(path)
        return f"{path.resolve()}:{mtime}"
    except OSError:
        return str(path.resolve())


def _meta_to_dict(meta: VideoMeta) -> dict[str, Any]:
    """将 VideoMeta 转换为可序列化的字典"""
    return {
        "path": str(meta.path),
        "duration": meta.duration,
        "has_video": meta.has_video,
        "has_audio": meta.has_audio,
        "video_codec": meta.video_codec,
        "width": meta.width,
        "height": meta.height,
        "fps": meta.fps,
        "pix_fmt": meta.pix_fmt,
        "sar": meta.sar,
        "audio_codec": meta.audio_codec,
        "audio_sample_rate": meta.audio_sample_rate,
        "audio_channels": meta.audio_channels,
        "audio_layout": meta.audio_layout,
        "audio_bit_rate": meta.audio_bit_rate,
        "probe_ok": meta.probe_ok,
        "error": meta.error,
    }


def _dict_to_meta(d: dict[str, Any]) -> VideoMeta:
    """将字典转换为 VideoMeta"""
    return VideoMeta(
        path=Path(d["path"]),
        duration=d.get("duration"),
        has_video=d.get("has_video", True),
        has_audio=d.get("has_audio", True),
        video_codec=d.get("video_codec"),
        width=d.get("width"),
        height=d.get("height"),
        fps=d.get("fps"),
        pix_fmt=d.get("pix_fmt"),
        sar=d.get("sar"),
        audio_codec=d.get("audio_codec"),
        audio_sample_rate=d.get("audio_sample_rate"),
        audio_channels=d.get("audio_channels"),
        audio_layout=d.get("audio_layout"),
        audio_bit_rate=d.get("audio_bit_rate"),
        probe_ok=d.get("probe_ok", True),
        error=d.get("error"),
    )


def probe_many(paths: list[str | Path]) -> list[VideoMeta]:
    """批量探测多个视频文件（支持缓存）"""
    if not paths:
        return []
    
    # 尝试从缓存加载
    cache_path = _get_cache_path(paths)
    cache = _load_cache(cache_path) if cache_path else {}
    
    results: list[VideoMeta] = []
    uncached_indices: list[int] = []
    uncached_paths: list[Path] = []
    
    # 检查缓存
    for i, path in enumerate(paths):
        key = _cache_key(Path(path))
        if key in cache:
            results.append(_dict_to_meta(cache[key]))
        else:
            results.append(None)  # placeholder
            uncached_indices.append(i)
            uncached_paths.append(Path(path))
    
    # 探测未缓存的文件
    if uncached_paths:
        if len(uncached_paths) <= 1:
            probed = [probe_one(path) for path in uncached_paths]
        else:
            max_workers = min(4, len(uncached_paths))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                probed = list(executor.map(probe_one, uncached_paths))
        
        # 更新结果和缓存
        for idx, meta in zip(uncached_indices, probed):
            results[idx] = meta
            if cache_path:
                cache[_cache_key(meta.path)] = _meta_to_dict(meta)
        
        # 保存缓存
        if cache_path:
            _save_cache(cache_path, cache)
    
    return results


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
        audio_bit_rate=_int_or_none(_stream_value(audio, "bit_rate")),
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
        audio_bit_rate=None,
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
