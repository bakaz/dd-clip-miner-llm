from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .ffmpeg import cut_audio, cut_video
from .models import ContentResult, create_song_result
from .paths import stage_input_for_ffmpeg
from .report import write_reports


def manual_cut(
    run_dir: str | Path,
    config: dict[str, Any],
    csv_path: str | Path | None = None,
    input_video: str | Path | None = None,
    output_dir: str | Path | None = None,
    content_type: str = "song",
) -> list[ContentResult]:
    run = Path(run_dir)
    
    # 确定 CSV 路径
    if csv_path:
        source_csv = Path(csv_path)
    else:
        # 尝试多种可能的报告路径
        possible_paths = [
            run / "04_reports" / f"{content_type}s.csv",
            run / "04_reports" / "songs.csv",  # 兼容旧项目
        ]
        source_csv = next((p for p in possible_paths if p.exists()), possible_paths[0])
    
    # 确定视频路径
    source_video = Path(input_video) if input_video else _input_video_from_manifest(run)
    
    # 确定输出目录
    out = Path(output_dir) if output_dir else run / "05_manual"
    audio_out = out / "audio"
    video_out = out / "video"
    reports_out = out / "reports"
    for path in [audio_out, video_out, reports_out]:
        path.mkdir(parents=True, exist_ok=True)
    source_video = stage_input_for_ffmpeg(source_video, out / "00_input").resolve()

    audio_ext = str(config["output"].get("audio_extension", "mp3")).lstrip(".")
    audio_bitrate_kbps = int(config["output"].get("audio_bitrate_kbps") or 320)
    video_ext = str(config["output"].get("video_extension", "mp4")).lstrip(".")
    video_codec = str(config["output"].get("video_codec", "copy"))

    results = _read_manual_rows(source_csv, content_type)
    for result in results:
        stem = _safe_manual_stem(result)
        if config["output"].get("audio_segments", True):
            target = audio_out / f"{stem}.{audio_ext}"
            copy_audio = audio_ext.lower() in {"aac", "m4a"}
            cut_audio(
                source_video,
                target,
                result.start,
                result.end,
                copy_codec=copy_audio,
                bitrate_kbps=audio_bitrate_kbps,
            )
            result.audio_path = target

        if config["output"].get("video_clips", True):
            target = video_out / f"{stem}.{video_ext}"
            cut_video(source_video, target, result.start, result.end, video_codec=video_codec)
            result.video_path = target

    write_reports(results, reports_out, content_type)
    return results


def _read_manual_rows(csv_path: Path, content_type: str = "song") -> list[ContentResult]:
    results: list[ContentResult] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for fallback_index, row in enumerate(reader, start=1):
            start = _parse_time(row.get("start", "0"))
            end = _parse_time(row.get("end", "0"))
            if end <= start:
                raise ValueError(f"Invalid manual time range in row {fallback_index}: {start} -> {end}")
            index = int(row.get("index") or fallback_index)
            
            # 使用工厂函数创建结果（兼容旧项目）
            if content_type == "song":
                result = create_song_result(
                    index=index,
                    title=row.get("title") or f"manual_{index:03d}",
                    artist=row.get("artist") or "",
                    start=start,
                    end=end,
                    duration=end - start,
                    lyrics_snippet=row.get("lyrics_snippet") or "",
                    confidence=float(row.get("confidence") or 1.0),
                    audio_path=None,
                    video_path=None,
                    transcript=row.get("transcript") or "",
                    errors=[],
                )
            else:
                result = ContentResult(
                    index=index,
                    content_type=content_type,
                    title=row.get("title") or f"manual_{index:03d}",
                    start=start,
                    end=end,
                    duration=end - start,
                    transcript=row.get("transcript") or "",
                    confidence=float(row.get("confidence") or 1.0),
                    tags=[t.strip() for t in (row.get("tags") or "").split("|") if t.strip()],
                    description=row.get("description") or "",
                    artist=row.get("artist") or "",
                    audio_path=None,
                    video_path=None,
                    errors=[],
                )
            results.append(result)
    return results


def _input_video_from_manifest(run_dir: Path) -> Path:
    manifest_path = run_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_video = Path(data["input_video"])
    if input_video.is_absolute():
        return input_video
    cwd_candidate = Path.cwd() / input_video
    if cwd_candidate.exists():
        return cwd_candidate
    run_candidate = run_dir / input_video
    if run_candidate.exists():
        return run_candidate
    return input_video


def _parse_time(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    raise ValueError(f"Unsupported time value: {value}")


def _safe_manual_stem(result: ContentResult) -> str:
    from .paths import safe_path_part

    bits = [f"{result.index:03d}", result.title]
    if result.artist:
        bits.append(result.artist)
    return safe_path_part(" - ".join(bits), fallback=f"manual_{result.index:03d}")
