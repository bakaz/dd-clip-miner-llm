from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from ..models import ContentMatch, ContentResult, TranscriptSegment
from ..ffmpeg import cut_audio, cut_video
from ..clip_naming import ClipNamingProfile, resolve_export_stem
from .utils import _safe_filename


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

    for target_dir, enabled in ((audio_dir_out, do_audio), (video_dir_out, do_video)):
        if not enabled or not target_dir.exists():
            continue
        for stale_file in target_dir.iterdir():
            if stale_file.is_file():
                stale_file.unlink()

    stem_counts: dict[str, int] = {}
    tasks: list[tuple[ContentResult, str, str]] = []
    for result in results:
        stem = resolve_export_stem(
            result, config, content_type, naming_profile,
            legacy_safe_filename=_safe_filename,
        )
        seen_count = stem_counts.get(stem, 0)
        stem_counts[stem] = seen_count + 1
        if seen_count:
            stem = f"{stem}_{result.index:03d}"
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


def _export_sus_clips(
    merge_events: list[dict[str, Any]],
    segments: list[TranscriptSegment],
    input_path: Path,
    clips_dir: Path,
    config: dict[str, Any],
    content_type: str,
) -> None:
    """导出被合并的未知歌曲原始片段到 sus/ 子文件夹。

    对于每个 force_merge_unknown_song 事件，从 merge_events 中读取
    原始 segment_indices，构建 ContentResult 并导出到 sus/ 子目录。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not merge_events:
        return

    audio_ext = str(config["output"].get("audio_extension", "mp3")).lstrip(".")
    audio_bitrate_kbps = int(config["output"].get("audio_bitrate_kbps") or 320)
    video_ext = str(config["output"].get("video_extension", "mp4")).lstrip(".")
    video_codec = str(config["output"].get("video_codec", "copy"))
    max_workers = int(config["output"].get("max_export_workers", 4))

    audio_dir_out = clips_dir / "audio" / content_type / "sus"
    video_dir_out = clips_dir / "video" / content_type / "sus"
    do_audio = config["output"].get("audio_segments", True)
    do_video = config["output"].get("video_clips", True)

    for target_dir, enabled in ((audio_dir_out, do_audio), (video_dir_out, do_video)):
        if not enabled:
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        # 清理旧的 sus 文件
        if target_dir.exists():
            for stale_file in target_dir.iterdir():
                if stale_file.is_file():
                    stale_file.unlink()

    # 构建原始片段的 ContentResult
    sus_results: list[tuple[ContentResult, str]] = []
    stem_counts: dict[str, int] = {}
    for event in merge_events:
        if event.get("type") != "force_merge_unknown_song":
            continue
        for key, title_key in [("prev_indices", "prev_title"), ("match_indices", "match_title")]:
            indices = event.get(key, [])
            if not indices:
                continue
            title = event.get(title_key, "未知歌曲")
            valid_indices = sorted({i for i in indices if 0 <= i < len(segments)})
            if not valid_indices:
                continue
            start = segments[valid_indices[0]].start
            end = segments[valid_indices[-1]].end
            transcript = " ".join(segments[i].text for i in valid_indices)
            result = ContentResult(
                index=0,
                content_type=content_type,
                title=title,
                start=start,
                end=end,
                duration=end - start,
                transcript=transcript,
                confidence=0.5,
                tags=["sus_original"],
            )
            stem = _safe_filename(f"{title}_sus")
            # 防止同名文件覆盖
            seen_count = stem_counts.get(stem, 0)
            stem_counts[stem] = seen_count + 1
            if seen_count:
                stem = f"{stem}_{seen_count + 1:02d}"
            sus_results.append((result, stem))

    if not sus_results:
        return

    # 导出
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

    tasks: list[tuple[ContentResult, str, str]] = []
    for result, stem in sus_results:
        if do_audio:
            tasks.append((result, stem, "audio"))
        if do_video:
            tasks.append((result, stem, "video"))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_export_one, result, stem, kind)
            for result, stem, kind in tasks
        ]
        for future in as_completed(futures):
            future.result()

    print(f"  [sus] 导出 {len(sus_results)} 个原始片段到 sus/ 文件夹")
