from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import ContentMatch, ContentResult, TranscriptSegment


def _format_timecode(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_reports(results: list[ContentResult], reports_dir: Path, content_type: str = "song") -> tuple[Path, Path]:
    """写入报告文件，兼容旧项目格式"""
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON 报告
    json_path = out / f"{content_type}s.json"
    try:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
    except PermissionError:
        json_path = _alternate_report_path(json_path)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)

    # CSV 报告（兼容旧项目格式）
    csv_path = out / f"{content_type}s.csv"
    try:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "index", "start", "end", "duration_seconds",
                "title", "artist", "confidence",
                "audio_path", "video_path", "transcript", "errors",
                "tags", "description",
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "index": r.index,
                    "start": _format_timecode(r.start),
                    "end": _format_timecode(r.end),
                    "duration_seconds": round(r.duration, 3),
                    "title": r.title,
                    "artist": r.artist,
                    "confidence": r.confidence,
                    "audio_path": str(r.audio_path) if r.audio_path else "",
                    "video_path": str(r.video_path) if r.video_path else "",
                    "transcript": r.transcript,
                    "errors": " | ".join(r.errors),
                    "tags": "|".join(r.tags),
                    "description": r.description,
                })
    except PermissionError:
        csv_path = _alternate_report_path(csv_path)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "index", "start", "end", "duration_seconds",
                "title", "artist", "confidence",
                "audio_path", "video_path", "transcript", "errors",
                "tags", "description",
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "index": r.index,
                    "start": _format_timecode(r.start),
                    "end": _format_timecode(r.end),
                    "duration_seconds": round(r.duration, 3),
                    "title": r.title,
                    "artist": r.artist,
                    "confidence": r.confidence,
                    "audio_path": str(r.audio_path) if r.audio_path else "",
                    "video_path": str(r.video_path) if r.video_path else "",
                    "transcript": r.transcript,
                    "errors": " | ".join(r.errors),
                    "tags": "|".join(r.tags),
                    "description": r.description,
                })

    return csv_path, json_path


def write_match_context_reports(
    matches: list[ContentMatch],
    segments: list[TranscriptSegment],
    llm_dir: Path,
    context_segments: int = 10,
    content_type: str = "content",
) -> tuple[Path, Path]:
    """写入匹配上下文报告，兼容旧项目格式"""
    out = Path(llm_dir)
    out.mkdir(parents=True, exist_ok=True)

    # matches.json
    matches_path = out / "matches.json"
    matches_path.write_text(
        json.dumps([m.to_dict() for m in matches], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 构建上下文数据
    rows = []
    payload = []
    for match_index, match in enumerate(matches, start=1):
        valid = sorted({i for i in match.segment_indices if 0 <= i < len(segments)})
        if not valid:
            continue
        first = valid[0]
        last = valid[-1]
        context_start = max(0, first - context_segments)
        context_end = min(len(segments) - 1, last + context_segments)

        context_items = []
        for idx in range(context_start, context_end + 1):
            segment = segments[idx]
            item = {
                "segment_index": idx,
                "start": segment.start,
                "end": segment.end,
                "start_timecode": _format_timecode(segment.start),
                "end_timecode": _format_timecode(segment.end),
                "is_match": idx in valid,
                "text": segment.text,
            }
            context_items.append(item)
            rows.append({
                "match_index": match_index,
                "title": match.title,
                "artist": match.artist,
                "confidence": match.confidence,
                "match_start": _format_timecode(segments[first].start),
                "match_end": _format_timecode(segments[last].end),
                **item,
            })

        payload.append({
            "match_index": match_index,
            "title": match.title,
            "artist": match.artist,
            "lyrics_snippet": match.lyrics_snippet,
            "confidence": match.confidence,
            "segment_indices": valid,
            "start": segments[first].start,
            "end": segments[last].end,
            "start_timecode": _format_timecode(segments[first].start),
            "end_timecode": _format_timecode(segments[last].end),
            "matched_segments": [context_items[i - context_start] for i in valid],
            "context_segments": context_items,
        })

    # match_context.json
    json_path = out / "match_context.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # match_context.csv
    csv_path = out / "match_context.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "match_index", "title", "artist", "confidence", "match_start", "match_end",
            "segment_index", "start", "end", "start_timecode", "end_timecode", "is_match", "text",
        ])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path, json_path


def _alternate_report_path(path: Path) -> Path:
    """生成备用报告路径（当原路径被占用时）"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")
