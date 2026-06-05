from __future__ import annotations

from typing import Any

from .config import get_padding_config
from .models import ContentMatch, ContentResult, TranscriptSegment


def _merge_adjacent_matches(
    matches: list[dict[str, Any]],
    merge_gap: float,
    max_duration: float | None = None,
) -> list[dict[str, Any]]:
    """合并相邻或重叠的内容片段"""
    if not matches:
        return []

    sorted_matches = sorted(matches, key=lambda s: s["start"])
    merged: list[dict[str, Any]] = [sorted_matches[0]]

    for match in sorted_matches[1:]:
        prev = merged[-1]

        # 检查 segment_indices 是否重叠
        prev_indices = set(range(prev["segment_start_idx"], prev["segment_end_idx"] + 1))
        curr_indices = set(range(match["segment_start_idx"], match["segment_end_idx"] + 1))
        has_overlap = bool(prev_indices & curr_indices)

        same_title_nearby = match["start"] - prev["end"] <= merge_gap and match["title"] == prev["title"]
        merged_duration = max(prev["end"], match["end"]) - min(prev["start"], match["start"])
        within_max_duration = max_duration is None or max_duration <= 0 or merged_duration <= max_duration

        # 如果重叠，或者 title 相同且间隔 ≤ merge_gap，就合并；但歌曲超长时保守拆开。
        if (has_overlap or same_title_nearby) and within_max_duration:
            prev["end"] = max(prev["end"], match["end"])
            prev["segment_end_idx"] = max(prev["segment_end_idx"], match["segment_end_idx"])
            prev["segment_start_idx"] = min(prev["segment_start_idx"], match["segment_start_idx"])
            prev["confidence"] = max(prev["confidence"], match["confidence"])
            prev["transcript"] += " " + match["transcript"]
            if len(match["title"]) > len(prev["title"]):
                prev["title"] = match["title"]
        else:
            merged.append(match)

    return merged


def _split_indices_by_time_gap(
    segments: list[TranscriptSegment],
    indices: list[int],
    merge_gap: float,
) -> list[list[int]]:
    if not indices:
        return []

    groups: list[list[int]] = [[indices[0]]]
    for index in indices[1:]:
        previous = groups[-1][-1]
        gap = float(segments[index].start) - float(segments[previous].end)
        if gap > merge_gap:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def build_content_results(
    segments: list[TranscriptSegment],
    matches: list[ContentMatch],
    total_duration: float,
    config: dict[str, Any],
    content_type: str,
) -> list[ContentResult]:
    """构建内容片段结果"""
    # 获取类型配置
    type_config = config.get(content_type, {})
    
    # 获取 padding 配置（兼容新旧配置结构）
    padding_config = get_padding_config(config, content_type)
    
    # 歌曲使用特殊的 padding 配置
    if content_type == "song":
        before_pad = float(padding_config.get("before_seconds", 15.0))
        after_pad = float(padding_config.get("after_seconds", 15.0))
        after_guard = float(padding_config.get("after_next_asr_end_guard_seconds", 2.0))
        min_duration = float(padding_config.get("min_song_seconds", 75.0))
        max_duration = float(padding_config.get("max_song_seconds", 360.0))
        merge_gap = float(padding_config.get("merge_gap_seconds", 20.0))
    else:
        # 其他类型使用简单 padding
        before_pad = float(padding_config.get("before_seconds", 1.0))
        after_pad = float(padding_config.get("after_seconds", 2.0))
        after_guard = 0.0
        min_duration = float(type_config.get("min_duration", padding_config.get("min_duration", 10.0)))
        max_duration = None
        merge_gap = float(type_config.get("merge_gap_seconds", padding_config.get("merge_gap_seconds", 10.0)))

    raw_matches: list[dict[str, Any]] = []

    for match in matches:
        if not match.segment_indices:
            continue

        valid_indices = sorted({i for i in match.segment_indices if 0 <= i < len(segments)})
        if not valid_indices:
            continue

        for group_indices in _split_indices_by_time_gap(segments, valid_indices, merge_gap):
            start = segments[min(group_indices)].start
            end = segments[max(group_indices)].end
            transcript = " ".join(segments[i].text for i in group_indices)

            raw_matches.append({
                "title": match.title,
                "content_type": match.content_type,
                "start": start,
                "end": end,
                "segment_start_idx": min(group_indices),
                "segment_end_idx": max(group_indices),
                "confidence": match.confidence,
                "transcript": transcript,
                "tags": match.tags,
                "description": match.description,
                "artist": match.artist,
                "lyrics_snippet": match.lyrics_snippet,
            })

    merged = _merge_adjacent_matches(raw_matches, merge_gap, max_duration=max_duration)

    results: list[ContentResult] = []
    for i, item in enumerate(merged):
        item_start = item["start"]
        item_end = item["end"]

        # 应用 padding
        if content_type == "song":
            # 歌曲使用复杂的 padding 逻辑
            # before_limit: 前一个 ASR 的 start + guard_seconds
            if item["segment_start_idx"] > 0:
                prev_segment = segments[item["segment_start_idx"] - 1]
                before_limit = prev_segment.start + after_guard  # 使用 start + guard
            else:
                before_limit = 0.0
            
            # after_limit: 下一个 ASR 的 end - guard_seconds
            if item["segment_end_idx"] + 1 < len(segments):
                next_segment = segments[item["segment_end_idx"] + 1]
                after_limit = max(item_end, next_segment.end - after_guard)
            else:
                after_limit = total_duration
            
            start = min(item_start, max(before_limit, item_start - before_pad))
            end = max(item_end, min(after_limit, item_end + after_pad))
        else:
            # 其他类型简单 padding
            start = max(0.0, item_start - before_pad)
            end = min(total_duration, item_end + after_pad)

        # 确保不超出总时长
        start = max(0.0, start)
        end = min(total_duration, end)

        duration = end - start

        if duration < min_duration:
            continue

        results.append(ContentResult(
            index=i + 1,
            content_type=item.get("content_type", content_type),
            title=item["title"],
            start=start,
            end=end,
            duration=duration,
            transcript=item["transcript"],
            confidence=item["confidence"],
            tags=item.get("tags", []),
            description=item.get("description", ""),
            artist=item.get("artist", ""),
            audio_path=None,
            video_path=None,
            errors=[],
        ))

    return results


# 兼容旧项目的函数别名
def build_song_results(
    segments: list[TranscriptSegment],
    matches: list[ContentMatch],
    total_duration: float,
    config: dict[str, Any],
) -> list[ContentResult]:
    """构建歌曲结果（兼容旧项目）"""
    return build_content_results(segments, matches, total_duration, config, "song")
