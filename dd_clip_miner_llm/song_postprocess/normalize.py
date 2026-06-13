from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..config import get_padding_config, get_song_normalization_config, is_risk_routed
from ..models import ContentMatch, TranscriptSegment


def _uncovered_segment_ranges(
    segment_count: int,
    matches: list[ContentMatch],
    min_gap_segments: int = 1,
) -> list[tuple[int, int]]:
    """Find ASR segment ranges not covered by any match."""
    covered: set[int] = set()
    for match in matches:
        covered.update(i for i in match.segment_indices if 0 <= i < segment_count)

    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(segment_count):
        if index in covered:
            if start is not None and index - start >= min_gap_segments:
                ranges.append((start, index - 1))
            start = None
            continue
        if start is None:
            start = index

    if start is not None and segment_count - start >= min_gap_segments:
        ranges.append((start, segment_count - 1))
    return ranges


def _split_segment_ranges(
    ranges: list[tuple[int, int]],
    batch_size: int,
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    batch_size = max(1, batch_size)
    for start, end in ranges:
        current = start
        while current <= end:
            chunk_end = min(end, current + batch_size - 1)
            result.append((current, chunk_end))
            current = chunk_end + 1
    return result


def _group_segment_ranges(
    ranges: list[tuple[int, int]],
    max_span_segments: int,
) -> list[list[tuple[int, int]]]:
    """把多个目标区间合并到有限跨度的请求窗口中。"""
    if not ranges:
        return []

    max_span_segments = max(1, max_span_segments)
    groups: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    group_start = 0

    for start, end in ranges:
        if not current:
            current = [(start, end)]
            group_start = start
            continue
        if end - group_start + 1 <= max_span_segments:
            current.append((start, end))
            continue
        groups.append(current)
        current = [(start, end)]
        group_start = start

    if current:
        groups.append(current)
    return groups


def _segment_range_duration_seconds(
    segments: list[TranscriptSegment],
    start: int,
    end: int,
) -> float:
    if not segments or start < 0 or end < start or end >= len(segments):
        return 0.0
    return max(0.0, float(segments[end].end) - float(segments[start].start))


def _filter_short_segment_ranges(
    segments: list[TranscriptSegment],
    ranges: list[tuple[int, int]],
    min_duration_seconds: float,
) -> tuple[list[tuple[int, int]], int]:
    if min_duration_seconds <= 0:
        return ranges, 0

    kept: list[tuple[int, int]] = []
    skipped = 0
    for start, end in ranges:
        if _segment_range_duration_seconds(segments, start, end) < min_duration_seconds:
            skipped += 1
            continue
        kept.append((start, end))
    return kept, skipped


def _expand_segment_range(
    start: int,
    end: int,
    segment_count: int,
    context_segments: int,
) -> tuple[int, int]:
    context_segments = max(0, context_segments)
    return (
        max(0, start - context_segments),
        min(segment_count - 1, end + context_segments),
    )


def _filter_matches_to_segment_range(
    matches: list[ContentMatch],
    start: int,
    end: int,
) -> list[ContentMatch]:
    filtered: list[ContentMatch] = []
    for match in matches:
        segment_indices = sorted({i for i in match.segment_indices if start <= i <= end})
        if not segment_indices:
            continue
        filtered.append(
            ContentMatch(
                content_type=match.content_type,
                title=match.title,
                segment_indices=segment_indices,
                confidence=match.confidence,
                tags=match.tags,
                description=match.description,
                artist=match.artist,
                lyrics_snippet=match.lyrics_snippet,
            )
        )
    return filtered


def _filter_matches_to_segment_ranges(
    matches: list[ContentMatch],
    ranges: list[tuple[int, int]],
) -> list[ContentMatch]:
    allowed: set[int] = set()
    for start, end in ranges:
        allowed.update(range(start, end + 1))

    filtered: list[ContentMatch] = []
    for match in matches:
        segment_indices = sorted(set(match.segment_indices) & allowed)
        if segment_indices:
            filtered.append(_clone_match_with_indices(match, segment_indices))
    return filtered


def _merge_adjacent_same_title_matches(
    segments: list[TranscriptSegment],
    matches: list[ContentMatch],
    max_gap_seconds: float,
) -> tuple[list[ContentMatch], list[dict[str, Any]]]:
    if max_gap_seconds < 0:
        return matches, []

    matches = [m for m in matches if m.segment_indices]
    if not matches:
        return [], []

    merged: list[ContentMatch] = []
    events: list[dict[str, Any]] = []
    for match in sorted(matches, key=lambda item: min(item.segment_indices)):
        candidate_index = None
        candidate_gap = None
        title_key = match.title.strip().casefold()
        for index in range(len(merged) - 1, -1, -1):
            previous = merged[index]
            if previous.title.strip().casefold() != title_key:
                continue
            previous_end = max(previous.segment_indices)
            match_start = min(match.segment_indices)
            if match_start <= previous_end:
                gap_seconds = 0.0
            else:
                gap_seconds = max(
                    0.0,
                    segments[match_start].start - segments[previous_end].end,
                )
            if gap_seconds <= max_gap_seconds:
                candidate_index = index
                candidate_gap = gap_seconds
            break

        if candidate_index is None:
            merged.append(match)
            continue

        previous = merged[candidate_index]
        combined = _clone_match_with_indices(
            previous if previous.confidence >= match.confidence else match,
            sorted(set(previous.segment_indices) | set(match.segment_indices)),
        )
        merged[candidate_index] = combined
        events.append({
            "type": "adjacent_same_title_merge",
            "title": combined.title,
            "gap_seconds": candidate_gap,
            "ranges": _indices_to_ranges(combined.segment_indices),
        })

    merged.sort(key=lambda item: min(item.segment_indices))
    return merged, events


def _is_invalid_audit_title(title: str) -> bool:
    normalized = title.strip().casefold()
    return normalized in {
        "",
        "unknown",
        "未知",
        "未知歌曲",
        "未知歌曲：代表歌词",
        "未知歌曲:代表歌词",
        "对话",
        "对话片段",
        "聊天",
        "聊天片段",
        "非歌曲",
        "无歌曲",
    }


def _clone_match_with_indices(
    match: ContentMatch,
    segment_indices: list[int],
) -> ContentMatch:
    return ContentMatch(
        content_type=match.content_type,
        title=match.title,
        segment_indices=segment_indices,
        confidence=match.confidence,
        tags=match.tags,
        description=match.description,
        artist=match.artist,
        lyrics_snippet=match.lyrics_snippet,
    )


def _content_match_from_dict(
    item: dict[str, Any],
    default_content_type: str = "song",
) -> ContentMatch:
    return ContentMatch(
        content_type=str(item.get("content_type", default_content_type)),
        title=str(item.get("title", "")),
        segment_indices=[
            int(index)
            for index in item.get("segment_indices", [])
            if not isinstance(index, bool)
        ],
        confidence=float(item.get("confidence", 0.5)),
        tags=list(item.get("tags", [])),
        description=str(item.get("description", "")),
        artist=str(item.get("artist", "")),
        lyrics_snippet=str(item.get("lyrics_snippet", "")),
    )


def _indices_to_ranges(indices: list[int]) -> list[list[int]]:
    values = sorted(set(indices))
    if not values:
        return []
    ranges: list[list[int]] = []
    start = previous = values[0]
    for index in values[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append([start, previous])
        start = previous = index
    ranges.append([start, previous])
    return ranges


def _match_key(match: ContentMatch) -> tuple[str, tuple[int, ...]]:
    return (match.title.strip().casefold(), tuple(sorted(set(match.segment_indices))))


def _matches_overlap(left: ContentMatch, right: ContentMatch) -> bool:
    return bool(set(left.segment_indices) & set(right.segment_indices))


def _normalize_song_matches(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    matches: list[ContentMatch],
    force_merge_same_title: set | None = None,
    preserve_time_gaps: bool = False,
) -> tuple[list[ContentMatch], list[dict[str, Any]], set[tuple[str, tuple[int, ...]]]]:
    """Normalize song matches: split disjoint ranges, deduplicate, flag suspicious entries.

    If force_merge_same_title is provided (set of _match_key for LLM-reviewed same-title that should be treated as one performance),
    skip the time-gap split for those, to respect review decision (e.g. for 囚鸟 case where LLM wanted merge across gap).

    Returns:
        (normalized_matches, events, suspicious_keys)
    """
    padding_config = get_padding_config(config, "song")
    merge_gap_seconds = float(padding_config.get("merge_gap_seconds", 40.0))
    max_song_seconds = float(padding_config.get("max_song_seconds", 360.0))
    normalized: list[ContentMatch] = []
    events: list[dict[str, Any]] = []
    suspicious: set[tuple[str, tuple[int, ...]]] = set()
    seen: set[tuple[str, tuple[int, ...]]] = set()
    force_merge_same_title = force_merge_same_title or set()

    for match in matches:
        valid_indices = sorted({i for i in match.segment_indices if 0 <= i < len(segments)})
        if not valid_indices:
            events.append({"type": "invalid_range", "title": match.title})
            continue
        key = _match_key(_clone_match_with_indices(match, valid_indices))
        is_v2 = is_risk_routed(config)
        preserve_adjudicated_range = (
            preserve_time_gaps or "temporal_adjudicated" in match.tags
        )
        if not is_v2 and (key in force_merge_same_title or preserve_adjudicated_range):
            groups = [valid_indices]
        elif is_v2:
            norm_cfg = get_song_normalization_config(config)
            if norm_cfg.get("chorus_aware_split", False):
                groups = _v2_chorus_aware_split(
                    segments, valid_indices, merge_gap_seconds,
                    chorus_gap=float(norm_cfg.get("chorus_gap_seconds", 120.0)),
                    similarity_threshold=float(norm_cfg.get("chorus_similarity_threshold", 0.3)),
                    context_segments=int(norm_cfg.get("chorus_context_segments", 3)),
                )
            else:
                groups = _split_indices_by_time_gap_for_recheck(
                    segments, valid_indices, merge_gap_seconds,
                )
        else:
            groups = _split_indices_by_time_gap_for_recheck(
                segments,
                valid_indices,
                merge_gap_seconds,
            )
        if len(groups) > 1:
            events.append({
                "type": "disjoint_ranges",
                "title": match.title,
                "ranges": [_indices_to_ranges(group) for group in groups],
            })
        for group in groups:
            candidate = _clone_match_with_indices(match, group)
            key = _match_key(candidate)
            if key in seen:
                events.append({
                    "type": "exact_duplicate",
                    "title": candidate.title,
                    "ranges": _indices_to_ranges(candidate.segment_indices),
                })
                continue
            seen.add(key)
            normalized.append(candidate)
            if (
                len(groups) > 1
                and key not in force_merge_same_title
                and not preserve_adjudicated_range
            ):
                suspicious.add(key)
            duration = _segment_range_duration_seconds(
                segments,
                min(group),
                max(group),
            )
            if max_song_seconds > 0 and duration > max_song_seconds:
                suspicious.add(key)
                events.append({
                    "type": "overlong",
                    "title": candidate.title,
                    "duration_seconds": duration,
                })

    # V2: 同名相邻强制合并（在 dedup 之后、sort 之前）
    if is_risk_routed(config):
        norm_cfg = get_song_normalization_config(config)
        if norm_cfg.get("force_merge_same_title", False):
            before_count = len(normalized)
            normalized = _v2_force_merge_adjacent_same_title(
                normalized, segments, gap_seconds=merge_gap_seconds,
            )
            if len(normalized) < before_count:
                events.append({
                    "type": "v2_force_merge_same_title",
                    "before_count": before_count,
                    "after_count": len(normalized),
                })

    normalized.sort(key=lambda match: min(match.segment_indices))
    return normalized, events, suspicious


def _split_indices_by_time_gap_for_recheck(
    segments: list[TranscriptSegment],
    indices: list[int],
    merge_gap_seconds: float,
) -> list[list[int]]:
    if not indices:
        return []

    groups: list[list[int]] = [[indices[0]]]
    for index in indices[1:]:
        previous = groups[-1][-1]
        gap = float(segments[index].start) - float(segments[previous].end)
        if gap > merge_gap_seconds:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def _match_groups_over_max_song_seconds(
    segments: list[TranscriptSegment],
    matches: list[ContentMatch],
    max_song_seconds: float,
    merge_gap_seconds: float,
) -> bool:
    if max_song_seconds <= 0:
        return False
    for match in matches:
        valid_indices = sorted({i for i in match.segment_indices if 0 <= i < len(segments)})
        for group in _split_indices_by_time_gap_for_recheck(
            segments,
            valid_indices,
            merge_gap_seconds,
        ):
            if _segment_range_duration_seconds(segments, min(group), max(group)) > max_song_seconds:
                return True
    return False


# ─── V2 专用函数 ───────────────────────────────────────────────


def _v2_force_merge_adjacent_same_title(
    matches: list[ContentMatch],
    segments: list[TranscriptSegment],
    gap_seconds: float,
) -> list[ContentMatch]:
    """合并相邻同名 match，不跨越其他候选。

    按 min(segment_indices) 排序后，相邻 match 标题 casefold() 相同
    且时间间隔 ≤ gap_seconds 时合并。
    合并后 segment_indices 取并集，confidence 取 max，元数据优先使用高置信结果。
    """
    if not matches:
        return []

    sorted_matches = sorted(matches, key=lambda m: min(m.segment_indices))
    merged: list[ContentMatch] = []

    for match in sorted_matches:
        if not merged:
            merged.append(match)
            continue

        prev = merged[-1]
        title_key = match.title.strip().casefold()
        prev_title_key = prev.title.strip().casefold()

        if title_key != prev_title_key:
            merged.append(match)
            continue

        # 计算时间间隔
        prev_end_idx = max(prev.segment_indices)
        match_start_idx = min(match.segment_indices)
        if match_start_idx <= prev_end_idx:
            gap = 0.0
        else:
            gap = max(0.0, float(segments[match_start_idx].start) - float(segments[prev_end_idx].end))

        if gap > gap_seconds:
            merged.append(match)
            continue

        # 合并：segment_indices 取并集
        combined_indices = sorted(set(prev.segment_indices) | set(match.segment_indices))
        # 高置信结果的元数据优先
        if match.confidence >= prev.confidence:
            merged[-1] = ContentMatch(
                content_type=match.content_type,
                title=match.title,
                segment_indices=combined_indices,
                confidence=match.confidence,
                tags=list(set(match.tags) | set(prev.tags)),
                description=match.description or prev.description,
                artist=match.artist or prev.artist,
                lyrics_snippet=match.lyrics_snippet or prev.lyrics_snippet,
            )
        else:
            merged[-1] = ContentMatch(
                content_type=prev.content_type,
                title=prev.title,
                segment_indices=combined_indices,
                confidence=prev.confidence,
                tags=list(set(prev.tags) | set(match.tags)),
                description=prev.description or match.description,
                artist=prev.artist or match.artist,
                lyrics_snippet=prev.lyrics_snippet or match.lyrics_snippet,
            )

    return merged


def _v2_text_similarity(texts_a: list[str], texts_b: list[str]) -> float:
    """计算两组文本的相似度（overlap coefficient）。

    中文（CJK 占比 > 30%）：去标点后的字符二元组
    英文：小写单词
    """
    import re as _re

    def _tokenize(texts: list[str]) -> set[str]:
        tokens: set[str] = set()
        for text in texts:
            text = text.strip()
            if not text:
                continue
            cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
            if cjk_count > len(text) * 0.3:
                clean = _re.sub(r"[^\u4e00-\u9fff\w]", "", text)
                for i in range(len(clean) - 1):
                    tokens.add(clean[i : i + 2].lower())
            else:
                tokens.update(w.lower() for w in text.split() if w.isalpha())
        return tokens

    ta = _tokenize(texts_a)
    tb = _tokenize(texts_b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _v2_chorus_aware_split(
    segments: list[TranscriptSegment],
    indices: list[int],
    merge_gap: float,
    *,
    chorus_gap: float = 120.0,
    similarity_threshold: float = 0.3,
    context_segments: int = 3,
) -> list[list[int]]:
    """副歌感知的区间拆分。

    - gap ≤ merge_gap：保持同一段
    - merge_gap < gap ≤ chorus_gap：检查文本相似度，≥ 阈值则不拆分（副歌重现）
    - gap > chorus_gap：直接拆分
    """
    if not indices:
        return []

    groups: list[list[int]] = [[indices[0]]]
    for i in range(1, len(indices)):
        prev_idx = groups[-1][-1]
        curr_idx = indices[i]
        gap = float(segments[curr_idx].start) - float(segments[prev_idx].end)

        if gap <= merge_gap:
            groups[-1].append(curr_idx)
        elif gap <= chorus_gap:
            before_texts = [segments[j].text for j in groups[-1][-context_segments:]]
            after_end = min(i + context_segments, len(indices))
            after_texts = [segments[indices[k]].text for k in range(i, after_end)]
            if _v2_text_similarity(before_texts, after_texts) >= similarity_threshold:
                groups[-1].append(curr_idx)
                # debug event for keep-together by chorus (for observability on overlong like 昨日青空)
                # caller can collect if needed; here we note in case of future event list
            else:
                groups.append([curr_idx])
        else:
            groups.append([curr_idx])

    return groups


