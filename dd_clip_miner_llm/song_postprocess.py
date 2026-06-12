"""Song normalization, review, overlong recheck, and missed-recheck post-processing."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import get_padding_config
from .models import ContentMatch, TranscriptSegment
from .song_adaptive import (
    ensure_song_adaptive_strategies,
    load_adaptive_strategies_cache,
    resolve_missed_recheck_strategy,
    resolve_review_transcript_scope,
)

from .profile_state import (
    _config_fingerprint,
    _fingerprint_payload,
    _transcript_fingerprint,
)

class _OffsetRecognizer:
    def __init__(self, recognizer: Any, offset: int) -> None:
        self._recognizer = recognizer
        self._offset = offset

    @property
    def name(self) -> str:
        return self._recognizer.name

    @property
    def default_config(self) -> dict[str, Any]:
        return getattr(self._recognizer, "default_config", {})

    def transcript_index_start(self, batch_start: int) -> int:
        return self._offset + batch_start

    def build_prompt(
        self,
        segments: list[TranscriptSegment],
        batch_start: int,
        config: dict[str, Any],
    ) -> str:
        return self._recognizer.build_prompt(segments, self._offset + batch_start, config)

    def build_system_prompt(self, config: dict[str, Any]) -> str | None:
        return self._recognizer.build_system_prompt(config)

    def parse_response(self, items: list[dict[str, Any]], config: dict[str, Any]) -> list[ContentMatch]:
        return self._recognizer.parse_response(items, config)

    def get_tools(self, config: dict[str, Any]) -> Any:
        return self._recognizer.get_tools(config)


class _SongReviewRecognizer:
    name = "song"

    def __init__(self, base_recognizer: Any, candidates: list[ContentMatch]) -> None:
        self._base_recognizer = base_recognizer
        self._candidates = candidates

    @property
    def default_config(self) -> dict[str, Any]:
        return getattr(self._base_recognizer, "default_config", {})

    def build_prompt(
        self,
        segments: list[TranscriptSegment],
        batch_start: int,
        config: dict[str, Any],
    ) -> str:
        lines = [
            f"[{batch_start + index}] ({segment.start:.1f}s-{segment.end:.1f}s) {segment.text}"
            for index, segment in enumerate(segments)
        ]
        candidates = []
        for match in self._candidates:
            candidates.append({
                "title": match.title,
                "artist": match.artist,
                "segment_ranges": _indices_to_ranges(match.segment_indices),
                "confidence": match.confidence,
            })
        return f"""你是歌曲识别结果复核器。请只裁决候选结果之间的冲突，不要总结整场内容。

候选结果：
{json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))}

要求：
1. 根据 ASR 歌词判断正确歌名和精确边界。
2. 删除重复、错误标题和非演唱片段。
3. 不同歌曲的 segment_ranges 不得重叠。
4. 只返回 JSON 数组，每项包含 content_type、title、artist、segment_ranges、confidence、tags、description。
5. 可以使用 search_lyrics 一次确认歌词；无法确认时保留“未知歌曲：代表歌词”。

完整 ASR 转写片段：
{chr(10).join(lines)}"""

    def build_system_prompt(self, config: dict[str, Any]) -> str | None:
        return None

    def parse_response(
        self,
        items: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[ContentMatch]:
        return self._base_recognizer.parse_response(items, config)

    def get_tools(self, config: dict[str, Any]) -> Any:
        return self._base_recognizer.get_tools(config)


class _SongFullReviewRecognizer:
    name = "song"

    def __init__(
        self,
        base_recognizer: Any,
        candidates: list[ContentMatch],
        *,
        target_start: int,
        target_end: int,
        allowed_start: int,
        allowed_end: int,
    ) -> None:
        self._base_recognizer = base_recognizer
        self._candidates = candidates
        self._target_start = target_start
        self._target_end = target_end
        self._allowed_start = allowed_start
        self._allowed_end = allowed_end

    @property
    def default_config(self) -> dict[str, Any]:
        return getattr(self._base_recognizer, "default_config", {})

    def build_prompt(
        self,
        segments: list[TranscriptSegment],
        batch_start: int,
        config: dict[str, Any],
    ) -> str:
        lines = [
            f"[{batch_start + index}] ({segment.start:.1f}s-{segment.end:.1f}s) {segment.text}"
            for index, segment in enumerate(segments)
        ]
        candidates = []
        for match in self._candidates:
            candidates.append({
                "title": match.title,
                "artist": match.artist,
                "segment_ranges": _indices_to_ranges(match.segment_indices),
                "confidence": match.confidence,
            })
        return f"""你是歌曲识别结果复核器。请只裁决候选结果之间的冲突，不要总结整场内容。

候选结果：
{json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))}

目标 segment ranges：
[[{self._target_start},{self._target_end}]]

允许输出的 segment ranges 必须落在：
[[{self._allowed_start},{self._allowed_end}]]

要求：
1. 根据 ASR 歌词判断正确歌名和精确边界。
2. 删除重复、错误标题和非演唱片段。
3. 不同歌曲的 segment_ranges 不得重叠。
4. 只返回 JSON 数组，每项包含 content_type、title、artist、segment_ranges、confidence、tags、description。
5. 可以使用 search_lyrics 一次确认歌词；无法确认时保留“未知歌曲：代表歌词”。

完整 ASR 转写片段：
{chr(10).join(lines)}"""

    def build_system_prompt(self, config: dict[str, Any]) -> str | None:
        return self._base_recognizer.build_system_prompt(config)

    def parse_response(
        self,
        items: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[ContentMatch]:
        return self._base_recognizer.parse_response(items, config)

    def get_tools(self, config: dict[str, Any]) -> Any:
        return self._base_recognizer.get_tools(config)


class _SongCoverageAuditRecognizer:
    name = "song"

    def __init__(
        self,
        base_recognizer: Any,
        target_ranges: list[tuple[int, int]],
        candidates: list[ContentMatch],
    ) -> None:
        self._base_recognizer = base_recognizer
        self._target_ranges = target_ranges
        self._candidates = candidates

    @property
    def default_config(self) -> dict[str, Any]:
        return getattr(self._base_recognizer, "default_config", {})

    def build_prompt(
        self,
        segments: list[TranscriptSegment],
        batch_start: int,
        config: dict[str, Any],
    ) -> str:
        lines = [
            f"[{batch_start + index}] ({segment.start:.1f}s-{segment.end:.1f}s) {segment.text}"
            for index, segment in enumerate(segments)
        ]
        target_ranges = [[start, end] for start, end in self._target_ranges]
        return f"""你是歌曲漏检覆盖审计器。完整 ASR 已经做过一次歌曲识别。

以下范围由已有歌曲结果的覆盖区间反向计算得到。只检查这些目标区间：
{json.dumps(target_ranges, ensure_ascii=False, separators=(",", ":"))}

要求：
1. 未覆盖区间通常包含聊天、报幕、静音和歌曲之间的空隙。不要为了覆盖目标区间而输出结果。
2. 必须按时间顺序检查每个目标区间。连续多句 ASR 呈现歌词、押韵、重复或明显演唱结构时，必须返回；不要只寻找能确认歌名的歌曲。
3. 目标区间可能包含完整漏检歌曲，也可能包含已有歌曲未覆盖的演唱续段。两者都要返回精确的演唱子区间。
4. 不要把整个目标区间直接当作歌曲。聊天、感谢、报幕、点歌、歌曲讨论和零散哼唱必须排除。
5. 不要返回目标区间之外的 segment。本地会再次裁剪并与已有结果去重。
6. 搜索工具只用于确认歌名。搜索失败或无法确认时仍必须返回“未知歌曲：代表歌词”。
7. 使用紧凑 segment_ranges，区间起止均包含。confidence 必须明确填写 0 到 1。
8. 只返回 JSON 数组，每项包含 content_type、title、artist、segment_ranges、confidence、tags、description。
9. 先完整扫描全部目标区间并形成候选列表，再决定是否使用唯一一次歌词搜索。不能发现第一首候选后就停止扫描。
10. 搜索只用于给候选命名，不能让最终结果只包含被搜索的对象。没有搜索的候选也必须保留。
11. 不要输出逐区间分析、解释或 Markdown。不需要搜索时立即返回 JSON。
12. 如果没有足够证据证明存在漏检歌曲，返回合法空数组 []。

完整 ASR 转写片段：
{chr(10).join(lines)}"""

    def build_system_prompt(self, config: dict[str, Any]) -> str | None:
        return self._base_recognizer.build_system_prompt(config)

    def parse_response(
        self,
        items: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[ContentMatch]:
        return self._base_recognizer.parse_response(items, config)

    def get_tools(self, config: dict[str, Any]) -> Any:
        return self._base_recognizer.get_tools(config)


def _uncovered_segment_ranges(
    segment_count: int,
    matches: list[ContentMatch],
    min_gap_segments: int = 1,
) -> list[tuple[int, int]]:
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


def _sanitize_full_transcript_review_results(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    matches: list[ContentMatch],
    candidate_matches: list[ContentMatch],
    target_ranges: list[tuple[int, int]],
) -> tuple[list[ContentMatch], list[dict[str, Any]]]:
    candidate_coverage = {
        index
        for candidate in candidate_matches
        for index in candidate.segment_indices
    }
    events: list[dict[str, Any]] = []
    sanitized: list[ContentMatch] = []
    for match in matches:
        if (
            match.content_type.strip().casefold()
            not in {"song", "music"}
            or _is_invalid_audit_title(match.title)
        ):
            events.append({
                "type": "invalid_audit_result",
                "title": match.title,
                "content_type": match.content_type,
                "ranges": _indices_to_ranges(match.segment_indices),
            })
            continue
        if candidate_coverage.intersection(match.segment_indices):
            sanitized.append(match)
            continue
        cropped = _filter_matches_to_segment_ranges([match], target_ranges)
        if not cropped:
            events.append({
                "type": "review_result_outside_audit_targets",
                "title": match.title,
                "ranges": _indices_to_ranges(match.segment_indices),
            })
            continue
        if cropped[0].segment_indices != match.segment_indices:
            events.append({
                "type": "review_result_cropped_to_audit_targets",
                "title": match.title,
                "before": _indices_to_ranges(match.segment_indices),
                "after": _indices_to_ranges(cropped[0].segment_indices),
            })
        sanitized.extend(cropped)

    sanitized, merge_events = _merge_adjacent_same_title_matches(
        segments,
        sanitized,
        float(
            get_padding_config(config, "song").get(
                "merge_gap_seconds",
                20.0,
            )
        ),
    )
    events.extend(merge_events)
    sanitized, normalization_events, _ = _normalize_song_matches(
        segments,
        config,
        sanitized,
    )
    events.extend(normalization_events)
    return sanitized, events


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


def _load_cached_identify_matches(
    debug_dir: Path,
    recognizer: Any,
    config: dict[str, Any],
    segments: list[TranscriptSegment],
    *,
    debug_phase: str | None = None,
) -> list[ContentMatch] | None:
    from .llm import (
        _try_load_cached_batch,
        build_llm_messages,
        build_providers,
        build_request_debug_metadata,
    )

    providers = [provider for provider in build_providers(config) if provider.api_key]
    if not providers:
        return None
    batch_size = config.get("llm", {}).get("batch_size")
    if batch_size in (None, "", 0, "0"):
        batches = [(0, segments)]
    else:
        size = int(batch_size)
        batches = [
            (start, segments[start:start + size])
            for start in range(0, len(segments), size)
        ]
    tools = recognizer.get_tools(config)
    matches: list[ContentMatch] = []
    for batch_start, batch_segments in batches:
        cached_items: list[dict[str, Any]] | None = None
        messages = build_llm_messages(
            recognizer,
            batch_segments,
            batch_start,
            config,
        )
        for provider in providers:
            metadata = build_request_debug_metadata(
                messages,
                config=config,
                provider=provider,
                recognizer=recognizer,
                segments=batch_segments,
                batch_start=batch_start,
                tools=tools,
                debug_phase=debug_phase,
            )
            cached = _try_load_cached_batch(
                debug_dir,
                batch_start,
                expected_metadata=metadata,
            )
            if cached is not None:
                _, cached_items = cached
                break
        if cached_items is None:
            return None
        matches.extend(recognizer.parse_response(cached_items, config))
    return matches


def _active_debug_files(
    debug_dir: Path,
    *,
    relative_to: Path,
) -> list[str]:
    manifest_path = debug_dir / "active_debug_files.json"
    try:
        values = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        path = debug_dir / str(value)
        if path.is_file():
            result.append(str(path.relative_to(relative_to)).replace("\\", "/"))
    return result


def _cache_reuse_count(debug_dir: Path) -> int:
    count = 0
    for path in debug_dir.glob("llm_batch_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        reuse = payload.get("cache_reuse")
        if isinstance(reuse, dict):
            count += int(reuse.get("count") or 0)
    return count


def _llm_debug_structural_failures(debug_dir: Path) -> list[str]:
    paths = sorted(debug_dir.glob("llm_batch_*.json"))
    if not paths:
        return ["missing_debug"]

    failures: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append("invalid_debug_json")
            continue
        if payload.get("error"):
            failures.append("api_error")
        if payload.get("parse_valid") is not True:
            failures.append("invalid_result_json")
        if payload.get("json_fix_rounds"):
            failures.append("json_repair")
        if payload.get("finish_reason") == "length":
            failures.append("output_truncated")
        if any(
            item.get("finish_reason") == "length"
            for item in payload.get("tool_rounds", [])
            if isinstance(item, dict)
        ):
            failures.append("output_truncated")
    return sorted(set(failures))


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
) -> tuple[list[ContentMatch], list[dict[str, Any]], set[tuple[str, tuple[int, ...]]]]:
    padding_config = get_padding_config(config, "song")
    merge_gap_seconds = float(padding_config.get("merge_gap_seconds", 20.0))
    max_song_seconds = float(padding_config.get("max_song_seconds", 360.0))
    normalized: list[ContentMatch] = []
    events: list[dict[str, Any]] = []
    suspicious: set[tuple[str, tuple[int, ...]]] = set()
    seen: set[tuple[str, tuple[int, ...]]] = set()

    for match in matches:
        valid_indices = sorted({i for i in match.segment_indices if 0 <= i < len(segments)})
        if not valid_indices:
            events.append({"type": "invalid_range", "title": match.title})
            continue
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
            if len(groups) > 1:
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

    normalized.sort(key=lambda match: min(match.segment_indices))
    return normalized, events, suspicious


def _build_song_review_clusters(
    matches: list[ContentMatch],
    suspicious: set[tuple[str, tuple[int, ...]]],
    max_span_segments: int = 500,
    nearby_title_conflict_gap_segments: int = -1,
) -> list[list[ContentMatch]]:
    adjacency: list[set[int]] = [set() for _ in matches]
    included: set[int] = set()
    for index, match in enumerate(matches):
        if _match_key(match) in suspicious:
            included.add(index)
    for left_index, left in enumerate(matches):
        for right_index in range(left_index + 1, len(matches)):
            right = matches[right_index]
            same_title = left.title.strip().casefold() == right.title.strip().casefold()
            same_suspicious_source = (
                same_title
                and _match_key(left) in suspicious
                and _match_key(right) in suspicious
                and (
                    max(max(left.segment_indices), max(right.segment_indices))
                    - min(min(left.segment_indices), min(right.segment_indices))
                    + 1
                    <= max(1, max_span_segments)
                )
            )
            left_start, left_end = min(left.segment_indices), max(left.segment_indices)
            right_start, right_end = min(right.segment_indices), max(right.segment_indices)
            if left_end < right_start:
                gap_segments = right_start - left_end - 1
            elif right_end < left_start:
                gap_segments = left_start - right_end - 1
            else:
                gap_segments = -1
            nearby_title_conflict = (
                not same_title
                and nearby_title_conflict_gap_segments >= 0
                and 0 <= gap_segments <= nearby_title_conflict_gap_segments
                and max(left_end, right_end) - min(left_start, right_start) + 1
                <= max(1, max_span_segments)
            )
            if (
                same_suspicious_source
                or (not same_title and _matches_overlap(left, right))
                or nearby_title_conflict
            ):
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
                included.update({left_index, right_index})

    clusters: list[list[ContentMatch]] = []
    visited: set[int] = set()
    for start in sorted(included):
        if start in visited:
            continue
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)
        clusters.append([matches[index] for index in sorted(component)])
    return clusters


def _llm_debug_has_structural_issue(debug_root: Path) -> bool:
    for path in debug_root.glob("llm_batch_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("json_fix_rounds"):
            return True
        if payload.get("finish_reason") == "length":
            return True
        for tool_round in payload.get("tool_rounds", []):
            if tool_round.get("finish_reason") == "length":
                return True
    return False


def _review_debug_succeeded(debug_root: Path) -> bool:
    paths = list(debug_root.glob("llm_batch_*.json"))
    if not paths:
        return False
    try:
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    except (OSError, json.JSONDecodeError):
        return False
    return all(not payload.get("error") and payload.get("parse_valid") is True for payload in payloads)


def _load_searched_titles(debug_root: Path) -> set[str]:
    titles: set[str] = set()
    for path in debug_root.rglob("llm_batch_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for call in payload.get("tool_calls_log", []):
            arguments = call.get("arguments", {})
            title = str(arguments.get("title", "")).strip()
            if title:
                titles.add(title.casefold())
    return titles


def _local_best_song_cluster(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    cluster: list[ContentMatch],
    searched_titles: set[str],
) -> tuple[list[ContentMatch], list[dict[str, Any]]]:
    padding_config = get_padding_config(config, "song")
    max_song_seconds = float(padding_config.get("max_song_seconds", 360.0))
    ranked: list[tuple[tuple[Any, ...], int, ContentMatch]] = []
    for index, match in enumerate(cluster):
        duration = _segment_range_duration_seconds(
            segments,
            min(match.segment_indices),
            max(match.segment_indices),
        )
        title = match.title.strip()
        known_title = not title.startswith("未知歌曲")
        has_search_evidence = any(
            title.casefold() in searched or searched in title.casefold()
            for searched in searched_titles
        )
        duration_valid = max_song_seconds <= 0 or duration <= max_song_seconds
        score = (
            int(has_search_evidence),
            int(known_title),
            float(match.confidence),
            int(duration_valid),
            -abs(duration - min(max_song_seconds, duration)) if max_song_seconds > 0 else 0.0,
            -index,
        )
        ranked.append((score, index, match))

    selected: list[ContentMatch] = []
    decisions: list[dict[str, Any]] = []
    for score, original_index, match in sorted(ranked, key=lambda item: item[0], reverse=True):
        if any(_matches_overlap(match, kept) for kept in selected):
            decisions.append({
                "title": match.title,
                "action": "discard",
                "reason": "lower local evidence score in overlapping cluster",
                "score": list(score),
            })
            continue
        selected.append(match)
        decisions.append({
            "title": match.title,
            "action": "keep",
            "reason": "highest non-overlapping local evidence score",
            "score": list(score),
            "original_index": original_index,
        })
    selected.sort(key=lambda match: min(match.segment_indices))
    return selected, decisions


def _review_loses_known_title(
    cluster: list[ContentMatch],
    reviewed_matches: list[ContentMatch],
) -> bool:
    return (
        any(not match.title.strip().startswith("未知歌曲") for match in cluster)
        and not any(
            not match.title.strip().startswith("未知歌曲")
            for match in reviewed_matches
        )
    )


def _review_song_matches(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    llm_dir: Path,
    *,
    phase: str,
) -> list[ContentMatch]:
    review_config = config.get("song", {}).get("review", {})
    if review_config.get("enabled", False) is False:
        return matches

    normalized, normalization_events, suspicious = _normalize_song_matches(
        segments,
        config,
        matches,
    )
    if phase == "before_missed_recheck" and _llm_debug_has_structural_issue(llm_dir):
        suspicious.update(_match_key(match) for match in normalized)
        normalization_events.append({"type": "main_output_truncated_or_repaired"})
    max_cluster_span = max(
        1,
        int(review_config.get("max_window_segments", 500) or 500)
        - 2 * int(review_config.get("context_segments", 10) or 0),
    )
    clusters = _build_song_review_clusters(
        normalized,
        suspicious,
        max_span_segments=max_cluster_span,
        nearby_title_conflict_gap_segments=int(
            review_config.get("nearby_title_conflict_gap_segments", 2) or 0
        ),
    )
    review_root = llm_dir / "review" / phase
    review_root.mkdir(parents=True, exist_ok=True)
    full_audit: dict[str, Any] | None = None
    full_audit_candidate_keys: set[tuple[str, tuple[int, ...]]] | None = None
    missed_strategy = str(
        config.get("song", {}).get("missed_recheck", {}).get("strategy", "windowed")
    ).strip().lower()
    missed_audit_path = llm_dir / "missed_recheck" / "audit.json"
    missed_strategy_resolved = missed_strategy
    if missed_audit_path.exists():
        try:
            missed_audit = json.loads(missed_audit_path.read_text(encoding="utf-8"))
            missed_strategy_resolved = str(
                missed_audit.get("strategy_resolved")
                or missed_audit.get("strategy")
                or missed_strategy
            ).strip().lower()
        except (OSError, json.JSONDecodeError):
            missed_strategy_resolved = missed_strategy
    if (
        phase == "after_missed_recheck"
        and missed_strategy_resolved == "full_transcript"
    ):
        full_audit_path = llm_dir / "missed_recheck" / "audit.json"
        try:
            loaded_audit = json.loads(full_audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded_audit = None
        if isinstance(loaded_audit, dict):
            full_audit = loaded_audit
            full_audit_candidates = [
                _content_match_from_dict(item)
                for item in loaded_audit.get("candidates", [])
                if isinstance(item, dict)
            ]
            full_audit_candidate_keys = {
                _match_key(match) for match in full_audit_candidates
            }
    searched_titles = _load_searched_titles(llm_dir)
    context_segments = int(review_config.get("context_segments", 10) or 0)
    max_window_segments = int(review_config.get("max_window_segments", 500) or 500)
    transcript_scope_requested = str(
        review_config.get("transcript_scope", "local")
    ).strip().lower()
    scope_cost_details: dict[str, Any] = {}
    if phase == "before_missed_recheck":
        min_gap_segments = int(
            config.get("song", {}).get("missed_recheck", {}).get("min_gap_segments", 1) or 1
        )
        min_song_seconds = float(
            get_padding_config(config, "song").get("min_song_seconds", 75.0)
        )
        preview_ranges, _ = _filter_short_segment_ranges(
            segments,
            _uncovered_segment_ranges(
                len(segments),
                normalized,
                min_gap_segments=min_gap_segments,
            ),
            min_song_seconds,
        )
        adaptive_resolution = ensure_song_adaptive_strategies(
            llm_dir,
            config,
            clusters=clusters,
            segments=segments,
            matches=normalized,
            target_ranges=preview_ranges,
            recognizer=recognizer,
        )
    else:
        adaptive_resolution = load_adaptive_strategies_cache(llm_dir) or {}

    if adaptive_resolution.get("review_scope_resolved"):
        transcript_scope = str(adaptive_resolution["review_scope_resolved"])
        transcript_scope_reason = str(adaptive_resolution.get("reason", "cached_joint"))
        scope_cost_details = dict(adaptive_resolution.get("review_details") or {})
        if scope_cost_details:
            scope_cost_details["joint_resolution_mode"] = adaptive_resolution.get(
                "resolution_mode"
            )
            scope_cost_details["joint_combinations"] = adaptive_resolution.get(
                "combinations",
                []
            )
            scope_cost_details["joint_chosen_total_usd"] = adaptive_resolution.get(
                "chosen_total_usd"
            )
    else:
        transcript_scope, transcript_scope_reason, scope_cost_details = (
            resolve_review_transcript_scope(
                config,
                clusters=clusters,
                segments=segments,
                recognizer=recognizer,
            )
        )

    if (
        phase == "before_missed_recheck"
        and adaptive_resolution.get("resolution_mode") == "joint_cost_estimate"
    ):
        print(
            "  Song adaptive (joint): "
            f"review={adaptive_resolution.get('review_scope_resolved')} + "
            f"missed={adaptive_resolution.get('missed_strategy_resolved')} "
            f"({adaptive_resolution.get('reason')}, "
            f"pipeline=${adaptive_resolution.get('chosen_total_usd')}, "
            f"main=${adaptive_resolution.get('main_cost_usd')}, "
            f"review_before=${adaptive_resolution.get('review_before_cost_usd')}, "
            f"overlong=${adaptive_resolution.get('overlong_cost_usd')}, "
            f"missed=${adaptive_resolution.get('missed_cost_usd')}, "
            f"review_after=${adaptive_resolution.get('review_after_cost_usd')})"
        )
    elif transcript_scope != transcript_scope_requested:
        print(
            f"  Song review ({phase}): adaptive transcript_scope "
            f"{transcript_scope_requested} -> {transcript_scope} "
            f"({transcript_scope_reason}, clusters={len(clusters)})"
        )
    resolved: list[ContentMatch] = []
    cluster_member_ids = {id(match) for cluster in clusters for match in cluster}
    resolved.extend(match for match in normalized if id(match) not in cluster_member_ids)
    audit_clusters: list[dict[str, Any]] = []

    for cluster_index, cluster in enumerate(clusters, 1):
        target_start = min(min(match.segment_indices) for match in cluster)
        target_end = max(max(match.segment_indices) for match in cluster)
        context_start, context_end = _expand_segment_range(
            target_start,
            target_end,
            len(segments),
            context_segments,
        )
        audit: dict[str, Any] = {
            "cluster": cluster_index,
            "target_range": [target_start, target_end],
            "context_range": [context_start, context_end],
            "before": [match.to_dict() for match in cluster],
            "resolution": None,
            "after": [],
        }
        if (
            full_audit_candidate_keys is not None
            and all(_match_key(match) in full_audit_candidate_keys for match in cluster)
        ):
            resolved.extend(cluster)
            audit["resolution"] = "unchanged_pre_audit_cluster"
            audit["after"] = [match.to_dict() for match in cluster]
            audit_clusters.append(audit)
            continue
        reviewed_matches: list[ContentMatch] = []
        review_succeeded = False
        audit_path = review_root / f"cluster_{cluster_index:03d}.json"
        cluster_span = target_end - target_start + 1
        debug_phase = (
            "review_before"
            if phase == "before_missed_recheck"
            else "review_after"
        )
        if cluster_span <= max_window_segments:
            local_config = deepcopy(config)
            local_config["llm"]["max_tokens"] = int(
                review_config.get("max_completion_tokens", 4096) or 4096
            )
            local_config["llm"]["max_completion_tokens"] = int(
                review_config.get("max_completion_tokens", 4096) or 4096
            )
            local_config["llm"]["final_tool_max_tokens"] = int(
                review_config.get("max_completion_tokens", 4096) or 4096
            )
            local_config["llm"]["max_tool_rounds"] = int(
                review_config.get("max_tool_rounds", 1) or 0
            )
            local_config["llm"]["compact_segment_ranges"] = True
            cluster_debug_dir = review_root / f"cluster_{cluster_index:03d}"
            from .llm import identify_content
            if transcript_scope == "full":
                review_recognizer = _SongFullReviewRecognizer(
                    recognizer,
                    cluster,
                    target_start=target_start,
                    target_end=target_end,
                    allowed_start=context_start,
                    allowed_end=context_end,
                )
                reviewed_matches = identify_content(
                    segments,
                    local_config,
                    review_recognizer,
                    debug_dir=cluster_debug_dir,
                    debug_phase=debug_phase,
                )
            else:
                review_recognizer = _SongReviewRecognizer(recognizer, cluster)
                offset_recognizer = _OffsetRecognizer(review_recognizer, context_start)
                reviewed_matches = identify_content(
                    segments[context_start:context_end + 1],
                    local_config,
                    offset_recognizer,
                    debug_dir=cluster_debug_dir,
                    debug_phase=debug_phase,
                )
            reviewed_matches = _filter_matches_to_segment_range(
                reviewed_matches,
                context_start,
                context_end,
            )
            reviewed_matches, review_events, _ = _normalize_song_matches(
                segments,
                config,
                reviewed_matches,
            )
            has_conflict = any(
                left.title.strip().casefold() != right.title.strip().casefold()
                and _matches_overlap(left, right)
                for left_index, left in enumerate(reviewed_matches)
                for right in reviewed_matches[left_index + 1:]
            )
            loses_known_title = _review_loses_known_title(
                cluster,
                reviewed_matches,
            )
            review_succeeded = (
                _review_debug_succeeded(cluster_debug_dir)
                and not has_conflict
                and not loses_known_title
            )
            audit["review_normalization_events"] = review_events
            if loses_known_title:
                audit["review_rejected_reason"] = "known_title_downgraded_to_unknown"
        else:
            audit["skip_reason"] = "review cluster exceeds max_window_segments"

        if review_succeeded:
            replacement = reviewed_matches
            audit["resolution"] = "llm"
        else:
            replacement, decisions = _local_best_song_cluster(
                segments,
                config,
                cluster,
                searched_titles,
            )
            audit["resolution"] = "local_best"
            audit["local_decisions"] = decisions
        resolved.extend(replacement)
        audit["after"] = [match.to_dict() for match in replacement]
        audit_clusters.append(audit)
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    final_matches, final_events, _ = _normalize_song_matches(segments, config, resolved)
    residual_clusters = _build_song_review_clusters(final_matches, set())
    if residual_clusters:
        residual_ids = {id(match) for cluster in residual_clusters for match in cluster}
        conflict_free = [match for match in final_matches if id(match) not in residual_ids]
        for cluster in residual_clusters:
            replacement, decisions = _local_best_song_cluster(
                segments,
                config,
                cluster,
                searched_titles,
            )
            conflict_free.extend(replacement)
            final_events.append({
                "type": "residual_conflict_local_best",
                "decisions": decisions,
            })
        final_matches = sorted(
            conflict_free,
            key=lambda match: min(match.segment_indices),
        )
    sanitation_events: list[dict[str, Any]] = []
    if full_audit is not None:
        full_audit_candidates = [
            _content_match_from_dict(item)
            for item in full_audit.get("candidates", [])
            if isinstance(item, dict)
        ]
        target_ranges = [
            (int(item[0]), int(item[1]))
            for item in full_audit.get("target_ranges", [])
            if isinstance(item, list) and len(item) == 2
        ]
        final_matches, sanitation_events = _sanitize_full_transcript_review_results(
            segments,
            config,
            final_matches,
            full_audit_candidates,
            target_ranges,
        )
    summary = {
        "phase": phase,
        "input_count": len(matches),
        "normalized_count": len(normalized),
        "cluster_count": len(clusters),
        "output_count": len(final_matches),
        "transcript_scope_requested": transcript_scope_requested,
        "transcript_scope_resolved": transcript_scope,
        "transcript_scope_reason": transcript_scope_reason,
        **scope_cost_details,
        "normalization_events": normalization_events,
        "final_normalization_events": final_events,
        "full_transcript_sanitation_events": sanitation_events,
        "clusters": audit_clusters,
    }
    (review_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return final_matches


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


def _recheck_overlong_song_matches(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    llm_dir: Path,
) -> list[ContentMatch]:
    if not segments or not matches:
        return matches

    recheck_config = config.get("song", {}).get("missed_recheck", {})
    if recheck_config.get("enabled", True) is False:
        return matches

    padding_config = get_padding_config(config, "song")
    max_song_seconds = float(padding_config.get("max_song_seconds", 360.0))
    merge_gap_seconds = float(padding_config.get("merge_gap_seconds", 20.0))
    if max_song_seconds <= 0:
        return matches

    context_segments = int(recheck_config.get("context_segments", 10) or 0)

    from .llm import identify_content

    recheck_root = llm_dir / "overlong_recheck"
    replacement_matches: list[ContentMatch] = []
    rechecked_count = 0
    replaced_count = 0
    kept_count = 0
    all_rechecked_matches: list[ContentMatch] = []
    active_debug_files: list[str] = []

    for mi, match in enumerate(matches, 1):
        valid_indices = sorted({i for i in match.segment_indices if 0 <= i < len(segments)})
        groups = _split_indices_by_time_gap_for_recheck(
            segments,
            valid_indices,
            merge_gap_seconds,
        )
        if not groups:
            continue

        if not any(
            _segment_range_duration_seconds(segments, min(group), max(group)) > max_song_seconds
            for group in groups
        ):
            replacement_matches.append(match)
            continue

        recheck_root.mkdir(parents=True, exist_ok=True)
        for gi, group in enumerate(groups, 1):
            group_start = min(group)
            group_end = max(group)
            group_match = _clone_match_with_indices(match, group)
            if _segment_range_duration_seconds(segments, group_start, group_end) <= max_song_seconds:
                replacement_matches.append(group_match)
                continue

            rechecked_count += 1
            print(f"  Song overlong recheck: match {mi}/{len(matches)}, group {gi}/{len(groups)} (segments {group_start}-{group_end})...")
            context_start, context_end = _expand_segment_range(
                group_start,
                group_end,
                len(segments),
                context_segments,
            )
            chunk = segments[context_start:context_end + 1]
            debug_dir = recheck_root / f"{group_start:06d}_{group_end:06d}"
            offset_recognizer = _OffsetRecognizer(recognizer, context_start)
            raw_rechecked_matches = identify_content(
                chunk,
                config,
                offset_recognizer,
                debug_dir=debug_dir,
                debug_phase="overlong",
            )
            active_debug_files.extend(
                _active_debug_files(debug_dir, relative_to=recheck_root)
            )
            rechecked_matches = _filter_matches_to_segment_range(
                raw_rechecked_matches,
                group_start,
                group_end,
            )
            all_rechecked_matches.extend(rechecked_matches)

            if rechecked_matches and not _match_groups_over_max_song_seconds(
                segments,
                rechecked_matches,
                max_song_seconds,
                merge_gap_seconds,
            ):
                replacement_matches.extend(rechecked_matches)
                replaced_count += 1
            else:
                replacement_matches.append(group_match)
                kept_count += 1

    if not rechecked_count:
        return matches

    if all_rechecked_matches:
        (recheck_root / "matches.json").write_text(
            json.dumps([m.to_dict() for m in all_rechecked_matches], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (recheck_root / "audit.json").write_text(
        json.dumps(
            {
                "status": "success",
                "active_debug_files": sorted(set(active_debug_files)),
                "rechecked_count": rechecked_count,
                "replaced_count": replaced_count,
                "kept_count": kept_count,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "  Song overlong recheck: "
        f"checked {rechecked_count} range(s), replaced {replaced_count}, kept original {kept_count}"
    )
    return replacement_matches


def _run_windowed_missed_recheck(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    recheck_root: Path,
) -> tuple[list[ContentMatch], list[str]]:
    recheck_config = config.get("song", {}).get("missed_recheck", {})
    min_gap_segments = int(recheck_config.get("min_gap_segments", 1) or 1)
    context_segments = int(recheck_config.get("context_segments", 10) or 0)
    padding_config = get_padding_config(config, "song")
    min_song_seconds = float(padding_config.get("min_song_seconds", 75.0))
    batch_size_value = recheck_config.get("batch_size", config.get("llm", {}).get("batch_size") or 500)
    batch_size = int(batch_size_value or 500)
    ranges = _split_segment_ranges(
        _uncovered_segment_ranges(len(segments), matches, min_gap_segments=min_gap_segments),
        batch_size,
    )
    ranges, skipped_short = _filter_short_segment_ranges(segments, ranges, min_song_seconds)
    if skipped_short:
        print(
            f"  Song missed recheck: skipped {skipped_short} short ASR range(s) "
            f"below min_song_seconds={min_song_seconds:g}"
        )
    if not ranges:
        return [], []

    from .llm import identify_content

    range_groups = _group_segment_ranges(ranges, batch_size)
    print(
        f"  Song missed recheck: {len(ranges)} uncovered ASR range(s), "
        f"combined into {len(range_groups)} request window(s)"
    )
    extra_matches: list[ContentMatch] = []
    recheck_root.mkdir(parents=True, exist_ok=True)
    active_debug_files: list[str] = []

    for ri, target_ranges in enumerate(range_groups, 1):
        start = target_ranges[0][0]
        end = target_ranges[-1][1]
        context_start, context_end = _expand_segment_range(
            start,
            end,
            len(segments),
            context_segments,
        )
        chunk = segments[context_start:context_end + 1]
        debug_dir = recheck_root / f"{start:06d}_{end:06d}"
        offset_recognizer = _OffsetRecognizer(recognizer, context_start)
        print(
            f"  Song missed recheck: processing window {ri}/{len(range_groups)} "
            f"(segments {start}-{end}, {len(target_ranges)} target range(s))..."
        )
        rechecked_matches = identify_content(
            chunk,
            config,
            offset_recognizer,
            debug_dir=debug_dir,
            debug_phase="missed_recheck",
        )
        active_debug_files.extend(
            _active_debug_files(debug_dir, relative_to=recheck_root)
        )
        for target_start, target_end in target_ranges:
            extra_matches.extend(
                _filter_matches_to_segment_range(
                    rechecked_matches,
                    target_start,
                    target_end,
                )
            )
        print(
            f"  Song missed recheck: window {ri}/{len(range_groups)} done, "
            f"found {len(rechecked_matches)} match(es)"
        )

    if extra_matches:
        (recheck_root / "matches.json").write_text(
            json.dumps([m.to_dict() for m in extra_matches], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Song missed recheck: found {len(extra_matches)} additional match(es)")
        return extra_matches, active_debug_files

    print("  Song missed recheck: no additional matches")
    return [], active_debug_files


def _finalize_windowed_missed_recheck_matches(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    matches: list[ContentMatch],
    extra_matches: list[ContentMatch],
) -> tuple[list[ContentMatch], dict[str, Any]]:
    """Normalize windowed missed-recheck output before returning."""
    combined = [*matches, *extra_matches]
    before_count = len(combined)
    normalized, normalization_events, _ = _normalize_song_matches(
        segments,
        config,
        combined,
    )
    merge_gap_seconds = float(
        get_padding_config(config, "song").get("merge_gap_seconds", 20.0)
    )
    merged, merge_events = _merge_adjacent_same_title_matches(
        segments,
        normalized,
        merge_gap_seconds,
    )
    events = [*normalization_events, *merge_events]
    return merged, {
        "before_count": before_count,
        "after_count": len(merged),
        "normalization_events": events,
    }


def _missed_recheck_fingerprint(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    matches: list[ContentMatch],
    target_ranges: list[tuple[int, int]],
) -> dict[str, str]:
    candidates = sorted(
        (
            {
                "title": match.title,
                "artist": match.artist,
                "segment_ranges": _indices_to_ranges(match.segment_indices),
                "confidence": match.confidence,
            }
            for match in matches
        ),
        key=lambda item: (
            item["segment_ranges"][0][0] if item["segment_ranges"] else -1,
            item["title"],
        ),
    )
    return {
        "protocol": "full_transcript_coverage_audit_v5",
        "transcript": _transcript_fingerprint(segments),
        "candidates": _fingerprint_payload(candidates),
        "target_ranges": _fingerprint_payload(target_ranges),
        "config": _config_fingerprint(config),
    }


def _load_missed_recheck_audit(
    audit_path: Path,
    fingerprints: dict[str, str],
) -> dict[str, Any] | None:
    if not audit_path.exists():
        return None
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if audit.get("fingerprints") != fingerprints:
        return None
    return audit


def _recheck_uncovered_song_segments(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    llm_dir: Path,
) -> list[ContentMatch]:
    recheck_config = config.get("song", {}).get("missed_recheck", {})
    if recheck_config.get("enabled", True) is False:
        return matches

    strategy_requested = str(recheck_config.get("strategy", "windowed")).strip().lower()

    recheck_root = llm_dir / "missed_recheck"
    recheck_root.mkdir(parents=True, exist_ok=True)

    min_gap_segments = int(recheck_config.get("min_gap_segments", 1) or 1)
    min_song_seconds = float(
        get_padding_config(config, "song").get("min_song_seconds", 75.0)
    )
    preview_ranges, _ = _filter_short_segment_ranges(
        segments,
        _uncovered_segment_ranges(
            len(segments),
            matches,
            min_gap_segments=min_gap_segments,
        ),
        min_song_seconds,
    )
    adaptive_resolution = load_adaptive_strategies_cache(llm_dir) or {}
    if not adaptive_resolution:
        review_config = config.get("song", {}).get("review", {})
        normalized, _, suspicious = _normalize_song_matches(segments, config, matches)
        max_cluster_span = max(
            1,
            int(review_config.get("max_window_segments", 500) or 500)
            - 2 * int(review_config.get("context_segments", 10) or 0),
        )
        clusters = (
            _build_song_review_clusters(
                normalized,
                suspicious,
                max_span_segments=max_cluster_span,
                nearby_title_conflict_gap_segments=int(
                    review_config.get("nearby_title_conflict_gap_segments", 2) or 0
                ),
            )
            if review_config.get("enabled", False)
            else []
        )
        adaptive_resolution = ensure_song_adaptive_strategies(
            llm_dir,
            config,
            clusters=clusters,
            segments=segments,
            matches=normalized,
            target_ranges=preview_ranges,
            recognizer=recognizer,
        )
    if adaptive_resolution.get("missed_strategy_resolved"):
        strategy = str(adaptive_resolution["missed_strategy_resolved"])
        strategy_reason = str(adaptive_resolution.get("reason", "cached_joint"))
        strategy_cost_details = dict(adaptive_resolution.get("missed_details") or {})
        if strategy_cost_details:
            strategy_cost_details["joint_resolution_mode"] = adaptive_resolution.get(
                "resolution_mode"
            )
            strategy_cost_details["joint_combinations"] = adaptive_resolution.get(
                "combinations",
                []
            )
            strategy_cost_details["joint_chosen_total_usd"] = adaptive_resolution.get(
                "chosen_total_usd"
            )
    else:
        strategy, strategy_reason, strategy_cost_details = resolve_missed_recheck_strategy(
            config,
            segments=segments,
            matches=matches,
            target_ranges=preview_ranges,
            recognizer=recognizer,
        )
    if (
        strategy != strategy_requested
        and adaptive_resolution.get("resolution_mode") != "joint_cost_estimate"
    ):
        print(
            f"  Song missed recheck: adaptive strategy {strategy_requested} -> "
            f"{strategy} ({strategy_reason}, targets={len(preview_ranges)})"
        )

    if strategy == "windowed":
        extra_matches, active_files = _run_windowed_missed_recheck(
            segments,
            config,
            recognizer,
            matches,
            recheck_root,
        )
        finalized_matches, normalization_summary = _finalize_windowed_missed_recheck_matches(
            segments,
            config,
            matches,
            extra_matches,
        )
        (recheck_root / "audit.json").write_text(
            json.dumps(
                {
                    "strategy": "windowed",
                    "strategy_requested": strategy_requested,
                    "strategy_resolved": strategy,
                    "strategy_reason": strategy_reason,
                    **strategy_cost_details,
                    "status": "success",
                    "fallback_used": False,
                    "active_debug_files": active_files,
                    "additional_match_count": len(extra_matches),
                    **normalization_summary,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return finalized_matches

    target_ranges, skipped_short = _filter_short_segment_ranges(
        segments,
        _uncovered_segment_ranges(
            len(segments),
            matches,
            min_gap_segments=min_gap_segments,
        ),
        min_song_seconds,
    )
    if skipped_short:
        print(
            f"  Song missed recheck: skipped {skipped_short} short ASR range(s) "
            f"below min_song_seconds={min_song_seconds:g}"
        )
    if not target_ranges:
        return matches

    fingerprints = _missed_recheck_fingerprint(
        segments,
        config,
        matches,
        target_ranges,
    )
    audit_path = recheck_root / "audit.json"
    cached_audit = _load_missed_recheck_audit(audit_path, fingerprints)
    fallback_strategy = str(
        recheck_config.get(
            "fallback_strategy",
            "windowed_on_structural_failure",
        )
    ).strip().lower()

    if cached_audit and cached_audit.get("status") == "fallback_success":
        extra_matches, active_files = _run_windowed_missed_recheck(
            segments,
            config,
            recognizer,
            matches,
            recheck_root / "fallback_windowed",
        )
        cached_audit["active_debug_files"] = [
            f"fallback_windowed/{path}" for path in active_files
        ]
        audit_path.write_text(
            json.dumps(cached_audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return [*matches, *extra_matches]

    audit_recognizer = _SongCoverageAuditRecognizer(
        recognizer,
        target_ranges,
        matches,
    )
    audit_debug_dir = recheck_root / "full_transcript"
    raw_matches: list[ContentMatch] | None = None
    reuse_count_before = _cache_reuse_count(audit_debug_dir)

    if raw_matches is None:
        from .llm import identify_content

        local_config = deepcopy(config)
        max_completion_tokens = int(
            recheck_config.get("max_completion_tokens", 4096) or 4096
        )
        local_config["llm"]["max_tokens"] = max_completion_tokens
        local_config["llm"]["max_completion_tokens"] = max_completion_tokens
        local_config["llm"]["final_tool_max_tokens"] = max_completion_tokens
        local_config["llm"]["max_tool_rounds"] = int(
            recheck_config.get("max_tool_rounds", 1) or 0
        )
        local_config["llm"]["force_final_tool_round"] = True
        local_config["llm"]["final_tool_instruction"] = (
            "歌词搜索只用于确认某个候选的名称，不能替代完整覆盖审计。"
            "现在重新检查原任务列出的每一个目标区间，返回所有有明确连续演唱证据的候选；"
            "每个 segment_ranges 必须与目标区间相交，不能只返回刚刚搜索的对象，"
            "也不能返回已有覆盖区间。不要再调用工具。只返回 JSON 数组，不要解释或 Markdown。"
        )
        local_config["llm"]["retry_empty_with_reasoning"] = False
        local_config["llm"]["json_fix_rounds"] = 0
        raw_matches = identify_content(
            segments,
            local_config,
            audit_recognizer,
            debug_dir=audit_debug_dir,
            debug_phase="missed_recheck",
        )

    cache_reused = _cache_reuse_count(audit_debug_dir) > reuse_count_before
    structural_failures = _llm_debug_structural_failures(audit_debug_dir)
    active_debug_files = _active_debug_files(
        audit_debug_dir,
        relative_to=recheck_root,
    )
    audit: dict[str, Any] = {
        "strategy": "full_transcript",
        "strategy_requested": strategy_requested,
        "strategy_resolved": strategy,
        "strategy_reason": strategy_reason,
        **strategy_cost_details,
        "status": "success" if not structural_failures else "structural_failure",
        "fallback_strategy": fallback_strategy,
        "fallback_used": False,
        "fingerprints": fingerprints,
        "target_ranges": [[start, end] for start, end in target_ranges],
        "candidate_count": len(matches),
        "candidates": [match.to_dict() for match in matches],
        "cache_reused": cache_reused,
        "structural_failures": structural_failures,
        "active_debug_files": active_debug_files,
    }
    review_trigger_matches: list[ContentMatch] = []

    if (
        structural_failures
        and fallback_strategy == "windowed_on_structural_failure"
    ):
        print(
            "  Song missed recheck: full transcript audit failed structurally "
            f"({', '.join(structural_failures)}); falling back to windowed"
        )
        extra_matches, fallback_files = _run_windowed_missed_recheck(
            segments,
            config,
            recognizer,
            matches,
            recheck_root / "fallback_windowed",
        )
        audit["status"] = "fallback_success"
        audit["fallback_used"] = True
        audit["active_debug_files"] = [
            f"fallback_windowed/{path}" for path in fallback_files
        ]
    elif structural_failures:
        extra_matches = []
    else:
        cropped_matches = _filter_matches_to_segment_ranges(
            raw_matches,
            target_ranges,
        )
        review_trigger_matches = [
            match
            for match in cropped_matches
            if _is_invalid_audit_title(match.title)
        ]
        rejected_titles = [
            {
                "title": match.title,
                "ranges": _indices_to_ranges(match.segment_indices),
            }
            for match in review_trigger_matches
        ]
        if rejected_titles:
            audit["rejected_invalid_titles"] = rejected_titles
        extra_matches = [
            match
            for match in cropped_matches
            if not _is_invalid_audit_title(match.title)
        ]
        if config.get("song", {}).get("review", {}).get("enabled", False):
            audit["review_trigger_count"] = len(review_trigger_matches)
        else:
            review_trigger_matches = []

    audit["additional_match_count"] = len(extra_matches)
    combined_matches = [
        *matches,
        *extra_matches,
        *review_trigger_matches,
    ]
    if not structural_failures:
        combined_matches, merge_events = _merge_adjacent_same_title_matches(
            segments,
            combined_matches,
            float(
                get_padding_config(config, "song").get(
                    "merge_gap_seconds",
                    20.0,
                )
            ),
        )
        audit["same_title_merge_events"] = merge_events
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if extra_matches:
        (recheck_root / "matches.json").write_text(
            json.dumps(
                [match.to_dict() for match in extra_matches],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  Song missed recheck: found {len(extra_matches)} additional match(es)")
    else:
        print("  Song missed recheck: no additional matches")
    return combined_matches


