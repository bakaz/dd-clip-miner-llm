from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..models import ContentMatch, TranscriptSegment
from .normalize import (
    _clone_match_with_indices,
    _indices_to_ranges,
    _normalize_song_matches,
)


class _SongTemporalAdjudicationRecognizer:
    """Full-transcript, no-tool recognizer concerned only with performance timing."""

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
        preserve_names = config.get("song", {}).get("pipeline", {}).get(
            "temporal_adjudication", {}
        ).get("preserve_source_names", False)

        lines = [
            f"[{batch_start + index}] ({segment.start:.1f}s-{segment.end:.1f}s) {segment.text}"
            for index, segment in enumerate(segments)
        ]
        candidates = []
        for idx, match in enumerate(self._candidates):
            if not match.segment_indices:
                continue
            entry: dict[str, Any] = {
                "candidate_id": idx,
                "segment_ranges": _indices_to_ranges(match.segment_indices),
                "confidence": match.confidence,
            }
            if preserve_names:
                entry["title"] = match.title
                entry["artist"] = match.artist
            candidates.append(entry)

        if preserve_names:
            rule_8 = (
                '8. 如果区间边界基本对应某个第一轮候选，使用其 title 和 artist。'
                '新增区间填写"未知歌曲"。不允许因无法确认名称而删除有演唱证据的区间。'
            )
        else:
            rule_8 = '8. title 固定填写"未知歌曲：时序裁决"，artist 为空，tags 为空，description 为空。'

        return f"""你是直播歌曲演唱区间的时序裁决器。主要判断演唱分段。

第一轮候选区间：
{json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))}

任务：重新按时间顺序检查完整 ASR，返回最终的连续演唱区间。

严格要求：
1. 默认输入是聊天。只有连续多句呈现歌词、重复副歌、稳定节奏或明确演唱结构时才标记。
2. 商品讨论、感谢、报幕、点歌、歌曲名称讨论、普通聊天和单句哼唱不是歌曲区间。
3. 第一轮候选是参考，不是事实：修正其起止边界，删除误报，并补充明确漏检的完整演唱区间。
4. 同一首歌的连续主歌、副歌和桥段合为一个区间；中间短暂停顿可以保留。
5. 不同歌曲必须拆开。长时间聊天、报幕或明显换歌必须形成边界。
6. 不要逐句输出，不要为聊天生成低置信度候选。没有充分证据就不输出。
7. 每个对象只包含 content_type、title、artist、segment_ranges、confidence、tags、description。
{rule_8}
9. segment_ranges 必须是连续、精确、互不重叠的区间，起止均包含。
10. 只返回 JSON 数组，不要解释、Markdown 或分析过程。

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

    def get_tools(self, config: dict[str, Any]) -> None:
        return None


def _temporal_overlap(left: ContentMatch, right: ContentMatch) -> int:
    return len(set(left.segment_indices) & set(right.segment_indices))


def _temporal_tags(match: ContentMatch) -> list[str]:
    return [*dict.fromkeys([*match.tags, "temporal_adjudicated"])]


def _split_temporal_at_source_boundaries(
    temporal_matches: list[ContentMatch],
    source_matches: list[ContentMatch],
) -> tuple[list[ContentMatch], list[dict[str, Any]]]:
    """Recover strong first-pass boundaries swallowed by temporal consolidation."""
    refined: list[ContentMatch] = []
    events: list[dict[str, Any]] = []
    for temporal in temporal_matches:
        temporal_indices = sorted(set(temporal.segment_indices))
        if not temporal_indices:
            continue
        temporal_set = set(temporal_indices)
        supporting = []
        for source in source_matches:
            source_indices = sorted(set(source.segment_indices))
            if not source_indices:
                continue
            overlap = len(temporal_set & set(source_indices))
            if overlap / len(source_indices) < 0.5:
                continue
            supporting.append(source)
        supporting.sort(key=lambda item: min(item.segment_indices))
        distinct_titles = {
            source.title.strip().casefold() for source in supporting
        }
        source_spans = [
            (min(source.segment_indices), max(source.segment_indices))
            for source in supporting
        ]
        non_overlapping = all(
            left[1] < right[0]
            for left, right in zip(source_spans, source_spans[1:])
        )
        if (
            len(supporting) < 2
            or len(distinct_titles) < 2
            or not non_overlapping
        ):
            refined.append(temporal)
            continue

        cuts = [
            (left[1] + right[0]) // 2
            for left, right in zip(source_spans, source_spans[1:])
        ]
        starts = [temporal_indices[0], *(cut + 1 for cut in cuts)]
        ends = [*cuts, temporal_indices[-1]]
        parts = [
            list(range(max(temporal_indices[0], start), min(temporal_indices[-1], end) + 1))
            for start, end in zip(starts, ends)
        ]
        parts = [part for part in parts if part]
        refined.extend(
            _clone_match_with_indices(temporal, part) for part in parts
        )
        events.append({
            "type": "temporal_source_boundary_split",
            "temporal_ranges": _indices_to_ranges(temporal_indices),
            "source_candidates": [
                {
                    "title": source.title,
                    "ranges": _indices_to_ranges(source.segment_indices),
                }
                for source in supporting
            ],
            "result_ranges": [_indices_to_ranges(part) for part in parts],
        })
    return refined, events


def _restore_temporal_titles(
    temporal_matches: list[ContentMatch],
    source_matches: list[ContentMatch],
) -> tuple[list[ContentMatch], list[dict[str, Any]]]:
    restored: list[ContentMatch] = []
    events: list[dict[str, Any]] = []
    for temporal in temporal_matches:
        ranked = sorted(
            (
                (_temporal_overlap(temporal, source), index, source)
                for index, source in enumerate(source_matches)
                if source.segment_indices
            ),
            key=lambda item: (item[0], item[2].confidence),
            reverse=True,
        )
        overlap, source_index, source = ranked[0] if ranked else (0, -1, None)

        temporal_is_known = bool(temporal.title and not temporal.title.strip().casefold().startswith("未知歌曲"))
        source_is_known = bool(source and source.title and not source.title.strip().casefold().startswith("未知歌曲"))

        if temporal_is_known:
            # temporal 已有名称
            if source_is_known and overlap > 0:
                if temporal.title.strip().casefold() == source.title.strip().casefold():
                    # 名称一致，保留 temporal（边界更精确）
                    restored_match = ContentMatch(
                        content_type="song",
                        title=temporal.title,
                        segment_indices=sorted(set(temporal.segment_indices)),
                        confidence=max(temporal.confidence, source.confidence),
                        tags=_temporal_tags(temporal),
                        description=temporal.description or source.description,
                        artist=temporal.artist or source.artist,
                        lyrics_snippet=temporal.lyrics_snippet or source.lyrics_snippet,
                    )
                    events.append({
                        "type": "temporal_name_consistent",
                        "title": temporal.title,
                        "source_ranges": _indices_to_ranges(source.segment_indices),
                        "temporal_ranges": _indices_to_ranges(temporal.segment_indices),
                    })
                else:
                    source_is_anchor = "anchor_expanded" in (source.tags if source else [])
                    if source_is_anchor:
                        # anchor 来源的标题质量较低，保留 temporal 名称
                        restored_match = ContentMatch(
                            content_type="song",
                            title=temporal.title,
                            segment_indices=sorted(set(temporal.segment_indices)),
                            confidence=max(temporal.confidence, source.confidence),
                            tags=_temporal_tags(temporal),
                            description=temporal.description or source.description,
                            artist=temporal.artist or source.artist,
                            lyrics_snippet=temporal.lyrics_snippet or source.lyrics_snippet,
                        )
                        events.append({
                            "type": "temporal_name_override_anchor",
                            "temporal_title": temporal.title,
                            "anchor_title": source.title,
                            "temporal_ranges": _indices_to_ranges(temporal.segment_indices),
                        })
                    else:
                        # 名称不一致，用 source（首轮识别更可靠）
                        restored_match = ContentMatch(
                            content_type="song",
                            title=source.title,
                            segment_indices=sorted(set(temporal.segment_indices)),
                            confidence=max(temporal.confidence, source.confidence),
                            tags=_temporal_tags(source),
                            description=source.description,
                            artist=source.artist,
                            lyrics_snippet=source.lyrics_snippet,
                        )
                        events.append({
                            "type": "temporal_name_conflict_use_source",
                            "temporal_title": temporal.title,
                            "source_title": source.title,
                            "temporal_ranges": _indices_to_ranges(temporal.segment_indices),
                        })
            else:
                # temporal 有名称但无 source 重叠（或 source 未知），保留 temporal
                restored_match = _clone_match_with_indices(temporal, sorted(set(temporal.segment_indices)))
                restored_match.tags = _temporal_tags(restored_match)
                events.append({
                    "type": "temporal_name_preserved_no_source",
                    "title": temporal.title,
                    "ranges": _indices_to_ranges(temporal.segment_indices),
                })
        else:
            # temporal 无名称或为"未知歌曲"
            if overlap > 0 and source:
                # 有重叠 → 保留 source 名称（含歌词描述的"未知歌曲：xxx"也比通用的"时序裁决新增区间"好）
                restored_match = ContentMatch(
                    content_type="song",
                    title=source.title,
                    segment_indices=sorted(set(temporal.segment_indices)),
                    confidence=max(temporal.confidence, source.confidence),
                    tags=_temporal_tags(source),
                    description=source.description,
                    artist=source.artist,
                    lyrics_snippet=source.lyrics_snippet,
                )
                events.append({
                    "type": "temporal_boundary_replacement" if source_is_known else "temporal_unknown_preserved",
                    "title": source.title,
                    "source_ranges": _indices_to_ranges(source.segment_indices),
                    "temporal_ranges": _indices_to_ranges(temporal.segment_indices),
                    "overlap_segments": overlap,
                })
            else:
                # 无重叠 → 新增区间
                restored_match = _clone_match_with_indices(temporal, sorted(set(temporal.segment_indices)))
                restored_match.title = "未知歌曲：时序裁决新增区间"
                restored_match.tags = _temporal_tags(restored_match)
                events.append({
                    "type": "temporal_new_performance",
                    "ranges": _indices_to_ranges(temporal.segment_indices),
                })
        restored.append(restored_match)

    # A second pass may miss a high-quality first-pass candidate. Preserve it rather
    # than treating one LLM omission as explicit negative evidence.
    for index, source in enumerate(source_matches):
        if any(_temporal_overlap(temporal, source) > 0 for temporal in temporal_matches):
            continue
        if source.confidence < 0.65:
            continue
        restored.append(source)
        events.append({
            "type": "temporal_source_preserved",
            "title": source.title,
            "ranges": _indices_to_ranges(source.segment_indices),
            "reason": "no temporal overlap but source confidence >= 0.65",
        })
    return restored, events


def run_temporal_adjudication(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    llm_dir: Path,
) -> tuple[list[ContentMatch], dict[str, Any]]:
    settings = config.get("song", {}).get("pipeline", {}).get(
        "temporal_adjudication", {}
    )
    if settings.get("enabled", True) is False:
        return matches, {"status": "disabled"}

    from ..llm import identify_content

    local_config = deepcopy(config)
    max_completion = int(settings.get("max_completion_tokens", 8192) or 8192)
    local_config["llm"]["max_tokens"] = max_completion
    local_config["llm"]["max_completion_tokens"] = max_completion
    local_config["llm"]["max_tool_rounds"] = 0
    local_config["llm"]["use_tools"] = False
    local_config["llm"]["retry_empty_with_reasoning"] = False
    local_config["llm"]["json_fix_rounds"] = 0
    debug_dir = llm_dir / "temporal_adjudication"
    temporal = identify_content(
        segments,
        local_config,
        _SongTemporalAdjudicationRecognizer(recognizer, matches),
        debug_dir=debug_dir,
        debug_phase="temporal_adjudication",
    )
    debug_files = sorted(debug_dir.glob("llm_batch_*.json"))
    valid = False
    structural_failures: list[str] = []
    for path in debug_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            structural_failures.append("invalid_debug")
            continue
        if payload.get("finish_reason") == "length":
            structural_failures.append("output_truncated")
        if payload.get("parse_valid") is not True:
            structural_failures.append("invalid_result_json")
        if not payload.get("error") and payload.get("parse_valid") is True:
            valid = True
    if not valid or structural_failures:
        audit = {
            "status": "fallback_source",
            "input_count": len(matches),
            "output_count": len(matches),
            "structural_failures": sorted(set(structural_failures)),
        }
        (debug_dir / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return matches, audit

    refined_temporal, boundary_events = _split_temporal_at_source_boundaries(
        temporal, matches,
    )
    restored, events = _restore_temporal_titles(refined_temporal, matches)
    normalized, normalization_events, _ = _normalize_song_matches(
        segments, config, restored,
    )
    audit = {
        "status": "success",
        "input_count": len(matches),
        "temporal_count": len(temporal),
        "refined_temporal_count": len(refined_temporal),
        "output_count": len(normalized),
        "events": [*boundary_events, *events],
        "normalization_events": normalization_events,
    }
    (debug_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return normalized, audit
