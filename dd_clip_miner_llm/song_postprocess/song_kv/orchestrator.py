from __future__ import annotations

import json
from typing import Any

from ...models import ContentMatch, TranscriptSegment
from ..normalize import (
    _filter_short_segment_ranges,
    _indices_to_ranges,
    _uncovered_segment_ranges,
)
from .recognizers import _compact_json
from .validation import (
    _addition_has_evidence,
    _analyze_coordinate_fields,
    _confidence,
    _dedupe_objects,
    _discovery_candidates,
    _ranges_to_indices,
    _sanitize_ranges,
    _validate_adjudication,
)


def _assign_discovery_ids(payload: dict[str, Any], segment_count: int) -> list[dict[str, Any]]:
    candidates = _discovery_candidates(payload, segment_count)
    candidates.sort(key=lambda item: item["segment_ranges"][0])
    for index, item in enumerate(candidates, 1):
        item["candidate_id"] = f"P{index:03d}"
        item["source"] = "precision_discovery"
    return candidates


def _candidate_matches(candidates: list[dict[str, Any]], tag: str) -> list[ContentMatch]:
    result: list[ContentMatch] = []
    for item in candidates:
        anchor = str(item.get("anchor_text") or "").strip()
        candidate_id = str(item.get("candidate_id") or "")
        result.append(ContentMatch(
            content_type="song",
            title=f"未知歌曲：{anchor or candidate_id}",
            segment_indices=_ranges_to_indices(item["segment_ranges"]),
            confidence=_confidence(item.get("confidence")),
            tags=[tag, candidate_id] if candidate_id else [tag],
            description="",
            artist="",
            lyrics_snippet=anchor,
        ))
    return result


def _build_recall_targets(
    segments: list[TranscriptSegment],
    discovery: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    matches = _candidate_matches(discovery, "precision_discovery")
    ranges = _uncovered_segment_ranges(len(segments), matches, min_gap_segments=1)
    minimum = float(
        config.get("song", {}).get("missed_recheck", {}).get(
            "min_uncovered_seconds", 10.0
        ) or 10.0
    )
    ranges, _ = _filter_short_segment_ranges(segments, ranges, minimum)
    return [
        {"target_id": f"U{index:03d}", "segment_range": [start, end]}
        for index, (start, end) in enumerate(ranges, 1)
    ]


def _sanitize_recall_anchors(
    payload: dict[str, Any],
    targets: list[dict[str, Any]],
    segment_count: int,
) -> list[dict[str, Any]]:
    target_map = {item["target_id"]: item["segment_range"] for item in targets}
    anchors: list[dict[str, Any]] = []
    for item in payload.get("anchors", []):
        if not isinstance(item, dict):
            continue
        target_id = str(item.get("target_id") or "")
        target = target_map.get(target_id)
        if target is None:
            continue
        ranges = _sanitize_ranges(item.get("evidence_ranges"), segment_count)
        cropped = [
            [max(start, target[0]), min(end, target[1])]
            for start, end in ranges
            if max(start, target[0]) <= min(end, target[1])
        ]
        if not cropped:
            continue
        anchors.append({
            "target_id": target_id,
            "evidence_ranges": cropped,
            "segment_ranges": cropped,
            "confidence": _confidence(item.get("confidence")),
            "anchor_text": str(item.get("anchor_text") or "").strip()[:200],
        })
    anchors = _dedupe_objects(
        anchors,
        lambda item: (item["target_id"], tuple(map(tuple, item["evidence_ranges"]))),
    )
    anchors.sort(key=lambda item: item["evidence_ranges"][0])
    for index, item in enumerate(anchors, 1):
        item["candidate_id"] = f"R{index:03d}"
        item["source"] = "recall_audit"
    return anchors


def _continuation_for_discovery(
    items: list[dict[str, Any]], segment_count: int, overlap: int,
) -> str:
    sanitized = _discovery_candidates({"candidates": items}, segment_count)
    last_end = max(
        (end for item in sanitized for _, end in item["segment_ranges"]),
        default=-1,
    )
    resume = max(0, last_end - max(0, overlap) + 1)
    return (
        f"上一响应因长度截断。只继续扫描尚未完成的部分，从 segment {resume} 开始复查到 {segment_count - 1}；"
        f"可向前覆盖 {max(0, overlap)} 段以避免边界遗漏，但不要重复已完整候选。"
        "返回相同 discovery JSON object 协议，并在完成时设置 scan_complete=true。"
    )


def _continuation_for_recall(items: list[dict[str, Any]], targets: list[dict[str, Any]]) -> str:
    completed = {str(item.get("target_id") or "") for item in items}
    remaining = [item["target_id"] for item in targets if item["target_id"] not in completed]
    return (
        "上一响应因长度截断。只审计这些尚未完成的 target_id："
        f"{_compact_json(remaining)}。返回相同 recall JSON object 协议，完成后 audit_complete=true。"
    )


def _decision_ids(items: list[dict[str, Any]]) -> set[str]:
    return {
        str(candidate_id)
        for item in items
        if isinstance(item, dict) and isinstance(item.get("candidate_ids"), list)
        for candidate_id in item["candidate_ids"]
    }


def _continuation_for_adjudication(
    items: list[dict[str, Any]], candidate_ids: list[str],
) -> str:
    remaining = [item for item in candidate_ids if item not in _decision_ids(items)]
    return (
        "上一响应因长度截断。只裁决这些尚未处理的 candidate ID："
        f"{_compact_json(remaining)}。不要再次处理已完成 ID；返回相同 adjudication JSON object 协议。"
    )


def _apply_adjudication(
    payload: dict[str, Any],
    candidates: list[dict[str, Any]],
    segments: list[TranscriptSegment],
    config: dict[str, Any],
) -> list[ContentMatch]:
    source_map = {item["candidate_id"]: item for item in candidates}
    matches: list[ContentMatch] = []
    for decision in payload.get("decisions", []):
        action = str(decision.get("action"))
        if action == "reject":
            continue
        ids = [str(value) for value in decision.get("candidate_ids", [])]
        sources = [source_map[value] for value in ids]
        anchor = next((str(item.get("anchor_text") or "").strip() for item in sources if item.get("anchor_text")), ids[0])
        ranges = _sanitize_ranges(decision.get("segment_ranges"), len(segments))
        matches.append(ContentMatch(
            content_type="song",
            title=f"未知歌曲：{anchor}",
            segment_indices=_ranges_to_indices(ranges),
            confidence=_confidence(decision.get("confidence")),
            tags=["kv_adjudicated", action, *ids],
            description="",
            artist="",
            lyrics_snippet=anchor,
        ))
    if config.get("song", {}).get("pipeline", {}).get("allow_final_discovery", True):
        for item in payload.get("additions", []):
            if not isinstance(item, dict) or not _addition_has_evidence(item, segments, len(segments)):
                continue
            anchor = str(item.get("anchor_text") or "").strip()[:200]
            ranges = _sanitize_ranges(item.get("segment_ranges"), len(segments))
            matches.append(ContentMatch(
                content_type="song",
                title=f"未知歌曲：{anchor or 'final_discovery'}",
                segment_indices=_ranges_to_indices(ranges),
                confidence=_confidence(item.get("confidence")),
                tags=["kv_adjudicated", "final_discovery"],
                description="",
                artist="",
                lyrics_snippet=anchor,
            ))
    return matches
