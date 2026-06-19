from __future__ import annotations

import json
import math
from typing import Any, Callable

from ...models import TranscriptSegment


def _sanitize_ranges(value: Any, segment_count: int) -> list[list[int]]:
    if not isinstance(value, list) or segment_count <= 0:
        return []
    result: list[list[int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        if isinstance(item[0], bool) or isinstance(item[1], bool):
            continue
        try:
            start = int(item[0])
            end = int(item[1])
        except (TypeError, ValueError):
            continue
        if 0 <= start <= end < segment_count:
            result.append([start, end])
    result.sort()
    return result


def _classify_range(
    value: Any,
    *,
    segment_count: int,
    total_duration_seconds: float,
) -> tuple[str, list[int]] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if isinstance(value[0], bool) or isinstance(value[1], bool):
        return None
    try:
        start = int(value[0])
        end = int(value[1])
    except (TypeError, ValueError):
        return None
    if start > end or start < 0:
        return "invalid", [start, end]
    if end < segment_count:
        return "index", [start, end]
    duration_limit = max(segment_count - 1, int(math.ceil(total_duration_seconds)))
    if end <= duration_limit:
        return "seconds", [start, end]
    return "invalid", [start, end]


def _analyze_coordinate_fields(
    payload: Any,
    fields: list[str],
    *,
    segment_count: int,
    total_duration_seconds: float,
) -> dict[str, Any]:
    diagnostics = {
        "coordinate_mode": "empty",
        "seconds_like_range_count": 0,
        "example_seconds_like_ranges": [],
        "invalid_range_count": 0,
        "example_invalid_ranges": [],
        "valid_range_count": 0,
    }
    if not isinstance(payload, dict):
        return diagnostics
    for value in payload.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            for field_name in fields:
                field_value = item.get(field_name)
                if not isinstance(field_value, list):
                    continue
                for range_value in field_value:
                    classified = _classify_range(
                        range_value,
                        segment_count=segment_count,
                        total_duration_seconds=total_duration_seconds,
                    )
                    if classified is None:
                        diagnostics["invalid_range_count"] += 1
                        if len(diagnostics["example_invalid_ranges"]) < 3:
                            diagnostics["example_invalid_ranges"].append([None, None])
                        continue
                    kind, normalized = classified
                    if kind == "index":
                        diagnostics["valid_range_count"] += 1
                    elif kind == "seconds":
                        diagnostics["seconds_like_range_count"] += 1
                        if len(diagnostics["example_seconds_like_ranges"]) < 3:
                            diagnostics["example_seconds_like_ranges"].append(normalized)
                    else:
                        diagnostics["invalid_range_count"] += 1
                        if len(diagnostics["example_invalid_ranges"]) < 3:
                            diagnostics["example_invalid_ranges"].append(normalized)
    if diagnostics["seconds_like_range_count"] and diagnostics["valid_range_count"]:
        diagnostics["coordinate_mode"] = "mixed_coordinate_mode"
    elif diagnostics["seconds_like_range_count"]:
        diagnostics["coordinate_mode"] = "seconds_coordinate_drift"
    elif diagnostics["invalid_range_count"]:
        diagnostics["coordinate_mode"] = "invalid_coordinate_range"
    elif diagnostics["valid_range_count"]:
        diagnostics["coordinate_mode"] = "segment_index"
    return diagnostics


def _ranges_to_indices(ranges: list[list[int]]) -> list[int]:
    return sorted({index for start, end in ranges for index in range(start, end + 1)})


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _extract_array_objects(raw: str, field: str) -> list[dict[str, Any]]:
    marker = f'"{field}"'
    marker_index = raw.find(marker)
    if marker_index < 0:
        return []
    array_start = raw.find("[", marker_index + len(marker))
    if array_start < 0:
        return []
    decoder = json.JSONDecoder()
    result: list[dict[str, Any]] = []
    index = array_start + 1
    while index < len(raw):
        while index < len(raw) and raw[index] in " \t\r\n,":
            index += 1
        if index >= len(raw) or raw[index] == "]":
            break
        if raw[index] != "{":
            index += 1
            continue
        try:
            value, end = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            result.append(value)
        index += end
    return result


def _dedupe_objects(items: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for item in items:
        item_key = key(item)
        if item_key in seen:
            continue
        seen.add(item_key)
        result.append(item)
    return result


def _discovery_candidates(payload: Any, segment_count: int) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        return []
    candidates: list[dict[str, Any]] = []
    for item in payload["candidates"]:
        if not isinstance(item, dict):
            continue
        ranges = _sanitize_ranges(item.get("segment_ranges"), segment_count)
        if not ranges:
            continue
        candidates.append({
            "segment_ranges": ranges,
            "confidence": _confidence(item.get("confidence")),
            "anchor_text": str(item.get("anchor_text") or "").strip()[:200],
        })
    return _dedupe_objects(candidates, lambda item: tuple(map(tuple, item["segment_ranges"])))


def _validate_discovery(
    payload: Any,
    segment_count: int,
    total_duration_seconds: float,
) -> tuple[bool, str | None, dict[str, Any]]:
    diagnostics = _analyze_coordinate_fields(
        payload,
        ["segment_ranges"],
        segment_count=segment_count,
        total_duration_seconds=total_duration_seconds,
    )
    if not isinstance(payload, dict):
        return False, "discovery_not_object", diagnostics
    if diagnostics["coordinate_mode"] == "mixed_coordinate_mode":
        return False, "mixed_coordinate_mode", diagnostics
    if diagnostics["coordinate_mode"] == "seconds_coordinate_drift":
        return False, "seconds_coordinate_drift", diagnostics
    if diagnostics["coordinate_mode"] == "invalid_coordinate_range":
        return False, "invalid_coordinate_range", diagnostics
    if payload.get("scan_complete") is not True:
        return False, "discovery_scan_incomplete", diagnostics
    try:
        complete_through = int(payload.get("complete_through_segment"))
    except (TypeError, ValueError):
        return False, "discovery_missing_complete_through", diagnostics
    if complete_through != segment_count - 1:
        return False, "discovery_incomplete_coverage", diagnostics
    if not isinstance(payload.get("candidates"), list):
        return False, "discovery_candidates_not_array", diagnostics
    return True, None, diagnostics


def _validate_recall(
    payload: Any,
    targets: list[dict[str, Any]],
    segment_count: int,
    total_duration_seconds: float,
) -> tuple[bool, str | None, dict[str, Any]]:
    diagnostics = _analyze_coordinate_fields(
        payload,
        ["evidence_ranges"],
        segment_count=segment_count,
        total_duration_seconds=total_duration_seconds,
    )
    if not isinstance(payload, dict):
        return False, "recall_not_object", diagnostics
    if diagnostics["coordinate_mode"] == "mixed_coordinate_mode":
        return False, "mixed_coordinate_mode", diagnostics
    if diagnostics["coordinate_mode"] == "seconds_coordinate_drift":
        return False, "seconds_coordinate_drift", diagnostics
    if diagnostics["coordinate_mode"] == "invalid_coordinate_range":
        return False, "invalid_coordinate_range", diagnostics
    if payload.get("audit_complete") is not True:
        return False, "recall_incomplete", diagnostics
    if not isinstance(payload.get("anchors"), list):
        return False, "recall_anchors_not_array", diagnostics
    target_map = {item["target_id"]: item["segment_range"] for item in targets}
    for item in payload["anchors"]:
        if not isinstance(item, dict):
            return False, "recall_invalid_anchor", diagnostics
        target = target_map.get(str(item.get("target_id") or ""))
        if target is None:
            return False, "recall_unknown_target", diagnostics
        ranges = _sanitize_ranges(item.get("evidence_ranges"), segment_count)
        if not ranges:
            return False, "recall_missing_evidence", diagnostics
        if any(start < target[0] or end > target[1] for start, end in ranges):
            return False, "recall_evidence_outside_target", diagnostics
    return True, None, diagnostics


def _candidate_explosion(
    candidates: list[dict[str, Any]],
    segments: list[TranscriptSegment],
    config: dict[str, Any],
) -> bool:
    if not candidates or not segments:
        return False
    guard = config.get("song", {}).get("pipeline", {}).get("protocol_guard", {})
    duration_hours = max(1.0 / 60.0, (segments[-1].end - segments[0].start) / 3600.0)
    limit = max(
        int(guard.get("min_candidate_limit", 64) or 64),
        int(math.ceil(duration_hours * float(guard.get("max_candidates_per_hour", 40.0) or 40.0))),
    )
    short_count = sum(
        1
        for item in candidates
        if sum(end - start + 1 for start, end in item.get("segment_ranges", [])) <= 2
    )
    ratio = short_count / len(candidates)
    return len(candidates) > limit and ratio >= float(
        guard.get("short_candidate_ratio_threshold", 0.70) or 0.70
    )


def _unpack_validation_result(
    result: tuple[bool, str | None] | tuple[bool, str | None, dict[str, Any]],
) -> tuple[bool, str | None, dict[str, Any]]:
    if len(result) == 3:
        valid, reason, diagnostics = result
        return valid, reason, diagnostics
    valid, reason = result
    return valid, reason, {}


def _validate_adjudication(
    payload: Any,
    candidate_ids: list[str],
    segments: list[TranscriptSegment],
    config: dict[str, Any],
) -> tuple[bool, str | None, dict[str, Any]]:
    diagnostics = _analyze_coordinate_fields(
        payload,
        ["segment_ranges", "evidence_ranges"],
        segment_count=len(segments),
        total_duration_seconds=max((float(segment.end) for segment in segments), default=0.0),
    )
    if not isinstance(payload, dict):
        return False, "adjudication_not_object", diagnostics
    if diagnostics["coordinate_mode"] == "mixed_coordinate_mode":
        return False, "mixed_coordinate_mode", diagnostics
    if diagnostics["coordinate_mode"] == "seconds_coordinate_drift":
        return False, "seconds_coordinate_drift", diagnostics
    if diagnostics["coordinate_mode"] == "invalid_coordinate_range":
        return False, "invalid_coordinate_range", diagnostics
    if payload.get("adjudication_complete") is not True:
        return False, "adjudication_incomplete", diagnostics
    decisions = payload.get("decisions")
    additions = payload.get("additions")
    if not isinstance(decisions, list) or not isinstance(additions, list):
        return False, "adjudication_arrays_missing", diagnostics
    expected = set(candidate_ids)
    used: list[str] = []
    valid_actions = {"accept", "reject", "adjust", "split", "merge"}
    for item in decisions:
        if not isinstance(item, dict) or str(item.get("action")) not in valid_actions:
            return False, "adjudication_invalid_action", diagnostics
        ids = item.get("candidate_ids")
        if not isinstance(ids, list) or not ids:
            return False, "adjudication_missing_ids", diagnostics
        normalized_ids = [str(value) for value in ids]
        if any(value not in expected for value in normalized_ids):
            return False, "adjudication_unknown_id", diagnostics
        if str(item.get("action")) == "merge" and len(normalized_ids) < 2:
            return False, "adjudication_invalid_merge", diagnostics
        if str(item.get("action")) != "merge" and len(normalized_ids) != 1:
            return False, "adjudication_multi_id_non_merge", diagnostics
        ranges = _sanitize_ranges(item.get("segment_ranges"), len(segments))
        if str(item.get("action")) == "reject":
            if ranges:
                return False, "adjudication_reject_has_ranges", diagnostics
        elif not ranges:
            return False, "adjudication_missing_ranges", diagnostics
        used.extend(normalized_ids)
    if len(used) != len(set(used)):
        return False, "adjudication_duplicate_id", diagnostics
    if set(used) != expected:
        return False, "adjudication_missing_id", diagnostics

    guard = config.get("song", {}).get("pipeline", {}).get("protocol_guard", {})
    if len(additions) > int(guard.get("max_final_additions", 10) or 10):
        return False, "adjudication_too_many_additions", diagnostics
    if any(
        not isinstance(item, dict)
        or not _addition_has_evidence(item, segments, len(segments))
        for item in additions
    ):
        return False, "adjudication_addition_without_evidence", diagnostics
    return True, None, diagnostics


def _addition_has_evidence(
    item: dict[str, Any], segments: list[TranscriptSegment], segment_count: int,
) -> bool:
    if item.get("final_discovery") is not True:
        return False
    evidence = _sanitize_ranges(item.get("evidence_ranges"), segment_count)
    final_ranges = _sanitize_ranges(item.get("segment_ranges"), segment_count)
    if not evidence or not final_ranges:
        return False
    evidence_indices = _ranges_to_indices(evidence)
    if len(evidence_indices) >= 2:
        return True
    start, end = evidence[0]
    return float(segments[end].end) - float(segments[start].start) >= 10.0
