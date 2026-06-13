from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import get_llm_config
from ..llm import (
    _attach_request_debug,
    _build_openai_clients,
    _record_cache_reuse,
    _record_usage,
    _write_active_debug_files,
    batch_debug_is_reusable,
    build_llm_messages,
    build_providers,
    build_request_debug_metadata,
    call_llm,
    llm_response_debug,
    parse_llm_json,
    write_llm_debug,
)
from ..models import ContentMatch, TranscriptSegment
from ..profile_state import _fingerprint_payload
from ..recognizers.base import BaseRecognizer
from .normalize import (
    _filter_short_segment_ranges,
    _indices_to_ranges,
    _uncovered_segment_ranges,
)
from .pipeline import BoundaryRiskStage, FinalAdjudicationStage, SearchVerificationStage, SongPipelineContext


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _transcript_text(segments: list[TranscriptSegment], batch_start: int = 0) -> str:
    return "\n".join(
        f"[{batch_start + offset}] ({segment.start:.1f}s-{segment.end:.1f}s) {segment.text}"
        for offset, segment in enumerate(segments)
    )


class _V3Recognizer(BaseRecognizer):
    stage = "v3"

    @property
    def name(self) -> str:
        return "song"

    def task_instructions(self, config: dict[str, Any]) -> str:
        raise NotImplementedError

    def build_prompt(
        self,
        segments: list[TranscriptSegment],
        batch_start: int,
        config: dict[str, Any],
    ) -> str:
        return (
            f"{self.task_instructions(config)}\n\n"
            "只返回一个 JSON object，不要 Markdown、解释或代码块。\n\n"
            f"完整 ASR 转写片段：\n{_transcript_text(segments, batch_start)}"
        )


class _PrecisionDiscoveryRecognizer(_V3Recognizer):
    stage = "v3_discovery"

    def task_instructions(self, config: dict[str, Any]) -> str:
        return """你是一个面向演唱会、直播和长视频的歌曲分段专家。

下面是一整段视频的 Whisper ASR 转写片段，每行格式为：
[序号] (开始秒-结束秒) ASR文本

任务：按时间顺序完整扫描 ASR，找出所有可能是连续演唱的歌曲片段，返回纯 JSON 数组。

尽量覆盖同一次演唱的完整片段，边界可以略宽，但不要跨到明显聊天、换歌、报幕或另一首歌。

Whisper ASR 可能存在错字、漏字、同音字替换、外语误识别、断句错误、重复切分等问题。判断时不要只依赖文本是否像标准歌词，而要结合上下文连续性、重复结构、押韵/节奏感、旋律化表达、外语段落、说唱结构、哼唱/拟声等线索综合判断。

识别原则：

1. 只要是在唱歌的段落都应识别出来，即使无法确定歌名。
2. 同一首歌的连续演唱段落必须合并成一个候选，不要把主歌、副歌、桥段拆成多首。
3. 不要把单句歌词、点歌讨论、歌曲介绍、聊天、感谢、报幕、直播结束口播单独当作歌曲候选。

此阶段不识别歌名、不搜索歌词。

协议：
{"candidates":[{"segment_ranges":[[241,271]],"confidence":0.82,"anchor_text":"我终于鼓起勇气"}],"scan_complete":true,"complete_through_segment":2710}

字段限制：candidate 只能包含 segment_ranges、confidence、anchor_text。segment_ranges 起止均包含并且必须来自输入。没有候选时 candidates 为 []。

输出要求：
- 只返回 JSON 数组，不要 Markdown，不要解释，不要代码块。

完整 ASR 转写片段：
"""


@dataclass
class _RecallAuditRecognizer(_V3Recognizer):
    targets: list[dict[str, Any]]
    stage = "v3_recall_audit"

    def task_instructions(self, config: dict[str, Any]) -> str:
        return f"""你负责歌曲分段 V3 的第二轮 Recall Audit。
第一轮已经确定的歌曲不能修改。你只检查下列未覆盖目标区间，寻找可能漏掉的演唱证据，并只返回短 evidence_ranges；不要推测整首歌边界，不要识别歌名，不要搜索歌词。

目标区间：{_compact_json(self.targets)}

普通聊天、感谢、报幕、歌曲讨论、点歌和单个感叹不能成为 anchor。没有证据的 target 不需要输出。

协议：
{{"anchors":[{{"target_id":"U003","evidence_ranges":[[655,660]],"confidence":0.76,"anchor_text":"执念的雨"}}],"audit_complete":true}}

每个 anchor 只能包含 target_id、evidence_ranges、confidence、anchor_text；target_id 必须来自目标区间，evidence_ranges 必须位于对应目标区间内。"""


@dataclass
class _SegmentationAdjudicationRecognizer(_V3Recognizer):
    candidates: list[dict[str, Any]]
    allow_final_discovery: bool
    stage = "v3_adjudication"

    def task_instructions(self, config: dict[str, Any]) -> str:
        additions = (
            "允许有限 additions，但每项必须给出精确 segment_ranges、evidence_ranges、至少两段歌词或约 10 秒连续演唱证据，并标记 final_discovery=true。"
            if self.allow_final_discovery
            else "additions 必须为空数组。"
        )
        return f"""你负责歌曲分段 V3 的第三轮 Segmentation Adjudication。
结合完整 ASR，统一裁决第一轮候选 P 和第二轮证据锚点 R。输入：{_compact_json(self.candidates)}

每个输入 ID 必须且只能被一个 decision 处理。action 只能是 accept、reject、adjust、split、merge：
- accept/adjust/split 通常处理一个 ID；merge 必须处理两个或更多 ID。
- reject 的 segment_ranges 必须为空。
- 其他 action 必须返回最终精确 segment_ranges，不跨过聊天、感谢、报幕或另一首歌。
- split 可返回多个互不重叠区间；merge 只合并确属同一次连续演唱的输入。
{additions}
此阶段不识别歌名、不搜索歌词，分段完整性优先。

协议：
{{"decisions":[{{"candidate_ids":["P003","R002"],"action":"merge","segment_ranges":[[655,693]],"confidence":0.84}}],"additions":[],"adjudication_complete":true}}

decision 只能包含 candidate_ids、action、segment_ranges、confidence。addition 只能包含 segment_ranges、evidence_ranges、confidence、anchor_text、final_discovery。"""


def _sanitize_ranges(value: Any, segment_count: int) -> list[list[int]]:
    if not isinstance(value, list) or segment_count <= 0:
        return []
    result: list[list[int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            start = int(item[0])
            end = int(item[1])
        except (TypeError, ValueError):
            continue
        start = max(0, start)
        end = min(segment_count - 1, end)
        if start <= end:
            result.append([start, end])
    result.sort()
    return result


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


def _validate_discovery(payload: Any, segment_count: int) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return False, "discovery_not_object"
    if payload.get("scan_complete") is not True:
        return False, "discovery_scan_incomplete"
    try:
        complete_through = int(payload.get("complete_through_segment"))
    except (TypeError, ValueError):
        return False, "discovery_missing_complete_through"
    if complete_through < segment_count - 1:
        return False, "discovery_incomplete_coverage"
    if not isinstance(payload.get("candidates"), list):
        return False, "discovery_candidates_not_array"
    return True, None


def _validate_recall(
    payload: Any,
    targets: list[dict[str, Any]],
    segment_count: int,
) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return False, "recall_not_object"
    if payload.get("audit_complete") is not True:
        return False, "recall_incomplete"
    if not isinstance(payload.get("anchors"), list):
        return False, "recall_anchors_not_array"
    target_map = {item["target_id"]: item["segment_range"] for item in targets}
    for item in payload["anchors"]:
        if not isinstance(item, dict):
            return False, "recall_invalid_anchor"
        target = target_map.get(str(item.get("target_id") or ""))
        if target is None:
            return False, "recall_unknown_target"
        ranges = _sanitize_ranges(item.get("evidence_ranges"), segment_count)
        if not ranges:
            return False, "recall_missing_evidence"
        if any(start < target[0] or end > target[1] for start, end in ranges):
            return False, "recall_evidence_outside_target"
    return True, None


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


class _V3StageRunner:
    def __init__(self, segments: list[TranscriptSegment], config: dict[str, Any]) -> None:
        self.segments = segments
        self.config = config
        self.providers = build_providers(config)
        if not self.providers:
            raise RuntimeError("LLM API key not configured. Set llm.api_key in config.")
        self.clients = _build_openai_clients(self.providers)

    def run(
        self,
        recognizer: _V3Recognizer,
        debug_dir: Path,
        *,
        validate: Callable[[Any], tuple[bool, str | None]],
        partial_field: str,
        continuation_instruction: Callable[[list[dict[str, Any]]], str],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        debug_dir.mkdir(parents=True, exist_ok=True)
        llm_config = get_llm_config(self.config)
        store_requests = bool(llm_config.get("debug_store_requests", False))
        reuse = bool(llm_config.get("reuse_valid_batches", True))
        max_rounds = max(0, int(llm_config.get("max_continuation_rounds", 8) or 0))
        continuation_enabled = bool(llm_config.get("continuation_on_length", True))
        batch_debug: dict[str, Any] = {
            "batch_start": 0,
            "batch_end": len(self.segments) - 1,
            "segment_count": len(self.segments),
            "phase": recognizer.stage,
            "provider": None,
            "raw_response": None,
            "parsed_json": None,
            "continuation_rounds": [],
            "usage": [],
            "error": None,
        }
        last_error: Exception | None = None

        for provider in self.providers:
            if not provider.api_key:
                continue
            client = self.clients[provider.api_key]
            batch_debug["error"] = None
            messages = build_llm_messages(recognizer, self.segments, 0, self.config)
            metadata = build_request_debug_metadata(
                messages,
                config=self.config,
                provider=provider,
                recognizer=recognizer,
                segments=self.segments,
                batch_start=0,
                tools=None,
                debug_phase=recognizer.stage,
            )
            metadata["v3_stage_config_fingerprint"] = _fingerprint_payload({
                "pipeline": self.config.get("song", {}).get("pipeline", {}),
                "missed_recheck": self.config.get("song", {}).get("missed_recheck", {}),
                "normalization": self.config.get("song", {}).get("normalization", {}),
                "risk": self.config.get("song", {}).get("risk", {}),
            })
            cache_path = debug_dir / "llm_batch_000000.json"
            if reuse and cache_path.exists():
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    cached = None
                if isinstance(cached, dict) and batch_debug_is_reusable(
                    cached, expected_metadata=metadata,
                ):
                    payload = cached.get("parsed_json")
                    valid, _ = validate(payload)
                    if valid:
                        _record_cache_reuse(debug_dir, 0, cached)
                        _write_active_debug_files(debug_dir, [0])
                        return payload, cached

            _attach_request_debug(batch_debug, messages, store_requests=store_requests, metadata=metadata)
            batch_debug["provider"] = {
                "base_url": provider.base_url or "openai",
                "model": provider.model,
            }
            accumulated: list[dict[str, Any]] = []
            current_messages = messages
            final_payload: dict[str, Any] | None = None
            final_reason: str | None = None
            raw_parts: list[str] = []
            try:
                for round_index in range(max_rounds + 1):
                    response = call_llm(
                        client,
                        provider,
                        current_messages,
                        tools=None,
                        max_tokens_override=(
                            provider.max_completion_tokens
                            if provider.max_completion_tokens is not None
                            else provider.max_tokens
                        ),
                    )
                    response_debug = llm_response_debug(response)
                    _record_usage(
                        batch_debug,
                        "initial" if round_index == 0 else "continuation",
                        response_debug,
                        round=round_index,
                    )
                    raw = response_debug["content"] or response_debug["reasoning_content"]
                    raw_parts.append(raw)
                    parsed = parse_llm_json(raw)
                    partial = (
                        parsed.get(partial_field, [])
                        if isinstance(parsed, dict) and isinstance(parsed.get(partial_field), list)
                        else _extract_array_objects(raw, partial_field)
                    )
                    accumulated.extend(item for item in partial if isinstance(item, dict))
                    merged_payload = dict(parsed) if isinstance(parsed, dict) else {}
                    merged_payload[partial_field] = accumulated
                    valid, reason = validate(merged_payload)

                    if recognizer.stage == "v3_discovery":
                        partial_candidates = _discovery_candidates(
                            {"candidates": accumulated}, len(self.segments)
                        )
                        if _candidate_explosion(partial_candidates, self.segments, self.config):
                            reason = "candidate_protocol_explosion"
                            batch_debug["structural_failure"] = True
                            batch_debug["structural_failure_reason"] = reason
                            valid = False
                            final_reason = response_debug["finish_reason"]
                            break

                    if valid and response_debug["finish_reason"] != "length":
                        final_payload = merged_payload
                        final_reason = response_debug["finish_reason"]
                        break

                    if response_debug["finish_reason"] != "length":
                        batch_debug["structural_failure"] = True
                        batch_debug["structural_failure_reason"] = reason or "invalid_protocol"
                        final_reason = response_debug["finish_reason"]
                        break
                    if not continuation_enabled or round_index >= max_rounds:
                        batch_debug["scan_incomplete"] = True
                        batch_debug["structural_failure_reason"] = "continuation_limit"
                        final_reason = "length"
                        break

                    instruction = continuation_instruction(accumulated)
                    batch_debug["continuation_rounds"].append({
                        "round": round_index + 1,
                        "finish_reason": response_debug["finish_reason"],
                        "partial_count": len(accumulated),
                        "instruction": instruction,
                    })
                    current_messages = [*messages, {"role": "user", "content": instruction}]

                batch_debug["finish_reason"] = final_reason
                batch_debug["raw_response"] = "\n".join(raw_parts)
                batch_debug["parsed_json"] = final_payload
                batch_debug["parse_valid"] = final_payload is not None
                if final_payload is None and not batch_debug.get("error"):
                    batch_debug["error"] = batch_debug.get("structural_failure_reason") or "invalid_protocol"
                write_llm_debug(debug_dir, 0, batch_debug)
                _write_active_debug_files(debug_dir, [0])
                return final_payload, batch_debug
            except Exception as exc:
                last_error = exc
                batch_debug["error"] = str(exc)
                continue

        batch_debug["error"] = str(last_error or batch_debug.get("error") or "all providers failed")
        write_llm_debug(debug_dir, 0, batch_debug)
        _write_active_debug_files(debug_dir, [0])
        return None, batch_debug


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


def _validate_adjudication(
    payload: Any,
    candidate_ids: list[str],
    segments: list[TranscriptSegment],
    config: dict[str, Any],
) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return False, "adjudication_not_object"
    if payload.get("adjudication_complete") is not True:
        return False, "adjudication_incomplete"
    decisions = payload.get("decisions")
    additions = payload.get("additions")
    if not isinstance(decisions, list) or not isinstance(additions, list):
        return False, "adjudication_arrays_missing"
    expected = set(candidate_ids)
    used: list[str] = []
    valid_actions = {"accept", "reject", "adjust", "split", "merge"}
    for item in decisions:
        if not isinstance(item, dict) or str(item.get("action")) not in valid_actions:
            return False, "adjudication_invalid_action"
        ids = item.get("candidate_ids")
        if not isinstance(ids, list) or not ids:
            return False, "adjudication_missing_ids"
        normalized_ids = [str(value) for value in ids]
        if any(value not in expected for value in normalized_ids):
            return False, "adjudication_unknown_id"
        if str(item.get("action")) == "merge" and len(normalized_ids) < 2:
            return False, "adjudication_invalid_merge"
        if str(item.get("action")) != "merge" and len(normalized_ids) != 1:
            return False, "adjudication_multi_id_non_merge"
        ranges = _sanitize_ranges(item.get("segment_ranges"), len(segments))
        if str(item.get("action")) == "reject":
            if ranges:
                return False, "adjudication_reject_has_ranges"
        elif not ranges:
            return False, "adjudication_missing_ranges"
        used.extend(normalized_ids)
    if len(used) != len(set(used)):
        return False, "adjudication_duplicate_id"
    if set(used) != expected:
        return False, "adjudication_missing_id"

    guard = config.get("song", {}).get("pipeline", {}).get("protocol_guard", {})
    if len(additions) > int(guard.get("max_final_additions", 10) or 10):
        return False, "adjudication_too_many_additions"
    if any(
        not isinstance(item, dict)
        or not _addition_has_evidence(item, segments, len(segments))
        for item in additions
    ):
        return False, "adjudication_addition_without_evidence"
    return True, None


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
            tags=["v3_adjudicated", action, *ids],
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
                tags=["v3_adjudicated", "final_discovery"],
                description="",
                artist="",
                lyrics_snippet=anchor,
            ))
    return matches


def run_risk_routed_v3_pipeline(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    llm_dir: Path,
) -> list[ContentMatch]:
    """Run the strict three-round KV song segmentation protocol."""
    v3_dir = llm_dir / "v3"
    v3_dir.mkdir(parents=True, exist_ok=True)
    runner = _V3StageRunner(segments, config)
    overlap = int(
        config.get("song", {}).get("pipeline", {}).get(
            "continuation_overlap_segments", 50
        ) or 50
    )
    history: list[dict[str, Any]] = []

    discovery_recognizer = _PrecisionDiscoveryRecognizer()
    discovery_payload, discovery_debug = runner.run(
        discovery_recognizer,
        v3_dir / "discovery",
        validate=lambda value: _validate_discovery(value, len(segments)),
        partial_field="candidates",
        continuation_instruction=lambda items: _continuation_for_discovery(
            items, len(segments), overlap
        ),
    )
    if discovery_payload is None:
        audit = {
            "strategy": "risk_routed_v3",
            "status": "discovery_structural_failure",
            "stages": [{"stage": "precision_discovery", "status": "failed", "error": discovery_debug.get("error")}],
            "final_count": 0,
        }
        (v3_dir / "pipeline.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(
            "V3 precision discovery failed: "
            f"{discovery_debug.get('error') or 'invalid protocol'}"
        )

    discovery = _assign_discovery_ids(discovery_payload, len(segments))
    history.append({"stage": "precision_discovery", "status": "complete", "candidate_count": len(discovery)})
    (llm_dir / "initial_matches.json").write_text(
        json.dumps([match.to_dict() for match in _candidate_matches(discovery, "precision_discovery")], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    targets = _build_recall_targets(segments, discovery, config)
    recall_recognizer = _RecallAuditRecognizer(targets)
    recall_payload, recall_debug = runner.run(
        recall_recognizer,
        v3_dir / "recall_audit",
        validate=lambda value: _validate_recall(value, targets, len(segments)),
        partial_field="anchors",
        continuation_instruction=lambda items: _continuation_for_recall(items, targets),
    )
    recall_failed = recall_payload is None
    recall = _sanitize_recall_anchors(recall_payload or {"anchors": []}, targets, len(segments))
    history.append({
        "stage": "recall_audit",
        "status": "recall_incomplete" if recall_failed else "complete",
        "target_count": len(targets),
        "anchor_count": len(recall),
        "error": recall_debug.get("error") if recall_failed else None,
    })

    combined = [*discovery, *recall]
    candidate_ids = [item["candidate_id"] for item in combined]
    adjudication_recognizer = _SegmentationAdjudicationRecognizer(
        combined,
        bool(config.get("song", {}).get("pipeline", {}).get("allow_final_discovery", True)),
    )
    adjudication_payload, adjudication_debug = runner.run(
        adjudication_recognizer,
        v3_dir / "adjudication",
        validate=lambda value: _validate_adjudication(
            value, candidate_ids, segments, config
        ),
        partial_field="decisions",
        continuation_instruction=lambda items: _continuation_for_adjudication(items, candidate_ids),
    )
    if adjudication_payload is None:
        matches = [
            *_candidate_matches(discovery, "precision_discovery"),
            *_candidate_matches(recall, "unadjudicated_recall_anchor"),
        ]
        adjudication_status = "adjudication_incomplete"
    else:
        matches = _apply_adjudication(adjudication_payload, combined, segments, config)
        adjudication_status = "complete"
    history.append({
        "stage": "segmentation_adjudication",
        "status": adjudication_status,
        "input_count": len(combined),
        "output_count": len(matches),
        "error": adjudication_debug.get("error") if adjudication_payload is None else None,
    })

    context = SongPipelineContext(segments, config, recognizer, llm_dir, matches)
    BoundaryRiskStage("v3_final", "v3_adjudication").run(context)
    FinalAdjudicationStage().run(context)

    # 搜索验证命名（在所有分段和冲突裁决完成之后）
    from ..config import get_song_search_config
    if get_song_search_config(config).get("enabled", False):
        SearchVerificationStage().run(context)

    history.extend(context.stage_history)

    audit = {
        "strategy": "risk_routed_v3",
        "status": adjudication_status if not recall_failed else "recall_incomplete",
        "stages": history,
        "discovery_candidates": discovery,
        "recall_targets": targets,
        "recall_anchors": recall,
        "final_count": len(context.matches),
        "anchor_boundary_expansion": False,
        "search_enabled": get_song_search_config(config).get("enabled", False),
    }
    (v3_dir / "pipeline.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return context.matches


__all__ = [
    "_PrecisionDiscoveryRecognizer",
    "_RecallAuditRecognizer",
    "_SegmentationAdjudicationRecognizer",
    "_candidate_explosion",
    "_validate_adjudication",
    "run_risk_routed_v3_pipeline",
]
