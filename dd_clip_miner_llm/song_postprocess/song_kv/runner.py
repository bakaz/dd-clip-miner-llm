from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ...config import get_llm_config
from ...llm import (
    _attach_request_debug,
    _build_openai_clients,
    _get_client,
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
from ...models import TranscriptSegment
from ...profile_state import _fingerprint_payload
from .recognizers import _KVRecognizer
from .validation import (
    _candidate_explosion,
    _discovery_candidates,
    _extract_array_objects,
    _unpack_validation_result,
)

_PROTOCOL_RETRY_REASONS = {
    "invalid_protocol",
    "invalid_coordinate_range",
    "mixed_coordinate_mode",
    "seconds_coordinate_drift",
}


def _continuation_for_discovery_coverage(
    complete_through_segment: int,
    expected_last_segment: int,
) -> str:
    resume = max(0, complete_through_segment + 1)
    return (
        "上一响应的 JSON 完整结束，但 coverage 协议尚未完成。"
        f"你已经检查到 segment {complete_through_segment}；"
        f"现在只检查剩余闭区间 [{resume},{expected_last_segment}]。"
        "不要重新输出之前的候选，只输出剩余区间中新发现的 candidates。"
        "无论剩余内容是否为演唱，都必须检查到最后一行，并返回一个 JSON object："
        f'{{"candidates":[],"scan_complete":true,'
        f'"complete_through_segment":{expected_last_segment}}}。'
        "complete_through_segment 表示最后检查过的输入索引，不是最后一个歌曲候选索引。"
    )


class _KVStageRunner:
    def __init__(self, segments: list[TranscriptSegment], config: dict[str, Any]) -> None:
        self.segments = segments
        self.config = config
        self.providers = build_providers(config)
        if not self.providers:
            raise RuntimeError("LLM API key not configured. Set llm.api_key in config.")
        self.clients = _build_openai_clients(self.providers)

    def run(
        self,
        recognizer: _KVRecognizer,
        debug_dir: Path,
        *,
        validate: Callable[
            [Any],
            tuple[bool, str | None] | tuple[bool, str | None, dict[str, Any]],
        ],
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
            "provider_attempts": [],
        }
        last_error: Exception | None = None

        for provider in self.providers:
            if not provider.api_key:
                continue
            client = _get_client(provider, self.clients)
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
            metadata["kv_stage_config_fingerprint"] = _fingerprint_payload({
                "pipeline": self.config.get("song", {}).get("pipeline", {}),
                "missed_recheck": self.config.get("song", {}).get("missed_recheck", {}),
                "normalization": self.config.get("song", {}).get("normalization", {}),
                "risk": self.config.get("song", {}).get("risk", {}),
            })
            cache_path = debug_dir / "llm_batch_000000.json"
            recovery_seed: dict[str, Any] | None = None
            if reuse and cache_path.exists():
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    cached = None
                if isinstance(cached, dict) and batch_debug_is_reusable(
                    cached, expected_metadata=metadata,
                ):
                    payload = cached.get("parsed_json")
                    valid, _, _ = _unpack_validation_result(validate(payload))
                    if valid:
                        _record_cache_reuse(debug_dir, 0, cached)
                        _write_active_debug_files(debug_dir, [0])
                        return payload, cached
                stable_recovery_keys = (
                    "transcript_batch_fingerprint",
                    "tools_schema_fingerprint",
                    "recognizer_protocol",
                    "phase",
                )
                recovery_metadata_matches = isinstance(cached, dict) and all(
                    cached.get(key) == metadata.get(key)
                    for key in stable_recovery_keys
                )
                cached_provider = cached.get("provider") if isinstance(cached, dict) else None
                recovery_metadata_matches = recovery_metadata_matches and isinstance(
                    cached_provider, dict
                ) and (
                    cached_provider.get("base_url") == (provider.base_url or "openai")
                    and cached_provider.get("model") == provider.model
                )
                recovery_allowed = (
                    recovery_metadata_matches
                    and recognizer.stage == "kv_discovery"
                    and cached.get("error") == "discovery_incomplete_coverage"
                    and cached.get("finish_reason") != "length"
                    and not cached.get("json_fix_rounds")
                )
                if recovery_allowed:
                    raw_response = cached.get("raw_response")
                    reparsed = (
                        parse_llm_json(raw_response)
                        if isinstance(raw_response, str) and raw_response.strip()
                        else None
                    )
                    if isinstance(reparsed, dict):
                        try:
                            complete_through = int(
                                reparsed.get("complete_through_segment")
                            )
                        except (TypeError, ValueError):
                            complete_through = -1
                        if -1 <= complete_through < len(self.segments) - 1:
                            recovery_seed = {
                                "payload": reparsed,
                                "raw_response": raw_response,
                                "usage": list(cached.get("usage") or []),
                                "complete_through_segment": complete_through,
                            }
            result_retries = max(0, int(provider.result_retries or 0))
            total_result_attempts = result_retries + 1

            for result_attempt in range(1, total_result_attempts + 1):
                if result_attempt > 1:
                    import random as _random
                    import time as _time

                    backoff_list = provider.retry_backoff_seconds or [2, 5]
                    base = float(backoff_list[min(result_attempt - 2, len(backoff_list) - 1)])
                    delta = base * float(provider.retry_jitter_ratio or 0.0) * (2 * _random.random() - 1)
                    _time.sleep(max(0.0, base + delta))

                batch_debug["error"] = None
                batch_debug["raw_response"] = None
                batch_debug["parsed_json"] = None
                batch_debug["continuation_rounds"] = []
                batch_debug["usage"] = []
                batch_debug["provider"] = {
                    "base_url": provider.base_url or "openai",
                    "model": provider.model,
                    "result_retries": result_retries,
                    "result_attempt": result_attempt,
                }
                batch_debug["parse_valid"] = False
                batch_debug["protocol_valid"] = False
                batch_debug["coordinate_mode"] = "empty"
                batch_debug["seconds_like_range_count"] = 0
                batch_debug["example_seconds_like_ranges"] = []
                batch_debug.pop("invalid_range_count", None)
                batch_debug.pop("example_invalid_ranges", None)
                batch_debug.pop("valid_range_count", None)
                batch_debug.pop("structural_failure", None)
                batch_debug.pop("structural_failure_reason", None)
                batch_debug.pop("scan_incomplete", None)
                batch_debug.pop("continued_from_cached_incomplete_response", None)
                _attach_request_debug(
                    batch_debug,
                    messages,
                    store_requests=store_requests,
                    metadata=metadata,
                )

                accumulated: list[dict[str, Any]] = []
                current_messages = messages
                final_payload: dict[str, Any] | None = None
                final_reason: str | None = None
                raw_parts: list[str] = []
                protocol_reason: str | None = None
                active_recovery_seed = recovery_seed if result_attempt == 1 else None
                if active_recovery_seed is not None:
                    seed_payload = active_recovery_seed["payload"]
                    seed_items = seed_payload.get(partial_field, [])
                    accumulated.extend(
                        item for item in seed_items if isinstance(item, dict)
                    )
                    raw_parts.append(str(active_recovery_seed["raw_response"]))
                    batch_debug["usage"] = list(active_recovery_seed["usage"])
                    instruction = _continuation_for_discovery_coverage(
                        int(active_recovery_seed["complete_through_segment"]),
                        len(self.segments) - 1,
                    )
                    batch_debug["continued_from_cached_incomplete_response"] = True
                    batch_debug["continuation_rounds"].append({
                        "round": 1,
                        "finish_reason": "stop",
                        "reason": "discovery_incomplete_coverage",
                        "partial_count": len(accumulated),
                        "instruction": instruction,
                    })
                    current_messages = [*messages, {"role": "user", "content": instruction}]
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
                            (
                                "continuation"
                                if active_recovery_seed is not None or round_index > 0
                                else "initial"
                            ),
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
                        valid, reason, diagnostics = _unpack_validation_result(validate(merged_payload))
                        batch_debug.update(diagnostics)
                        batch_debug["protocol_valid"] = valid

                        if recognizer.stage == "kv_discovery":
                            partial_candidates = _discovery_candidates(
                                {"candidates": accumulated}, len(self.segments)
                            )
                            if _candidate_explosion(partial_candidates, self.segments, self.config):
                                reason = "candidate_protocol_explosion"
                                batch_debug["structural_failure"] = True
                                batch_debug["structural_failure_reason"] = reason
                                valid = False
                                batch_debug["protocol_valid"] = False
                                final_reason = response_debug["finish_reason"]
                                break

                        if valid and response_debug["finish_reason"] != "length":
                            final_payload = merged_payload
                            final_reason = response_debug["finish_reason"]
                            batch_debug["protocol_valid"] = True
                            break

                        coverage_incomplete = (
                            recognizer.stage == "kv_discovery"
                            and reason == "discovery_incomplete_coverage"
                        )
                        should_continue = (
                            response_debug["finish_reason"] == "length"
                            or coverage_incomplete
                        )
                        if not should_continue:
                            protocol_reason = reason or "invalid_protocol"
                            batch_debug["structural_failure"] = True
                            batch_debug["structural_failure_reason"] = protocol_reason
                            final_reason = response_debug["finish_reason"]
                            break
                        if not continuation_enabled or round_index >= max_rounds:
                            batch_debug["scan_incomplete"] = True
                            batch_debug["structural_failure_reason"] = (
                                reason if coverage_incomplete else "continuation_limit"
                            )
                            protocol_reason = batch_debug["structural_failure_reason"]
                            final_reason = response_debug["finish_reason"]
                            break

                        if coverage_incomplete:
                            try:
                                complete_through = int(
                                    merged_payload.get("complete_through_segment")
                                )
                            except (TypeError, ValueError):
                                complete_through = -1
                            instruction = _continuation_for_discovery_coverage(
                                complete_through,
                                len(self.segments) - 1,
                            )
                        else:
                            instruction = continuation_instruction(accumulated)
                        batch_debug["continuation_rounds"].append({
                            "round": round_index + 1,
                            "finish_reason": response_debug["finish_reason"],
                            "reason": reason,
                            "partial_count": len(accumulated),
                            "instruction": instruction,
                        })
                        current_messages = [*messages, {"role": "user", "content": instruction}]
                except Exception as exc:
                    last_error = exc
                    batch_debug["error"] = str(exc)
                    batch_debug.setdefault("provider_attempts", []).append({
                        "provider": batch_debug["provider"],
                        "status": "exception",
                        "error": str(exc),
                    })
                    # Result retries are for malformed LLM payloads. Transport
                    # failures have already exhausted call_llm's timeout schedule,
                    # so replaying them here would multiply a bad provider's delay.
                    break

                batch_debug["finish_reason"] = final_reason
                batch_debug["raw_response"] = "\n".join(raw_parts)
                batch_debug["parsed_json"] = final_payload
                batch_debug["parse_valid"] = final_payload is not None
                if final_payload is not None:
                    batch_debug.setdefault("provider_attempts", []).append({
                        "provider": batch_debug["provider"],
                        "status": "success",
                        "finish_reason": final_reason,
                    })
                    write_llm_debug(debug_dir, 0, batch_debug)
                    _write_active_debug_files(debug_dir, [0])
                    return final_payload, batch_debug

                batch_debug["error"] = (
                    batch_debug.get("structural_failure_reason")
                    or protocol_reason
                    or "invalid_protocol"
                )
                batch_debug["protocol_valid"] = False
                batch_debug.setdefault("provider_attempts", []).append({
                    "provider": batch_debug["provider"],
                    "status": "invalid_result",
                    "error": batch_debug["error"],
                    "coordinate_mode": batch_debug.get("coordinate_mode"),
                    "seconds_like_range_count": batch_debug.get("seconds_like_range_count"),
                })
                if batch_debug["error"] in _PROTOCOL_RETRY_REASONS and result_attempt < total_result_attempts:
                    continue
                break

        batch_debug["error"] = str(last_error or batch_debug.get("error") or "all providers failed")
        write_llm_debug(debug_dir, 0, batch_debug)
        _write_active_debug_files(debug_dir, [0])
        return None, batch_debug
