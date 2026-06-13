"""LLM 调试、缓存和指纹工具。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import TranscriptSegment
from .recognizers.base import BaseRecognizer


def llm_response_debug(response: Any) -> dict[str, Any]:
    """提取 LLM response 的调试信息"""
    choice = response.choices[0] if response.choices else None
    message = choice.message if choice is not None else None
    message_data = message.model_dump() if message is not None else {}
    usage = response.usage.model_dump() if getattr(response, "usage", None) else None
    content = message_data.get("content") or ""
    reasoning = message_data.get("reasoning_content") or ""
    return {
        "model": getattr(response, "model", None),
        "finish_reason": getattr(choice, "finish_reason", None) if choice is not None else None,
        "content": content,
        "content_length": len(content),
        "reasoning_content": reasoning,
        "reasoning_content_length": len(reasoning),
        "message_keys": list(message_data.keys()),
        "usage": usage,
        "tool_calls": message_data.get("tool_calls"),
    }


def _record_usage(
    batch_debug: dict[str, Any],
    phase: str,
    debug: dict[str, Any],
    **details: Any,
) -> None:
    usage = debug.get("usage")
    if not isinstance(usage, dict):
        return
    batch_debug.setdefault("usage", []).append({
        "phase": phase,
        **details,
        **usage,
    })


def _message_lengths(messages: list[dict[str, Any]]) -> list[dict[str, str | int]]:
    lengths: list[dict[str, str | int]] = []
    for index, message in enumerate(messages):
        content = message.get("content", "")
        lengths.append({
            "index": index,
            "role": str(message.get("role", "")),
            "chars": len(str(content)),
        })
    return lengths


def _transcript_batch_fingerprint(
    segments: list[TranscriptSegment],
    batch_start: int,
    recognizer: BaseRecognizer,
) -> str:
    from .profile_state import _fingerprint_payload
    index_start = batch_start
    resolve_start = getattr(recognizer, "transcript_index_start", None)
    if callable(resolve_start):
        index_start = int(resolve_start(batch_start))
    payload = [
        {
            "index": index_start + offset,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        for offset, segment in enumerate(segments)
    ]
    return _fingerprint_payload(payload)


def _tools_schema_fingerprint(tools: list[dict[str, Any]] | None) -> str | None:
    from .profile_state import _fingerprint_payload
    if not tools:
        return None
    return _fingerprint_payload(tools)


def _provider_request_fingerprint(provider: Any, config: dict[str, Any]) -> str:
    from .profile_state import _fingerprint_payload
    llm_config = config.get("llm", {})
    return _fingerprint_payload({
        "base_url": provider.base_url or "openai",
        "model": provider.model,
        "temperature": provider.temperature,
        "max_tokens": provider.max_tokens,
        "max_completion_tokens": provider.max_completion_tokens,
        "thinking": provider.thinking,
        "max_tool_rounds": llm_config.get("max_tool_rounds"),
        "final_tool_max_tokens": llm_config.get("final_tool_max_tokens"),
        "force_final_tool_round": llm_config.get("force_final_tool_round"),
        "final_tool_instruction": llm_config.get("final_tool_instruction"),
        "json_fix_rounds": llm_config.get("json_fix_rounds"),
        "retry_empty_with_reasoning": llm_config.get("retry_empty_with_reasoning"),
    })


def _recognizer_protocol_fingerprint(recognizer: BaseRecognizer) -> str:
    return (
        f"{recognizer.__class__.__module__}."
        f"{recognizer.__class__.__name__}:{recognizer.name}"
    )


def build_request_debug_metadata(
    messages: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    provider: Any,
    recognizer: BaseRecognizer,
    segments: list[TranscriptSegment],
    batch_start: int,
    tools: list[dict[str, Any]] | None,
    debug_phase: str | None,
) -> dict[str, Any]:
    from .profile_state import _fingerprint_payload
    llm_config = config.get("llm", {})
    metadata: dict[str, Any] = {
        "request_fingerprint": _fingerprint_payload(messages),
        "message_count": len(messages),
        "message_lengths": _message_lengths(messages),
        "transcript_batch_fingerprint": _transcript_batch_fingerprint(
            segments, batch_start, recognizer,
        ),
        "tools_schema_fingerprint": _tools_schema_fingerprint(tools),
        "provider_request_fingerprint": _provider_request_fingerprint(provider, config),
        "recognizer_protocol": _recognizer_protocol_fingerprint(recognizer),
        "cache_friendly_prompt_layout": bool(
            llm_config.get("cache_friendly_prompt_layout", False)
        ),
        "compact_segment_ranges": bool(llm_config.get("compact_segment_ranges", False)),
    }
    if debug_phase:
        metadata["phase"] = debug_phase
    return metadata


def _attach_request_debug(
    batch_debug: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    store_requests: bool,
    metadata: dict[str, Any],
) -> None:
    batch_debug.update(metadata)
    if store_requests:
        batch_debug["request_messages"] = messages
    else:
        batch_debug.pop("request_messages", None)


def batch_debug_is_reusable(
    payload: dict[str, Any],
    *,
    expected_metadata: dict[str, Any],
) -> bool:
    if payload.get("error"):
        return False
    if payload.get("parse_valid") is not True:
        return False
    if payload.get("json_fix_rounds"):
        return False
    if payload.get("reasoning_followups"):
        return False
    if payload.get("scan_incomplete"):
        return False
    if payload.get("finish_reason") == "length":
        return False
    if any(
        item.get("finish_reason") == "length"
        for item in payload.get("tool_rounds", [])
        if isinstance(item, dict)
    ):
        return False
    for key, value in expected_metadata.items():
        if payload.get(key) != value:
            return False
    return True


def _try_load_cached_batch(
    debug_path: Path,
    batch_start: int,
    *,
    expected_metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    path = debug_path / f"llm_batch_{batch_start:06d}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not batch_debug_is_reusable(payload, expected_metadata=expected_metadata):
        return None
    items = payload.get("parsed_items")
    if not isinstance(items, list):
        return None
    return payload, [item for item in items if isinstance(item, dict)]


def _record_cache_reuse(
    debug_path: Path,
    batch_start: int,
    payload: dict[str, Any],
) -> None:
    reuse = payload.get("cache_reuse")
    if not isinstance(reuse, dict):
        reuse = {}
    payload["cache_reuse"] = {
        "count": int(reuse.get("count") or 0) + 1,
        "last_reused_at": datetime.now(timezone.utc).isoformat(),
    }
    write_llm_debug(debug_path, batch_start, payload)


def _write_active_debug_files(
    debug_path: Path | None,
    batch_starts: list[int],
) -> list[str]:
    if debug_path is None:
        return []
    relative_paths = [
        f"llm_batch_{batch_start:06d}.json"
        for batch_start in batch_starts
        if (debug_path / f"llm_batch_{batch_start:06d}.json").is_file()
    ]
    (debug_path / "active_debug_files.json").write_text(
        json.dumps(relative_paths, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return relative_paths


def _cache_usage_summary(batch_debug: dict[str, Any]) -> str | None:
    hit_tokens = 0
    miss_tokens = 0
    for usage in batch_debug.get("usage", []):
        if not isinstance(usage, dict):
            continue
        hit_tokens += int(usage.get("prompt_cache_hit_tokens") or 0)
        miss_tokens += int(usage.get("prompt_cache_miss_tokens") or 0)
    total = hit_tokens + miss_tokens
    if total <= 0:
        return None
    return (
        f"KV cache hit {hit_tokens}/{total} input tokens "
        f"({hit_tokens / total:.1%})"
    )


def _format_transcript_for_cache(
    segments: list[TranscriptSegment],
    batch_start: int,
    recognizer: BaseRecognizer,
) -> str:
    index_start = batch_start
    resolve_start = getattr(recognizer, "transcript_index_start", None)
    if callable(resolve_start):
        index_start = int(resolve_start(batch_start))
    return "\n".join(
        f"[{index_start + i}] ({seg.start:.1f}s-{seg.end:.1f}s) {seg.text}"
        for i, seg in enumerate(segments)
    )


def _extract_task_instructions(prompt: str) -> str | None:
    for marker in ("\n完整 ASR 转写片段：\n", "\nASR 转写：\n"):
        if marker in prompt:
            instructions, _ = prompt.rsplit(marker, 1)
            return instructions.strip()
    return None


def write_llm_debug(debug_dir: Path, batch_start: int, payload: dict[str, Any]) -> None:
    """写入 LLM 调试信息"""
    target = debug_dir / f"llm_batch_{batch_start:06d}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
