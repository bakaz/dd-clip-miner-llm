"""内容识别模块

通用的内容识别逻辑，包含 provider 路由、产物验证链、缓存等。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import get_llm_config
from ..llm_debug import (
    _attach_request_debug,
    _cache_usage_summary,
    _record_cache_reuse,
    _record_usage,
    _try_load_cached_batch,
    _write_active_debug_files,
    build_request_debug_metadata,
    llm_response_debug,
    write_llm_debug,
)
from ..models import ContentMatch, TranscriptSegment
from ..recognizers.base import BaseRecognizer
from .parse import parse_llm_json, parse_llm_response_with_status, write_llm_debug as _write_debug
from .prompt import build_llm_messages
from .provider import (
    LLMProvider,
    build_providers,
    _ensure_openai_clients,
    _get_client,
    _build_openai_clients,
)
from .repair import (
    _continue_truncated_json_array,
    _continue_truncated_json_object,
    fix_json_with_llm,
    fix_structured_json_with_llm,
    run_reasoning_followups,
)
from .tools import run_llm_with_tools
from .transport import call_llm_with_transport_retry


def identify_songs(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    debug_dir: Path | None = None,
) -> list[ContentMatch]:
    """识别歌曲片段（兼容旧接口）"""
    from ..recognizers import get_recognizer
    recognizer = get_recognizer("song")
    if recognizer is None:
        raise RuntimeError("Song recognizer not found")
    return identify_content(segments, config, recognizer, debug_dir)


def identify_dialogues(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    debug_dir: Path | None = None,
) -> list[ContentMatch]:
    """识别对话片段（兼容旧接口）"""
    from ..recognizers import get_recognizer
    recognizer = get_recognizer("dialogue")
    if recognizer is None:
        raise RuntimeError("Dialogue recognizer not found")
    return identify_content(segments, config, recognizer, debug_dir)


def identify_structured_content(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: BaseRecognizer,
    debug_dir: Path | None = None,
) -> dict[str, Any]:
    """通用结构化内容生成，返回 JSON object。"""
    content_type = recognizer.name

    providers = build_providers(config)
    if not providers:
        raise RuntimeError("LLM API key not configured. Set llm.api_key in config.")

    clients: dict[tuple[str | None, str, str | None], Any] = {}
    _ensure_openai_clients(providers, clients)

    debug_path = Path(debug_dir) if debug_dir is not None else None
    if debug_path is not None:
        debug_path.mkdir(parents=True, exist_ok=True)

    batch_debug: dict[str, Any] = {
        "batch_start": 0,
        "batch_end": len(segments) - 1,
        "segment_count": len(segments),
        "provider": None,
        "raw_response": None,
        "parsed_json": None,
        "json_fix_rounds": [],
        "usage": [],
        "error": None,
    }

    content = None
    last_error = None
    client = None
    provider = None

    messages = build_llm_messages(recognizer, segments, 0, config)
    store_requests = bool(get_llm_config(config).get("debug_store_requests", False))

    for candidate in providers:
        if not candidate.api_key:
            continue

        request_metadata = build_request_debug_metadata(
            messages, config=config, provider=candidate, recognizer=recognizer,
            segments=segments, batch_start=0, tools=None, debug_phase=content_type,
        )
        _attach_request_debug(batch_debug, messages, store_requests=store_requests, metadata=request_metadata)

        result_retries = candidate.result_retries
        for result_attempt in range(1, result_retries + 2):
            try:
                prov_client = _get_client(candidate, clients)
                if prov_client is None:
                    break
                response, content = call_llm_with_transport_retry(
                    prov_client, candidate, messages,
                    batch_debug=batch_debug,
                )
                debug = llm_response_debug(response)
                batch_debug["finish_reason"] = debug["finish_reason"]
                content = _continue_truncated_json_object(
                    prov_client,
                    candidate,
                    config,
                    messages,
                    content,
                    debug["finish_reason"],
                    batch_debug,
                    max_tokens=(
                        candidate.max_completion_tokens
                        if candidate.max_completion_tokens is not None
                        else candidate.max_tokens
                    ),
                )
            except Exception as exc:
                last_error = exc
                print(f"  [warn] provider={candidate.name} failed: {exc}")
                break

            if content.strip():
                parsed_check = parse_llm_json(content)
                if isinstance(parsed_check, dict) and parsed_check:
                    provider = candidate
                    client = prov_client
                    batch_debug["provider"] = {
                        "name": candidate.name,
                        "base_url": candidate.base_url or "openai",
                        "model": candidate.model,
                        "timeout_schedule": candidate.timeout_schedule,
                        "result_retries": result_retries,
                    }
                    break
                last_error = RuntimeError(f"Invalid result from {candidate.name}")
            else:
                last_error = RuntimeError(f"Empty result from {candidate.name}")

        if content and provider and client:
            break

    if content is None or provider is None or client is None:
        batch_debug["error"] = str(last_error)
        if debug_path is not None:
            write_llm_debug(debug_path, 0, batch_debug)
            _write_active_debug_files(debug_path, [0])
        print(f"  [error] All LLM providers failed for {content_type}. Last error: {last_error}")
        return {
            "content_type": content_type,
            "title": config.get(content_type, {}).get("title", content_type),
            "error": str(last_error),
        }

    parsed = parse_llm_json(content)
    if not isinstance(parsed, dict) and content.strip():
        parsed, content = fix_structured_json_with_llm(
            client, provider, config, content, content_type, batch_debug
        )

    if not isinstance(parsed, dict) or not parsed:
        parsed = {
            "content_type": content_type,
            "title": config.get(content_type, {}).get("title", content_type),
            "error": "LLM did not return a JSON object",
            "raw_response": content,
        }

    batch_debug["parsed_json"] = parsed
    batch_debug["raw_response"] = content
    if debug_path is not None:
        write_llm_debug(debug_path, 0, batch_debug)
        _write_active_debug_files(debug_path, [0])
    cache_summary = _cache_usage_summary(batch_debug)
    if cache_summary:
        print(f"  LLM {cache_summary}")

    return parsed


def _process_single_batch(
    *,
    batch_idx: int,
    batch_count: int,
    batch_start: int,
    batch_segments: list[TranscriptSegment],
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: BaseRecognizer,
    providers: list[LLMProvider],
    clients: dict[tuple[str | None, str, str | None], Any],
    tools: list[dict[str, Any]] | None,
    debug_path: Path | None,
    debug_phase: str | None,
    store_requests: bool,
    reuse_valid_batches: bool,
    content_type: str,
) -> list[ContentMatch] | None:
    """Process a single LLM batch. Returns matches or None on failure."""
    print(f"  LLM batch {batch_idx}/{batch_count}: segments {batch_start}-{batch_start + len(batch_segments) - 1} (total {len(segments)})...")
    batch_debug: dict[str, Any] = {
        "batch_start": batch_start,
        "batch_end": batch_start + len(batch_segments) - 1,
        "segment_count": len(batch_segments),
        "provider": None,
        "raw_response": None,
        "parsed_items": [],
        "tool_calls_log": [],
        "tool_rounds": [],
        "reasoning_followups": [],
        "json_fix_rounds": [],
        "usage": [],
        "error": None,
    }
    if debug_phase:
        batch_debug["phase"] = debug_phase

    llm_config = get_llm_config(config)

    first_provider = next((p for p in providers if p.api_key), None)
    if first_provider is None:
        batch_debug["error"] = "No provider with API key"
        if debug_path is not None:
            write_llm_debug(debug_path, batch_start, batch_debug)
        return None

    messages = build_llm_messages(recognizer, batch_segments, batch_start, config)
    request_metadata = build_request_debug_metadata(
        messages, config=config, provider=first_provider, recognizer=recognizer,
        segments=batch_segments, batch_start=batch_start, tools=tools,
        debug_phase=debug_phase,
    )
    if reuse_valid_batches and debug_path is not None and request_metadata is not None:
        cached_batch = _try_load_cached_batch(debug_path, batch_start, expected_metadata=request_metadata)
        if cached_batch is not None:
            cached_payload, cached_items = cached_batch
            _record_cache_reuse(debug_path, batch_start, cached_payload)
            matches = recognizer.parse_response(cached_items, config)
            print(f"  LLM batch {batch_idx}/{batch_count}: reused cached result, found {len(matches)} match(es)")
            return matches

    _attach_request_debug(batch_debug, messages, store_requests=store_requests, metadata=request_metadata)

    max_tokens_val = (
        int(llm_config["final_tool_max_tokens"])
        if llm_config.get("final_tool_max_tokens") not in (None, "")
        else (
            first_provider.max_completion_tokens
            if first_provider.max_completion_tokens is not None
            else first_provider.max_tokens
        )
    )

    last_error = None
    for provider in providers:
        if not provider.api_key:
            continue
        client = _get_client(provider, clients)
        if client is None:
            continue

        result_retries = provider.result_retries
        total_result_attempts = result_retries + 1

        for result_attempt in range(1, total_result_attempts + 1):
            transport_schedule = (
                provider.timeout_schedule
                or [provider.timeout] * provider.max_retries
            )

            if result_attempt > 1:
                import time as _time
                import random as _random
                backoff_list = provider.retry_backoff_seconds or [2, 5]
                jitter = provider.retry_jitter_ratio
                base = backoff_list[-1]
                delta = base * jitter * (2 * _random.random() - 1)
                wait = max(0.0, base + delta)
                print(
                    f"  provider={provider.name} "
                    f"result_attempt={result_attempt}/{total_result_attempts} "
                    f"result invalid: replaying in {wait:.1f}s",
                    flush=True,
                )
                _time.sleep(wait)

            content = None
            try:
                if tools and content_type == "song":
                    from ..search_tools import execute_tool
                    content = run_llm_with_tools(
                        client, provider, messages, tools, execute_tool, batch_debug,
                        max_tool_rounds=int(llm_config.get("max_tool_rounds", 2) or 0),
                        final_max_tokens=max_tokens_val,
                        force_final_round=bool(llm_config.get("force_final_tool_round", False)),
                        final_instruction=(str(llm_config["final_tool_instruction"]) if llm_config.get("final_tool_instruction") else None),
                        config=config,
                    )
                    batch_debug["provider"] = {
                        "name": provider.name,
                        "base_url": provider.base_url or "openai",
                        "model": provider.model,
                        "timeout_schedule": transport_schedule,
                        "result_retries": result_retries,
                    }
                else:
                    response, content = call_llm_with_transport_retry(
                        client, provider, messages,
                        batch_debug=batch_debug,
                    )
                    batch_debug["provider"] = {
                        "name": provider.name,
                        "base_url": provider.base_url or "openai",
                        "model": provider.model,
                        "timeout_schedule": transport_schedule,
                        "result_retries": result_retries,
                    }
                    debug = llm_response_debug(response)
                    batch_debug["finish_reason"] = debug["finish_reason"]
                    content = _continue_truncated_json_array(
                        client, provider, config, messages, content,
                        debug["finish_reason"], batch_debug,
                        max_tokens=max_tokens_val,
                    )
            except Exception as exc:
                last_error = exc
                from .error import _classify_error
                _, reason = _classify_error(exc)
                if not _classify_error(exc)[0]:
                    print(f"  [warn] provider={provider.name} non-retryable ({reason}), switching", flush=True)
                    break
                if result_attempt < total_result_attempts:
                    continue
                print(
                    f"  provider={provider.name} "
                    f"result_attempt={result_attempt}/{total_result_attempts} "
                    f"transport exhausted, switching provider",
                    flush=True,
                )
                break

            # 产物验证链
            if not content.strip():
                reasoning_content = ""
                if batch_debug.get("tool_rounds"):
                    for tr in batch_debug["tool_rounds"]:
                        if tr.get("reasoning_content"):
                            reasoning_content = tr["reasoning_content"]
                            break
                content = run_reasoning_followups(client, provider, config, reasoning_content, "", batch_debug)

            items, is_valid_array = parse_llm_response_with_status(content)
            if not is_valid_array:
                if batch_debug.get("tool_rounds"):
                    for tr in batch_debug["tool_rounds"]:
                        rc = tr.get("reasoning_content", "")
                        if rc.strip():
                            items, is_valid_array = parse_llm_response_with_status(rc)
                            if is_valid_array:
                                break
                if not is_valid_array:
                    reasoning_content = ""
                    if batch_debug.get("tool_rounds"):
                        for tr in batch_debug["tool_rounds"]:
                            if tr.get("reasoning_content"):
                                reasoning_content = tr["reasoning_content"]
                                break
                    if reasoning_content.strip():
                        followup_content = run_reasoning_followups(client, provider, config, reasoning_content, content, batch_debug)
                        if followup_content.strip():
                            content = followup_content
                            items, is_valid_array = parse_llm_response_with_status(content)

            if not is_valid_array and content.strip():
                items, content = fix_json_with_llm(client, provider, config, content, content_type, batch_debug)
                _, is_valid_array = parse_llm_response_with_status(content)

            if is_valid_array:
                matches = recognizer.parse_response(items, config)
                batch_debug["parsed_items"] = items
                batch_debug["parse_valid"] = True
                batch_debug["raw_response"] = content
                if debug_path is not None:
                    write_llm_debug(debug_path, batch_start, batch_debug)
                cache_summary = _cache_usage_summary(batch_debug)
                cache_suffix = f", {cache_summary}" if cache_summary else ""
                print(f"  LLM batch {batch_idx}/{batch_count}: done, found {len(matches)} match(es){cache_suffix}")
                return matches

            if result_attempt < total_result_attempts:
                print(
                    f"  provider={provider.name} "
                    f"result_attempt={result_attempt}/{total_result_attempts} "
                    f"result invalid: discovery_incomplete_coverage, replaying",
                    flush=True,
                )

    batch_debug["error"] = str(last_error)
    if debug_path is not None:
        write_llm_debug(debug_path, batch_start, batch_debug)
    print(f"  [error] All LLM providers failed for batch {batch_start}. Last error: {last_error}")
    return None


def identify_content(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: BaseRecognizer,
    debug_dir: Path | None = None,
    *,
    debug_phase: str | None = None,
) -> list[ContentMatch]:
    """通用内容识别"""
    content_type = recognizer.name

    providers = build_providers(config)
    if not providers:
        raise RuntimeError("LLM API key not configured. Set llm.api_key in config.")

    clients = _build_openai_clients(providers)

    batch_size = config["llm"].get("batch_size")
    if batch_size in (None, "", 0, "0"):
        batches = [(0, segments)]
    else:
        batch_size = int(batch_size)
        batches = [
            (batch_start, segments[batch_start:batch_start + batch_size])
            for batch_start in range(0, len(segments), batch_size)
        ]

    all_matches: list[ContentMatch] = []
    debug_path = Path(debug_dir) if debug_dir is not None else None
    if debug_path is not None:
        debug_path.mkdir(parents=True, exist_ok=True)

    tools = recognizer.get_tools(config)
    llm_config = get_llm_config(config)
    store_requests = bool(llm_config.get("debug_store_requests", False))
    reuse_valid_batches = bool(llm_config.get("reuse_valid_batches", True))

    for batch_idx, (batch_start, batch_segments) in enumerate(batches, 1):
        matches = _process_single_batch(
            batch_idx=batch_idx, batch_count=len(batches),
            batch_start=batch_start, batch_segments=batch_segments,
            segments=segments, config=config, recognizer=recognizer,
            providers=providers, clients=clients, tools=tools,
            debug_path=debug_path, debug_phase=debug_phase,
            store_requests=store_requests, reuse_valid_batches=reuse_valid_batches,
            content_type=content_type,
        )
        if matches is not None:
            all_matches.extend(matches)

    _write_active_debug_files(debug_path, [batch_start for batch_start, _ in batches])
    return all_matches
