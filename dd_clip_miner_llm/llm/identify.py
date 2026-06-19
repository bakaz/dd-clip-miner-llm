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


def _is_kv_v2_main_song(
    config: dict[str, Any],
    *,
    content_type: str,
    debug_phase: str | None,
) -> bool:
    return (
        config.get("_profile_name") == "kv_v2"
        and debug_phase == "main"
        and content_type == "song"
    )


_SINGING_STRONG_KEYWORDS = (
    "歌词", "副歌", "主歌", "原曲", "歌名", "唱歌", "在唱",
    "rap", "verse", "chorus",
)

_SINGING_VOCAL_MARKERS = (
    "啦啦", "呜呜", "哼哼", "啊啊", "呀呀", "哦哦",
    "lalala", "la la", "na na", "woo", "yeah",
)


def _has_repeated_character(text: str) -> bool:
    previous = ""
    repeat_count = 0
    for char in text:
        if char.isspace():
            continue
        if char == previous:
            repeat_count += 1
            if repeat_count >= 2:
                return True
        else:
            previous = char
            repeat_count = 0
    return False


def _is_short_lyric_like_line(text: str) -> bool:
    compact = "".join(text.split())
    if not compact:
        return False
    return len(compact) <= 28


def _asr_has_singing_evidence(segments: list[TranscriptSegment]) -> bool:
    """Heuristic: does this ASR transcript contain obvious singing evidence?"""
    strong_count = 0
    vocal_count = 0
    consecutive_short = 0
    max_consecutive_short = 0
    seen_lines: set[str] = set()
    repeated_line_count = 0
    for seg in segments:
        text = seg.text.strip()
        if not text:
            consecutive_short = 0
            continue
        lowered = text.casefold()
        is_strong = any(kw in lowered for kw in _SINGING_STRONG_KEYWORDS)
        is_vocal = (
            any(marker in lowered for marker in _SINGING_VOCAL_MARKERS)
            or _has_repeated_character(text)
        )
        if is_strong:
            strong_count += 1
        if is_vocal:
            vocal_count += 1
        if _is_short_lyric_like_line(text):
            consecutive_short += 1
            max_consecutive_short = max(max_consecutive_short, consecutive_short)
        else:
            consecutive_short = 0
        normalized = "".join(lowered.split())
        if normalized in seen_lines:
            repeated_line_count += 1
        seen_lines.add(normalized)

    if strong_count >= 2:
        return True
    if strong_count >= 1 and max_consecutive_short >= 2:
        return True
    if vocal_count >= 2:
        return True
    if repeated_line_count >= 1 and max_consecutive_short >= 2:
        return True
    return False


def _validate_song_match_schema(
    items: list[Any],
    matches: list[ContentMatch],
    *,
    segments: list[TranscriptSegment] | None = None,
) -> dict[str, Any]:
    raw_item_count = len(items)
    parsed_match_count = len(matches)
    details: dict[str, Any] = {
        "schema_valid": True,
        "schema_error_reason": None,
        "raw_item_count": raw_item_count,
        "parsed_match_count": parsed_match_count,
    }
    if not items:
        if segments is not None and _asr_has_singing_evidence(segments):
            details["schema_valid"] = False
            details["schema_error_reason"] = "empty_song_match_array"
            return details
        details["schema_valid"] = True
        details["schema_error_reason"] = None
        return details
    if any(not isinstance(item, dict) for item in items):
        details["schema_valid"] = False
        details["schema_error_reason"] = "invalid_song_match_schema"
        return details
    if any(
        "segment_ranges" not in item and "segment_indices" not in item
        for item in items
    ):
        details["schema_valid"] = False
        details["schema_error_reason"] = "missing_song_segments"
        return details
    if not matches:
        details["schema_valid"] = False
        details["schema_error_reason"] = "zero_parsed_song_matches"
        return details
    return details


def _validate_song_match_content(
    content: str,
    recognizer: BaseRecognizer,
    config: dict[str, Any],
    *,
    segments: list[TranscriptSegment] | None = None,
) -> tuple[bool, dict[str, Any]]:
    parsed = parse_llm_json(content)
    if not isinstance(parsed, list):
        return False, {
            "schema_valid": False,
            "reason": "invalid_json_array",
            "schema_error_reason": "invalid_json_array",
            "raw_item_count": 0,
            "parsed_match_count": 0,
        }
    matches = recognizer.parse_response(parsed, config)
    details = _validate_song_match_schema(parsed, matches, segments=segments)
    if details["schema_valid"]:
        return True, details
    details["reason"] = details["schema_error_reason"]
    return False, details


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

    messages = build_llm_messages(
        recognizer, segments, 0, config, debug_phase=content_type,
    )
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

    messages = build_llm_messages(
        recognizer, batch_segments, batch_start, config, debug_phase=debug_phase,
    )
    request_metadata = build_request_debug_metadata(
        messages, config=config, provider=first_provider, recognizer=recognizer,
        segments=batch_segments, batch_start=batch_start, tools=tools,
        debug_phase=debug_phase,
    )
    if reuse_valid_batches and debug_path is not None and request_metadata is not None:
        cached_batch = _try_load_cached_batch(debug_path, batch_start, expected_metadata=request_metadata)
        if cached_batch is not None:
            cached_payload, cached_items = cached_batch
            raw_cached_response = cached_payload.get("raw_response")
            raw_cached_items = (
                parse_llm_json(str(raw_cached_response))
                if isinstance(raw_cached_response, str)
                else None
            )
            if not isinstance(raw_cached_items, list):
                raw_cached_items = cached_payload.get("parsed_items")
            if not isinstance(raw_cached_items, list):
                raw_cached_items = cached_items
            matches = recognizer.parse_response(cached_items, config)
            if _is_kv_v2_main_song(
                config, content_type=content_type, debug_phase=debug_phase,
            ):
                schema_details = _validate_song_match_schema(
                    raw_cached_items, matches, segments=batch_segments,
                )
                if not schema_details["schema_valid"]:
                    print(
                        f"  LLM batch {batch_idx}/{batch_count}: cached result "
                        f"invalid song schema ({schema_details['raw_item_count']} raw items, "
                        f"{schema_details['parsed_match_count']} parsed matches), ignoring cache",
                        flush=True,
                    )
                    cached_batch = None
                else:
                    cached_payload["schema_valid"] = True
            if cached_batch is not None:
                _record_cache_reuse(debug_path, batch_start, cached_payload)
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
                    initial_tool_choice = (
                        "none"
                        if _is_kv_v2_main_song(
                            config, content_type=content_type, debug_phase=debug_phase,
                        )
                        else None
                    )
                    content_validator = (
                        (
                            lambda text, _recognizer=recognizer, _config=config, _segments=batch_segments:
                            _validate_song_match_content(text, _recognizer, _config, segments=_segments)
                        )
                        if _is_kv_v2_main_song(
                            config, content_type=content_type, debug_phase=debug_phase,
                        )
                        else None
                    )
                    content = run_llm_with_tools(
                        client, provider, messages, tools, execute_tool, batch_debug,
                        max_tool_rounds=int(llm_config.get("max_tool_rounds", 2) or 0),
                        final_max_tokens=max_tokens_val,
                        force_final_round=bool(llm_config.get("force_final_tool_round", False)),
                        final_instruction=(str(llm_config["final_tool_instruction"]) if llm_config.get("final_tool_instruction") else None),
                        config=config,
                        initial_tool_choice=initial_tool_choice,
                        content_validator=content_validator,
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
                if _is_kv_v2_main_song(
                    config, content_type=content_type, debug_phase=debug_phase,
                ):
                    raw_items = parse_llm_json(content)
                    if not isinstance(raw_items, list):
                        raw_items = items
                    schema_details = _validate_song_match_schema(
                        raw_items, matches, segments=batch_segments,
                    )
                    batch_debug.update(schema_details)
                    if not schema_details["schema_valid"]:
                        reason = str(schema_details["schema_error_reason"])
                        last_error = RuntimeError(reason)
                        batch_debug.setdefault("schema_validation_failures", []).append({
                            "result_attempt": result_attempt,
                            "provider": provider.name,
                            **schema_details,
                        })
                        if debug_path is not None:
                            write_llm_debug(debug_path, batch_start, batch_debug)
                        action = (
                            "replaying"
                            if result_attempt < total_result_attempts
                            else "switching provider"
                        )
                        print(
                            f"  LLM batch {batch_idx}/{batch_count}: "
                            f"invalid song schema "
                            f"({schema_details['raw_item_count']} raw items, "
                            f"{schema_details['parsed_match_count']} parsed matches), "
                            f"{action}",
                            flush=True,
                        )
                        continue
                else:
                    batch_debug["schema_valid"] = True
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
