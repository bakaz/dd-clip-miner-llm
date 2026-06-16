"""LLM 调用包

提供与 OpenAI 兼容 API 的调用逻辑，包括：
- Provider 管理（多 key、fallback）
- 传输层（重试、流式）
- 工具调用
- Reasoning followup
- JSON 修复
- 内容识别

本模块是 llm 包的公共 API 入口，所有公共函数和类从子模块 re-export。
"""
from __future__ import annotations

# ── error ──
from .error import _classify_error

# ── provider ──
from .provider import (
    LLMProvider,
    build_providers,
    _resolve_provider_from_config,
    _ensure_openai_clients,
    _get_client,
    _build_openai_clients,
)

# ── transport ──
from .transport import (
    StreamInterruptedError,
    _build_request_kwargs,
    _call_llm_raw,
    call_llm,
    call_llm_with_transport_retry,
    _collect_stream,
    _build_llm_response,
)

# ── prompt ──
from .prompt import _CACHE_SYSTEM_PROMPT, build_llm_messages

# ── parse ──
from .parse import (
    parse_llm_json,
    parse_llm_response,
    parse_llm_response_with_status,
    write_llm_debug,
)

# ── repair ──
from .repair import (
    _extract_complete_json_objects,
    _continuation_item_key,
    _merge_continuation_items,
    _continue_truncated_json_array,
    _continue_truncated_json_object,
    build_reasoning_followup_prompt,
    reasoning_followup_settings,
    run_reasoning_followups,
    fix_json_with_llm,
    fix_structured_json_with_llm,
)

# ── tools ──
from .tools import run_llm_with_tools

# ── identify ──
from .identify import (
    identify_songs,
    identify_dialogues,
    identify_structured_content,
    _process_single_batch,
    identify_content,
)

# ── llm_debug (re-export for backward compatibility) ──
from ..llm_debug import (
    _attach_request_debug,
    _cache_usage_summary,
    _extract_task_instructions,
    _format_transcript_for_cache,
    _record_cache_reuse,
    _record_usage,
    _try_load_cached_batch,
    _write_active_debug_files,
    batch_debug_is_reusable,
    build_request_debug_metadata,
    llm_response_debug,
)

__all__ = [
    # error
    "_classify_error",
    # provider
    "LLMProvider",
    "build_providers",
    "_resolve_provider_from_config",
    "_ensure_openai_clients",
    "_get_client",
    "_build_openai_clients",
    # transport
    "StreamInterruptedError",
    "_build_request_kwargs",
    "_call_llm_raw",
    "call_llm",
    "call_llm_with_transport_retry",
    "_collect_stream",
    "_build_llm_response",
    # prompt
    "_CACHE_SYSTEM_PROMPT",
    "build_llm_messages",
    # parse
    "parse_llm_json",
    "parse_llm_response",
    "parse_llm_response_with_status",
    "write_llm_debug",
    # repair
    "_extract_complete_json_objects",
    "_continuation_item_key",
    "_merge_continuation_items",
    "_continue_truncated_json_array",
    "_continue_truncated_json_object",
    "build_reasoning_followup_prompt",
    "reasoning_followup_settings",
    "run_reasoning_followups",
    "fix_json_with_llm",
    "fix_structured_json_with_llm",
    # tools
    "run_llm_with_tools",
    # identify
    "identify_songs",
    "identify_dialogues",
    "identify_structured_content",
    "_process_single_batch",
    "identify_content",
    # llm_debug (backward compat)
    "_attach_request_debug",
    "_cache_usage_summary",
    "_extract_task_instructions",
    "_format_transcript_for_cache",
    "_record_cache_reuse",
    "_record_usage",
    "_try_load_cached_batch",
    "_write_active_debug_files",
    "batch_debug_is_reusable",
    "build_request_debug_metadata",
    "llm_response_debug",
]
