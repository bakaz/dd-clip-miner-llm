"""LLM 调用工具

提供与 OpenAI 兼容 API 的调用逻辑，包括：
- Provider 管理（多 key、fallback）
- 工具调用
- Reasoning followup
- JSON 修复
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import get_llm_config
from .llm_debug import (
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
    write_llm_debug,
)
from .models import ContentMatch, TranscriptSegment
from .profile_state import _fingerprint_payload
from .recognizers.base import BaseRecognizer


class StreamInterruptedError(Exception):
    """流式响应中断异常。

    当流式传输过程中发生中断（如网络错误、超时）时抛出，
    保留已收到的部分内容供后续处理。
    """

    def __init__(self, original_error: Exception, partial_content: str):
        super().__init__(str(original_error))
        self.original_error = original_error
        self.partial_content = partial_content


_CACHE_SYSTEM_PROMPT = (
    "你将先收到一份带全局序号和时间范围的 ASR 转写，再收到具体分析任务。"
    "必须只依据该转写完成任务，不得使用输入中不存在的 segment index。"
)


@dataclass
class LLMProvider:
    name: str = "default"
    api_key: str = ""
    base_url: str | None = None
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 4096
    max_completion_tokens: int | None = None
    thinking: str | None = None
    timeout: float = 300.0
    max_retries: int = 3
    proxy: str | None = None
    stream: bool = False
    # 重试退避与路由配置
    timeout_schedule: list[float] | None = None
    retry_backoff_seconds: list[float] | None = None
    retry_jitter_ratio: float = 0.25
    result_retries: int = 2


# ============ Provider 管理 ============

def _resolve_provider_from_config(
    name: str,
    provider_cfg: dict[str, Any],
    llm_shared: dict[str, Any],
) -> LLMProvider | None:
    """从 provider 配置构建 LLMProvider，返回 None 如果没有 api_key。"""
    api_key = provider_cfg.get("api_key", "")
    api_key_env = provider_cfg.get("api_key_env")
    if not api_key and api_key_env:
        api_key = os.environ.get(str(api_key_env), "")
    if not api_key:
        return None

    base_url = provider_cfg.get("base_url") or llm_shared.get("base_url")
    model = provider_cfg.get("model") or llm_shared.get("model", "gpt-4o")
    temperature = float(provider_cfg.get("temperature", llm_shared.get("temperature", 0.3)))
    max_tokens = int(provider_cfg.get("max_tokens", llm_shared.get("max_tokens", 4096)))
    max_completion_tokens_value = provider_cfg.get("max_completion_tokens")
    if max_completion_tokens_value in (None, ""):
        max_completion_tokens_value = llm_shared.get("max_completion_tokens")
    max_completion_tokens = (
        int(max_completion_tokens_value)
        if max_completion_tokens_value not in (None, "")
        else None
    )
    thinking = provider_cfg.get("thinking")
    if thinking in (None, ""):
        thinking = llm_shared.get("thinking")
    timeout = float(provider_cfg.get("timeout", llm_shared.get("timeout", 300)))
    max_retries = max(1, int(provider_cfg.get("max_retries", llm_shared.get("max_retries", 3))))

    # 新字段：重试退避与路由
    timeout_schedule_raw = provider_cfg.get("timeout_schedule")
    if timeout_schedule_raw is None:
        timeout_schedule_raw = llm_shared.get("timeout_schedule")
    timeout_schedule: list[float] | None = None
    if isinstance(timeout_schedule_raw, (list, tuple)) and timeout_schedule_raw:
        timeout_schedule = [float(v) for v in timeout_schedule_raw]

    retry_backoff_raw = provider_cfg.get("retry_backoff_seconds")
    if retry_backoff_raw is None:
        retry_backoff_raw = llm_shared.get("retry_backoff_seconds", [2, 5])
    retry_backoff_seconds: list[float] | None = None
    if isinstance(retry_backoff_raw, (list, tuple)) and retry_backoff_raw:
        retry_backoff_seconds = [float(v) for v in retry_backoff_raw]

    retry_jitter_ratio = float(
        provider_cfg.get("retry_jitter_ratio", llm_shared.get("retry_jitter_ratio", 0.25))
    )
    result_retries = max(0, int(
        provider_cfg.get("result_retries", llm_shared.get("result_retries", 2))
    ))

    proxy_raw = provider_cfg.get("proxy")
    if proxy_raw is None:
        proxy_raw = llm_shared.get("proxy")
    proxy = str(proxy_raw).strip() if proxy_raw not in (None, "") else None

    stream_raw = provider_cfg.get("stream")
    if stream_raw is None:
        stream_raw = llm_shared.get("stream")
    stream = bool(stream_raw) if stream_raw not in (None, "") else False

    # 向后兼容：旧配置没有 timeout_schedule 时，不自动创建
    # 让 call_llm 回退到 max_retries 行为
    # call_llm_with_transport_retry 内部会按 max_retries 重复 provider.timeout

    return LLMProvider(
        name=name,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
        thinking=str(thinking) if thinking not in (None, "") else None,
        timeout=timeout,
        max_retries=max_retries,
        proxy=proxy,
        stream=stream,
        timeout_schedule=timeout_schedule,
        retry_backoff_seconds=retry_backoff_seconds,
        retry_jitter_ratio=retry_jitter_ratio,
        result_retries=result_retries,
    )


def build_providers(config: dict[str, Any]) -> list[LLMProvider]:
    """构建 LLM provider 列表。

    优先使用 provider_route 按名称顺序路由。
    未配置 provider_route 时回退到 active_provider + fallbacks（旧行为）。
    """
    llm_config = config["llm"]
    providers_cfg = llm_config.get("providers", {})

    # 共享参数（provider 内未指定时作为默认值）
    llm_shared: dict[str, Any] = {
        k: v for k, v in llm_config.items()
        if k not in ("providers", "provider_route", "active_provider", "fallbacks")
    }

    provider_route = llm_config.get("provider_route")
    if provider_route:
        # 新路由模式：按 provider_route 列表顺序构建
        result: list[LLMProvider] = []
        for name in provider_route:
            name = str(name).strip()
            if not name:
                continue
            provider_cfg = providers_cfg.get(name, {})
            provider = _resolve_provider_from_config(name, provider_cfg, llm_shared)
            if provider is not None:
                result.append(provider)
        return result

    # 旧模式：active_provider + fallbacks
    active_name = str(llm_config.get("active_provider") or "").strip()
    result = []
    if active_name and active_name in providers_cfg:
        provider = _resolve_provider_from_config(
            active_name, providers_cfg[active_name], llm_shared
        )
        if provider is not None:
            result.append(provider)

    for fb in llm_config.get("fallbacks", []):
        provider = _resolve_provider_from_config("fallback", fb, llm_shared)
        if provider is not None:
            result.append(provider)

    # 兜底：如果既无 route 也无 active_provider，返回所有有 key 的 provider
    if not result and providers_cfg:
        for name, cfg in providers_cfg.items():
            if not isinstance(cfg, dict):
                continue
            provider = _resolve_provider_from_config(name, cfg, llm_shared)
            if provider is not None:
                result.append(provider)

    # 最终兜底：旧版平铺格式（llm 直接含 api_key，无 providers 字典）
    if not result:
        api_key = llm_config.get("api_key", "")
        api_key_env = llm_config.get("api_key_env")
        if not api_key and api_key_env:
            api_key = os.environ.get(str(api_key_env), "")
        if api_key:
            provider = _resolve_provider_from_config("default", llm_config, llm_shared)
            if provider is not None:
                result.append(provider)

    return result


# ============ LLM 调用 ============

def _build_request_kwargs(
    provider: LLMProvider,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    max_tokens_override: int | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """构建 OpenAI 请求参数。"""
    token_limit = (
        max_tokens_override
        if max_tokens_override is not None
        else (
            provider.max_completion_tokens
            if provider.max_completion_tokens is not None
            else provider.max_tokens
        )
    )
    uses_deepseek_api = "deepseek.com" in str(provider.base_url or "").casefold()
    token_args = (
        {"max_tokens": token_limit}
        if uses_deepseek_api or provider.max_completion_tokens is None
        else {"max_completion_tokens": token_limit}
    )
    kwargs: dict[str, Any] = {
        "model": provider.model,
        "messages": messages,
        "temperature": provider.temperature,
        **token_args,
    }
    if uses_deepseek_api and provider.thinking:
        kwargs["extra_body"] = {"thinking": {"type": provider.thinking}}
    if tools:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
    kwargs["timeout"] = provider.timeout if timeout is None else float(timeout)
    return kwargs


def _call_llm_raw(client: Any, kwargs: dict[str, Any]) -> Any:
    """唯一底层请求入口。所有 LLM API 调用必须经过此函数。"""
    return client.chat.completions.create(**kwargs)


def call_llm(
    client: Any,
    provider: LLMProvider,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens_override: int | None = None,
    max_retries: int | None = None,
    tool_choice: Any = None,
    timeout: float | None = None,
) -> Any:
    """调用 LLM，返回完整的 response 对象。

    旧路径：无 timeout_schedule 时按 max_retries 指数退避。
    新路径：有 timeout_schedule 或启用流式时委托 call_llm_with_transport_retry。
    所有请求最终经过 _call_llm_raw，测试可拦截。
    """
    effective_max_retries = (
        provider.max_retries if max_retries is None else max(1, int(max_retries))
    )

    # 流式响应必须由传输级重试负责收集，即使使用旧版超时配置。
    if provider.timeout_schedule or (provider.stream and not tools):
        response, _content = call_llm_with_transport_retry(
            client, provider, messages,
            tools=tools, tool_choice=tool_choice,
            max_tokens_override=max_tokens_override,
            timeout=timeout,
            max_attempts=effective_max_retries,
        )
        return response

    # 旧路径：无 timeout_schedule 时按 max_retries 指数退避
    import time as _time

    kwargs = _build_request_kwargs(
        provider, messages, tools=tools, tool_choice=tool_choice,
        max_tokens_override=max_tokens_override, timeout=timeout,
    )
    last_exc = None
    for retry in range(effective_max_retries):
        print(
            f"  [llm] Request attempt {retry + 1}/{effective_max_retries} "
            f"(timeout={kwargs['timeout']:g}s, provider={provider.name})",
            flush=True,
        )
        try:
            return _call_llm_raw(client, kwargs)
        except Exception as exc:
            last_exc = exc
            if retry < effective_max_retries - 1:
                wait_time = 2 ** retry
                print(
                    f"  [llm] API call failed "
                    f"(attempt {retry + 1}/{effective_max_retries}, wait {wait_time}s): {exc}",
                    flush=True,
                )
                _time.sleep(wait_time)
    if last_exc is None:
        raise RuntimeError("call_llm was invoked with max_retries=0; no attempt was made")
    raise last_exc


def call_llm_with_transport_retry(
    client: Any,
    provider: LLMProvider,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    max_tokens_override: int | None = None,
    *,
    timeout: float | None = None,
    batch_debug: dict[str, Any] | None = None,
    max_attempts: int | None = None,
) -> tuple[Any, str]:
    """传输级重试：timeout_schedule 逐步升级超时 + 退避抖动。

    返回 (response, content)。content 已含 reasoning fallback。
    每次传输尝试通过 _call_llm_raw，测试可拦截。
    支持流式接收（provider.stream=True 且无 tools）：
      - 逐 chunk 收集内容
      - 流式中断时继续传输重试（不立即触发续写）
      - 传输重试耗尽且有部分内容时，返回 finish_reason="length" 触发续写
    传输耗尽后 raise 最后一个异常（非流式）或返回部分内容（流式）。
    """
    import random
    import time

    kwargs = _build_request_kwargs(
        provider, messages, tools=tools, tool_choice=tool_choice,
        max_tokens_override=max_tokens_override, timeout=timeout,
    )
    fallback_attempts = provider.max_retries if max_attempts is None else max(1, max_attempts)
    schedule = provider.timeout_schedule or [provider.timeout] * fallback_attempts
    backoff_list = provider.retry_backoff_seconds or [2, 5]
    jitter = provider.retry_jitter_ratio
    use_stream = provider.stream and not tools

    if use_stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

    last_exc: Exception | None = None
    last_partial_content = ""
    for attempt, timeout_val in enumerate(schedule):
        kwargs["timeout"] = timeout_val
        print(
            f"  [llm] transport attempt {attempt + 1}/{len(schedule)} "
            f"(timeout={timeout_val:g}s, provider={provider.name})",
            flush=True,
        )
        try:
            if use_stream:
                response, content = _collect_stream(client, kwargs)
            else:
                response = _call_llm_raw(client, kwargs)
                content = ""
                choice = response.choices[0] if response.choices else None
                message = choice.message if choice is not None else None
                if message:
                    content = message.content or ""
                    if not content.strip():
                        reasoning = getattr(message, "reasoning_content", None) or ""
                        if reasoning.strip():
                            content = reasoning
            if batch_debug is not None:
                _record_usage(batch_debug, "transport", llm_response_debug(response), attempt=attempt + 1)
            return response, content
        except StreamInterruptedError as exc:
            last_exc = exc.original_error
            # 流式中断时，保留已收到的部分内容
            if len(exc.partial_content) > len(last_partial_content):
                last_partial_content = exc.partial_content

            retryable, reason = _classify_error(exc.original_error)
            if not retryable:
                print(f"  [llm] non-retryable ({reason}): {exc.original_error}", flush=True)
                raise exc.original_error

            backoff_base = backoff_list[min(attempt, len(backoff_list) - 1)]
            jitter_delta = backoff_base * jitter * (2 * random.random() - 1)
            wait_time = max(0.0, backoff_base + jitter_delta)

            retry_after = None
            resp = getattr(exc.original_error, "response", None)
            if resp is not None:
                ra = resp.headers.get("retry-after") if hasattr(resp, "headers") else None
                if ra is not None:
                    try:
                        retry_after = min(float(ra), 60.0)
                    except (ValueError, TypeError):
                        pass
            if retry_after is not None and retry_after > 0:
                wait_time = retry_after

            if attempt < len(schedule) - 1:
                print(
                    f"  [llm] transport failed ({reason}); retrying in {wait_time:.1f}s",
                    flush=True,
                )
                if batch_debug is not None:
                    batch_debug.setdefault("transport_retries", []).append({
                        "attempt": attempt + 1,
                        "timeout": timeout_val,
                        "reason": reason,
                        "wait_seconds": round(wait_time, 1),
                    })
                time.sleep(wait_time)
                continue  # 继续下一次传输尝试
        except Exception as exc:
            last_exc = exc
            retryable, reason = _classify_error(exc)
            if not retryable:
                print(f"  [llm] non-retryable ({reason}): {exc}", flush=True)
                raise

            backoff_base = backoff_list[min(attempt, len(backoff_list) - 1)]
            jitter_delta = backoff_base * jitter * (2 * random.random() - 1)
            wait_time = max(0.0, backoff_base + jitter_delta)

            retry_after = None
            resp = getattr(exc, "response", None)
            if resp is not None:
                ra = resp.headers.get("retry-after") if hasattr(resp, "headers") else None
                if ra is not None:
                    try:
                        retry_after = min(float(ra), 60.0)
                    except (ValueError, TypeError):
                        pass
            if retry_after is not None and retry_after > 0:
                wait_time = retry_after

            if attempt < len(schedule) - 1:
                print(
                    f"  [llm] transport failed ({reason}); retrying in {wait_time:.1f}s",
                    flush=True,
                )
                if batch_debug is not None:
                    batch_debug.setdefault("transport_retries", []).append({
                        "attempt": attempt + 1,
                        "timeout": timeout_val,
                        "reason": reason,
                        "wait_seconds": round(wait_time, 1),
                    })
                time.sleep(wait_time)
                continue  # 继续下一次传输尝试
    # 传输重试耗尽
    if use_stream and last_partial_content:
        # 流式模式：返回部分内容，触发续写
        print(
            f"  [llm] stream interrupted, returning partial content "
            f"({len(last_partial_content)} chars) for continuation",
            flush=True,
        )
        response = _build_llm_response(
            content=last_partial_content,
            finish_reason="length",
            usage=None,
            model=provider.model,
        )
        return response, last_partial_content

    if last_exc is None:
        raise RuntimeError("call_llm_with_transport_retry: no attempt was made")
    raise last_exc


def _collect_stream(client: Any, kwargs: dict[str, Any]) -> tuple[Any, str]:
    """收集流式响应，返回 (response, content)。

    如果流式中断，抛出的异常会附带 _partial_content 属性。
    """
    stream = _call_llm_raw(client, kwargs)
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = None
    usage = None
    model = None

    try:
        for chunk in stream:
            model = getattr(chunk, "model", None) or model
            if not chunk.choices:
                # usage-only chunk
                usage = getattr(chunk, "usage", None) or usage
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta:
                if delta.content:
                    content_parts.append(delta.content)
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    reasoning_parts.append(rc)
            fr = getattr(choice, "finish_reason", None)
            if fr:
                finish_reason = fr
            usage = getattr(chunk, "usage", None) or usage
    except Exception as exc:
        # 流式中断，保留已收到的内容
        partial = "".join(content_parts) or "".join(reasoning_parts)
        raise StreamInterruptedError(exc, partial) from exc

    content = "".join(content_parts)
    if not content.strip() and reasoning_parts:
        content = "".join(reasoning_parts)

    response = _build_llm_response(
        content=content,
        finish_reason=finish_reason or "stop",
        usage=usage,
        model=model or "",
    )
    return response, content


def _build_llm_response(
    content: str,
    finish_reason: str = "stop",
    usage: Any = None,
    model: str = "",
) -> Any:
    """构建 LLM 响应对象。

    支持正常完成和部分内容（finish_reason="length" 触发续写）。
    """
    from types import SimpleNamespace

    message = SimpleNamespace(
        content=content,
        reasoning_content="",
        tool_calls=None,
    )
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
    )
    return SimpleNamespace(
        choices=[choice],
        usage=usage,
        model=model,
    )


# ============ 重试退避与路由 ============

def _classify_error(exc: Exception) -> tuple[bool, str]:
    """分类异常是否可重试，返回 (retryable, reason)。

    可重试：网络错误、超时、429、5xx。
    不可重试：401、403、确定的请求参数错误。
    """
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            PermissionDeniedError,
            RateLimitError,
            APIStatusError,
        )
    except ImportError:
        msg = str(exc).lower()
        if any(k in msg for k in ("timeout", "connection", "network")):
            return True, "network_error"
        return False, "unknown_error"

    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True, "network_error"
    if isinstance(exc, TimeoutError):
        return True, "timeout"
    if isinstance(exc, (ConnectionError, OSError)):
        return True, "network_error"
    if isinstance(exc, RateLimitError):
        return True, "rate_limited"
    if isinstance(exc, AuthenticationError):
        return False, "auth_error"
    if isinstance(exc, PermissionDeniedError):
        return False, "permission_denied"
    if isinstance(exc, BadRequestError):
        return False, "bad_request"
    if isinstance(exc, APIStatusError):
        code = exc.status_code
        if code == 429:
            return True, "rate_limited"
        if code >= 500:
            return True, f"server_error_{code}"
        if code in (401, 403):
            return False, f"http_{code}"
        if 400 <= code < 500:
            return False, f"client_error_{code}"
        return True, f"http_{code}"
    return False, "unknown_error"


def _ensure_openai_clients(
    providers: list[LLMProvider],
    clients: dict[tuple[str | None, str, str | None], Any],
) -> None:
    """确保所有 provider 对应的 OpenAI 客户端已创建。

    按 (base_url, api_key, proxy) 缓存，避免相同密钥不同代理复用错误客户端。
    """
    from openai import OpenAI
    for provider in providers:
        cache_key = (provider.base_url, provider.api_key, provider.proxy)
        if not provider.api_key or cache_key in clients:
            continue
        client_kwargs: dict[str, Any] = {"api_key": provider.api_key}
        if provider.base_url:
            client_kwargs["base_url"] = provider.base_url
        if provider.proxy:
            import httpx
            client_kwargs["http_client"] = httpx.Client(proxy=provider.proxy)
        clients[cache_key] = OpenAI(**client_kwargs)


def _get_client(
    provider: LLMProvider,
    clients: dict[tuple[str | None, str, str | None], Any],
) -> Any:
    """从缓存获取 OpenAI 客户端。"""
    return clients.get((provider.base_url, provider.api_key, provider.proxy))


def build_llm_messages(
    recognizer: BaseRecognizer,
    segments: list[TranscriptSegment],
    batch_start: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """构建请求消息；缓存友好模式把可复用 ASR 长文本放在任务指令之前。"""
    prompt = recognizer.build_prompt(segments, batch_start, config)
    llm_config = get_llm_config(config)
    if not llm_config.get("cache_friendly_prompt_layout", False):
        system_prompt = recognizer.build_system_prompt(config)
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    instructions = _extract_task_instructions(prompt)
    if not instructions:
        return [{"role": "user", "content": prompt}]

    recognizer_system_prompt = recognizer.build_system_prompt(config)
    if recognizer_system_prompt:
        instructions = f"{recognizer_system_prompt}\n\n{instructions}"

    transcript = _format_transcript_for_cache(segments, batch_start, recognizer)
    return [
        {"role": "system", "content": _CACHE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"ASR 转写开始：\n{transcript}\nASR 转写结束。\n\n"
                f"{instructions}\n\n请基于上面的完整 ASR 转写执行任务。"
            ),
        },
    ]


def _extract_complete_json_objects(text: str) -> list[dict[str, Any]]:
    """Extract complete top-level objects from a possibly truncated JSON array."""
    objects: list[dict[str, Any]] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    value = json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    value = None
                if isinstance(value, dict):
                    objects.append(value)
                start = None
    return objects


def _continuation_item_key(item: dict[str, Any]) -> str:
    if item.get("content_type") == "scan_checkpoint":
        return f"checkpoint:{item.get('scan_id', '')}"
    ranges = item.get("segment_ranges")
    if not isinstance(ranges, list):
        ranges = item.get("segment_indices")
    return _fingerprint_payload({
        "content_type": item.get("content_type"),
        "scan_id": item.get("scan_id"),
        "title": item.get("title"),
        "ranges": ranges,
    })


def _merge_continuation_items(
    target: list[dict[str, Any]],
    seen: set[str],
    content: str,
) -> bool:
    items, valid = parse_llm_response_with_status(content)
    if not valid:
        items = _extract_complete_json_objects(content)
    for item in items:
        key = _continuation_item_key(item)
        if key in seen:
            continue
        seen.add(key)
        target.append(item)
    return valid


def _continue_truncated_json_array(
    client: Any,
    provider: LLMProvider,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    content: str,
    finish_reason: str | None,
    batch_debug: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int | None = None,
) -> str:
    """Continue a truncated JSON array while preserving the original request prefix."""
    llm_config = get_llm_config(config)
    if finish_reason != "length" or not llm_config.get("continuation_on_length", False):
        return content

    max_rounds = max(0, int(llm_config.get("max_continuation_rounds", 0) or 0))
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    _merge_continuation_items(merged, seen, content)
    current_reason = finish_reason
    continuation_debug = batch_debug.setdefault("continuation_rounds", [])

    for round_index in range(max_rounds):
        checkpoints = [
            str(item.get("scan_id"))
            for item in merged
            if item.get("content_type") == "scan_checkpoint" and item.get("scan_id")
        ]
        completed = [
            {
                "scan_id": item.get("scan_id"),
                "title": item.get("title"),
                "segment_ranges": item.get("segment_ranges"),
                "segment_indices": item.get("segment_indices"),
            }
            for item in merged
            if item.get("content_type") != "scan_checkpoint"
        ]
        continuation_prompt = (
            "上一轮 JSON 数组因为输出长度限制而截断。请继续完成同一个任务，只返回尚未输出的对象，"
            "不要重复下列已完成对象。仍然只返回纯 JSON 数组。"
            f"\n已完成 scan checkpoint: {json.dumps(checkpoints, ensure_ascii=False, separators=(',', ':'))}"
            f"\n已完成对象: {json.dumps(completed[-200:], ensure_ascii=False, separators=(',', ':'))}"
        )
        response = call_llm(
            client,
            provider,
            [*messages, {"role": "user", "content": continuation_prompt}],
            tools=tools,
            tool_choice="none" if tools else None,
            max_tokens_override=max_tokens,
        )
        debug = llm_response_debug(response)
        _record_usage(batch_debug, "continuation", debug, round=round_index + 1)
        continuation_content = debug["content"] or debug["reasoning_content"]
        valid = _merge_continuation_items(merged, seen, continuation_content)
        current_reason = debug["finish_reason"]
        continuation_debug.append({
            "round": round_index + 1,
            "finish_reason": current_reason,
            "parse_valid": valid,
            "content": continuation_content,
            "usage": debug["usage"],
        })
        if current_reason != "length" and valid:
            batch_debug["continuation_complete"] = True
            batch_debug["finish_reason"] = current_reason
            return json.dumps(merged, ensure_ascii=False, separators=(",", ":"))

    batch_debug["scan_incomplete"] = True
    batch_debug["continuation_complete"] = False
    batch_debug["finish_reason"] = current_reason
    if merged:
        return json.dumps(merged, ensure_ascii=False, separators=(",", ":"))
    return content


# ============ 工具调用 ============

def run_llm_with_tools(
    client: Any,
    provider: LLMProvider,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_executor: Any,
    batch_debug: dict[str, Any],
    max_tool_rounds: int = 2,
    final_max_tokens: int | None = None,
    force_final_round: bool = False,
    final_instruction: str | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    """调用 LLM，处理 tool calls
    
    Args:
        client: OpenAI 客户端
        provider: LLM provider
        messages: 消息列表
        tools: 工具定义
        tool_executor: 工具执行器，接受 (name, arguments) 返回结果字符串
        batch_debug: 调试信息字典
        max_tool_rounds: 最大工具调用轮数
    """
    for tool_round in range(max_tool_rounds + 1):
        is_last = (tool_round == max_tool_rounds)

        call_tools = tools
        tool_choice = "none" if is_last else "auto"
        last_round_tokens = final_max_tokens if is_last else None
        if is_last and tool_round > 0:
            messages = messages + [{
                "role": "user",
                "content": final_instruction or (
                    "搜索已完成。现在请根据已有的搜索结果，直接返回识别结果的JSON数组。"
                    "不要再调用任何工具。只返回JSON数组，不要其他文字。"
                ),
            }]

        response = call_llm(
            client,
            provider,
            messages,
            max_tokens_override=last_round_tokens,
            tools=call_tools,
            tool_choice=tool_choice,
        )
        debug = llm_response_debug(response)
        _record_usage(batch_debug, "tool", debug, round=tool_round + 1)
        batch_debug["finish_reason"] = debug["finish_reason"]
        batch_debug.setdefault("tool_rounds", []).append({
            "round": tool_round + 1,
            "content": debug["content"][:200],
            "reasoning_content": debug["reasoning_content"][:200],
            "finish_reason": debug["finish_reason"],
            "has_tool_calls": bool(debug.get("tool_calls")),
            "usage": debug["usage"],
        })

        content = debug["content"]
        tool_calls_data = debug.get("tool_calls")

        if not tool_calls_data:
            if not content.strip() and debug["reasoning_content"].strip():
                content = debug["reasoning_content"]
            if not is_last and force_final_round:
                _, is_valid_array = parse_llm_response_with_status(content)
                if not is_valid_array:
                    continue
            if config is not None:
                content = _continue_truncated_json_array(
                    client, provider, config, messages, content,
                    debug["finish_reason"], batch_debug, tools=tools,
                    max_tokens=final_max_tokens,
                )
            return content

        if is_last:
            if not content.strip() and debug["reasoning_content"].strip():
                content = debug["reasoning_content"]
            if config is not None:
                content = _continue_truncated_json_array(
                    client, provider, config, messages, content,
                    debug["finish_reason"], batch_debug, tools=tools,
                    max_tokens=final_max_tokens,
                )
            return content

        choice = response.choices[0] if response.choices else None
        message = choice.message if choice is not None else None
        if not message or not message.tool_calls:
            return content

        messages.append(message.model_dump())
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = tool_executor(tc.function.name, args)
            tool_log: dict[str, Any] = {
                "round": tool_round + 1,
                "function": tc.function.name,
                "arguments": args,
                "result_preview": result[:1000],
                "result_length": len(result),
            }
            if tc.function.name == "search_lyrics":
                try:
                    search_payload = json.loads(result)
                except (TypeError, json.JSONDecodeError):
                    search_payload = None
                if isinstance(search_payload, dict):
                    tool_log["result_summary"] = {
                        "query": search_payload.get("query", ""),
                        "results": [
                            {
                                "title": item.get("title", ""),
                                "snippet": str(item.get("snippet", ""))[:240],
                                "url": item.get("url", ""),
                            }
                            for item in search_payload.get("results", [])[:3]
                            if isinstance(item, dict)
                        ],
                        "lyrics_hints": [
                            str(item)[:240]
                            for item in search_payload.get("lyrics_hints", [])[:2]
                        ],
                    }
            batch_debug.setdefault("tool_calls_log", []).append(tool_log)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return ""


# ============ Reasoning Followup ============

def build_reasoning_followup_prompt(reasoning_content: str, partial_content: str = "") -> str:
    """构建 reasoning followup 提示词"""
    partial_block = (
        f"\n\n上一轮已经生成但可能被截断或格式不完整的内容：\n{partial_content}"
        if partial_content.strip()
        else ""
    )
    return f"""下面是上一轮模型对内容识别任务的分析内容。它可能是不完整的，但里面已经包含了内容边界判断。

不要继续分析，不要解释，不要输出思考过程。请只把分析中已经确定的内容整理成 JSON 数组。

输出必须是纯 JSON 数组，不要 Markdown，不要代码块，不要额外文字。

上一轮分析内容：
{reasoning_content}{partial_block}"""


def reasoning_followup_settings(config: dict[str, Any]) -> tuple[bool, int, int | None]:
    """获取 reasoning followup 配置"""
    llm_config = config["llm"]
    enabled = bool(llm_config.get("retry_empty_with_reasoning", True))
    rounds = int(llm_config.get("reasoning_followup_rounds", 2))
    tokens_value = llm_config.get("reasoning_followup_max_tokens", 8192)
    tokens = int(tokens_value) if tokens_value not in (None, "") else None
    return enabled, max(0, rounds), tokens


def run_reasoning_followups(
    client: Any,
    provider: LLMProvider,
    config: dict[str, Any],
    reasoning_content: str,
    partial_content: str,
    batch_debug: dict[str, Any],
) -> str:
    """运行 reasoning followup 轮次"""
    retry_reasoning, followup_rounds, followup_tokens = reasoning_followup_settings(config)
    if not retry_reasoning:
        return ""

    content = ""
    material = reasoning_content
    partial = partial_content
    for _ in range(followup_rounds):
        if not material.strip() and not partial.strip():
            break

        followup_prompt = build_reasoning_followup_prompt(material, partial)
        try:
            followup_response = call_llm(
                client,
                provider,
                [{"role": "user", "content": followup_prompt}],
                max_tokens_override=followup_tokens,
            )
            followup_debug = llm_response_debug(followup_response)
            _record_usage(
                batch_debug,
                "reasoning_followup",
                followup_debug,
                round=len(batch_debug["reasoning_followups"]) + 1,
            )
            content = followup_debug["content"]
            content = _continue_truncated_json_array(
                client,
                provider,
                config,
                [{"role": "user", "content": followup_prompt}],
                content or followup_debug["reasoning_content"],
                followup_debug["finish_reason"],
                batch_debug,
                max_tokens=followup_tokens,
            )
            batch_debug["reasoning_followups"].append({
                "round": len(batch_debug["reasoning_followups"]) + 1,
                "content": content[:500],
                "reasoning_content": followup_debug["reasoning_content"][:500],
                "usage": followup_debug["usage"],
            })
            batch_debug["raw_response"] = content
        except Exception as exc:
            batch_debug["reasoning_followups"].append({
                "round": len(batch_debug["reasoning_followups"]) + 1,
                "error": str(exc),
            })
            return ""

        _, is_valid_array = parse_llm_response_with_status(content)
        if content.strip() and is_valid_array:
            return content

        material = str(followup_debug.get("reasoning_content") or "")
        partial = content

    return content


# ============ JSON 修复 ============

def fix_json_with_llm(
    client: Any,
    provider: LLMProvider,
    config: dict[str, Any],
    raw_content: str,
    content_type: str,
    batch_debug: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """当 LLM 返回非 JSON 时，让它把内容转换成 JSON 格式"""
    max_rounds = int(config["llm"].get("json_fix_rounds", 3))
    if max_rounds <= 0:
        return [], raw_content

    fix_prompt = f"""下面是之前对{content_type}识别任务的回复，但它不是纯JSON格式。
请把其中的信息提取出来，转换成纯JSON数组。

输出必须是纯JSON数组，不要Markdown，不要代码块，不要额外文字。

之前的回复：
{raw_content}"""

    content = raw_content
    for round_num in range(max_rounds):
        try:
            response = call_llm(
                client, provider,
                [{"role": "user", "content": fix_prompt}],
                max_tokens_override=(
                    provider.max_completion_tokens
                    if provider.max_completion_tokens is not None
                    else provider.max_tokens
                ),
            )
            debug = llm_response_debug(response)
            _record_usage(batch_debug, "json_fix", debug, round=round_num + 1)
            new_content = debug["content"] or debug["reasoning_content"]
            new_content = _continue_truncated_json_array(
                client,
                provider,
                config,
                [{"role": "user", "content": fix_prompt}],
                new_content,
                debug["finish_reason"],
                batch_debug,
                max_tokens=(
                    provider.max_completion_tokens
                    if provider.max_completion_tokens is not None
                    else provider.max_tokens
                ),
            )
            batch_debug.setdefault("json_fix_rounds", []).append({
                "round": round_num + 1,
                "content": new_content[:500],
                "finish_reason": debug["finish_reason"],
                "usage": debug["usage"],
            })

            items, is_valid_array = parse_llm_response_with_status(new_content)
            if is_valid_array:
                return items, new_content

            if new_content.strip():
                content = new_content
                fix_prompt = f"""下面的回复仍然不是纯JSON格式。请直接返回纯JSON数组，不要任何其他文字。

{new_content}"""
        except Exception as exc:
            batch_debug.setdefault("json_fix_rounds", []).append({
                "round": round_num + 1,
                "error": str(exc),
            })
            break

    return [], content


def fix_structured_json_with_llm(
    client: Any,
    provider: LLMProvider,
    config: dict[str, Any],
    raw_content: str,
    content_type: str,
    batch_debug: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """当 LLM 返回非 JSON 时，让它把内容转换成 JSON object。"""
    max_rounds = int(config["llm"].get("json_fix_rounds", 3))
    if max_rounds <= 0:
        return {
            "content_type": content_type,
            "title": config.get(content_type, {}).get("title", content_type),
            "error": "LLM JSON repair disabled",
            "raw_response": raw_content,
        }, raw_content

    fix_prompt = f"""下面是之前对{content_type}任务的回复，但它不是纯JSON object。
请把其中的信息提取出来，转换成纯JSON object。

输出必须是纯JSON object，不要Markdown，不要代码块，不要额外文字。

之前的回复：
{raw_content}"""

    content = raw_content
    for round_num in range(max_rounds):
        try:
            response = call_llm(
                client, provider,
                [{"role": "user", "content": fix_prompt}],
                max_tokens_override=(
                    provider.max_completion_tokens
                    if provider.max_completion_tokens is not None
                    else provider.max_tokens
                ),
            )
            debug = llm_response_debug(response)
            _record_usage(batch_debug, "json_fix", debug, round=round_num + 1)
            new_content = debug["content"] or debug["reasoning_content"]
            batch_debug.setdefault("json_fix_rounds", []).append({
                "round": round_num + 1,
                "content": new_content[:500],
                "finish_reason": debug["finish_reason"],
                "usage": debug["usage"],
            })

            parsed = parse_llm_json(new_content)
            if isinstance(parsed, dict) and parsed:
                return parsed, new_content

            if new_content.strip():
                content = new_content
                fix_prompt = f"""下面的回复仍然不是纯JSON object。请直接返回纯JSON object，不要任何其他文字。

{new_content}"""
        except Exception as exc:
            batch_debug.setdefault("json_fix_rounds", []).append({
                "round": round_num + 1,
                "error": str(exc),
            })
            break

    return {
        "content_type": content_type,
        "title": config.get(content_type, {}).get("title", content_type),
        "error": "LLM JSON repair failed",
        "raw_response": content,
    }, content


# ============ 响应解析 ============

def parse_llm_json(text: str) -> Any:
    """解析 LLM 响应为 JSON，兼容代码块和前后解释文字。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates_with_start: list[tuple[int, str]] = []
    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start != -1 and object_end > object_start:
        candidates_with_start.append((object_start, text[object_start:object_end + 1]))

    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end > array_start:
        candidates_with_start.append((array_start, text[array_start:array_end + 1]))

    for _, candidate in sorted(candidates_with_start, key=lambda item: item[0]):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def parse_llm_response(text: str) -> list[dict[str, Any]]:
    """解析 LLM 响应为 JSON 数组"""
    items, _ = parse_llm_response_with_status(text)
    return items


def parse_llm_response_with_status(text: str) -> tuple[list[dict[str, Any]], bool]:
    """解析 JSON 数组，并区分合法空数组与解析失败。"""
    result = parse_llm_json(text)
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)], True
    return [], False


# ============ 调试工具 ============

def write_llm_debug(debug_dir: Path, batch_start: int, payload: dict[str, Any]) -> None:
    """写入 LLM 调试信息"""
    target = debug_dir / f"llm_batch_{batch_start:06d}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ============ 兼容旧接口 ============

def identify_songs(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    debug_dir: Path | None = None,
) -> list[ContentMatch]:
    """识别歌曲片段（兼容旧接口）"""
    from .recognizers import get_recognizer
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
    from .recognizers import get_recognizer
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
            except Exception as exc:
                last_error = exc
                print(f"  [warn] provider={candidate.name} failed: {exc}")
                break

            # 产物验证：结构化内容需要是 JSON object
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


# ============ 核心识别逻辑 ============

def _build_openai_clients(providers: list[LLMProvider]) -> dict[tuple[str | None, str, str | None], Any]:
    """Pre-create OpenAI clients keyed by (base_url, api_key)."""
    clients: dict[tuple[str | None, str, str | None], Any] = {}
    _ensure_openai_clients(providers, clients)
    return clients


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
    """Process a single LLM batch. Returns matches or None on failure.

    Provider 路由 + 传输重试 + 完整产物验证链 + 产物重放。
    产物重放在续写、reasoning follow-up、JSON 修复和 recognizer 业务校验全部失败后触发。
    """
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

    # 缓存检查在重试路由之前（缓存结果不需重新验证）
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

    # ── Provider 路由 + 产物重放循环 ──
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

            # 产物重放退避（非首次产物尝试）
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

            # ── 传输请求 ──
            content = None
            try:
                if tools and content_type == "song":
                    # 工具模式：直接执行工具流程（无预检请求）
                    from .search_tools import execute_tool
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
                    # 非工具模式：传输级重试
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
                    # 截断续写
                    content = _continue_truncated_json_array(
                        client, provider, config, messages, content,
                        debug["finish_reason"], batch_debug,
                        max_tokens=max_tokens_val,
                    )
            except Exception as exc:
                last_error = exc
                _, reason = _classify_error(exc)
                # 不可重试异常（401/403/400）→ 立即切 provider，不重放
                if not _classify_error(exc)[0]:
                    print(f"  [warn] provider={provider.name} non-retryable ({reason}), switching", flush=True)
                    break
                # 可重试传输异常 → 如果还有产物重放机会，继续重放
                if result_attempt < total_result_attempts:
                    continue
                # 传输和产物重放都耗尽 → 切 provider
                print(
                    f"  provider={provider.name} "
                    f"result_attempt={result_attempt}/{total_result_attempts} "
                    f"transport exhausted, switching provider",
                    flush=True,
                )
                break

            # ── 完整产物验证链 ──
            # 1. 空响应 → reasoning follow-up
            if not content.strip():
                reasoning_content = ""
                if batch_debug.get("tool_rounds"):
                    for tr in batch_debug["tool_rounds"]:
                        if tr.get("reasoning_content"):
                            reasoning_content = tr["reasoning_content"]
                            break
                content = run_reasoning_followups(client, provider, config, reasoning_content, "", batch_debug)

            # 2. 解析 JSON 数组
            items, is_valid_array = parse_llm_response_with_status(content)
            if not is_valid_array:
                # 尝试从 tool rounds 的 reasoning content 解析
                if batch_debug.get("tool_rounds"):
                    for tr in batch_debug["tool_rounds"]:
                        rc = tr.get("reasoning_content", "")
                        if rc.strip():
                            items, is_valid_array = parse_llm_response_with_status(rc)
                            if is_valid_array:
                                break
                # reasoning follow-up
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

            # 3. JSON 修复
            if not is_valid_array and content.strip():
                items, content = fix_json_with_llm(client, provider, config, content, content_type, batch_debug)
                _, is_valid_array = parse_llm_response_with_status(content)

            # 4. 业务校验（recognizer.parse_response）
            if is_valid_array:
                matches = recognizer.parse_response(items, config)
                # 成功！记录并返回
                batch_debug["parsed_items"] = items
                batch_debug["parse_valid"] = True
                batch_debug["raw_response"] = content
                if debug_path is not None:
                    write_llm_debug(debug_path, batch_start, batch_debug)
                cache_summary = _cache_usage_summary(batch_debug)
                cache_suffix = f", {cache_summary}" if cache_summary else ""
                print(f"  LLM batch {batch_idx}/{batch_count}: done, found {len(matches)} match(es){cache_suffix}")
                return matches

            # 产物验证失败 → 继续产物重放（如果有剩余次数）
            if result_attempt < total_result_attempts:
                print(
                    f"  provider={provider.name} "
                    f"result_attempt={result_attempt}/{total_result_attempts} "
                    f"result invalid: discovery_incomplete_coverage, replaying",
                    flush=True,
                )

        # 当前 provider 产物重放耗尽或不可重试异常 → 切下一个 provider

    # 所有 provider 耗尽
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
    """通用内容识别
    
    Args:
        segments: ASR 转写片段列表
        config: 完整配置字典
        recognizer: 识别器实例
        debug_dir: 调试信息输出目录
    """
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
