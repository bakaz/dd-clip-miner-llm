"""传输层模块

LLM 调用、重试、流式处理。
"""
from __future__ import annotations

import time as _time
from typing import Any

from .error import _classify_error
from .provider import LLMProvider


class StreamInterruptedError(Exception):
    """流式响应中断异常。"""

    def __init__(self, original_error: Exception, partial_content: str):
        super().__init__(str(original_error))
        self.original_error = original_error
        self.partial_content = partial_content


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
    """调用 LLM，返回完整的 response 对象。"""
    effective_max_retries = (
        provider.max_retries if max_retries is None else max(1, int(max_retries))
    )

    # 流式响应必须由传输级重试负责收集
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
    """传输级重试：timeout_schedule 逐步升级超时 + 退避抖动。"""
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
                from ..llm_debug import _record_usage, llm_response_debug
                _record_usage(batch_debug, "transport", llm_response_debug(response), attempt=attempt + 1)
            return response, content
        except StreamInterruptedError as exc:
            last_exc = exc.original_error
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
                print(f"  [llm] transport failed ({reason}); retrying in {wait_time:.1f}s", flush=True)
                if batch_debug is not None:
                    batch_debug.setdefault("transport_retries", []).append({
                        "attempt": attempt + 1, "timeout": timeout_val,
                        "reason": reason, "wait_seconds": round(wait_time, 1),
                    })
                time.sleep(wait_time)
                continue
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
                print(f"  [llm] transport failed ({reason}); retrying in {wait_time:.1f}s", flush=True)
                if batch_debug is not None:
                    batch_debug.setdefault("transport_retries", []).append({
                        "attempt": attempt + 1, "timeout": timeout_val,
                        "reason": reason, "wait_seconds": round(wait_time, 1),
                    })
                time.sleep(wait_time)
                continue

    # 传输重试耗尽
    if use_stream and last_partial_content:
        print(
            f"  [llm] stream interrupted, returning partial content "
            f"({len(last_partial_content)} chars) for continuation",
            flush=True,
        )
        response = _build_llm_response(
            content=last_partial_content, finish_reason="length",
            usage=None, model=provider.model,
        )
        return response, last_partial_content

    if last_exc is None:
        raise RuntimeError("call_llm_with_transport_retry: no attempt was made")
    raise last_exc


def _collect_stream(client: Any, kwargs: dict[str, Any]) -> tuple[Any, str]:
    """收集流式响应，返回 (response, content)。"""
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
        partial = "".join(content_parts) or "".join(reasoning_parts)
        raise StreamInterruptedError(exc, partial) from exc

    content = "".join(content_parts)
    if not content.strip() and reasoning_parts:
        content = "".join(reasoning_parts)

    response = _build_llm_response(
        content=content, finish_reason=finish_reason or "stop",
        usage=usage, model=model or "",
    )
    return response, content


def _build_llm_response(
    content: str,
    finish_reason: str = "stop",
    usage: Any = None,
    model: str = "",
) -> Any:
    """构建 LLM 响应对象。"""
    from types import SimpleNamespace

    message = SimpleNamespace(
        content=content, reasoning_content="", tool_calls=None,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage, model=model)
