"""传输层模块

LLM 调用、重试、流式处理。
"""
from __future__ import annotations

from typing import Any

from .error import _classify_error
from .provider import LLMProvider
from ..errors import LLMError


class StreamInterruptedError(LLMError):
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


def _call_with_hard_timeout(func: Any, timeout_seconds: float | None) -> Any:
    if timeout_seconds is None or timeout_seconds <= 0:
        return func()

    import queue
    import threading

    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put((True, func()))
        except BaseException as exc:  # pragma: no cover - re-raised in caller
            result_queue.put((False, exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(float(timeout_seconds))
    if thread.is_alive():
        raise TimeoutError(
            f"LLM request exceeded hard timeout of {float(timeout_seconds):g}s"
        )
    ok, value = result_queue.get_nowait()
    if ok:
        return value
    raise value


def _next_with_idle_timeout(iterator: Any, timeout_seconds: float | None) -> tuple[bool, Any]:
    try:
        return True, _call_with_hard_timeout(lambda: next(iterator), timeout_seconds)
    except StopIteration:
        return False, None


def _is_stream_unsupported_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    text = f"{type(exc).__name__} {exc}".casefold()
    if not any(key in text for key in ("stream", "streaming", "stream_options")):
        return False
    return any(
        key in text
        for key in (
            "unsupported",
            "not support",
            "does not support",
            "unknown parameter",
            "unrecognized",
            "invalid parameter",
            "not allowed",
        )
    )


def _extract_response_content(response: Any) -> str:
    content = ""
    choice = response.choices[0] if response.choices else None
    message = choice.message if choice is not None else None
    if message:
        content = message.content or ""
        if not content.strip():
            reasoning = getattr(message, "reasoning_content", None) or ""
            if reasoning.strip():
                content = reasoning
    return content


def _call_non_stream_without_hard_timeout(client: Any, kwargs: dict[str, Any]) -> tuple[Any, str]:
    request_kwargs = dict(kwargs)
    request_kwargs.pop("stream", None)
    request_kwargs.pop("stream_options", None)
    request_kwargs.pop("timeout", None)
    response = _call_llm_raw(client, request_kwargs)
    return response, _extract_response_content(response)


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

    # Stream-first transport for all requests. Tool-call streaming is supported
    # via _collect_stream with accumulated tool_calls. Providers that don't
    # support streaming fall back to non-stream automatically.
    response, _content = call_llm_with_transport_retry(
        client, provider, messages,
        tools=tools, tool_choice=tool_choice,
        max_tokens_override=max_tokens_override,
        timeout=timeout,
        max_attempts=effective_max_retries,
    )
    return response


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
    configured_schedule = provider.timeout_schedule or [provider.timeout] * fallback_attempts
    schedule = configured_schedule
    backoff_list = provider.retry_backoff_seconds or [2, 5]
    jitter = provider.retry_jitter_ratio
    # 始终优先尝试流式。流式支持 tool_calls 累积，若不支持则走 fallback。
    use_stream = True

    if use_stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

    last_exc: Exception | None = None
    last_partial_content = ""
    for attempt, timeout_val in enumerate(schedule):
        kwargs["timeout"] = timeout_val
        mode = "stream" if use_stream else "non-stream"
        print(
            f"  [llm] transport attempt {attempt + 1}/{len(schedule)} "
            f"({mode}, timeout={timeout_val:g}s, provider={provider.name})",
            flush=True,
        )
        try:
            if use_stream:
                response, content = _collect_stream(
                    client,
                    kwargs,
                    idle_timeout=float(timeout_val),
                )
            else:
                response, content = _call_non_stream_without_hard_timeout(client, kwargs)
            if batch_debug is not None:
                from ..llm_debug import _record_usage, llm_response_debug
                _record_usage(batch_debug, "transport", llm_response_debug(response), attempt=attempt + 1)
            return response, content
        except StreamInterruptedError as exc:
            last_exc = exc.original_error
            if len(exc.partial_content) > len(last_partial_content):
                last_partial_content = exc.partial_content

            if use_stream and not exc.partial_content and _is_stream_unsupported_error(exc.original_error):
                print(
                    f"  [llm] stream unsupported; falling back to non-stream "
                    f"(provider={provider.name})",
                    flush=True,
                )
                response, content = _call_non_stream_without_hard_timeout(client, kwargs)
                if batch_debug is not None:
                    from ..llm_debug import _record_usage, llm_response_debug
                    _record_usage(
                        batch_debug,
                        "transport_non_stream_fallback",
                        llm_response_debug(response),
                        attempt=attempt + 1,
                    )
                return response, content

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
            if use_stream and _is_stream_unsupported_error(exc):
                print(
                    f"  [llm] stream unsupported; falling back to non-stream "
                    f"(provider={provider.name})",
                    flush=True,
                )
                response, content = _call_non_stream_without_hard_timeout(client, kwargs)
                if batch_debug is not None:
                    from ..llm_debug import _record_usage, llm_response_debug
                    _record_usage(
                        batch_debug,
                        "transport_non_stream_fallback",
                        llm_response_debug(response),
                        attempt=attempt + 1,
                    )
                return response, content

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

    if last_partial_content:
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


def _collect_stream(
    client: Any,
    kwargs: dict[str, Any],
    *,
    idle_timeout: float | None = None,
) -> tuple[Any, str]:
    """收集流式响应，返回 (response, content)。支持 tool_calls 累积。"""
    import time

    stream = _call_with_hard_timeout(
        lambda: _call_llm_raw(client, kwargs),
        idle_timeout,
    )
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = None
    usage = None
    model = None
    last_meaningful_at = time.monotonic()
    last_progress_log_at = last_meaningful_at
    meaningful_chars = 0
    heartbeat_grace_seconds = (
        float(idle_timeout) * 5.0
        if idle_timeout is not None and idle_timeout > 0
        else None
    )
    # tool_calls 累积：{index: {"id": str, "name": str, "arguments": str}}
    accrued_tool_calls: dict[int, dict[str, str]] = {}

    try:
        try:
            iterator = iter(stream)
        except TypeError as exc:
            raise RuntimeError(
                "streaming unsupported: provider returned a non-stream response"
            ) from exc
        while True:
            has_chunk, chunk = _next_with_idle_timeout(iterator, idle_timeout)
            if not has_chunk:
                break
            now = time.monotonic()
            model = getattr(chunk, "model", None) or model
            if not chunk.choices:
                usage = getattr(chunk, "usage", None) or usage
                if (
                    heartbeat_grace_seconds is not None
                    and now - last_meaningful_at > heartbeat_grace_seconds
                ):
                    raise TimeoutError(
                        f"LLM stream produced only heartbeat chunks for "
                        f"{heartbeat_grace_seconds:g}s"
                    )
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            got_text = False
            if delta:
                if delta.content:
                    content_parts.append(delta.content)
                    meaningful_chars += len(delta.content)
                    got_text = True
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    reasoning_parts.append(rc)
                    meaningful_chars += len(rc)
                    got_text = True
                # 累积流式 tool_calls
                delta_tcs = getattr(delta, "tool_calls", None)
                if delta_tcs:
                    for tc in delta_tcs:
                        idx = getattr(tc, "index", 0) or 0
                        entry = accrued_tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        tc_id = getattr(tc, "id", None)
                        if tc_id:
                            entry["id"] = tc_id
                        tc_func = getattr(tc, "function", None)
                        if tc_func is not None:
                            fname = getattr(tc_func, "name", None)
                            if fname:
                                entry["name"] = fname
                            fargs = getattr(tc_func, "arguments", None)
                            if fargs:
                                entry["arguments"] += fargs
                    # tool_calls 也算有意义的心跳
                    last_meaningful_at = now
            if got_text:
                last_meaningful_at = now
                if now - last_progress_log_at >= 30.0:
                    print(
                        f"  [llm] stream received {meaningful_chars} chars "
                        f"(provider={kwargs.get('model', '')})",
                        flush=True,
                    )
                    last_progress_log_at = now
            fr = getattr(choice, "finish_reason", None)
            if fr:
                finish_reason = fr
            usage = getattr(chunk, "usage", None) or usage
            if (
                heartbeat_grace_seconds is not None
                and now - last_meaningful_at > heartbeat_grace_seconds
            ):
                raise TimeoutError(
                    f"LLM stream produced only heartbeat chunks for "
                    f"{heartbeat_grace_seconds:g}s"
                )
    except Exception as exc:
        partial = "".join(content_parts) or "".join(reasoning_parts)
        raise StreamInterruptedError(exc, partial) from exc

    content = "".join(content_parts)
    if not content.strip() and reasoning_parts:
        content = "".join(reasoning_parts)

    # 组装 tool_calls
    tool_calls = None
    if finish_reason == "tool_calls" and accrued_tool_calls:
        from types import SimpleNamespace as _SN
        assembled = []
        for idx in sorted(accrued_tool_calls):
            entry = accrued_tool_calls[idx]
            assembled.append(_SN(
                id=entry["id"],
                type="function",
                function=_SN(
                    name=entry["name"],
                    arguments=entry["arguments"],
                ),
            ))
        tool_calls = assembled

    response = _build_llm_response(
        content=content, finish_reason=finish_reason or "stop",
        usage=usage, model=model or "", tool_calls=tool_calls,
    )
    return response, content


def _build_llm_response(
    content: str,
    finish_reason: str = "stop",
    usage: Any = None,
    model: str = "",
    tool_calls: Any = None,
) -> Any:
    """构建 LLM 响应对象。"""
    from types import SimpleNamespace

    message = SimpleNamespace(
        content=content,
        reasoning_content="",
        tool_calls=tool_calls,
        model_dump=lambda: {
            "content": content,
            "reasoning_content": "",
            "tool_calls": (
                [
                    {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ]
                if tool_calls
                else None
            ),
        },
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage, model=model)
