"""Provider 管理模块

LLMProvider 数据类、provider 配置解析、OpenAI 客户端缓存。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


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
    timeout_schedule: list[float] | None = None
    retry_backoff_seconds: list[float] | None = None
    retry_jitter_ratio: float = 0.25
    result_retries: int = 2


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

    return LLMProvider(
        name=name, api_key=api_key, base_url=base_url, model=model,
        temperature=temperature, max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
        thinking=str(thinking) if thinking not in (None, "") else None,
        timeout=timeout, max_retries=max_retries, proxy=proxy, stream=stream,
        timeout_schedule=timeout_schedule, retry_backoff_seconds=retry_backoff_seconds,
        retry_jitter_ratio=retry_jitter_ratio, result_retries=result_retries,
    )


def build_providers(config: dict[str, Any]) -> list[LLMProvider]:
    """构建 LLM provider 列表。"""
    llm_config = config["llm"]
    providers_cfg = llm_config.get("providers", {})
    llm_shared: dict[str, Any] = {
        k: v for k, v in llm_config.items()
        if k not in ("providers", "provider_route", "active_provider", "fallbacks")
    }

    provider_route = llm_config.get("provider_route")
    if provider_route:
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

    if not result and providers_cfg:
        for name, cfg in providers_cfg.items():
            if not isinstance(cfg, dict):
                continue
            provider = _resolve_provider_from_config(name, cfg, llm_shared)
            if provider is not None:
                result.append(provider)

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


def _ensure_openai_clients(
    providers: list[LLMProvider],
    clients: dict[tuple[str | None, str, str | None], Any],
) -> None:
    """确保所有 provider 对应的 OpenAI 客户端已创建。"""
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


def _build_openai_clients(providers: list[LLMProvider]) -> dict[tuple[str | None, str, str | None], Any]:
    """Pre-create OpenAI clients keyed by (base_url, api_key)."""
    clients: dict[tuple[str | None, str, str | None], Any] = {}
    _ensure_openai_clients(providers, clients)
    return clients
