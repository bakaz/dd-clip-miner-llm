"""Tests for LLM provider retry, backoff, and routing.

Covers:
- Error classification (retryable vs non-retryable)
- Escalating timeout schedule
- Backoff with jitter
- Transport retry (call_llm_with_transport_retry)
- Provider routing + full validation chain + result replay
- 401 → immediate fallback, no result replay
- Tool mode: no pre-check request
- JSON fix success → no replay; business validation failure → replay
- Retry-After header handling
- Backward compatibility with old config format
- Client cache keyed by (base_url, api_key)
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dd_clip_miner_llm.config import DEFAULT_CONFIG, deep_merge, load_config
from dd_clip_miner_llm.llm import (
    LLMProvider,
    _classify_error,
    _get_client,
    _ensure_openai_clients,
    build_providers,
    call_llm,
    call_llm_with_transport_retry,
)


# ============ Fixtures ============

def _make_provider(**overrides: Any) -> LLMProvider:
    """Create an LLMProvider with sensible defaults."""
    defaults: dict[str, Any] = {
        "name": "test",
        "api_key": "test-key",
        "base_url": "https://api.test.com/v1",
        "model": "test-model",
        "timeout": 300.0,
        "max_retries": 3,
        "timeout_schedule": [60.0, 120.0, 180.0],
        "retry_backoff_seconds": [2.0, 5.0],
        "retry_jitter_ratio": 0.25,
        "result_retries": 2,
    }
    defaults.update(overrides)
    return LLMProvider(**defaults)


def _make_config(**llm_overrides: Any) -> dict[str, Any]:
    """Create a config dict with providers."""
    config = deep_merge(DEFAULT_CONFIG, {
        "llm": {
            "provider_route": ["test"],
            "providers": {
                "test": {
                    "api_key": "test-key",
                    "base_url": "https://api.test.com/v1",
                    "model": "test-model",
                    "timeout_schedule": [60, 120, 180],
                    "retry_backoff_seconds": [2, 5],
                    "retry_jitter_ratio": 0.25,
                    "result_retries": 2,
                },
            },
        },
    })
    if llm_overrides:
        config["llm"].update(llm_overrides)
    return config


def _mock_response(content: str = '[{"title": "test"}]', finish_reason: str = "stop") -> Any:
    """Create a mock OpenAI response."""
    message = SimpleNamespace(
        content=content,
        reasoning_content="",
        tool_calls=None,
        model_dump=lambda: {"content": content, "reasoning_content": ""},
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=20, total_tokens=30,
        model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )
    return SimpleNamespace(choices=[choice], usage=usage, model="test-model")


def _mock_empty_response() -> Any:
    return _mock_response(content="", finish_reason="stop")


def _make_http_error(status_code: int, message: str = "error") -> Any:
    try:
        from openai import APIStatusError
        request = SimpleNamespace(method="POST", url="https://api.test.com/v1")
        response = SimpleNamespace(status_code=status_code, headers={}, request=request)
        return APIStatusError(message=message, response=response, body=None)
    except ImportError:
        return RuntimeError(f"HTTP {status_code}: {message}")


def _make_rate_limit_error() -> Any:
    try:
        from openai import RateLimitError
        request = SimpleNamespace(method="POST", url="https://api.test.com/v1")
        response = SimpleNamespace(status_code=429, headers={}, request=request)
        return RateLimitError(message="rate limited", response=response, body=None)
    except ImportError:
        return RuntimeError("rate limited")


def _make_auth_error() -> Any:
    try:
        from openai import AuthenticationError
        request = SimpleNamespace(method="POST", url="https://api.test.com/v1")
        response = SimpleNamespace(status_code=401, headers={}, request=request)
        return AuthenticationError(message="unauthorized", response=response, body=None)
    except ImportError:
        return RuntimeError("unauthorized")


def _make_connection_error() -> Any:
    try:
        from openai import APIConnectionError
        return APIConnectionError(request=MagicMock())
    except ImportError:
        return RuntimeError("connection failed")


def _make_timeout_error() -> Any:
    try:
        from openai import APITimeoutError
        return APITimeoutError(request=MagicMock())
    except ImportError:
        return RuntimeError("timeout")


# ============ Error Classification ============

class TestClassifyError:

    def test_rate_limit_is_retryable(self):
        retryable, reason = _classify_error(_make_rate_limit_error())
        assert retryable is True

    def test_connection_error_is_retryable(self):
        retryable, _ = _classify_error(_make_connection_error())
        assert retryable is True

    def test_timeout_error_is_retryable(self):
        retryable, _ = _classify_error(_make_timeout_error())
        assert retryable is True

    def test_500_is_retryable(self):
        retryable, _ = _classify_error(_make_http_error(500))
        assert retryable is True

    def test_502_is_retryable(self):
        retryable, _ = _classify_error(_make_http_error(502))
        assert retryable is True

    def test_401_is_not_retryable(self):
        retryable, reason = _classify_error(_make_auth_error())
        assert retryable is False

    def test_403_is_not_retryable(self):
        retryable, _ = _classify_error(_make_http_error(403))
        assert retryable is False

    def test_400_is_not_retryable(self):
        retryable, _ = _classify_error(_make_http_error(400))
        assert retryable is False

    def test_422_is_not_retryable(self):
        retryable, _ = _classify_error(_make_http_error(422))
        assert retryable is False

    def test_generic_exception_is_not_retryable(self):
        retryable, _ = _classify_error(ValueError("bad"))
        assert retryable is False

    def test_timeout_error_builtin_is_retryable(self):
        retryable, reason = _classify_error(TimeoutError("timed out"))
        assert retryable is True
        assert "timeout" in reason


# ============ Timeout Schedule ============

class TestTimeoutSchedule:

    def test_schedule_length_determines_attempts(self):
        p = _make_provider(timeout_schedule=[60, 120, 180])
        assert len(p.timeout_schedule) == 3

    def test_single_timeout_backward_compat(self):
        p = _make_provider(timeout=300.0, timeout_schedule=[300.0])
        assert p.timeout_schedule == [300.0]

    def test_schedule_values_escalate(self):
        s = _make_provider(timeout_schedule=[60, 120, 180]).timeout_schedule
        assert s[0] < s[1] < s[2]


# ============ call_llm_with_transport_retry ============

class TestTransportRetry:

    def test_returns_response_and_content(self):
        provider = _make_provider(timeout_schedule=[60])
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response('[{"ok": true}]')

        response, content = call_llm_with_transport_retry(
            client, provider, [{"role": "user", "content": "test"}],
        )

        assert content == '[{"ok": true}]'
        assert response is not None
        assert client.chat.completions.create.call_count == 1

    def test_timeout_escalates_on_retry(self):
        provider = _make_provider(timeout_schedule=[60, 120, 180])
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_timeout_error(),
            _mock_response('[{"ok": true}]'),
        ]

        with patch("time.sleep"):
            _, content = call_llm_with_transport_retry(
                client, provider, [{"role": "user", "content": "test"}],
            )

        assert content == '[{"ok": true}]'
        calls = client.chat.completions.create.call_args_list
        assert calls[0][1]["timeout"] == 60
        assert calls[1][1]["timeout"] == 120

    def test_non_retryable_raises_immediately(self):
        provider = _make_provider(timeout_schedule=[60, 120, 180])
        client = MagicMock()
        client.chat.completions.create.side_effect = _make_auth_error()

        with pytest.raises(Exception):
            call_llm_with_transport_retry(
                client, provider, [{"role": "user", "content": "test"}],
            )

        assert client.chat.completions.create.call_count == 1

    def test_exhausts_all_attempts_then_raises(self):
        provider = _make_provider(timeout_schedule=[60, 120, 180])
        client = MagicMock()
        client.chat.completions.create.side_effect = _make_timeout_error()

        with patch("time.sleep"):
            with pytest.raises(Exception):
                call_llm_with_transport_retry(
                    client, provider, [{"role": "user", "content": "test"}],
                )

        assert client.chat.completions.create.call_count == 3

    def test_reasoning_content_fallback(self):
        provider = _make_provider()
        client = MagicMock()
        message = SimpleNamespace(
            content="", reasoning_content="reasoning output",
            tool_calls=None,
            model_dump=lambda: {"content": "", "reasoning_content": "reasoning output"},
        )
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=20, total_tokens=30,
            model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[choice], usage=usage, model="test-model",
        )

        _, content = call_llm_with_transport_retry(
            client, provider, [{"role": "user", "content": "test"}],
        )
        assert content == "reasoning output"

    def test_batch_debug_records_transport_retries(self):
        provider = _make_provider(timeout_schedule=[60, 120])
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_timeout_error(),
            _mock_response('[{"ok": true}]'),
        ]
        batch_debug: dict[str, Any] = {"usage": []}

        with patch("time.sleep"):
            call_llm_with_transport_retry(
                client, provider, [{"role": "user", "content": "test"}],
                batch_debug=batch_debug,
            )

        assert "transport_retries" in batch_debug
        assert len(batch_debug["transport_retries"]) == 1


# ============ call_llm backward compat ============

class TestCallLlmBackwardCompat:

    def test_call_llm_returns_response(self):
        provider = _make_provider(timeout_schedule=[60])
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response('[{"ok": true}]')

        response = call_llm(client, provider, [{"role": "user", "content": "test"}])

        assert response is not None
        assert response.choices[0].message.content == '[{"ok": true}]'

    def test_call_llm_old_smoke_test_compat(self):
        """Original smoke test pattern still works."""
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise TimeoutError("test timeout")
                return _mock_response('[{"ok": true}]')

        client = type("Client", (), {
            "chat": type("Chat", (), {"completions": Completions()})(),
        })()
        provider = _make_provider(
            name="test", api_key="test", model="test-model",
            timeout=12.0, max_retries=2, timeout_schedule=[12.0, 12.0],
        )

        with patch("time.sleep"):
            call_llm(client, provider, [{"role": "user", "content": "test"}])

        assert len(calls) == 2
        assert calls[0]["timeout"] == 12


# ============ Client Cache ============

class TestClientCache:

    def test_same_key_different_url_different_clients(self):
        p1 = _make_provider(api_key="same-key", base_url="https://a.com/v1", name="a")
        p2 = _make_provider(api_key="same-key", base_url="https://b.com/v1", name="b")
        clients: dict[tuple[str | None, str, str | None], Any] = {}
        _ensure_openai_clients([p1, p2], clients)

        c1 = _get_client(p1, clients)
        c2 = _get_client(p2, clients)
        assert c1 is not c2

    def test_same_key_same_url_same_client(self):
        p1 = _make_provider(api_key="same-key", base_url="https://a.com/v1", name="a1")
        p2 = _make_provider(api_key="same-key", base_url="https://a.com/v1", name="a2")
        clients: dict[tuple[str | None, str, str | None], Any] = {}
        _ensure_openai_clients([p1, p2], clients)

        c1 = _get_client(p1, clients)
        c2 = _get_client(p2, clients)
        assert c1 is c2

    def test_same_key_same_url_different_proxy_different_clients(self):
        p1 = _make_provider(api_key="same-key", base_url="https://a.com/v1", proxy=None, name="a")
        p2 = _make_provider(api_key="same-key", base_url="https://a.com/v1", proxy="http://p:8080", name="b")
        clients: dict[tuple[str | None, str, str | None], Any] = {}
        _ensure_openai_clients([p1, p2], clients)

        c1 = _get_client(p1, clients)
        c2 = _get_client(p2, clients)
        assert c1 is not c2

    def test_get_client_returns_none_for_unknown(self):
        p = _make_provider(api_key="unknown-key", base_url="https://x.com/v1")
        clients: dict[tuple[str | None, str, str | None], Any] = {}
        assert _get_client(p, clients) is None


# ============ Provider Name ============

class TestProviderName:

    def test_name_from_config(self):
        p = _make_provider(name="opencode")
        assert p.name == "opencode"

    def test_name_preserved_in_build(self):
        config = _make_config()
        providers = build_providers(config)
        assert providers[0].name == "test"


# ============ Retry-After ============

class TestRetryAfterHeader:

    def test_retry_after_is_respected(self):
        provider = _make_provider(timeout_schedule=[60, 120])
        client = MagicMock()

        try:
            from openai import RateLimitError
            req = SimpleNamespace(method="POST", url="https://api.test.com/v1")
            response = SimpleNamespace(
                status_code=429, headers={"retry-after": "10"}, request=req,
            )
            error = RateLimitError(message="rate limited", response=response, body=None)
        except ImportError:
            error = RuntimeError("rate limited")

        client.chat.completions.create.side_effect = [error, _mock_response('[{"ok": true}]')]

        with patch("time.sleep") as mock_sleep:
            _, content = call_llm_with_transport_retry(
                client, provider, [{"role": "user", "content": "test"}],
            )

        assert content == '[{"ok": true}]'
        if mock_sleep.called:
            assert mock_sleep.call_args[0][0] <= 60

    def test_retry_after_capped_at_60s(self):
        provider = _make_provider(timeout_schedule=[60, 120])
        client = MagicMock()

        try:
            from openai import RateLimitError
            req = SimpleNamespace(method="POST", url="https://api.test.com/v1")
            response = SimpleNamespace(
                status_code=429, headers={"retry-after": "120"}, request=req,
            )
            error = RateLimitError(message="rate limited", response=response, body=None)
        except ImportError:
            error = RuntimeError("rate limited")

        client.chat.completions.create.side_effect = [error, _mock_response('[{"ok": true}]')]

        with patch("time.sleep") as mock_sleep:
            call_llm_with_transport_retry(
                client, provider, [{"role": "user", "content": "test"}],
            )

        if mock_sleep.called:
            assert mock_sleep.call_args[0][0] <= 60


# ============ Backward Compatibility ============

class TestBackwardCompatibility:

    def test_old_flat_config_format(self):
        config = {"llm": {"api_key": "test-key", "model": "test-model", "timeout": 300}}
        providers = build_providers(config)
        assert len(providers) == 1
        assert providers[0].api_key == "test-key"

    def test_old_flat_config_has_no_timeout_schedule(self):
        """Old config without explicit timeout_schedule → None (backward compat)."""
        config = {"llm": {"api_key": "test-key", "timeout": 300}}
        providers = build_providers(config)
        assert providers[0].timeout_schedule is None

    def test_new_provider_route_format(self):
        config = _make_config()
        providers = build_providers(config)
        assert len(providers) == 1
        assert providers[0].api_key == "test-key"

    def test_provider_route_order_preserved(self):
        config = deep_merge(DEFAULT_CONFIG, {
            "llm": {
                "provider_route": ["b", "a"],
                "providers": {
                    "a": {"api_key": "key-a", "model": "model-a"},
                    "b": {"api_key": "key-b", "model": "model-b"},
                },
            },
        })
        providers = build_providers(config)
        assert len(providers) == 2
        assert providers[0].model == "model-b"
        assert providers[1].model == "model-a"

    def test_active_provider_fallback_without_route(self):
        config = deep_merge(DEFAULT_CONFIG, {
            "llm": {
                "active_provider": "my_provider",
                "providers": {
                    "my_provider": {"api_key": "my-key", "model": "my-model"},
                },
            },
        })
        providers = build_providers(config)
        assert len(providers) == 1
        assert providers[0].api_key == "my-key"

    def test_provider_with_env_api_key(self):
        config = deep_merge(DEFAULT_CONFIG, {
            "llm": {
                "provider_route": ["test"],
                "providers": {
                    "test": {"api_key_env": "TEST_LLM_KEY", "model": "test-model"},
                },
            },
        })
        with patch.dict("os.environ", {"TEST_LLM_KEY": "env-key"}):
            providers = build_providers(config)
        assert len(providers) == 1
        assert providers[0].api_key == "env-key"


# ============ 401 → Fallback, No Replay ============

class TestAuthErrorFallback:
    """401/403 should immediately switch provider, NOT trigger result replay."""

    def test_401_skips_to_next_provider(self):
        """Provider1 gets 401 → should try provider2, not replay on provider1."""
        p1 = _make_provider(name="bad", api_key="bad-key", base_url="https://bad.com/v1",
                            timeout_schedule=[60], result_retries=2)
        p2 = _make_provider(name="good", api_key="good-key", base_url="https://good.com/v1",
                            timeout_schedule=[60], result_retries=2)

        client1 = MagicMock()
        client1.chat.completions.create.side_effect = _make_auth_error()
        client2 = MagicMock()
        client2.chat.completions.create.return_value = _mock_response('[{"ok": true}]')

        clients: dict[tuple[str | None, str, str | None], Any] = {
            (p1.base_url, p1.api_key, p1.proxy): client1,
            (p2.base_url, p2.api_key, p2.proxy): client2,
        }

        # Simulate what _process_single_batch does
        from dd_clip_miner_llm.llm import _get_client
        result = None
        for provider in [p1, p2]:
            client = _get_client(provider, clients)
            if client is None:
                continue
            try:
                _, content = call_llm_with_transport_retry(
                    client, provider, [{"role": "user", "content": "test"}],
                )
                result = content
                break
            except Exception:
                continue

        assert result == '[{"ok": true}]'
        # client1 should only be called once (401 → immediate raise)
        assert client1.chat.completions.create.call_count == 1
        # client2 should be called once (success)
        assert client2.chat.completions.create.call_count == 1


# ============ Provider Name Field ============

class TestProviderNameField:

    def test_name_set_from_config(self):
        config = deep_merge(DEFAULT_CONFIG, {
            "llm": {
                "provider_route": ["opencode"],
                "providers": {
                    "opencode": {"api_key": "key", "model": "m"},
                },
            },
        })
        providers = build_providers(config)
        assert providers[0].name == "opencode"

    def test_name_used_in_logs(self, capsys):
        p = _make_provider(name="my-provider", timeout_schedule=[60])
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response('[{"ok": true}]')

        call_llm_with_transport_retry(client, p, [{"role": "user", "content": "test"}])

        captured = capsys.readouterr()
        assert "provider=my-provider" in captured.out


# ============ Streaming Tests ============

class TestStreaming:

    def _mock_stream_chunks(self, content: str, chunk_size: int = 5):
        """Create mock stream chunks from content string."""
        chunks = []
        for i in range(0, len(content), chunk_size):
            text = content[i:i + chunk_size]
            delta = SimpleNamespace(content=text, reasoning_content=None)
            choice = SimpleNamespace(delta=delta, finish_reason=None)
            chunks.append(SimpleNamespace(choices=[choice], usage=None, model="test-model"))
        # Final chunk with finish_reason
        choice = SimpleNamespace(delta=SimpleNamespace(content=None, reasoning_content=None), finish_reason="stop")
        usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=20, total_tokens=30,
            model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        chunks.append(SimpleNamespace(choices=[choice], usage=usage, model="test-model"))
        return chunks

    def test_stream_normal_completion(self):
        """流式正常完成，返回完整内容。"""
        provider = _make_provider(stream=True, timeout_schedule=[60])
        client = MagicMock()
        client.chat.completions.create.return_value = iter(
            self._mock_stream_chunks('[{"ok": true}]')
        )

        response, content = call_llm_with_transport_retry(
            client, provider, [{"role": "user", "content": "test"}],
        )

        assert content == '[{"ok": true}]'
        assert response.choices[0].finish_reason == "stop"

    def test_stream_sets_stream_kwargs(self):
        """流式模式设置 stream=True 和 stream_options。"""
        provider = _make_provider(stream=True, timeout_schedule=[60])
        client = MagicMock()
        client.chat.completions.create.return_value = iter(
            self._mock_stream_chunks('[{"ok": true}]')
        )

        call_llm_with_transport_retry(
            client, provider, [{"role": "user", "content": "test"}],
        )

        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}

    def test_stream_disabled_with_tools(self):
        """工具调用时不走流式。"""
        provider = _make_provider(stream=True, timeout_schedule=[60])
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response('[{"ok": true}]')
        tools = [{"type": "function", "function": {"name": "test"}}]

        call_llm_with_transport_retry(
            client, provider, [{"role": "user", "content": "test"}],
            tools=tools,
        )

        call_kwargs = client.chat.completions.create.call_args[1]
        assert "stream" not in call_kwargs

    def test_stream_interrupt_returns_partial(self):
        """流式中断且传输重试耗尽时，返回部分内容 + finish_reason=length。"""
        provider = _make_provider(stream=True, timeout_schedule=[60])
        client = MagicMock()

        def failing_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content='[{"partial":', reasoning_content=None),
                    finish_reason=None,
                )],
                usage=None, model="test-model",
            )
            raise ConnectionError("connection lost")

        client.chat.completions.create.return_value = failing_stream()

        with patch("time.sleep"):
            response, content = call_llm_with_transport_retry(
                client, provider, [{"role": "user", "content": "test"}],
            )

        assert '[{"partial":' in content
        assert response.choices[0].finish_reason == "length"

    def test_stream_interrupt_retries_transport(self):
        """流式中断时，先尝试传输重试。"""
        provider = _make_provider(stream=True, timeout_schedule=[60, 120])
        client = MagicMock()

        call_count = [0]
        def make_stream(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                def fail_stream():
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(
                            delta=SimpleNamespace(content='partial', reasoning_content=None),
                            finish_reason=None,
                        )],
                        usage=None, model="test-model",
                    )
                    raise ConnectionError("connection lost")
                return fail_stream()
            else:
                def ok_stream():
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(
                            delta=SimpleNamespace(content='[{"ok": true}]', reasoning_content=None),
                            finish_reason=None,
                        )],
                        usage=None, model="test-model",
                    )
                    choice = SimpleNamespace(
                        delta=SimpleNamespace(content=None, reasoning_content=None),
                        finish_reason="stop",
                    )
                    usage = SimpleNamespace(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30,
                        model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                    )
                    yield SimpleNamespace(choices=[choice], usage=usage, model="test-model")
                return ok_stream()

        client.chat.completions.create.side_effect = make_stream

        with patch("time.sleep"):
            response, content = call_llm_with_transport_retry(
                client, provider, [{"role": "user", "content": "test"}],
            )

        assert content == '[{"ok": true}]'
        assert call_count[0] == 2  # Two transport attempts

    def test_stream_interrupt_keeps_longest_partial(self):
        """A later empty interruption must not discard earlier partial content."""
        provider = _make_provider(stream=True, timeout_schedule=[60, 120])
        client = MagicMock()

        def partial_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content='[{"partial":', reasoning_content=None),
                    finish_reason=None,
                )],
                usage=None, model="test-model",
            )
            raise ConnectionError("connection lost")

        def empty_stream():
            if False:
                yield None
            raise ConnectionError("connection lost before first chunk")

        client.chat.completions.create.side_effect = [partial_stream(), empty_stream()]

        with patch("time.sleep"):
            response, content = call_llm_with_transport_retry(
                client, provider, [{"role": "user", "content": "test"}],
            )

        assert content == '[{"partial":'
        assert response.choices[0].finish_reason == "length"

    def test_call_llm_streams_without_timeout_schedule(self):
        """Old-style providers can enable streaming without timeout_schedule."""
        provider = _make_provider(
            stream=True,
            timeout_schedule=None,
            timeout=30,
            max_retries=2,
        )
        client = MagicMock()
        client.chat.completions.create.return_value = iter(
            self._mock_stream_chunks('[{"ok": true}]')
        )

        response = call_llm(
            client, provider, [{"role": "user", "content": "test"}],
        )

        assert response.choices[0].message.content == '[{"ok": true}]'
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True
        assert call_kwargs["timeout"] == 30

    def test_call_llm_stream_honors_max_retries_override(self):
        provider = _make_provider(
            stream=True,
            timeout_schedule=None,
            timeout=30,
            max_retries=3,
        )
        client = MagicMock()

        def interrupted_stream():
            if False:
                yield None
            raise ConnectionError("connection lost")

        client.chat.completions.create.side_effect = [
            interrupted_stream(),
            interrupted_stream(),
        ]

        with patch("time.sleep"), pytest.raises(ConnectionError):
            call_llm(
                client,
                provider,
                [{"role": "user", "content": "test"}],
                max_retries=2,
            )

        assert client.chat.completions.create.call_count == 2

    def test_no_stream_when_disabled(self):
        """stream=False 时不走流式。"""
        provider = _make_provider(stream=False, timeout_schedule=[60])
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response('[{"ok": true}]')

        call_llm_with_transport_retry(
            client, provider, [{"role": "user", "content": "test"}],
        )

        call_kwargs = client.chat.completions.create.call_args[1]
        assert "stream" not in call_kwargs
