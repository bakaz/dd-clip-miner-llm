"""Test stream fallback and interruption handling.

New behavior (transport.py):
- Non-stream is the default path
- On timeout, falls back to stream with idle_timeout
- Stream interrupted → partial content → finish_reason="length"
"""
import time
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

from dd_clip_miner_llm.llm import LLMProvider, call_llm_with_transport_retry, StreamInterruptedError


def _mock_response(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    """Create a mock non-stream response."""
    message = SimpleNamespace(
        content=content, reasoning_content="",
        tool_calls=None, model_dump=lambda: {"content": content},
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=20, total_tokens=30,
        model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )
    return SimpleNamespace(choices=[choice], usage=usage, model="test-model")


def make_stream_chunks(content_parts, fail_at=None):
    """Create a generator that yields stream chunks, optionally failing."""
    def stream():
        for i, text in enumerate(content_parts):
            yield SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=text, reasoning_content=None),
                    finish_reason=None,
                )],
                usage=None, model="test-model",
            )
            if fail_at is not None and i >= fail_at - 1:
                raise TimeoutError("read timeout")
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None, reasoning_content=None),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=20, total_tokens=30,
                model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            ),
            model="test-model",
        )
    return stream()


class TestNonStreamSuccess:
    """Non-stream path succeeds directly."""

    def test_non_stream_returns_response(self):
        provider = LLMProvider(
            name="test", api_key="key", base_url="https://test.com/v1",
            model="test", stream=True, timeout_schedule=[60], result_retries=0,
        )
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response('[{"ok": true}]')

        response, content = call_llm_with_transport_retry(
            client, provider, [{"role": "user", "content": "test"}],
        )
        assert content == '[{"ok": true}]'
        assert response.choices[0].finish_reason == "stop"
        # Should NOT set stream=True on the request
        call_kwargs = client.chat.completions.create.call_args[1]
        assert "stream" not in call_kwargs


class TestStreamFallback:
    """Non-stream times out → falls back to stream."""

    def test_timeout_triggers_stream_fallback(self):
        provider = LLMProvider(
            name="test", api_key="key", base_url="https://test.com/v1",
            model="test", stream=True, timeout_schedule=[1], result_retries=0,
        )
        client = MagicMock()

        call_count = [0]
        def mock_create(*args, **kwargs):
            call_count[0] += 1
            if kwargs.get("stream"):
                # Stream path succeeds
                return make_stream_chunks(['[{"ok": true}]'])
            else:
                # Non-stream times out
                raise TimeoutError("request timed out")

        client.chat.completions.create.side_effect = mock_create

        with patch("time.sleep"):
            response, content = call_llm_with_transport_retry(
                client, provider, [{"role": "user", "content": "test"}],
            )
        assert content == '[{"ok": true}]'
        assert call_count[0] == 2  # non-stream + stream fallback


class TestStreamInterruptPartial:
    """Stream fallback interrupted → returns partial content."""

    def test_stream_interrupt_returns_partial(self):
        provider = LLMProvider(
            name="test", api_key="key", base_url="https://test.com/v1",
            model="test", stream=True, timeout_schedule=[1], result_retries=0,
        )
        client = MagicMock()

        def mock_create(*args, **kwargs):
            if kwargs.get("stream"):
                return make_stream_chunks(['[{"partial": "data"]'], fail_at=1)
            else:
                raise TimeoutError("request timed out")

        client.chat.completions.create.side_effect = mock_create

        with patch("time.sleep"):
            response, content = call_llm_with_transport_retry(
                client, provider, [{"role": "user", "content": "test"}],
            )
        assert '[{"partial":' in content
        assert response.choices[0].finish_reason == "length"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
