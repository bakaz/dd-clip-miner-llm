"""Test stream interruption handling."""
import time
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

from dd_clip_miner_llm.llm import LLMProvider, call_llm_with_transport_retry, StreamInterruptedError


def make_stream_chunks(content_parts, fail_at=None):
    """Create a generator that yields stream chunks, optionally failing.
    
    Args:
        content_parts: List of content strings to yield
        fail_at: If set, raise TimeoutError after yielding this many chunks
    """
    def stream():
        for i, text in enumerate(content_parts):
            yield SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=text, reasoning_content=None),
                    finish_reason=None,
                )],
                usage=None, model="test-model",
            )
            # Fail after yielding the chunk at index fail_at-1
            if fail_at is not None and i >= fail_at - 1:
                raise TimeoutError("read timeout")
        # Final chunk (only reached if no failure)
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


class TestStreamCompletion:
    """Test normal stream completion."""

    def test_normal_stream_completion(self):
        """Test 1: Stream completes normally."""
        provider = LLMProvider(
            name="test", api_key="key", base_url="https://test.com/v1",
            model="test", stream=True, timeout_schedule=[60], result_retries=0,
        )
        client = MagicMock()
        client.chat.completions.create.return_value = make_stream_chunks(["[", '{"ok": true}', "]"])

        with patch("time.sleep"):
            response, content = call_llm_with_transport_retry(
                client, provider, [{"role": "user", "content": "test"}],
            )
        assert content == '[{"ok": true}]', f"Expected content, got: {repr(content)}"
        assert response.choices[0].finish_reason == "stop"


class TestStreamRetry:
    """Test stream interruption and retry."""

    def test_stream_interrupt_retry_succeeds(self):
        """Test 2: Stream interrupted, transport retry succeeds."""
        provider = LLMProvider(
            name="test", api_key="key", base_url="https://test.com/v1",
            model="test", stream=True, timeout_schedule=[60, 120], result_retries=0,
        )
        client = MagicMock()

        # Create separate generators for each call
        first_stream = make_stream_chunks(["partial"], fail_at=1)
        second_stream = make_stream_chunks(["[", '{"ok": true}', "]"])
        
        call_count = [0]
        def make_stream_with_retry(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return first_stream
            else:
                return second_stream

        client.chat.completions.create.side_effect = make_stream_with_retry

        with patch("time.sleep"):
            response, content = call_llm_with_transport_retry(
                client, provider, [{"role": "user", "content": "test"}],
            )
        assert content == '[{"ok": true}]', f"Expected content, got: {repr(content)}"
        assert call_count[0] == 2


class TestStreamPartial:
    """Test stream interruption returning partial content."""

    def test_stream_interrupt_returns_partial(self):
        """Test 3: Stream interrupted, all transport retries exhausted, return partial."""
        provider = LLMProvider(
            name="test", api_key="key", base_url="https://test.com/v1",
            model="test", stream=True, timeout_schedule=[60, 120, 180], result_retries=0,
        )
        client = MagicMock()
        
        # Create separate generators for each call to avoid exhausted generator issue
        def make_failing_stream(*args, **kwargs):
            return make_stream_chunks(['[{"partial": "data"]'], fail_at=1)
        
        client.chat.completions.create.side_effect = make_failing_stream

        with patch("time.sleep"):
            response, content = call_llm_with_transport_retry(
                client, provider, [{"role": "user", "content": "test"}],
            )
        assert '[{"partial":' in content, f"Expected partial content, got: {repr(content)}"
        assert response.choices[0].finish_reason == "length", f"Expected length, got: {response.choices[0].finish_reason}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])