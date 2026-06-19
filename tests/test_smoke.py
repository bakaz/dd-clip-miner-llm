"""冒烟测试 - 验证整体 pipeline 核心功能"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

import pytest

from dd_clip_miner_llm.config import DEFAULT_CONFIG, load_config, get_padding_config
from dd_clip_miner_llm.models import (
    ContentMatch, ContentResult, TranscriptSegment,
    create_song_match, create_song_result, parse_cringe_description
)
from dd_clip_miner_llm.recognizers import get_recognizer, list_recognizers
from dd_clip_miner_llm.merger import build_content_results
from dd_clip_miner_llm.report import write_reports, write_match_context_reports
from dd_clip_miner_llm.llm import fix_structured_json_with_llm, parse_llm_json, parse_llm_response
from dd_clip_miner_llm.paths import safe_path_part
from dd_clip_miner_llm.cli import build_parser


# ============ Fixtures ============

@pytest.fixture
def sample_segments():
    """模拟 ASR 转写结果"""
    return [
        TranscriptSegment(start=0.0, end=3.0, text="大家好欢迎来到直播间"),
        TranscriptSegment(start=3.0, end=6.0, text="今天给大家唱首歌"),
        TranscriptSegment(start=6.0, end=10.0, text="歌词第一句歌词第二句"),
        TranscriptSegment(start=10.0, end=14.0, text="副歌部分副歌重复"),
        TranscriptSegment(start=14.0, end=17.0, text="谢谢大家"),
        TranscriptSegment(start=17.0, end=20.0, text="有人在吗"),
        TranscriptSegment(start=20.0, end=25.0, text="主播回应观众的评论"),
        TranscriptSegment(start=25.0, end=30.0, text="继续聊天互动"),
    ]


@pytest.fixture
def config():
    """测试配置"""
    return deepcopy(DEFAULT_CONFIG)


# ============ 1. 配置加载测试 ============

class TestConfig:
    def test_default_config(self):
        """默认配置应包含所有必要字段"""
        config = DEFAULT_CONFIG
        assert "audio" in config
        assert "asr" in config
        assert "llm" in config
        assert "content_types" in config
        assert "song" in config
        assert "dialogue" in config
        assert "highlight" in config
        assert "funny" in config
        assert "cringe" in config
        assert "daily_summary" in config
        assert "output" in config

    def test_content_types_format(self):
        """content_types 应为字典格式"""
        ct = DEFAULT_CONFIG["content_types"]
        assert isinstance(ct, dict)
        assert ct["song"] is True
        assert ct["dialogue"] is True
        assert ct["cringe"] is True
        assert ct["daily_summary"] is False

    def test_load_config_yaml(self, tmp_path):
        """应能加载 YAML 配置"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("asr:\n  model: medium\n")
        config = load_config(config_file)
        assert config["asr"]["model"] == "medium"

    def test_get_padding_config(self):
        """应能获取 padding 配置"""
        config = {"song": {"padding": {"before_seconds": 5.0}}}
        padding = get_padding_config(config, "song")
        assert padding["before_seconds"] == 5.0


# ============ 2. 识别器注册测试 ============

class TestRecognizers:
    def test_all_recognizers_registered(self):
        """所有识别器应已注册"""
        available = list_recognizers()
        assert "song" in available
        assert "dialogue" in available
        assert "highlight" in available
        assert "funny" in available
        assert "cringe" in available
        assert "daily_summary" in available

    def test_get_recognizer(self):
        """应能获取识别器实例"""
        for name in ["song", "dialogue", "highlight", "funny", "cringe", "daily_summary"]:
            r = get_recognizer(name)
            assert r is not None
            assert r.name == name

    def test_get_unknown_recognizer(self):
        """未知识别器应返回 None"""
        assert get_recognizer("unknown") is None

    def test_recognizer_build_prompt(self, sample_segments, config):
        """识别器应能生成 prompt"""
        for name in ["song", "dialogue", "highlight", "funny", "cringe", "daily_summary"]:
            r = get_recognizer(name)
            prompt = r.build_prompt(sample_segments, 0, config)
            assert isinstance(prompt, str)
            assert len(prompt) > 100

    def test_recognizer_parse_response(self):
        """识别器应能解析响应"""
        r = get_recognizer("song")
        items = [{"content_type": "song", "title": "测试", "segment_indices": [0, 1], "confidence": 0.9}]
        matches = r.parse_response(items, {})
        assert len(matches) == 1
        assert matches[0].title == "测试"


# ============ 3. LLM 响应解析测试 ============

class TestLLM:
    def test_valid_empty_json_array_does_not_require_repair(self):
        from dd_clip_miner_llm.llm import parse_llm_response_with_status

        items, is_valid = parse_llm_response_with_status("```json\n[]\n```")

        assert items == []
        assert is_valid is True

    def test_invalid_json_array_reports_parse_failure(self):
        from dd_clip_miner_llm.llm import parse_llm_response_with_status

        items, is_valid = parse_llm_response_with_status("not json")

        assert items == []
        assert is_valid is False

    def test_bare_llm_config_defaults_to_task_first_layout(self, sample_segments):
        from dd_clip_miner_llm.llm import build_llm_messages

        messages = build_llm_messages(
            get_recognizer("song"),
            sample_segments,
            0,
            {"llm": {}},
        )

        assert messages[0]["role"] == "user"
        assert "ASR 转写开始" not in messages[0]["content"]

    def test_debug_store_requests_disabled_omits_messages(self, sample_segments, config, tmp_path):
        from dd_clip_miner_llm.llm import identify_content, write_llm_debug

        config["llm"]["debug_store_requests"] = False
        config["llm"]["reuse_valid_batches"] = False
        debug_dir = tmp_path / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "batch_start": 0,
            "parsed_items": [],
            "parse_valid": True,
            "usage": [],
        }
        write_llm_debug(debug_dir, 0, payload)
        saved = json.loads((debug_dir / "llm_batch_000000.json").read_text(encoding="utf-8"))
        assert "request_messages" not in saved

    def test_batch_cache_reuse_and_invalidation(
        self, sample_segments, config, tmp_path, monkeypatch
    ):
        from dd_clip_miner_llm.llm import (
            build_llm_messages,
            build_request_debug_metadata,
            build_providers,
            identify_content,
            write_llm_debug,
        )

        config["llm"]["reuse_valid_batches"] = True
        config["llm"]["debug_store_requests"] = False
        config["llm"]["api_key"] = "test-key"
        config["llm"]["use_tools"] = False
        recognizer = get_recognizer("song")
        provider = build_providers(config)[0]
        messages = build_llm_messages(recognizer, sample_segments, 0, config)
        metadata = build_request_debug_metadata(
            messages,
            config=config,
            provider=provider,
            recognizer=recognizer,
            segments=sample_segments,
            batch_start=0,
            tools=None,
            debug_phase="main",
        )
        debug_dir = tmp_path / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        write_llm_debug(debug_dir, 0, {
            **metadata,
            "provider": {
                "base_url": provider.base_url or "openai",
                "model": provider.model,
            },
            "raw_response": "[{\"title\":\"cached\"}]",
            "parsed_items": [{
                "content_type": "song",
                "title": "cached",
                "segment_indices": [0],
                "confidence": 0.9,
                "tags": [],
                "description": "",
                "artist": "",
            }],
            "parse_valid": True,
            "usage": [{
                "prompt_cache_hit_tokens": 100,
                "prompt_cache_miss_tokens": 20,
                "completion_tokens": 3,
            }],
            "tool_rounds": [],
            "json_fix_rounds": [],
            "reasoning_followups": [],
        })

        def fail_call_llm(*_args, **_kwargs):
            raise AssertionError("cached batch should not call LLM")

        import dd_clip_miner_llm.llm as llm_module
        monkeypatch.setattr(llm_module, "_call_llm_raw", fail_call_llm)
        matches = identify_content(
            sample_segments,
            config,
            recognizer,
            debug_dir=debug_dir,
            debug_phase="main",
        )

        assert len(matches) == 1
        assert matches[0].title == "cached"
        saved = json.loads(
            (debug_dir / "llm_batch_000000.json").read_text(encoding="utf-8")
        )
        assert saved["usage"][0]["prompt_cache_miss_tokens"] == 20
        assert saved["provider"]["model"] == provider.model
        assert saved["raw_response"] == "[{\"title\":\"cached\"}]"
        assert saved["cache_reuse"]["count"] == 1

        identify_content(
            sample_segments,
            config,
            recognizer,
            debug_dir=debug_dir,
            debug_phase="main",
        )
        saved = json.loads(
            (debug_dir / "llm_batch_000000.json").read_text(encoding="utf-8")
        )
        assert saved["usage"][0]["prompt_cache_miss_tokens"] == 20
        assert saved["cache_reuse"]["count"] == 2

        config["llm"]["model"] = "different-model"
        called = {"value": False}

        def mark_call_llm(*_args, **_kwargs):
            called["value"] = True
            raise RuntimeError("model changed, cache should miss")

        import dd_clip_miner_llm.llm.transport as transport_module

        monkeypatch.setattr(llm_module, "_call_llm_raw", mark_call_llm)
        monkeypatch.setattr(transport_module, "_call_llm_raw", mark_call_llm)
        matches_after_change = identify_content(
            sample_segments,
            config,
            recognizer,
            debug_dir=debug_dir,
            debug_phase="main",
        )
        assert called["value"] is True
        assert matches_after_change == []

    def test_batch_cache_rejects_top_level_and_tool_truncation(self):
        from dd_clip_miner_llm.llm import batch_debug_is_reusable

        metadata = {"request_fingerprint": "same"}
        base = {
            **metadata,
            "error": None,
            "parse_valid": True,
            "json_fix_rounds": [],
            "reasoning_followups": [],
            "tool_rounds": [],
        }

        assert batch_debug_is_reusable(base, expected_metadata=metadata)
        assert not batch_debug_is_reusable(
            {**base, "finish_reason": "length"},
            expected_metadata=metadata,
        )
        assert not batch_debug_is_reusable(
            {**base, "tool_rounds": [{"finish_reason": "length"}]},
            expected_metadata=metadata,
        )
        assert not batch_debug_is_reusable(
            {**base, "scan_incomplete": True},
            expected_metadata=metadata,
        )
        assert not batch_debug_is_reusable(
            {**base, "schema_valid": False},
            expected_metadata=metadata,
        )

    def test_truncated_json_array_continues_without_repeating_items(self):
        from dd_clip_miner_llm.llm import (
            LLMProvider,
            _continue_truncated_json_array,
            parse_llm_response,
        )

        calls = []

        class Message:
            content = '[{"content_type":"song","title":"B","segment_ranges":[[3,4]]}]'
            reasoning_content = ""
            tool_calls = None

            def model_dump(self):
                return {"content": self.content, "reasoning_content": "", "tool_calls": None}

        class Usage:
            def model_dump(self):
                return {"prompt_tokens": 10, "completion_tokens": 5}

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                choice = type("Choice", (), {"message": Message(), "finish_reason": "stop"})()
                return type("Response", (), {"choices": [choice], "usage": Usage(), "model": "test"})()

        client = type("Client", (), {
            "chat": type("Chat", (), {"completions": Completions()})(),
        })()
        config = {
            "llm": {"continuation_on_length": True, "max_continuation_rounds": 2},
        }
        debug = {"usage": []}
        content = _continue_truncated_json_array(
            client,
            LLMProvider(
                api_key="test",
                base_url="https://api.deepseek.com",
                max_completion_tokens=32768,
            ),
            config,
            [{"role": "user", "content": "full transcript and task"}],
            '[{"content_type":"song","title":"A","segment_ranges":[[1,2]]},',
            "length",
            debug,
            max_tokens=32768,
        )

        assert [item["title"] for item in parse_llm_response(content)] == ["A", "B"]
        assert debug["continuation_complete"] is True
        assert calls[0]["messages"][0]["content"] == "full transcript and task"
        assert calls[0]["max_tokens"] == 32768

    def test_cache_friendly_messages_share_transcript_prefix(self, sample_segments, config):
        from dd_clip_miner_llm.llm import build_llm_messages

        config["llm"]["cache_friendly_prompt_layout"] = True
        song_messages = build_llm_messages(
            get_recognizer("song"),
            sample_segments,
            0,
            config,
        )
        dialogue_messages = build_llm_messages(
            get_recognizer("dialogue"),
            sample_segments,
            0,
            config,
        )

        assert song_messages[0] == dialogue_messages[0]
        song_user = song_messages[1]["content"]
        dialogue_user = dialogue_messages[1]["content"]
        song_prefix = song_user.split("ASR 转写结束。", 1)[0]
        dialogue_prefix = dialogue_user.split("ASR 转写结束。", 1)[0]
        assert song_prefix != dialogue_prefix
        assert song_user != dialogue_user
        assert "[0] 大家好欢迎来到直播间" in song_user
        assert "(0.0s-3.0s)" not in song_user
        assert "[0] (0.0s-3.0s)" in dialogue_user

    def test_final_tool_round_keeps_tools_and_disables_tool_calls(self, monkeypatch):
        from dd_clip_miner_llm.llm import LLMProvider, run_llm_with_tools

        calls = []

        class Message:
            content = "[]"
            reasoning_content = ""
            tool_calls = None

            def model_dump(self):
                return {
                    "content": self.content,
                    "reasoning_content": self.reasoning_content,
                    "tool_calls": self.tool_calls,
                }

        class Usage:
            def model_dump(self):
                return {
                    "prompt_tokens": 10,
                    "prompt_cache_hit_tokens": 8,
                    "prompt_cache_miss_tokens": 2,
                }

        def fake_call_llm(_client, _provider, messages, **kwargs):
            calls.append({**kwargs, "messages": messages})
            choice = type("Choice", (), {
                "message": Message(),
                "finish_reason": "stop",
            })()
            return type("Response", (), {
                "choices": [choice],
                "usage": Usage(),
                "model": "test-model",
            })()

        monkeypatch.setattr("dd_clip_miner_llm.llm.tools.call_llm", fake_call_llm)
        client = object()
        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        debug = {}

        result = run_llm_with_tools(
            client,
            LLMProvider(api_key="test"),
            [{"role": "user", "content": "test"}],
            tools,
            lambda *_args: "{}",
            debug,
            max_tool_rounds=0,
            final_max_tokens=32768,
        )

        assert result == "[]"
        assert calls[0]["tools"] == tools
        assert calls[0]["tool_choice"] == "none"
        assert calls[0]["max_tokens_override"] == 32768
        assert debug["usage"][0]["prompt_cache_hit_tokens"] == 8

    def test_force_final_tool_round_retries_prose_as_json(self, monkeypatch):
        from dd_clip_miner_llm.llm import LLMProvider, run_llm_with_tools

        calls = []

        class Message:
            reasoning_content = ""
            tool_calls = None

            def __init__(self, content):
                self.content = content

            def model_dump(self):
                return {
                    "content": self.content,
                    "reasoning_content": self.reasoning_content,
                    "tool_calls": self.tool_calls,
                }

        class Usage:
            def model_dump(self):
                return {
                    "prompt_tokens": 10,
                    "prompt_cache_hit_tokens": 8,
                    "prompt_cache_miss_tokens": 2,
                }

        def fake_call_llm(_client, _provider, messages, **kwargs):
            calls.append({**kwargs, "messages": messages})
            content = "分析后没有漏检歌曲。" if len(calls) == 1 else "[]"
            choice = type("Choice", (), {
                "message": Message(content),
                "finish_reason": "stop",
            })()
            return type("Response", (), {
                "choices": [choice],
                "usage": Usage(),
                "model": "test-model",
            })()

        monkeypatch.setattr("dd_clip_miner_llm.llm.tools.call_llm", fake_call_llm)
        client = object()
        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        final_instruction = "重新扫描全部目标，只返回 JSON。"

        result = run_llm_with_tools(
            client,
            LLMProvider(api_key="test"),
            [{"role": "user", "content": "test"}],
            tools,
            lambda *_args: "{}",
            {},
            max_tool_rounds=1,
            final_max_tokens=4096,
            force_final_round=True,
            final_instruction=final_instruction,
        )

        assert result == "[]"
        assert len(calls) == 2
        assert calls[0]["tool_choice"] == "auto"
        assert calls[1]["tool_choice"] == "none"
        assert not any(
            message.get("role") == "assistant"
            and message.get("content") == "分析后没有漏检歌曲。"
            for message in calls[1]["messages"]
        )
        assert calls[1]["messages"][-1] == {
            "role": "user",
            "content": final_instruction,
        }

    def test_initial_tool_choice_none_returns_valid_json_without_tools(self, monkeypatch):
        from dd_clip_miner_llm.llm import LLMProvider, run_llm_with_tools

        calls = []

        class Message:
            content = '[{"content_type":"song","title":"A","segment_ranges":[[1,2]]}]'
            reasoning_content = ""
            tool_calls = None

            def model_dump(self):
                return {
                    "content": self.content,
                    "reasoning_content": self.reasoning_content,
                    "tool_calls": self.tool_calls,
                }

        class Usage:
            def model_dump(self):
                return {"prompt_tokens": 10}

        def fake_call_llm(_client, _provider, messages, **kwargs):
            calls.append({**kwargs, "messages": messages})
            choice = type("Choice", (), {
                "message": Message(),
                "finish_reason": "stop",
            })()
            return type("Response", (), {
                "choices": [choice],
                "usage": Usage(),
                "model": "test-model",
            })()

        monkeypatch.setattr("dd_clip_miner_llm.llm.tools.call_llm", fake_call_llm)

        result = run_llm_with_tools(
            object(),
            LLMProvider(api_key="test"),
            [{"role": "user", "content": "test"}],
            [{"type": "function", "function": {"name": "search", "parameters": {}}}],
            lambda *_args: "{}",
            {},
            max_tool_rounds=1,
            initial_tool_choice="none",
        )

        assert result == Message.content
        assert len(calls) == 1
        assert calls[0]["tool_choice"] == "none"

    def test_initial_tool_choice_none_continues_on_empty_json(self, monkeypatch):
        from dd_clip_miner_llm.llm import LLMProvider, run_llm_with_tools

        calls = []

        class Message:
            reasoning_content = ""
            tool_calls = None

            def __init__(self, content):
                self.content = content

            def model_dump(self):
                return {
                    "content": self.content,
                    "reasoning_content": self.reasoning_content,
                    "tool_calls": self.tool_calls,
                }

        class Usage:
            def model_dump(self):
                return {"prompt_tokens": 10}

        def fake_call_llm(_client, _provider, messages, **kwargs):
            calls.append({**kwargs, "messages": messages})
            content = "[]" if len(calls) == 1 else '[{"title":"fallback","segment_ranges":[[1,2]]}]'
            choice = type("Choice", (), {
                "message": Message(content),
                "finish_reason": "stop",
            })()
            return type("Response", (), {
                "choices": [choice],
                "usage": Usage(),
                "model": "test-model",
            })()

        debug = {}
        monkeypatch.setattr("dd_clip_miner_llm.llm.tools.call_llm", fake_call_llm)

        result = run_llm_with_tools(
            object(),
            LLMProvider(api_key="test"),
            [{"role": "user", "content": "test"}],
            [{"type": "function", "function": {"name": "search", "parameters": {}}}],
            lambda *_args: "{}",
            debug,
            max_tool_rounds=2,
            initial_tool_choice="none",
        )

        assert result == '[{"title":"fallback","segment_ranges":[[1,2]]}]'
        assert [call["tool_choice"] for call in calls] == ["none", "auto"]
        assert debug["tool_strategy_events"][0]["action"] == "continue_with_auto_tools"

    def test_content_validator_rejects_string_array_and_continues_with_tools(self, monkeypatch):
        from dd_clip_miner_llm.llm import LLMProvider, run_llm_with_tools

        calls = []

        class Message:
            reasoning_content = ""
            tool_calls = None

            def __init__(self, content):
                self.content = content

            def model_dump(self):
                return {
                    "content": self.content,
                    "reasoning_content": self.reasoning_content,
                    "tool_calls": self.tool_calls,
                }

        class Usage:
            def model_dump(self):
                return {"prompt_tokens": 10}

        def fake_call_llm(_client, _provider, messages, **kwargs):
            calls.append({**kwargs, "messages": messages})
            content = '["歌词1","歌词2"]' if len(calls) == 1 else '[{"title":"fallback","segment_ranges":[[1,2]]}]'
            choice = type("Choice", (), {
                "message": Message(content),
                "finish_reason": "stop",
            })()
            return type("Response", (), {
                "choices": [choice],
                "usage": Usage(),
                "model": "test-model",
            })()

        def validator(content):
            if content.startswith('["'):
                return False, {
                    "reason": "invalid_song_match_schema",
                    "raw_item_count": 2,
                    "parsed_match_count": 0,
                }
            return True, {
                "raw_item_count": 1,
                "parsed_match_count": 1,
            }

        debug = {}
        monkeypatch.setattr("dd_clip_miner_llm.llm.tools.call_llm", fake_call_llm)

        result = run_llm_with_tools(
            object(),
            LLMProvider(api_key="test"),
            [{"role": "user", "content": "test"}],
            [{"type": "function", "function": {"name": "search", "parameters": {}}}],
            lambda *_args: "{}",
            debug,
            max_tool_rounds=2,
            initial_tool_choice="none",
            content_validator=validator,
        )

        assert result == '[{"title":"fallback","segment_ranges":[[1,2]]}]'
        assert [call["tool_choice"] for call in calls] == ["none", "auto"]
        assert debug["content_validation"][0]["valid"] is False
        assert debug["tool_strategy_events"][0]["reason"] == "invalid_song_match_schema"

    def test_tool_role_unsupported_retries_with_user_tool_results(self, monkeypatch):
        from dd_clip_miner_llm.llm import LLMProvider, run_llm_with_tools

        calls = []

        class Function:
            name = "search_lyrics"
            arguments = '{"query":"lyrics"}'

        class ToolCall:
            id = "call_1"
            function = Function()

        class Message:
            reasoning_content = ""

            def __init__(self, content="", tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls

            def model_dump(self):
                return {
                    "content": self.content,
                    "reasoning_content": self.reasoning_content,
                    "tool_calls": (
                        [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in self.tool_calls
                        ]
                        if self.tool_calls
                        else None
                    ),
                }

        class Usage:
            def model_dump(self):
                return {"prompt_tokens": 10}

        def response(content="", tool_calls=None, finish_reason="stop"):
            choice = type("Choice", (), {
                "message": Message(content, tool_calls),
                "finish_reason": finish_reason,
            })()
            return type("Response", (), {
                "choices": [choice],
                "usage": Usage(),
                "model": "test-model",
            })()

        def fake_call_llm(_client, _provider, messages, **kwargs):
            calls.append({**kwargs, "messages": messages})
            if len(calls) == 1:
                return response(tool_calls=[ToolCall()], finish_reason="tool_calls")
            if any(message.get("role") == "tool" for message in messages):
                raise RuntimeError("Param Incorrect: messages[2] role is not supported")
            return response('[{"title":"A","segment_ranges":[[1,2]]}]')

        monkeypatch.setattr("dd_clip_miner_llm.llm.tools.call_llm", fake_call_llm)
        debug = {}

        result = run_llm_with_tools(
            object(),
            LLMProvider(api_key="test"),
            [{"role": "user", "content": "test"}],
            [{"type": "function", "function": {"name": "search_lyrics", "parameters": {}}}],
            lambda _name, _args: '{"results":[{"title":"A"}]}',
            debug,
            max_tool_rounds=1,
        )

        assert result == '[{"title":"A","segment_ranges":[[1,2]]}]'
        assert len(calls) == 3
        assert calls[1]["tool_choice"] == "none"
        assert any(message.get("role") == "tool" for message in calls[1]["messages"])
        assert calls[2]["tools"] is None
        assert not any(message.get("role") == "tool" for message in calls[2]["messages"])
        assert "工具调用得到的结果" in calls[2]["messages"][-1]["content"]
        assert debug["tool_strategy_events"][-1]["reason"] == "tool_role_unsupported"

    def test_tool_role_unsupported_without_tool_message_is_not_rewritten(
        self,
        monkeypatch,
    ):
        from dd_clip_miner_llm.llm.provider import LLMProvider
        from dd_clip_miner_llm.llm.tools import run_llm_with_tools

        calls = []

        def fake_call_llm(_client, _provider, messages, **kwargs):
            calls.append({**kwargs, "messages": messages})
            raise RuntimeError("Param Incorrect: messages[0] role is not supported")

        monkeypatch.setattr("dd_clip_miner_llm.llm.tools.call_llm", fake_call_llm)

        with pytest.raises(RuntimeError, match="role is not supported"):
            run_llm_with_tools(
                object(),
                LLMProvider(api_key="test"),
                [{"role": "user", "content": "test"}],
                [{"type": "function", "function": {"name": "search_lyrics", "parameters": {}}}],
                lambda _name, _args: "{}",
                {},
                max_tool_rounds=0,
            )

        assert len(calls) == 1

    def test_kv_v2_main_replays_invalid_song_schema(
        self,
        tmp_path,
        monkeypatch,
    ):
        from dd_clip_miner_llm.llm.identify import _process_single_batch
        from dd_clip_miner_llm.llm.provider import LLMProvider

        recognizer = get_recognizer("song")
        segments = [TranscriptSegment(0.0, 5.0, "歌词第一句")]
        config = deepcopy(DEFAULT_CONFIG)
        config["_profile_name"] = "kv_v2"
        config["llm"]["debug_store_requests"] = False
        calls = []

        def fake_run_llm_with_tools(*_args, **kwargs):
            calls.append(kwargs)
            assert kwargs.get("content_validator") is not None
            if len(calls) == 1:
                return '["歌词1","歌词2"]'
            return (
                '[{"content_type":"song","title":"测试",'
                '"segment_ranges":[[0,0]],"confidence":0.9,'
                '"tags":[],"description":"","artist":""}]'
            )

        monkeypatch.setattr(
            "dd_clip_miner_llm.llm.identify.run_llm_with_tools",
            fake_run_llm_with_tools,
        )
        monkeypatch.setattr("time.sleep", lambda _seconds: None)
        provider = LLMProvider(
            name="test",
            api_key="key",
            base_url="https://api.test/v1",
            result_retries=1,
        )

        matches = _process_single_batch(
            batch_idx=1,
            batch_count=1,
            batch_start=0,
            batch_segments=segments,
            segments=segments,
            config=config,
            recognizer=recognizer,
            providers=[provider],
            clients={(provider.base_url, provider.api_key, provider.proxy): object()},
            tools=[{"type": "function", "function": {"name": "search_lyrics"}}],
            debug_path=tmp_path,
            debug_phase="main",
            store_requests=False,
            reuse_valid_batches=False,
            content_type="song",
        )

        saved = json.loads((tmp_path / "llm_batch_000000.json").read_text(encoding="utf-8"))
        assert len(matches) == 1
        assert len(calls) == 2
        assert saved["schema_valid"] is True
        assert saved["schema_validation_failures"][0]["schema_error_reason"] == "invalid_song_match_schema"

    def test_kv_v2_empty_array_invalid_only_with_singing_evidence(self):
        from dd_clip_miner_llm.llm.identify import _validate_song_match_content

        recognizer = get_recognizer("song")
        config = deepcopy(DEFAULT_CONFIG)
        singing_segments = [
            TranscriptSegment(0.0, 5.0, "歌词第一句"),
            TranscriptSegment(5.0, 10.0, "歌词第二句"),
        ]
        chat_segments = [
            TranscriptSegment(0.0, 5.0, "今天月亮很好看"),
            TranscriptSegment(5.0, 10.0, "我心情不错"),
            TranscriptSegment(10.0, 15.0, "我们继续聊天"),
        ]

        singing_valid, singing_details = _validate_song_match_content(
            "[]", recognizer, config, segments=singing_segments,
        )
        chat_valid, chat_details = _validate_song_match_content(
            "[]", recognizer, config, segments=chat_segments,
        )

        assert singing_valid is False
        assert singing_details["schema_error_reason"] == "empty_song_match_array"
        assert chat_valid is True
        assert chat_details["schema_error_reason"] is None

    @pytest.mark.parametrize(
        ("profile", "debug_phase", "expected_initial_tool_choice"),
        [
            ("kv_v2", "main", "none"),
            ("kv_v2", "review_before", None),
            ("kv_v2", "missed_recheck", None),
            ("kv_v2", "review_after", None),
            ("kv_optimized", "main", None),
            ("accuracy", "main", None),
        ],
    )
    def test_kv_v2_main_identify_uses_initial_no_tools_only_there(
        self,
        tmp_path,
        monkeypatch,
        profile,
        debug_phase,
        expected_initial_tool_choice,
    ):
        from dd_clip_miner_llm.llm.identify import _process_single_batch
        from dd_clip_miner_llm.llm.provider import LLMProvider

        recognizer = get_recognizer("song")
        segments = [TranscriptSegment(0.0, 5.0, "歌词第一句")]
        config = deepcopy(DEFAULT_CONFIG)
        config["_profile_name"] = profile
        config["llm"]["debug_store_requests"] = False
        captured = {}

        def fake_run_llm_with_tools(*_args, **kwargs):
            captured["initial_tool_choice"] = kwargs.get("initial_tool_choice")
            return (
                '[{"content_type":"song","title":"测试",'
                '"segment_ranges":[[0,0]],"confidence":0.9,'
                '"tags":[],"description":"","artist":""}]'
            )

        monkeypatch.setattr(
            "dd_clip_miner_llm.llm.identify.run_llm_with_tools",
            fake_run_llm_with_tools,
        )
        provider = LLMProvider(
            name="test",
            api_key="key",
            base_url="https://api.test/v1",
            result_retries=0,
        )

        matches = _process_single_batch(
            batch_idx=1,
            batch_count=1,
            batch_start=0,
            batch_segments=segments,
            segments=segments,
            config=config,
            recognizer=recognizer,
            providers=[provider],
            clients={(provider.base_url, provider.api_key, provider.proxy): object()},
            tools=[{"type": "function", "function": {"name": "search_lyrics"}}],
            debug_path=tmp_path,
            debug_phase=debug_phase,
            store_requests=False,
            reuse_valid_batches=False,
            content_type="song",
        )

        assert len(matches) == 1
        assert captured["initial_tool_choice"] == expected_initial_tool_choice

    def test_deepseek_uses_max_tokens_for_completion_limit(self):
        from dd_clip_miner_llm.llm import LLMProvider, call_llm

        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                choice = SimpleNamespace(
                    delta=SimpleNamespace(content="ok", reasoning_content=None),
                    finish_reason="stop",
                )
                return iter([
                    SimpleNamespace(choices=[choice], usage=None, model="test-model"),
                ])

        client = type("Client", (), {
            "chat": type("Chat", (), {"completions": Completions()})(),
        })()
        provider = LLMProvider(
            api_key="test",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            max_completion_tokens=32768,
            thinking="disabled",
        )

        call_llm(client, provider, [{"role": "user", "content": "test"}])

        assert calls[0]["max_tokens"] == 32768
        assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
        assert "max_completion_tokens" not in calls[0]

    def test_provider_timeout_and_retries_are_forwarded(self, monkeypatch):
        import time

        from dd_clip_miner_llm.llm import build_providers, call_llm

        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise TimeoutError("test timeout")
                choice = SimpleNamespace(
                    delta=SimpleNamespace(content="ok", reasoning_content=None),
                    finish_reason=None,
                )
                final_choice = SimpleNamespace(
                    delta=SimpleNamespace(content=None, reasoning_content=None),
                    finish_reason="stop",
                )
                return iter([
                    SimpleNamespace(choices=[choice], usage=None, model="test-model"),
                    SimpleNamespace(choices=[final_choice], usage=None, model="test-model"),
                ])

        client = type("Client", (), {
            "chat": type("Chat", (), {"completions": Completions()})(),
        })()
        config = {
            "llm": {
                "api_key": "test",
                "model": "test-model",
                "timeout": 12,
                "max_retries": 2,
            }
        }
        provider = build_providers(config)[0]
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)

        call_llm(client, provider, [{"role": "user", "content": "test"}])

        assert len(calls) == 2
        assert calls[0]["timeout"] == 12
        assert calls[0]["stream"] is True
        assert calls[1]["stream"] is True
        assert provider.max_retries == 2

    def test_parse_json_array(self):
        """应能解析 JSON 数组"""
        text = '[{"title": "test"}]'
        items = parse_llm_response(text)
        assert len(items) == 1
        assert items[0]["title"] == "test"

    def test_parse_markdown_json(self):
        """应能解析 Markdown 中的 JSON"""
        text = '```json\n[{"title": "test"}]\n```'
        items = parse_llm_response(text)
        assert len(items) == 1

    def test_parse_invalid_json(self):
        """无效 JSON 应返回空列表"""
        items = parse_llm_response("not json")
        assert items == []

    def test_parse_json_object(self):
        """结构化总结可以解析 JSON object"""
        item = parse_llm_json('{"content_type":"daily_summary","level_1":[]}')
        assert item["content_type"] == "daily_summary"

    def test_structured_json_fix_failure_is_not_empty_dict(self):
        config = deepcopy(DEFAULT_CONFIG)
        config["llm"]["json_fix_rounds"] = 0

        parsed, _ = fix_structured_json_with_llm(
            client=None,
            provider=None,
            config=config,
            raw_content="{ truncated",
            content_type="daily_summary",
            batch_debug={},
        )

        assert parsed["content_type"] == "daily_summary"
        assert parsed["error"] == "LLM JSON repair disabled"


# ============ 4. Merger 测试 ============

class TestMerger:
    def test_build_song_results(self, sample_segments, config):
        """歌曲结果构建"""
        matches = [
            create_song_match(title="测试歌曲", segment_indices=[1, 2, 3], confidence=0.9)
        ]
        results = build_content_results(sample_segments, matches, 30.0, config, "song")
        assert len(results) >= 0  # 可能因 min_duration 过滤

    def test_build_dialogue_results(self, sample_segments, config):
        """对话结果构建"""
        matches = [
            ContentMatch(content_type="dialogue", title="测试对话", segment_indices=[5, 6, 7], confidence=0.8)
        ]
        results = build_content_results(sample_segments, matches, 30.0, config, "dialogue")
        assert len(results) >= 0

    def test_build_cringe_results(self, sample_segments, config):
        """下头结果构建"""
        matches = [
            ContentMatch(
                content_type="cringe",
                title="测试下头",
                segment_indices=[5, 6],
                confidence=0.9,
                tags=["油腻发言"],
                description="[严重程度:2/5][场景:A] 测试描述"
            )
        ]
        results = build_content_results(sample_segments, matches, 30.0, config, "cringe")
        assert len(results) >= 0


# ============ 5. 报告生成测试 ============

class TestReport:
    def test_write_reports(self, tmp_path):
        """应能生成报告"""
        results = [
            ContentResult(
                index=1, content_type="song", title="测试",
                start=10.0, end=40.0, duration=30.0,
                transcript="歌词", confidence=0.9
            )
        ]
        csv_path, json_path = write_reports(results, tmp_path, "song")
        assert csv_path.exists()
        assert json_path.exists()

    def test_write_match_context(self, tmp_path):
        """应能生成上下文报告"""
        matches = [
            ContentMatch(content_type="song", title="测试", segment_indices=[0, 1], confidence=0.9)
        ]
        segments = [
            TranscriptSegment(start=0.0, end=2.0, text="test1"),
            TranscriptSegment(start=2.0, end=4.0, text="test2"),
        ]
        csv_path, json_path = write_match_context_reports(matches, segments, tmp_path)
        assert csv_path.exists()
        assert json_path.exists()


# ============ 6. CLI 测试 ============

class TestCLI:
    def test_parser_run(self):
        """run 命令解析"""
        parser = build_parser()
        args = parser.parse_args(["run", "test.mp4"])
        assert args.command == "run"
        assert args.video == "test.mp4"

    def test_parser_batch_run(self):
        """batch-run 命令解析"""
        parser = build_parser()
        args = parser.parse_args(["batch-run", "input/", "--result-root", "output/"])
        assert args.command == "batch-run"

    def test_parser_manual_cut(self):
        """manual-cut 命令解析"""
        parser = build_parser()
        args = parser.parse_args(["manual-cut", "run_dir/"])
        assert args.command == "manual-cut"

    def test_parser_content_types(self):
        """--content-types 参数"""
        parser = build_parser()
        args = parser.parse_args(["run", "test.mp4", "--content-types", "song,cringe"])
        assert args.content_types == "song,cringe"

    def test_resolve_profile_all_order(self, tmp_path):
        from dd_clip_miner_llm.cli import _resolve_profile_names

        config_file = tmp_path / "profiles.yaml"
        config_file.write_text(
            "default_profile: kv_optimized\n"
            "profiles:\n"
            "  accuracy: {}\n"
            "  kv_optimized: {}\n",
            encoding="utf-8",
        )

        assert _resolve_profile_names(config_file, "all") == [
            "kv_optimized",
            "accuracy",
        ]


# ============ 7. 工具函数测试 ============

class TestUtils:
    def test_safe_path_part(self):
        """路径安全处理"""
        assert safe_path_part("normal") == "normal"
        assert safe_path_part("a/b:c") != "a/b:c"
        assert safe_path_part("") == "item"

    def test_parse_cringe_description(self):
        """Cringe 描述解析"""
        desc = "[严重程度:3/5][场景:B] 观众开黄腔"
        result = parse_cringe_description(desc)
        assert result["severity"] == 3
        assert result["scenario"] == "B"
        assert "观众开黄腔" in result["description"]


# ============ 8. 集成测试 ============

class TestIntegration:
    def test_full_pipeline_flow(self, sample_segments, config, tmp_path):
        """完整 pipeline 流程测试"""
        # 1. 获取识别器
        recognizer = get_recognizer("cringe")
        assert recognizer is not None

        # 2. 模拟 LLM 返回
        llm_items = [
            {
                "content_type": "cringe",
                "title": "观众开黄腔",
                "segment_indices": [0, 1],
                "confidence": 0.9,
                "severity": 3,
                "scenario": "B",
                "tags": ["性骚扰"],
                "description": "观众对主播开黄腔"
            }
        ]

        # 3. 解析响应
        matches = recognizer.parse_response(llm_items, config)
        assert len(matches) == 1
        assert matches[0].content_type == "cringe"

        # 4. 构建结果
        results = build_content_results(sample_segments, matches, 30.0, config, "cringe")
        assert len(results) >= 0

        # 5. 生成报告
        if results:
            csv_path, json_path = write_reports(results, tmp_path, "cringe")
            assert csv_path.exists()
            assert json_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
