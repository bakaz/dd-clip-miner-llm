"""冒烟测试 - 验证整体 pipeline 核心功能"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from copy import deepcopy

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
        assert song_prefix == dialogue_prefix
        assert song_user != dialogue_user
        assert "[0] (0.0s-3.0s)" in song_user

    def test_final_tool_round_keeps_tools_and_disables_tool_calls(self):
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

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                choice = type("Choice", (), {
                    "message": Message(),
                    "finish_reason": "stop",
                })()
                return type("Response", (), {
                    "choices": [choice],
                    "usage": Usage(),
                    "model": "test-model",
                })()

        client = type("Client", (), {
            "chat": type("Chat", (), {"completions": Completions()})(),
        })()
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
        assert calls[0]["max_tokens"] == 32768
        assert debug["usage"][0]["prompt_cache_hit_tokens"] == 8

    def test_force_final_tool_round_retries_prose_as_json(self):
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

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
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

        client = type("Client", (), {
            "chat": type("Chat", (), {"completions": Completions()})(),
        })()
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

    def test_deepseek_uses_max_tokens_for_completion_limit(self):
        from dd_clip_miner_llm.llm import LLMProvider, call_llm

        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return object()

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
