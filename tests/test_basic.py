"""dd_clip_miner_llm 基础测试"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dd_clip_miner_llm.config import DEFAULT_CONFIG, deep_merge, get_padding_config, load_config
from dd_clip_miner_llm import ffmpeg
from dd_clip_miner_llm.models import (
    ContentMatch,
    ContentResult,
    TranscriptSegment,
    create_song_match,
    create_song_result,
)
from dd_clip_miner_llm.paths import safe_path_part
from dd_clip_miner_llm.report import _format_timecode
from dd_clip_miner_llm.merger import build_content_results
from dd_clip_miner_llm.clip_naming import (
    ClipNamingProfile,
    build_clip_export_stem,
    extract_yymmdd_from_texts,
    is_valid_yymmdd,
    resolve_clip_naming_profile,
    text_similarity,
)


# ============ models.py 测试 ============

class TestTranscriptSegment:
    def test_create(self):
        seg = TranscriptSegment(start=1.0, end=2.0, text="hello")
        assert seg.start == 1.0
        assert seg.end == 2.0
        assert seg.text == "hello"

    def test_to_dict(self):
        seg = TranscriptSegment(start=1.0, end=2.0, text="hello")
        d = seg.to_dict()
        assert d == {"start": 1.0, "end": 2.0, "text": "hello"}


class TestContentMatch:
    def test_create(self):
        match = ContentMatch(
            content_type="song",
            title="Test Song",
            segment_indices=[1, 2, 3],
            confidence=0.85,
        )
        assert match.content_type == "song"
        assert match.title == "Test Song"
        assert match.segment_indices == [1, 2, 3]
        assert match.confidence == 0.85
        assert match.tags == []
        assert match.artist == ""

    def test_to_dict(self):
        match = ContentMatch(
            content_type="song",
            title="Test",
            segment_indices=[1],
            confidence=0.5,
        )
        d = match.to_dict()
        assert d["content_type"] == "song"
        assert d["title"] == "Test"

    def test_song_match_alias(self):
        """测试 SongMatch 类型别名"""
        match = ContentMatch(
            content_type="song",
            title="Test",
            segment_indices=[1],
            confidence=0.5,
        )
        assert isinstance(match, ContentMatch)


class TestContentResult:
    def test_create(self):
        result = ContentResult(
            index=1,
            content_type="song",
            title="Test",
            start=10.0,
            end=40.0,
            duration=30.0,
            transcript="lyrics",
            confidence=0.9,
        )
        assert result.index == 1
        assert result.content_type == "song"
        assert result.duration == 30.0

    def test_to_dict(self):
        result = ContentResult(
            index=1,
            content_type="song",
            title="Test",
            start=10.0,
            end=40.0,
            duration=30.0,
            transcript="lyrics",
            confidence=0.9,
            audio_path=Path("/path/to/audio.m4a"),
        )
        d = result.to_dict()
        assert d["audio_path"] == str(Path("/path/to/audio.m4a"))
        assert d["video_path"] is None


class TestFactoryFunctions:
    def test_create_song_match(self):
        match = create_song_match(
            title="Test Song",
            artist="Test Artist",
            segment_indices=[1, 2, 3],
            confidence=0.85,
        )
        assert match.content_type == "song"
        assert match.title == "Test Song"
        assert match.artist == "Test Artist"
        assert match.segment_indices == [1, 2, 3]

    def test_create_song_result(self):
        result = create_song_result(
            index=1,
            title="Test Song",
            artist="Test Artist",
            start=10.0,
            end=40.0,
            duration=30.0,
        )
        assert result.content_type == "song"
        assert result.title == "Test Song"
        assert result.artist == "Test Artist"
        assert result.duration == 30.0


# ============ config.py 测试 ============

class TestConfig:
    def test_default_config_structure(self):
        assert "audio" in DEFAULT_CONFIG
        assert "asr" in DEFAULT_CONFIG
        assert "llm" in DEFAULT_CONFIG
        assert "padding" in DEFAULT_CONFIG
        assert "song" in DEFAULT_CONFIG
        assert "dialogue" in DEFAULT_CONFIG
        assert "content_types" in DEFAULT_CONFIG
        assert "output" in DEFAULT_CONFIG

    def test_content_types_format(self):
        """测试 content_types 是字典格式"""
        content_types = DEFAULT_CONFIG["content_types"]
        assert isinstance(content_types, dict)
        assert "song" in content_types
        assert "dialogue" in content_types
        assert "highlight" in content_types
        assert "funny" in content_types
        assert "daily_summary" in content_types
        assert content_types["song"] is True
        assert content_types["daily_summary"] is False

    def test_deep_merge(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 4}, "e": 5}
        result = deep_merge(base, override)
        assert result["a"] == 1
        assert result["b"]["c"] == 4
        assert result["b"]["d"] == 3
        assert result["e"] == 5

    def test_get_padding_config_from_song(self):
        config = {
            "song": {
                "padding": {
                    "before_seconds": 5.0,
                    "after_seconds": 20.0,
                }
            }
        }
        padding = get_padding_config(config, "song")
        assert padding["before_seconds"] == 5.0
        assert padding["after_seconds"] == 20.0

    def test_get_padding_config_from_top_level(self):
        config = {
            "padding": {
                "before_seconds": 3.0,
                "after_seconds": 15.0,
            }
        }
        padding = get_padding_config(config, "song")
        assert padding["before_seconds"] == 3.0
        assert padding["after_seconds"] == 15.0

    def test_load_config_none(self):
        config = load_config(None)
        assert "audio" in config
        assert "padding" in config

    def test_load_config_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
asr:
  model: medium
padding:
  before_seconds: 5.0
""")
        config = load_config(config_file)
        assert config["asr"]["model"] == "medium"
        assert config["padding"]["before_seconds"] == 5.0

    def test_migrate_padding_config(self):
        """测试旧项目 padding 配置迁移到 song.padding"""
        from dd_clip_miner_llm.config import _migrate_padding_config
        
        config = {
            "padding": {
                "before_seconds": 5.0,
                "after_seconds": 20.0,
            },
            "song": {},
        }
        result = _migrate_padding_config(config)
        assert result["song"]["padding"]["before_seconds"] == 5.0
        assert result["song"]["padding"]["after_seconds"] == 20.0


class TestSongMissedRecheck:
    def test_uncovered_segment_ranges(self):
        from dd_clip_miner_llm.pipeline import _uncovered_segment_ranges

        matches = [
            ContentMatch(content_type="song", title="A", segment_indices=[1, 2], confidence=0.9),
            ContentMatch(content_type="song", title="B", segment_indices=[4], confidence=0.8),
        ]

        assert _uncovered_segment_ranges(6, matches) == [(0, 0), (3, 3), (5, 5)]

    def test_split_segment_ranges(self):
        from dd_clip_miner_llm.pipeline import _split_segment_ranges

        assert _split_segment_ranges([(0, 4), (8, 9)], 2) == [
            (0, 1),
            (2, 3),
            (4, 4),
            (8, 9),
        ]

    def test_default_config_enables_song_missed_recheck(self):
        recheck = DEFAULT_CONFIG["song"]["missed_recheck"]

        assert recheck["enabled"] is True
        assert recheck["batch_size"] == 500


# ============ ffmpeg.py 测试 ============

class TestFFmpeg:
    def test_auto_video_candidates_copy_first_then_hardware(self, monkeypatch):
        monkeypatch.setattr(
            ffmpeg,
            "detect_video_encoders",
            lambda _ffmpeg_bin: {"h264_nvenc", "h264_qsv", "h264_amf", "libx264"},
        )

        candidates = ffmpeg._video_encode_arg_candidates("ffmpeg", "auto")

        assert candidates[0] == ["-c:v", "copy"]
        assert candidates[1][1] == "h264_nvenc"
        assert candidates[2][1] == "h264_qsv"
        assert candidates[3][1] == "h264_amf"
        assert candidates[4][1] == "libx264"

    def test_concat_quick_probe_then_targeted_bad_segment_reencode(self, tmp_path, monkeypatch):
        calls = []

        def fake_run_command(args, timeout=3600):
            calls.append(args)

        monkeypatch.setattr(ffmpeg, "require_binary", lambda name: name)
        monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)
        monkeypatch.setattr(
            ffmpeg,
            "_find_bad_h264_segments",
            lambda _videos, _bin, tail_seconds=None: [1],
        )
        monkeypatch.setattr(
            ffmpeg,
            "detect_video_encoders",
            lambda _ffmpeg_bin: {"libx264"},
        )
        monkeypatch.setattr(ffmpeg, "_get_min_video_size", lambda _videos: (641, 361))

        output = tmp_path / "concat.mp4"
        ffmpeg.concat_videos(
            [tmp_path / "a.mp4", tmp_path / "b.mp4"],
            output,
            video_codec="auto",
            audio_bitrate_kbps=320,
        )

        assert calls[0][calls[0].index("-i") + 1] == str(tmp_path / "b.mp4")
        assert calls[0][calls[0].index("-c:v") + 1] == "libx264"
        assert calls[0][calls[0].index("-fflags") + 1] == "+discardcorrupt"
        assert calls[1][calls[1].index("-c") + 1] == "copy"
        assert not (tmp_path / "concat_list.txt").exists()

    def test_concat_targeted_reencode_skips_to_remux_when_no_bad_segments(self, tmp_path, monkeypatch):
        calls = []

        def fake_run_command(args, timeout=3600):
            calls.append(args)
            if len(calls) == 1:
                raise ffmpeg.FFmpegError("copy failed")

        monkeypatch.setattr(ffmpeg, "require_binary", lambda name: name)
        monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)
        monkeypatch.setattr(
            ffmpeg,
            "_find_bad_h264_segments",
            lambda _videos, _bin, tail_seconds=None: [],
        )
        monkeypatch.setattr(ffmpeg, "_get_min_video_size", lambda _videos: (641, 361))

        output = tmp_path / "concat.mp4"
        ffmpeg.concat_videos(
            [tmp_path / "a.mp4", tmp_path / "b.mp4"],
            output,
            video_codec="auto",
            audio_bitrate_kbps=320,
        )

        assert calls[1][calls[1].index("-c") + 1] == "copy"
        assert calls[1][calls[1].index("-avoid_negative_ts") + 1] == "make_zero"
        assert calls[2][calls[2].index("-c") + 1] == "copy"
        assert calls[3][calls[3].index("-c") + 1] == "copy"
        assert not (tmp_path / "concat_list.txt").exists()

    def test_concat_copy_codec_falls_back_to_auto_reencode(self, monkeypatch):
        monkeypatch.setattr(
            ffmpeg,
            "detect_video_encoders",
            lambda _ffmpeg_bin: {"h264_qsv", "libx264"},
        )

        candidates = ffmpeg._video_reencode_arg_candidates("ffmpeg", "copy")

        assert candidates[0][1] == "h264_qsv"
        assert candidates[-1][1] == "libx264"

    def test_concat_copy_codec_fallback_uses_auto_reencode(self, monkeypatch):
        monkeypatch.setattr(
            ffmpeg,
            "detect_video_encoders",
            lambda _ffmpeg_bin: {"h264_nvenc", "h264_qsv", "libx264"},
        )

        candidates = ffmpeg._concat_reencode_arg_candidates("ffmpeg", "copy")

        assert candidates[0][1] == "h264_nvenc"
        assert candidates[-1][1] == "libx264"

    def test_targeted_repair_prefers_hardware_for_auto(self, monkeypatch):
        monkeypatch.setattr(
            ffmpeg,
            "detect_video_encoders",
            lambda _ffmpeg_bin: {"h264_nvenc", "h264_qsv", "libx264"},
        )

        candidates = ffmpeg._targeted_repair_encode_candidates("ffmpeg", "auto")

        assert candidates[0][1] == "h264_nvenc"
        assert candidates[-1] == ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]


class TestBatch:
    def test_cleanup_concat_source_removes_concat_intermediates(self, tmp_path):
        from dd_clip_miner_llm.batch import _cleanup_concat_source

        concat_dir = tmp_path / "concat"
        concat_dir.mkdir()
        concat_file = concat_dir / "concat.mp4"
        concat_file.write_bytes(b"video")
        list_file = concat_dir / "concat_list.txt"
        list_file.write_text("file 'a.mp4'\n", encoding="utf-8")

        _cleanup_concat_source(concat_dir)

        assert not concat_file.exists()
        assert not list_file.exists()


# ============ merger.py 测试 ============

class TestMergerPadding:
    def test_song_before_padding_uses_previous_asr_start_guard(self):
        segments = [
            TranscriptSegment(start=90.0, end=95.0, text="上一句"),
            TranscriptSegment(start=100.0, end=110.0, text="唱歌"),
            TranscriptSegment(start=130.0, end=135.0, text="下一句"),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "padding": {
                    "before_seconds": 15.0,
                    "after_seconds": 15.0,
                    "after_next_asr_end_guard_seconds": 2.0,
                    "min_song_seconds": 0.0,
                },
            },
        })
        matches = [
            ContentMatch(
                content_type="song",
                title="测试歌曲",
                segment_indices=[1],
                confidence=0.9,
            )
        ]

        results = build_content_results(segments, matches, 150.0, config, "song")

        assert len(results) == 1
        assert results[0].start == 92.0
        assert results[0].end == 125.0


# ============ paths.py 测试 ============

class TestPaths:
    def test_safe_path_part_normal(self):
        assert safe_path_part("hello") == "hello"

    def test_safe_path_part_special_chars(self):
        result = safe_path_part('file<>:"/\\|?*name')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result

    def test_safe_path_part_empty(self):
        assert safe_path_part("") == "item"

    def test_safe_path_part_max_length(self):
        long_name = "a" * 200
        result = safe_path_part(long_name, max_length=50)
        assert len(result) <= 50


# ============ report.py 测试 ============

class TestReport:
    def test_format_timecode(self):
        assert _format_timecode(0) == "00:00:00"
        assert _format_timecode(61) == "00:01:01"
        assert _format_timecode(3661) == "01:01:01"

    def test_format_timecode_negative(self):
        assert _format_timecode(-1) == "00:00:00"


# ============ search_tools.py 测试 ============

class TestSearchTools:
    def test_get_tools(self):
        from dd_clip_miner_llm.search_tools import get_tools
        tools = get_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "search_lyrics"

    def test_execute_tool_unknown(self):
        from dd_clip_miner_llm.search_tools import execute_tool
        result = execute_tool("unknown_tool", {})
        assert "Unknown tool" in result


# ============ CLI 测试 ============

class TestCLI:
    def test_build_parser(self):
        from dd_clip_miner_llm.cli import build_parser
        parser = build_parser()
        # 测试 run 命令
        args = parser.parse_args(["run", "test.mp4"])
        assert args.command == "run"
        assert args.video == "test.mp4"

    def test_build_parser_batch_run(self):
        from dd_clip_miner_llm.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["batch-run", "input/", "--result-root", "output/"])
        assert args.command == "batch-run"
        assert args.input_root == "input/"

    def test_build_parser_manual_cut(self):
        from dd_clip_miner_llm.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["manual-cut", "run_dir/"])
        assert args.command == "manual-cut"
        assert args.run_dir == "run_dir/"

    def test_build_parser_init_config(self):
        from dd_clip_miner_llm.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["init-config", "--out", "config.yaml"])
        assert args.command == "init-config"
        assert args.out == "config.yaml"

    def test_build_parser_padding_args(self):
        from dd_clip_miner_llm.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["run", "test.mp4", "--padding-before", "5", "--padding-after", "20"])
        assert args.padding_before == 5.0
        assert args.padding_after == 20.0

    def test_build_parser_content_types(self):
        from dd_clip_miner_llm.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["run", "test.mp4", "--content-types", "song,dialogue,highlight"])
        assert args.content_types == "song,dialogue,highlight"


# ============ recognizers 测试 ============

class TestRecognizers:
    def test_list_recognizers(self):
        from dd_clip_miner_llm.recognizers import list_recognizers
        available = list_recognizers()
        assert "song" in available
        assert "dialogue" in available
        assert "highlight" in available
        assert "funny" in available
        assert "daily_summary" in available

    def test_get_recognizer(self):
        from dd_clip_miner_llm.recognizers import get_recognizer
        song = get_recognizer("song")
        assert song is not None
        assert song.name == "song"
        
        dialogue = get_recognizer("dialogue")
        assert dialogue is not None
        assert dialogue.name == "dialogue"
        
        highlight = get_recognizer("highlight")
        assert highlight is not None
        assert highlight.name == "highlight"
        
        funny = get_recognizer("funny")
        assert funny is not None
        assert funny.name == "funny"

        daily_summary = get_recognizer("daily_summary")
        assert daily_summary is not None
        assert daily_summary.name == "daily_summary"

    def test_get_recognizer_unknown(self):
        from dd_clip_miner_llm.recognizers import get_recognizer
        unknown = get_recognizer("unknown")
        assert unknown is None

    def test_recognizer_build_prompt(self, sample_segments, sample_config):
        from dd_clip_miner_llm.recognizers import get_recognizer
        
        song = get_recognizer("song")
        prompt = song.build_prompt(sample_segments, 0, sample_config)
        assert "歌曲识别专家" in prompt
        assert "[0]" in prompt
        
        dialogue = get_recognizer("dialogue")
        prompt = dialogue.build_prompt(sample_segments, 0, sample_config)
        assert "对话片段" in prompt
        
        highlight = get_recognizer("highlight")
        prompt = highlight.build_prompt(sample_segments, 0, sample_config)
        assert "高能时刻" in prompt
        
        funny = get_recognizer("funny")
        prompt = funny.build_prompt(sample_segments, 0, sample_config)
        assert "搞笑片段" in prompt

        daily_summary = get_recognizer("daily_summary")
        prompt = daily_summary.build_prompt(sample_segments, 0, sample_config)
        assert "金字塔结构" in prompt
        assert "level_1" in prompt
        assert "segment_indices 数组最多" in prompt
        assert "严禁列出连续长数组" in prompt

    def test_recognizer_parse_response(self):
        from dd_clip_miner_llm.recognizers import get_recognizer
        
        song = get_recognizer("song")
        items = [
            {"content_type": "song", "title": "Test Song", "segment_indices": [1, 2], "confidence": 0.9},
        ]
        matches = song.parse_response(items, {})
        assert len(matches) == 1
        assert matches[0].title == "Test Song"
        assert matches[0].content_type == "song"

    def test_cringe_title_is_short_summary(self):
        from dd_clip_miner_llm.recognizers import get_recognizer

        cringe = get_recognizer("cringe")
        items = [
            {
                "content_type": "cringe",
                "title": "主播连续说了很多让人非常不舒服的油腻发言",
                "segment_indices": [1],
                "confidence": 0.9,
                "severity": 2,
                "scenario": "A",
                "description": "主播连续说油腻话",
            }
        ]

        matches = cringe.parse_response(items, {})

        assert len(matches) == 1
        assert matches[0].content_type == "cringe"
        assert len(matches[0].title) < 20


# ============ clip_naming.py 测试 ============

class TestClipNaming:
    def test_is_valid_yymmdd(self):
        assert is_valid_yymmdd("260603")
        assert not is_valid_yymmdd("261399")

    def test_extract_yymmdd_from_folder(self):
        assert extract_yymmdd_from_texts(["2026_06_03", "live.mp4"]) == "260603"

    def test_extract_yymmdd_token(self):
        assert extract_yymmdd_from_texts(["live_250603_cut"]) == "250603"

    def test_build_clip_export_stem(self):
        profile = ClipNamingProfile(streamer="示例主播", date="260603")
        result = create_song_result(
            index=1,
            title="示例歌曲",
            artist="示例歌手",
            start=0.0,
            end=10.0,
            duration=10.0,
        )
        stem = build_clip_export_stem(result, profile)
        assert stem.startswith("【示例主播】")
        assert "示例歌曲" in stem
        assert "260603" in stem

    def test_resolve_profile_from_dictionary(self):
        with tempfile.TemporaryDirectory() as tmp:
            dict_path = Path(tmp) / "clip_dictionary.json"
            dict_path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "streamer": "示例主播",
                                "aliases": ["folder_keyword"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            video_path = Path(tmp) / "folder_keyword" / "2026_06_03" / "live.mp4"
            video_path.parent.mkdir(parents=True)
            video_path.write_bytes(b"test")
            config = {
                "output": {
                    "clip_naming": {
                        "enabled": True,
                        "dictionary_path": str(dict_path),
                        "min_score": 0.5,
                    }
                }
            }
            profile = resolve_clip_naming_profile(video_path, config)
            assert profile is not None
            assert profile.streamer == "示例主播"
            assert profile.date == "260603"
            assert profile.source == "dictionary"

    def test_no_yymmdd_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "no_date_folder" / "live.mp4"
            video_path.parent.mkdir(parents=True)
            video_path.write_bytes(b"test")
            config = {"output": {"clip_naming": {"enabled": True, "dictionary_path": ""}}}
            assert resolve_clip_naming_profile(video_path, config) is None

    def test_text_similarity_folder_alias(self):
        assert text_similarity("folder_keyword", "folder_keyword") == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
