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
from dd_clip_miner_llm.asr_backends import build_asr_backend
from dd_clip_miner_llm.asr_backends.faster_whisper import FasterWhisperBackend
from dd_clip_miner_llm.asr_backends.funasr_backend import FunASRBackend, funasr_result_to_segments


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
        assert DEFAULT_CONFIG["asr"]["backend"] == "funasr"
        assert DEFAULT_CONFIG["asr"]["funasr"]["model"] == "Qwen/Qwen3-ASR-0.6B"
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


class TestASRBackends:
    def test_build_default_backend(self):
        backend = build_asr_backend(DEFAULT_CONFIG["asr"])
        assert isinstance(backend, FunASRBackend)

    def test_build_faster_whisper_backend(self):
        backend = build_asr_backend({"backend": "faster_whisper"})
        assert isinstance(backend, FasterWhisperBackend)

    def test_build_funasr_backend_for_qwen3_alias(self):
        backend = build_asr_backend({"backend": "qwen3_asr", "funasr": {}})
        assert isinstance(backend, FunASRBackend)

    def test_funasr_timestamp_result_to_segments(self):
        result = [
            {
                "text": "hello world",
                "sentence_info": [
                    {"start": 0, "end": 1200, "text": "hello"},
                    {"start": 1200, "end": 2500, "text": "world"},
                ],
            }
        ]

        segments = funasr_result_to_segments(result, "missing.wav")

        assert segments == [
            TranscriptSegment(start=0.0, end=1.2, text="hello"),
            TranscriptSegment(start=1.2, end=2.5, text="world"),
        ]


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
        assert recheck["context_segments"] == 10
        assert DEFAULT_CONFIG["song"]["padding"]["max_song_seconds"] == 360.0

    def test_short_recheck_ranges_are_filtered_by_min_song_seconds(self):
        from dd_clip_miner_llm.pipeline import _filter_short_segment_ranges

        segments = [
            TranscriptSegment(start=0.0, end=10.0, text="短空隙"),
            TranscriptSegment(start=10.0, end=90.0, text="足够长"),
            TranscriptSegment(start=90.0, end=100.0, text="足够长结束"),
        ]

        kept, skipped = _filter_short_segment_ranges(
            segments,
            [(0, 0), (1, 2)],
            min_duration_seconds=75.0,
        )

        assert kept == [(1, 2)]
        assert skipped == 1

    def test_recheck_ranges_include_context_but_keep_core_matches(self, tmp_path, monkeypatch):
        from dd_clip_miner_llm.pipeline import _recheck_uncovered_song_segments

        segments = [
            TranscriptSegment(start=0.0, end=10.0, text="左上下文"),
            TranscriptSegment(start=10.0, end=30.0, text="漏识别开始"),
            TranscriptSegment(start=30.0, end=60.0, text="漏识别中间"),
            TranscriptSegment(start=60.0, end=90.0, text="漏识别结束"),
            TranscriptSegment(start=90.0, end=100.0, text="右上下文"),
        ]
        matches = [
            ContentMatch(content_type="song", title="已识别左", segment_indices=[0], confidence=0.9),
            ContentMatch(content_type="song", title="已识别右", segment_indices=[4], confidence=0.9),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "padding": {"min_song_seconds": 0.0},
                "missed_recheck": {
                    "enabled": True,
                    "batch_size": 500,
                    "min_gap_segments": 1,
                    "context_segments": 1,
                },
            },
        })
        seen = {}

        def fake_identify_content(chunk, _config, _recognizer, debug_dir=None):
            seen["range"] = (chunk[0].text, chunk[-1].text)
            seen["debug_dir"] = Path(debug_dir).name
            return [
                ContentMatch(
                    content_type="song",
                    title="二次识别",
                    segment_indices=[0, 1, 2, 3, 4],
                    confidence=0.9,
                )
            ]

        monkeypatch.setattr("dd_clip_miner_llm.llm.identify_content", fake_identify_content)

        result = _recheck_uncovered_song_segments(
            segments,
            config,
            object(),
            matches,
            tmp_path,
        )

        assert seen["range"] == ("左上下文", "右上下文")
        assert seen["debug_dir"] == "000001_000003"
        assert result[-1].segment_indices == [1, 2, 3]


# ============ ffmpeg.py 测试 ============

    def test_overlong_song_recheck_replaces_when_second_pass_splits(self, tmp_path, monkeypatch):
        from dd_clip_miner_llm.pipeline import _recheck_overlong_song_matches

        segments = [
            TranscriptSegment(start=0.0, end=100.0, text="a"),
            TranscriptSegment(start=100.0, end=200.0, text="b"),
            TranscriptSegment(start=200.0, end=300.0, text="c"),
            TranscriptSegment(start=300.0, end=470.0, text="d"),
        ]
        matches = [
            ContentMatch(content_type="song", title="too long", segment_indices=[0, 1, 2, 3], confidence=0.9)
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "padding": {
                    "max_song_seconds": 360.0,
                    "merge_gap_seconds": 20.0,
                },
                "missed_recheck": {"enabled": True, "context_segments": 1},
            },
        })

        def fake_identify_content(_chunk, _config, _recognizer, debug_dir=None):
            return [
                ContentMatch(content_type="song", title="part 1", segment_indices=[0, 1], confidence=0.9),
                ContentMatch(content_type="song", title="part 2", segment_indices=[2, 3], confidence=0.9),
            ]

        monkeypatch.setattr("dd_clip_miner_llm.llm.identify_content", fake_identify_content)

        result = _recheck_overlong_song_matches(segments, config, object(), matches, tmp_path)

        assert [match.title for match in result] == ["part 1", "part 2"]
        assert [match.segment_indices for match in result] == [[0, 1], [2, 3]]

    def test_overlong_song_recheck_keeps_first_pass_when_second_pass_still_overlong(self, tmp_path, monkeypatch):
        from dd_clip_miner_llm.pipeline import _recheck_overlong_song_matches

        segments = [
            TranscriptSegment(start=0.0, end=100.0, text="a"),
            TranscriptSegment(start=100.0, end=200.0, text="b"),
            TranscriptSegment(start=200.0, end=300.0, text="c"),
            TranscriptSegment(start=300.0, end=470.0, text="d"),
        ]
        matches = [
            ContentMatch(content_type="song", title="large model", segment_indices=[0, 1, 2, 3], confidence=0.9)
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "padding": {
                    "max_song_seconds": 360.0,
                    "merge_gap_seconds": 20.0,
                },
                "missed_recheck": {"enabled": True, "context_segments": 1},
            },
        })

        def fake_identify_content(_chunk, _config, _recognizer, debug_dir=None):
            return [
                ContentMatch(content_type="song", title="still too long", segment_indices=[0, 1, 2, 3], confidence=0.9),
            ]

        monkeypatch.setattr("dd_clip_miner_llm.llm.identify_content", fake_identify_content)

        result = _recheck_overlong_song_matches(segments, config, object(), matches, tmp_path)

        assert len(result) == 1
        assert result[0].title == "large model"
        assert result[0].segment_indices == [0, 1, 2, 3]


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

    def test_concat_uses_targeted_reencode_when_health_probe_finds_bad_segment(self, tmp_path, monkeypatch):
        from dd_clip_miner_llm.concat import pipeline as concat_pipeline
        from dd_clip_miner_llm.concat.models import VideoMeta

        calls = []

        def fake_run_command(args, timeout=3600, **_kwargs):
            calls.append(args)

        monkeypatch.setattr(ffmpeg, "require_binary", lambda name: name)
        monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)
        monkeypatch.setattr(
            ffmpeg,
            "_find_bad_h264_segments",
            lambda videos, _bin, tail_seconds=None: [0] if Path(videos[0]).name == "b.mp4" else [],
        )
        monkeypatch.setattr(
            concat_pipeline,
            "probe_many",
            lambda paths: [
                VideoMeta(Path(path), None, True, True, "h264", 640, 360, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo")
                for path in paths
            ],
        )
        monkeypatch.setattr(
            ffmpeg,
            "detect_video_encoders",
            lambda _ffmpeg_bin: {"libx264"},
        )
        monkeypatch.setattr(ffmpeg, "_get_video_fps", lambda _video: 60.0)
        monkeypatch.setattr(ffmpeg, "_has_audio_stream", lambda _video: True)
        monkeypatch.setattr(
            ffmpeg,
            "_concat_remuxed_copy",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ffmpeg.FFmpegError("Invalid NAL in remux")
            ),
        )
        monkeypatch.setattr(ffmpeg, "_get_min_video_size", lambda _videos: (641, 361))

        output = tmp_path / "concat.mp4"
        ffmpeg.concat_videos(
            [tmp_path / "a.mp4", tmp_path / "b.mp4"],
            output,
            video_codec="auto",
            audio_bitrate_kbps=320,
        )

        input_paths = [args[args.index("-i") + 1] for args in calls if "-i" in args]
        assert str(tmp_path / "b.mp4") in input_paths
        assert any("+genpts+igndts+discardcorrupt" in args for args in calls)
        assert any(args[args.index("-c") + 1] == "copy" for args in calls if "-c" in args)
        assert not (tmp_path / "concat_list.txt").exists()

    def test_concat_skips_mixed_repair_when_most_inputs_are_corrupt(self, tmp_path, monkeypatch):
        from dd_clip_miner_llm.concat import pipeline as concat_pipeline
        from dd_clip_miner_llm.concat.models import VideoMeta

        executed = []
        paths = [tmp_path / f"part_{i}.mp4" for i in range(4)]

        monkeypatch.setattr(ffmpeg, "require_binary", lambda name: name)
        monkeypatch.setattr(
            concat_pipeline,
            "probe_many",
            lambda probe_paths: [
                VideoMeta(Path(path), None, True, True, "h264", 640, 360, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo")
                for path in probe_paths
            ],
        )
        monkeypatch.setattr(
            ffmpeg,
            "_find_bad_h264_segments",
            lambda _videos, _bin, tail_seconds=None: [0, 1, 2],
        )
        monkeypatch.setattr(concat_pipeline.TargetedRepairStrategy, "execute", lambda *_args: False)
        monkeypatch.setattr(concat_pipeline.SelectiveNormalizeStrategy, "execute", lambda *_args: False)

        def fake_full_execute(self, context):
            executed.append(self.name)
            context.attempts.append(concat_pipeline.AttemptRecord(self.name, True, "ok"))
            return True

        monkeypatch.setattr(concat_pipeline.FullReencodeStrategy, "execute", fake_full_execute)

        output = tmp_path / "concat.mp4"
        ffmpeg.concat_videos(paths, output, video_codec="auto", audio_bitrate_kbps=320)

        assert executed == ["full reencode (demuxer + filter fallback)"]

    def test_concat_duration_validation_rejects_too_long_output(self, tmp_path, monkeypatch):
        output = tmp_path / "concat.mp4"
        monkeypatch.setattr(ffmpeg, "get_duration", lambda _path: 131.0)
        monkeypatch.setattr(ffmpeg, "_get_stream_duration", lambda *_args: None)

        with pytest.raises(ffmpeg.FFmpegError, match="too long"):
            ffmpeg._validate_concat_duration(output, 100.0)

    def test_concat_duration_validation_rejects_short_video_stream(self, tmp_path, monkeypatch):
        output = tmp_path / "concat.mp4"
        monkeypatch.setattr(ffmpeg, "get_duration", lambda _path: 100.0)
        monkeypatch.setattr(ffmpeg, "_get_stream_duration", lambda *_args: 69.0)

        with pytest.raises(ffmpeg.FFmpegError, match="video stream duration is too short"):
            ffmpeg._validate_concat_duration(output, 100.0)

    def test_output_duration_failure_detection(self):
        from dd_clip_miner_llm.concat.pipeline import _is_output_duration_failure

        assert _is_output_duration_failure("Concat output video stream duration is too short: 1")
        assert not _is_output_duration_failure("Invalid NAL unit size")

    def test_concat_falls_back_to_timestamp_remux_before_audio_reencode(self, tmp_path, monkeypatch):
        calls = []

        def fake_run_command(args, timeout=3600, **_kwargs):
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

        assert calls[1][calls[1].index("-fflags") + 1] == "+genpts+igndts+discardcorrupt"
        assert calls[1][calls[1].index("-err_detect") + 1] == "ignore_err"
        assert calls[1][calls[1].index("-c") + 1] == "copy"
        assert "-async" not in calls[1]
        assert not (tmp_path / "concat_list.txt").exists()

    def test_timestamp_audio_resync_uses_aresample_not_async(self, tmp_path, monkeypatch):
        calls = []

        def fake_run_command(args, timeout=3600, **_kwargs):
            calls.append(args)

        monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)
        output = tmp_path / "out.mp4"
        monkeypatch.setattr(ffmpeg, "get_duration", lambda path: 20.0 if Path(path) == output else 10.0)
        monkeypatch.setattr(ffmpeg, "_validate_audio_decodable", lambda *_args: None)
        monkeypatch.setattr(ffmpeg, "_get_stream_duration", lambda *_args: None)

        ffmpeg._concat_timestamp_remuxed_audio_resync(
            [tmp_path / "a.mp4", tmp_path / "b.mp4"],
            output,
            "ffmpeg",
            320,
            20.0,
        )

        final = calls[-1]
        assert final[final.index("-af") + 1] == "aresample=async=1000:first_pts=0"
        assert "-async" not in final

    def test_concat_keeps_mixed_repair_when_corrupt_duration_is_small(self, tmp_path, monkeypatch):
        from dd_clip_miner_llm.concat import pipeline as concat_pipeline
        from dd_clip_miner_llm.concat.models import VideoMeta

        executed = []
        paths = [tmp_path / f"part_{i}.mp4" for i in range(4)]
        durations = [1000.0, 10.0, 10.0, 10.0]

        monkeypatch.setattr(ffmpeg, "require_binary", lambda name: name)
        monkeypatch.setattr(
            concat_pipeline,
            "probe_many",
            lambda probe_paths: [
                VideoMeta(Path(path), durations[index], True, True, "h264", 640, 360, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo")
                for index, path in enumerate(probe_paths)
            ],
        )
        monkeypatch.setattr(
            ffmpeg,
            "_find_bad_h264_segments",
            lambda _videos, _bin, tail_seconds=None: [1, 2, 3],
        )

        def fake_targeted_execute(self, context):
            executed.append(self.name)
            context.attempts.append(concat_pipeline.AttemptRecord(self.name, True, "ok"))
            return True

        monkeypatch.setattr(concat_pipeline.TargetedRepairStrategy, "execute", fake_targeted_execute)
        monkeypatch.setattr(
            concat_pipeline.FullReencodeStrategy,
            "execute",
            lambda *_args: (_ for _ in ()).throw(AssertionError("full reencode should be skipped")),
        )

        ffmpeg.concat_videos(paths, tmp_path / "concat.mp4", video_codec="auto", audio_bitrate_kbps=320)

        assert executed == ["targeted H.264 repair"]

    def test_duration_failure_boundary_indexes_use_stream_duration(self, tmp_path):
        from dd_clip_miner_llm.concat import pipeline as concat_pipeline
        from dd_clip_miner_llm.concat.models import ConcatContext, ProblemProfile, TargetProfile, VideoMeta

        paths = [tmp_path / f"part_{i}.mp4" for i in range(3)]
        metas = [
            VideoMeta(paths[0], 100.0, True, True, "h264", 640, 360, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo"),
            VideoMeta(paths[1], 50.0, True, True, "h264", 640, 360, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo"),
            VideoMeta(paths[2], 25.0, True, True, "h264", 640, 360, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo"),
        ]
        context = ConcatContext(
            inputs=paths,
            metas=metas,
            output=tmp_path / "out.mp4",
            ffmpeg_bin="ffmpeg",
            video_codec="auto",
            audio_bitrate_kbps=320,
            single_file_policy="copy",
            force_normalize=False,
            expected_duration=175.0,
            target=TargetProfile(width=640, height=360, fps=60.0),
            target_size=(640, 360),
            profile=ProblemProfile(duration_truncated=True),
        )
        context.attempts.append(
            concat_pipeline.AttemptRecord(
                "direct concat copy",
                False,
                "Concat output video stream duration is too short: 100.000s, expected about 175.000s",
            )
        )

        assert concat_pipeline._duration_failure_boundary_indexes(context) == [0, 1]

    def test_full_reencode_candidates_do_not_overwrite_final_until_validated(self, tmp_path, monkeypatch):
        from dd_clip_miner_llm.concat import pipeline as concat_pipeline
        from dd_clip_miner_llm.concat.models import ConcatContext, TargetProfile, VideoMeta

        output = tmp_path / "concat.mp4"
        output.write_bytes(b"existing-final")
        concat_file = tmp_path / "concat_list.txt"
        inputs = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
        metas = [
            VideoMeta(inputs[0], 10.0, True, True, "h264", 640, 360, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo"),
            VideoMeta(inputs[1], 10.0, True, True, "h264", 640, 360, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo"),
        ]
        context = ConcatContext(
            inputs=inputs,
            metas=metas,
            output=output,
            ffmpeg_bin="ffmpeg",
            video_codec="auto",
            audio_bitrate_kbps=320,
            single_file_policy="copy",
            force_normalize=False,
            expected_duration=20.0,
            target=TargetProfile(width=640, height=360, fps=60.0),
            target_size=(640, 360),
            concat_file=concat_file,
        )
        candidates = [["-c:v", "h264_nvenc"]]
        seen_outputs = []

        monkeypatch.setattr(ffmpeg, "_concat_reencode_arg_candidates", lambda *_args: candidates)

        def fake_demuxer(output_path, *_args, **_kwargs):
            seen_outputs.append(Path(output_path))
            Path(output_path).write_bytes(b"candidate-bytes")
            if len(seen_outputs) == 1:
                raise ffmpeg.FFmpegError("Concat output video stream duration is too short: 10.000s")

        monkeypatch.setattr(concat_pipeline, "_concat_demuxer_full_reencode", fake_demuxer)

        def fake_filter_commands(_inputs, output_path, *_args, **_kwargs):
            return [["ffmpeg", "-y", str(output_path)]]

        def fake_run_command(args, timeout=3600, **_kwargs):
            Path(args[-1]).write_bytes(b"filter-candidate-bytes")

        monkeypatch.setattr(ffmpeg, "_concat_filter_commands", fake_filter_commands)
        monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)
        monkeypatch.setattr(concat_pipeline, "_validate_output", lambda *_args, **_kwargs: None)

        assert concat_pipeline.FullReencodeStrategy().execute(context) is True

        assert seen_outputs[0] != output
        assert not seen_outputs[0].exists()
        assert not (tmp_path / "_concat_candidates").exists()
        assert output.read_bytes() == b"filter-candidate-bytes"
        logs = sorted((tmp_path / "concat_attempts").glob("*.log"))
        assert len(logs) == 1
        assert "candidate_00" in logs[0].stem

    def test_concat_short_output_triggers_video_bitstream_scan(self):
        """Legacy internal test for _has_video_bitstream_failure (kept for compat during refactor).

        New code uses ProblemProfile + classify_ffmpeg_output in ConcatPipeline.
        """
        from dd_clip_miner_llm.concat.models import ConcatAttempt
        from dd_clip_miner_llm.concat.pipeline import _has_video_bitstream_failure

        attempts = [
            ConcatAttempt(
                "direct concat copy",
                False,
                "Concat output duration is too short: 7363.447s, expected about 11028.857s",
            )
        ]

        assert _has_video_bitstream_failure(attempts) is True

    def test_ffmpeg_failure_segment_numbers_parse_as_zero_based(self):
        profile = ffmpeg.classify_ffmpeg_output(
            "[concat] Detected possible corrupt H.264 segment(s) 2; "
            "repairing only those segment(s)."
        )

        assert profile.bitstream_corrupt_indexes == [1]

    def test_ffmpeg_failure_invalid_nal_is_bitstream_problem(self):
        profile = ffmpeg.classify_ffmpeg_output(
            "[h264 @ 000001] Invalid NAL unit size (10445 > 2608)."
        )

        assert profile.bitstream_corruption is True
        assert profile.is_bitstream_problem() is True

    def test_ffmpeg_failure_invalid_data_only_is_demux_not_bitstream(self):
        profile = ffmpeg.classify_ffmpeg_output(
            "Error opening input: Invalid data found when processing input"
        )

        assert profile.demux_errors is True
        assert profile.bitstream_corruption is False
        assert profile.is_bitstream_problem() is False

    def test_write_ffconcat_list_does_not_write_duration_by_default(self, tmp_path):
        list_file = tmp_path / "concat.ffconcat"

        ffmpeg._write_ffconcat_list(list_file, [tmp_path / "a.mp4"], [12.345])

        text = list_file.read_text(encoding="utf-8")
        assert "ffconcat version" not in text
        assert "duration" not in text

        ffmpeg._write_ffconcat_list(
            list_file,
            [tmp_path / "a.mp4"],
            [12.345],
            include_duration=True,
        )

        text = list_file.read_text(encoding="utf-8")
        assert text.startswith("ffconcat version 1.0")
        assert "duration 12.345000" in text

    def test_bitstream_scan_selects_bsf_by_codec(self, tmp_path, monkeypatch):
        files = [tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"]
        codec_by_file = {
            files[0]: "h264",
            files[1]: "hevc",
            files[2]: "vp9",
        }
        commands = []

        monkeypatch.setattr(ffmpeg, "_get_video_codec", lambda path: codec_by_file[Path(path)])

        def fake_run(args, **_kwargs):
            commands.append(args)
            return type("Completed", (), {"returncode": 0, "stderr": "Invalid NAL unit size"})()

        monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

        bad = ffmpeg._find_bad_h264_segments(files, "ffmpeg")

        assert bad == [0, 1]
        assert [cmd[cmd.index("-bsf:v") + 1] for cmd in commands] == [
            "h264_mp4toannexb",
            "hevc_mp4toannexb",
        ]

    def test_file_matches_profile_checks_fps_and_optional_audio_bitrate(self, tmp_path):
        from dd_clip_miner_llm.concat.models import TargetProfile, VideoMeta
        from dd_clip_miner_llm.concat.planner import file_matches_profile

        target = TargetProfile(width=640, height=360, fps=60.0, audio_bitrate_kbps=320)
        base = dict(
            path=tmp_path / "a.mp4",
            duration=10.0,
            has_video=True,
            has_audio=True,
            video_codec="h264",
            width=640,
            height=360,
            fps=60.02,
            pix_fmt="yuv420p",
            sar="1:1",
            audio_codec="aac",
            audio_sample_rate=48000,
            audio_channels=2,
            audio_layout="stereo",
        )

        assert file_matches_profile(VideoMeta(**base), target, 320) is True
        assert file_matches_profile(VideoMeta(**{**base, "fps": 59.0}), target, 320) is False
        assert file_matches_profile(VideoMeta(**{**base, "audio_bit_rate": 260000}), target, 320) is False
        assert file_matches_profile(VideoMeta(**{**base, "audio_bit_rate": 300000}), target, 320) is True

    def test_ffmpeg_auto_inserted_bitstream_filter_is_not_corruption(self):
        profile = ffmpeg.classify_ffmpeg_output(
            "[mov,mp4,m4a,3gp,3g2,mj2 @ 000001] "
            "Auto-inserting h264_mp4toannexb bitstream filter"
        )

        assert profile.bitstream_corruption is False
        assert profile.is_bitstream_problem() is False

    def test_health_probe_scans_only_supported_bitstream_inputs(self, tmp_path, monkeypatch):
        from dd_clip_miner_llm.concat.models import VideoMeta
        from dd_clip_miner_llm.concat.pipeline import _build_health_profile

        inputs = [tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"]
        metas = [
            VideoMeta(inputs[0], 1.0, True, True, "hevc", 1920, 1080, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo"),
            VideoMeta(inputs[1], 1.0, True, True, "h264", 1920, 1080, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo"),
            VideoMeta(inputs[2], 1.0, True, True, "vp9", 1920, 1080, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo"),
        ]
        scanned = []

        def fake_find_bad(videos, _ffmpeg_bin, tail_seconds=None):
            scanned.extend(videos)
            return [0] if Path(videos[0]).name == "b.mp4" else []

        monkeypatch.setattr(ffmpeg, "_find_bad_h264_segments", fake_find_bad)

        health = _build_health_profile(inputs, metas, "ffmpeg")

        assert scanned == [inputs[0], inputs[1]]
        assert health[0].is_bitstream_corrupt is False
        assert health[1].is_bitstream_corrupt is True
        assert health[2].is_bitstream_corrupt is False

    def test_selective_normalize_forces_known_corrupt_indexes(self, tmp_path, monkeypatch):
        from dd_clip_miner_llm.concat.models import TargetProfile, VideoMeta
        from dd_clip_miner_llm.concat.pipeline import _selective_normalize_concat

        inputs = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
        metas = [
            VideoMeta(inputs[0], None, True, True, "h264", 640, 360, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo"),
            VideoMeta(inputs[1], None, True, True, "h264", 640, 360, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo"),
        ]
        normalized = []

        def fake_normalize(source, output, *_args, **_kwargs):
            normalized.append(source)
            output.write_bytes(b"normalized")

        monkeypatch.setattr("dd_clip_miner_llm.concat.pipeline._normalize_to_profile", fake_normalize)
        monkeypatch.setattr("dd_clip_miner_llm.concat.pipeline._concat_copy_with_list", lambda *_args, **_kwargs: None)

        _selective_normalize_concat(
            inputs,
            metas,
            TargetProfile(width=640, height=360, fps=60.0),
            (640, 360),
            tmp_path / "out.mp4",
            "ffmpeg",
            "auto",
            320,
            None,
            force_indexes={1},
        )

        assert normalized == [inputs[1]]

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
        assert candidates[-1] == ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]


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

    def test_song_padding_expands_into_long_asr_silence_gap(self):
        segments = [
            TranscriptSegment(start=0.0, end=2.0, text="before"),
            TranscriptSegment(start=54.0, end=60.0, text="song"),
            TranscriptSegment(start=120.0, end=122.0, text="after"),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "padding": {
                    "before_seconds": 15.0,
                    "after_seconds": 15.0,
                    "after_next_asr_end_guard_seconds": 2.0,
                    "adaptive_silence_padding": True,
                    "adaptive_silence_gap_threshold_seconds": 25.0,
                    "adaptive_silence_gap_ratio": 0.5,
                    "adaptive_max_before_seconds": 45.0,
                    "adaptive_max_after_seconds": 45.0,
                    "min_song_seconds": 0.0,
                },
            },
        })
        matches = [
            ContentMatch(
                content_type="song",
                title="adaptive",
                segment_indices=[1],
                confidence=0.9,
            )
        ]

        results = build_content_results(segments, matches, 140.0, config, "song")

        assert len(results) == 1
        assert results[0].start == 28.0
        assert results[0].end == 90.0

    def test_song_adaptive_padding_is_capped(self):
        segments = [
            TranscriptSegment(start=0.0, end=2.0, text="before"),
            TranscriptSegment(start=120.0, end=130.0, text="song"),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "padding": {
                    "before_seconds": 15.0,
                    "adaptive_silence_padding": True,
                    "adaptive_silence_gap_threshold_seconds": 25.0,
                    "adaptive_silence_gap_ratio": 0.8,
                    "adaptive_max_before_seconds": 45.0,
                    "min_song_seconds": 0.0,
                },
            },
        })
        matches = [
            ContentMatch(
                content_type="song",
                title="capped",
                segment_indices=[1],
                confidence=0.9,
            )
        ]

        results = build_content_results(segments, matches, 150.0, config, "song")

        assert len(results) == 1
        assert results[0].start == 75.0

    def test_song_merge_respects_max_song_seconds(self):
        segments = [
            TranscriptSegment(start=0.0, end=180.0, text="第一首"),
            TranscriptSegment(start=190.0, end=370.0, text="第二首"),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "padding": {
                    "before_seconds": 0.0,
                    "after_seconds": 0.0,
                    "min_song_seconds": 0.0,
                    "max_song_seconds": 360.0,
                    "merge_gap_seconds": 20.0,
                },
            },
        })
        matches = [
            ContentMatch(content_type="song", title="同名歌曲", segment_indices=[0], confidence=0.9),
            ContentMatch(content_type="song", title="同名歌曲", segment_indices=[1], confidence=0.9),
        ]

        results = build_content_results(segments, matches, 400.0, config, "song")

        assert len(results) == 2
        assert [result.duration for result in results] == [180.0, 180.0]

    def test_song_match_with_large_internal_gap_is_split_before_merge(self):
        segments = [
            TranscriptSegment(start=0.0, end=100.0, text="第一段"),
            TranscriptSegment(start=100.0, end=180.0, text="第一段结束"),
            TranscriptSegment(start=500.0, end=620.0, text="第二段"),
            TranscriptSegment(start=620.0, end=700.0, text="第二段结束"),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "padding": {
                    "before_seconds": 0.0,
                    "after_seconds": 0.0,
                    "min_song_seconds": 0.0,
                    "max_song_seconds": 360.0,
                    "merge_gap_seconds": 20.0,
                },
            },
        })
        matches = [
            ContentMatch(
                content_type="song",
                title="同一条 LLM 返回",
                segment_indices=[0, 1, 2, 3],
                confidence=0.9,
            )
        ]

        results = build_content_results(segments, matches, 800.0, config, "song")

        assert len(results) == 2
        assert [(result.start, result.end, result.duration) for result in results] == [
            (0.0, 180.0, 180.0),
            (500.0, 700.0, 200.0),
        ]


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
            dict_path = Path(tmp) / "streamer_dictionary.json"
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
