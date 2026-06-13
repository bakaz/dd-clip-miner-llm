"""dd_clip_miner_llm 基础测试"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
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

    def test_load_config_profile_merges_common_and_override(self, tmp_path):
        config_file = tmp_path / "profiles.yaml"
        config_file.write_text(
            """
default_profile: accuracy
llm:
  model: shared-model
  max_tokens: 100
profiles:
  accuracy:
    llm:
      cache_friendly_prompt_layout: false
  kv_optimized:
    llm:
      cache_friendly_prompt_layout: true
      compact_segment_ranges: true
      max_completion_tokens: 32768
""",
            encoding="utf-8",
        )

        accuracy = load_config(config_file)
        optimized = load_config(config_file, profile="kv_optimized")

        assert accuracy["_profile_name"] == "accuracy"
        assert accuracy["llm"]["model"] == "shared-model"
        assert accuracy["llm"]["cache_friendly_prompt_layout"] is False
        assert optimized["_profile_name"] == "kv_optimized"
        assert optimized["llm"]["cache_friendly_prompt_layout"] is True
        assert optimized["llm"]["compact_segment_ranges"] is True
        assert optimized["llm"]["max_completion_tokens"] == 32768

    def test_load_config_rejects_unknown_profile(self, tmp_path):
        config_file = tmp_path / "profiles.yaml"
        config_file.write_text(
            "profiles:\n  accuracy: {}\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Unknown config profile"):
            load_config(config_file, profile="missing")

    def test_shipped_accuracy_profile_keeps_legacy_layout_with_review(self):
        config = load_config("config.deepseek.example.yaml", profile="accuracy")

        assert config["llm"]["cache_friendly_prompt_layout"] is False
        assert config["llm"]["compact_segment_ranges"] is False
        assert config["song"]["review"]["enabled"] is True

    def test_legacy_config_does_not_enable_profile_isolation(self, tmp_path):
        config_file = tmp_path / "legacy.yaml"
        config_file.write_text("llm:\n  model: legacy\n", encoding="utf-8")

        config = load_config(config_file)

        assert config["llm"]["model"] == "legacy"
        assert "_profile_name" not in config

    def test_load_config_rejects_reserved_profile_all(self, tmp_path):
        config_file = tmp_path / "profiles.yaml"
        config_file.write_text("profiles:\n  accuracy: {}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="reserved CLI value"):
            load_config(config_file, profile="all")

    def test_list_profile_names_puts_default_first(self, tmp_path):
        from dd_clip_miner_llm.config import list_profile_names

        loaded = {
            "default_profile": "kv_optimized",
            "profiles": {"accuracy": {}, "kv_optimized": {}},
        }
        assert list_profile_names(loaded) == ["kv_optimized", "accuracy"]

    def test_accuracy_profile_declares_local_review_scope(self):
        config = load_config("config.example.yaml", profile="accuracy")

        assert config["song"]["review"]["transcript_scope"] == "local"

    def test_profile_name_does_not_apply_hidden_song_overrides(self, tmp_path):
        config_file = tmp_path / "profiles.yaml"
        config_file.write_text(
            """
profiles:
  accuracy:
    song:
      review:
        transcript_scope: full
      missed_recheck:
        strategy: full_transcript
""",
            encoding="utf-8",
        )

        config = load_config(config_file, profile="accuracy")

        assert config["song"]["review"]["transcript_scope"] == "full"
        assert config["song"]["missed_recheck"]["strategy"] == "full_transcript"

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

    def test_resolve_asr_model_name_for_funasr(self):
        from dd_clip_miner_llm.asr_backends import resolve_asr_model_name

        name = resolve_asr_model_name({
            "backend": "funasr",
            "model": "small",
            "funasr": {"model": "Qwen/Qwen3-ASR-0.6B"},
        })
        assert name == "Qwen/Qwen3-ASR-0.6B"

    def test_apply_asr_model_override_updates_funasr_subconfig(self):
        from dd_clip_miner_llm.asr_backends import apply_asr_model_override

        asr = {
            "mode": "local",
            "local": {"backend": "funasr", "funasr": {"model": "old-model"}},
        }
        apply_asr_model_override(asr, "new-model")
        assert asr["model"] == "new-model"
        assert asr["local"]["funasr"]["model"] == "new-model"

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

    def test_resolve_hardware_gpu_section(self, monkeypatch):
        from dd_clip_miner_llm.asr_backends import _resolve_hardware_local_config
        monkeypatch.setattr("dd_clip_miner_llm.asr_backends._is_gpu_available", lambda: True)
        cfg = {
            "faster_whisper": {"model": "small", "device": "auto"},
            "gpu": {"faster_whisper": {"device": "cuda", "compute_type": "float16"}},
            "cpu": {"faster_whisper": {"device": "cpu", "compute_type": "int8"}},
        }
        resolved = _resolve_hardware_local_config(cfg)
        assert resolved["faster_whisper"]["device"] == "cuda"
        assert resolved["faster_whisper"]["compute_type"] == "float16"

    def test_resolve_hardware_cpu_section_fallback(self, monkeypatch):
        from dd_clip_miner_llm.asr_backends import _resolve_hardware_local_config
        monkeypatch.setattr("dd_clip_miner_llm.asr_backends._is_gpu_available", lambda: False)
        cfg = {
            "faster_whisper": {"model": "small"},
            "cpu": {"faster_whisper": {"device": "cpu", "compute_type": "int8", "cpu_threads": 4}},
        }
        resolved = _resolve_hardware_local_config(cfg)
        assert resolved["faster_whisper"]["device"] == "cpu"
        assert resolved["faster_whisper"]["compute_type"] == "int8"

    def test_resolve_nested_hardware_section(self, monkeypatch):
        from dd_clip_miner_llm.asr_backends import _resolve_hardware_local_config

        monkeypatch.setattr("dd_clip_miner_llm.asr_backends._is_gpu_available", lambda: False)
        cfg = {
            "backend": "faster_whisper",
            "faster_whisper": {
                "model": "small",
                "device": "auto",
                "cpu": {"device": "cpu", "compute_type": "int8"},
            },
        }

        resolved = _resolve_hardware_local_config(cfg)

        assert resolved["faster_whisper"]["device"] == "cpu"
        assert resolved["faster_whisper"]["compute_type"] == "int8"

    def test_explicit_device_is_not_overridden(self, monkeypatch):
        from dd_clip_miner_llm.asr_backends import _resolve_hardware_local_config

        monkeypatch.setattr("dd_clip_miner_llm.asr_backends._is_gpu_available", lambda: True)
        cfg = {
            "backend": "faster_whisper",
            "faster_whisper": {"device": "cpu"},
            "gpu": {"faster_whisper": {"device": "cuda"}},
        }

        resolved = _resolve_hardware_local_config(cfg)

        assert resolved["faster_whisper"]["device"] == "cpu"

    def test_build_with_hardware_sections(self, monkeypatch):
        from dd_clip_miner_llm.asr_backends import build_asr_backend, _resolve_hardware_local_config
        monkeypatch.setattr("dd_clip_miner_llm.asr_backends._is_gpu_available", lambda: False)
        cfg = {
            "mode": "local",
            "local": {
                "backend": "faster_whisper",
                "faster_whisper": {"model": "small", "device": "auto"},
                "cpu": {"faster_whisper": {"device": "cpu", "compute_type": "int8"}},
            }
        }
        backend = build_asr_backend(cfg["local"])
        assert isinstance(backend, FasterWhisperBackend)
        # after resolution in build, settings should have cpu values
        assert backend.settings.get("device") == "cpu"
        assert backend.settings.get("compute_type") == "int8"

    def test_batched_transcription_requests_timestamps_and_splits_word_gaps(self):
        backend = FasterWhisperBackend({
            "batch_size": 8,
            "word_gap_seconds": 2.0,
            "max_segment_seconds": 15.0,
        })
        calls = {}
        words = [
            SimpleNamespace(start=1.0, end=1.5, word="你"),
            SimpleNamespace(start=1.5, end=2.0, word="好"),
            SimpleNamespace(start=10.0, end=10.5, word="下"),
            SimpleNamespace(start=10.5, end=11.0, word="句"),
        ]
        segment = SimpleNamespace(start=1.0, end=11.0, text="你好下句", words=words)

        class BatchedModel:
            def transcribe(self, _audio_path, **kwargs):
                calls.update(kwargs)
                return [segment], object()

        backend._get_batched_model = lambda: BatchedModel()

        result = backend.transcribe("audio.wav")

        assert calls["without_timestamps"] is False
        assert calls["word_timestamps"] is True
        assert result == [
            TranscriptSegment(start=1.0, end=2.0, text="你好"),
            TranscriptSegment(start=10.0, end=11.0, text="下句"),
        ]


class TestSongMissedRecheck:
    def test_uncovered_segment_ranges(self):
        from dd_clip_miner_llm.song_postprocess import _uncovered_segment_ranges

        matches = [
            ContentMatch(content_type="song", title="A", segment_indices=[1, 2], confidence=0.9),
            ContentMatch(content_type="song", title="B", segment_indices=[4], confidence=0.8),
        ]

        assert _uncovered_segment_ranges(6, matches) == [(0, 0), (3, 3), (5, 5)]

    def test_split_segment_ranges(self):
        from dd_clip_miner_llm.song_postprocess import _split_segment_ranges

        assert _split_segment_ranges([(0, 4), (8, 9)], 2) == [
            (0, 1),
            (2, 3),
            (4, 4),
            (8, 9),
        ]

    def test_windowed_finalize_dedupes_splits_and_merges_same_title(self):
        from dd_clip_miner_llm.song_postprocess import _finalize_windowed_missed_recheck_matches

        segments = [
            TranscriptSegment(start=0.0, end=10.0, text="a"),
            TranscriptSegment(start=12.0, end=22.0, text="b"),
            TranscriptSegment(start=50.0, end=60.0, text="c"),
            TranscriptSegment(start=62.0, end=72.0, text="d"),
        ]
        matches = [
            ContentMatch(content_type="song", title="known", segment_indices=[0], confidence=0.9),
        ]
        extra_matches = [
            ContentMatch(content_type="song", title="known", segment_indices=[1], confidence=0.8),
            ContentMatch(content_type="song", title="other", segment_indices=[2, 3], confidence=0.7),
            ContentMatch(content_type="song", title="known", segment_indices=[1], confidence=0.8),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {"padding": {"merge_gap_seconds": 20.0}},
        })

        finalized, summary = _finalize_windowed_missed_recheck_matches(
            segments,
            config,
            matches,
            extra_matches,
        )

        assert summary["before_count"] == 4
        assert [(match.title, match.segment_indices) for match in finalized] == [
            ("known", [0, 1]),
            ("other", [2, 3]),
        ]
        assert any(event["type"] == "exact_duplicate" for event in summary["normalization_events"])

    def test_group_segment_ranges_combines_targets_within_request_span(self):
        from dd_clip_miner_llm.song_postprocess import _group_segment_ranges

        assert _group_segment_ranges(
            [(0, 20), (40, 60), (490, 499), (500, 520)],
            500,
        ) == [
            [(0, 20), (40, 60), (490, 499)],
            [(500, 520)],
        ]

    def test_default_config_enables_song_missed_recheck(self):
        recheck = DEFAULT_CONFIG["song"]["missed_recheck"]

        assert recheck["enabled"] is True
        assert recheck["strategy"] == "windowed"
        assert recheck["fallback_strategy"] == "windowed_on_structural_failure"
        assert recheck["batch_size"] == 500
        assert recheck["context_segments"] == 10
        assert recheck["max_completion_tokens"] == 32768
        assert recheck["max_tool_rounds"] == 1
        assert DEFAULT_CONFIG["song"]["padding"]["max_song_seconds"] == 360.0

    def test_shipped_kv_profile_uses_fixed_risk_routed_pipeline(self):
        config = load_config("config.example.yaml", profile="kv_optimized")

        assert config["song"]["pipeline"]["strategy"] == "risk_routed_v3"
        assert config["song"]["pipeline"]["runtime_adaptive"] == "fixed_three_stage"
        assert config["song"]["pipeline"]["stages"] == {
            "discovery": "precision",
            "recall_audit": "uncovered_evidence",
            "adjudication": "full_transcript",
        }
        assert config["song"]["pipeline"]["continuation_overlap_segments"] == 50
        assert config["song"]["pipeline"]["anchor_boundary_expansion"] is False
        assert config["song"]["search"]["enabled"] is False
        assert config["song"]["missed_recheck"]["enabled"] is True
        assert config["song"]["review"]["enabled"] is False
        assert config["song"]["review"]["transcript_scope"] == "local"
        assert config["song"]["padding"]["merge_gap_seconds"] == 40.0

    def test_accuracy_profile_keeps_legacy_song_pipeline(self):
        config = load_config("config.example.yaml", profile="accuracy")

        assert config["song"]["pipeline"]["strategy"] == "legacy"
        assert config["song"]["review"]["transcript_scope"] == "local"
        assert config["song"]["review"]["enabled"] is True
        assert config["song"]["missed_recheck"]["strategy"] == "windowed"
        assert config["song"]["missed_recheck"]["enabled"] is True
        assert config["song"]["padding"]["merge_gap_seconds"] == 40.0

    def test_song_normalization_splits_disjoint_ranges_and_deduplicates(self):
        from dd_clip_miner_llm.song_postprocess import _normalize_song_matches

        segments = [
            TranscriptSegment(start=float(i * 10), end=float(i * 10 + 5), text=str(i))
            for i in range(8)
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "padding": {
                    "merge_gap_seconds": 20.0,
                    "max_song_seconds": 360.0,
                }
            }
        })
        matches = [
            ContentMatch(
                content_type="song",
                title="下雨天",
                segment_indices=[0, 1, 6, 7],
                confidence=0.9,
            ),
            ContentMatch(
                content_type="song",
                title="下雨天",
                segment_indices=[0, 1],
                confidence=0.9,
            ),
        ]

        normalized, events, suspicious = _normalize_song_matches(
            segments,
            config,
            matches,
        )

        assert [match.segment_indices for match in normalized] == [[0, 1], [6, 7]]
        assert any(event["type"] == "disjoint_ranges" for event in events)
        assert any(event["type"] == "exact_duplicate" for event in events)
        assert len(suspicious) == 2

    def test_review_clusters_connect_different_title_overlaps(self):
        from dd_clip_miner_llm.song_postprocess import _build_song_review_clusters

        matches = [
            ContentMatch(content_type="song", title="七月七日晴", segment_indices=[1, 2], confidence=0.9),
            ContentMatch(content_type="song", title="说了再见", segment_indices=[2, 3], confidence=0.8),
            ContentMatch(content_type="song", title="天后", segment_indices=[8, 9], confidence=0.9),
        ]

        clusters = _build_song_review_clusters(matches, set())

        assert len(clusters) == 1
        assert {match.title for match in clusters[0]} == {"七月七日晴", "说了再见"}

    def test_review_clusters_group_nearby_suspicious_ranges_from_same_title(self):
        from dd_clip_miner_llm.song_postprocess import _build_song_review_clusters, _match_key

        matches = [
            ContentMatch(content_type="song", title="天后", segment_indices=[10, 11], confidence=0.9),
            ContentMatch(content_type="song", title="天后", segment_indices=[20, 21], confidence=0.9),
            ContentMatch(content_type="song", title="天后", segment_indices=[800, 801], confidence=0.9),
        ]
        suspicious = {_match_key(match) for match in matches}

        clusters = _build_song_review_clusters(
            matches,
            suspicious,
            max_span_segments=100,
        )

        assert [[match.segment_indices for match in cluster] for cluster in clusters] == [
            [[10, 11], [20, 21]],
            [[800, 801]],
        ]

    def test_review_clusters_connect_adjacent_different_titles_when_enabled(self):
        from dd_clip_miner_llm.song_postprocess import _build_song_review_clusters

        matches = [
            ContentMatch(content_type="song", title="七月七日晴", segment_indices=[999, 1002], confidence=0.85),
            ContentMatch(content_type="song", title="说了再见", segment_indices=[1003, 1059], confidence=0.9),
            ContentMatch(content_type="song", title="天后", segment_indices=[1200, 1250], confidence=0.9),
        ]

        clusters = _build_song_review_clusters(
            matches,
            set(),
            nearby_title_conflict_gap_segments=2,
        )

        assert len(clusters) == 1
        assert {match.title for match in clusters[0]} == {"七月七日晴", "说了再见"}

    def test_review_clusters_leave_adjacent_different_titles_disabled_by_default(self):
        from dd_clip_miner_llm.song_postprocess import _build_song_review_clusters

        matches = [
            ContentMatch(content_type="song", title="七月七日晴", segment_indices=[1, 2], confidence=0.9),
            ContentMatch(content_type="song", title="说了再见", segment_indices=[3, 4], confidence=0.8),
        ]

        assert _build_song_review_clusters(matches, set()) == []

    def test_local_best_prefers_known_higher_confidence_candidate(self):
        from dd_clip_miner_llm.song_postprocess import _local_best_song_cluster

        segments = [
            TranscriptSegment(start=0.0, end=30.0, text="a"),
            TranscriptSegment(start=30.0, end=60.0, text="b"),
        ]
        cluster = [
            ContentMatch(content_type="song", title="未知歌曲：一句歌词", segment_indices=[0, 1], confidence=0.95),
            ContentMatch(content_type="song", title="七月七日晴", segment_indices=[0, 1], confidence=0.9),
        ]

        selected, decisions = _local_best_song_cluster(
            segments,
            DEFAULT_CONFIG,
            cluster,
            set(),
        )

        assert [match.title for match in selected] == ["七月七日晴"]
        assert any(item["action"] == "discard" for item in decisions)

    def test_review_cluster_audit_does_not_bypass_request_cache(
        self,
        tmp_path,
        monkeypatch,
    ):
        from dd_clip_miner_llm.song_postprocess import _review_song_matches
        from dd_clip_miner_llm.recognizers import get_recognizer

        segments = [
            TranscriptSegment(start=0.0, end=10.0, text="a"),
            TranscriptSegment(start=10.0, end=20.0, text="b"),
            TranscriptSegment(start=20.0, end=30.0, text="c"),
        ]
        matches = [
            ContentMatch("song", "A", [0, 1], 0.9),
            ContentMatch("song", "B", [1, 2], 0.8),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "review": {
                    "enabled": True,
                    "context_segments": 0,
                    "max_window_segments": 500,
                },
            },
        })
        review_dir = tmp_path / "review" / "before_missed_recheck"
        review_dir.mkdir(parents=True)
        (review_dir / "cluster_001.json").write_text(
            json.dumps({
                "before": [match.to_dict() for match in matches],
                "resolution": "llm",
                "after": [matches[0].to_dict()],
            }),
            encoding="utf-8",
        )
        called = {"value": False}

        def fake_identify(
            _segments,
            _config,
            _recognizer,
            debug_dir=None,
            **_kwargs,
        ):
            called["value"] = True
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "llm_batch_000000.json").write_text(
                json.dumps({
                    "error": None,
                    "parse_valid": True,
                    "parsed_items": [],
                }),
                encoding="utf-8",
            )
            return [matches[0]]

        monkeypatch.setattr(
            "dd_clip_miner_llm.llm.identify_content",
            fake_identify,
        )

        result = _review_song_matches(
            segments,
            config,
            get_recognizer("song"),
            matches,
            tmp_path,
            phase="before_missed_recheck",
        )

        assert called["value"] is True
        assert [match.title for match in result] == ["A"]

    def test_profile_state_requires_matching_fingerprints(self, tmp_path):
        from dd_clip_miner_llm.profile_state import _profile_state_matches, _write_profile_state

        state_path = tmp_path / "profile.json"
        config = deep_merge(DEFAULT_CONFIG, {
            "_profile_name": "accuracy",
            "_profile_enabled": True,
        })
        input_path = tmp_path / "input.mp4"
        _write_profile_state(
            state_path,
            input_path=input_path,
            config=config,
            config_fingerprint="config-a",
            transcript_fingerprint="asr-a",
            status="complete",
        )

        assert _profile_state_matches(
            state_path,
            input_path=input_path,
            config_fingerprint="config-a",
            transcript_fingerprint="asr-a",
        )
        assert not _profile_state_matches(
            state_path,
            input_path=input_path,
            config_fingerprint="config-b",
            transcript_fingerprint="asr-a",
        )

    def test_profile_comparison_report_is_written_for_two_profiles(self, tmp_path):
        from dd_clip_miner_llm.profile_state import _write_profile_comparison

        for name, title, hit, miss in [
            ("accuracy", "七月七日晴", 10, 90),
            ("kv_optimized", "七月七日晴", 98, 2),
        ]:
            profile_dir = tmp_path / name
            song_dir = profile_dir / "song"
            song_dir.mkdir(parents=True)
            (profile_dir / "profile.json").write_text(
                json.dumps({"profile": name, "status": "complete", "model": "same-model"}),
                encoding="utf-8",
            )
            (song_dir / "matches.json").write_text(
                json.dumps([{
                    "title": title,
                    "segment_indices": [1, 2],
                }]),
                encoding="utf-8",
            )
            (song_dir / "llm_batch_000000.json").write_text(
                json.dumps({
                    "usage": [{
                        "prompt_cache_hit_tokens": hit,
                        "prompt_cache_miss_tokens": miss,
                        "completion_tokens": 5,
                    }]
                }),
                encoding="utf-8",
            )

        _write_profile_comparison(tmp_path)

        comparison = json.loads(
            (tmp_path / "profile_comparison.json").read_text(encoding="utf-8")
        )
        assert comparison["profiles"]["accuracy"]["cache_hit_ratio"] == 0.1
        assert comparison["profiles"]["kv_optimized"]["cache_hit_ratio"] == 0.98
        assert (tmp_path / "profile_comparison.md").exists()

    def test_short_recheck_ranges_are_filtered_by_min_song_seconds(self):
        from dd_clip_miner_llm.song_postprocess import _filter_short_segment_ranges

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
        from dd_clip_miner_llm.song_postprocess import _recheck_uncovered_song_segments
        from dd_clip_miner_llm.recognizers import get_recognizer

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
        stale_dir = tmp_path / "missed_recheck" / "000001_000003"
        stale_dir.mkdir(parents=True)
        (stale_dir / "llm_batch_000000.json").write_text(
            json.dumps({
                "error": None,
                "parse_valid": True,
                "parsed_items": [{
                    "content_type": "song",
                    "title": "stale",
                    "segment_indices": [1, 2, 3],
                    "confidence": 0.9,
                }],
            }),
            encoding="utf-8",
        )
        seen = {}

        def fake_identify_content(chunk, _config, _recognizer, debug_dir=None, **_kwargs):
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
            get_recognizer("song"),
            matches,
            tmp_path,
        )

        assert seen["range"] == ("左上下文", "右上下文")
        assert seen["debug_dir"] == "000001_000003"
        rechecked = [match for match in result if match.title == "二次识别"]
        assert len(rechecked) == 1
        assert rechecked[0].segment_indices == [1, 2, 3]

    def test_full_transcript_audit_keeps_main_asr_prefix_and_tools(self, sample_segments, sample_config):
        from dd_clip_miner_llm.llm import build_llm_messages
        from dd_clip_miner_llm.song_postprocess import _SongCoverageAuditRecognizer
        from dd_clip_miner_llm.recognizers import get_recognizer

        sample_config["llm"]["cache_friendly_prompt_layout"] = True
        sample_config["llm"]["compact_segment_ranges"] = True
        recognizer = get_recognizer("song")
        audit = _SongCoverageAuditRecognizer(recognizer, [(1, 2)], [])

        main_messages = build_llm_messages(recognizer, sample_segments, 0, sample_config)
        audit_messages = build_llm_messages(audit, sample_segments, 0, sample_config)

        marker = "ASR 转写结束。"
        assert main_messages[0] == audit_messages[0]
        assert main_messages[1]["content"].split(marker, 1)[0] == (
            audit_messages[1]["content"].split(marker, 1)[0]
        )
        assert recognizer.get_tools(sample_config) == audit.get_tools(sample_config)
        assert '"title":' not in audit_messages[1]["content"]
        assert "[[1,2]]" in audit_messages[1]["content"]

    def test_full_review_shares_main_asr_prefix_and_tools(self, sample_segments, sample_config):
        from dd_clip_miner_llm.llm import build_llm_messages
        from dd_clip_miner_llm.song_postprocess import _SongFullReviewRecognizer
        from dd_clip_miner_llm.recognizers import get_recognizer

        sample_config["llm"]["cache_friendly_prompt_layout"] = True
        recognizer = get_recognizer("song")
        cluster = [ContentMatch("song", "A", [1, 2], 0.8)]
        review = _SongFullReviewRecognizer(
            recognizer,
            cluster,
            target_start=1,
            target_end=2,
            allowed_start=0,
            allowed_end=2,
        )

        main_messages = build_llm_messages(recognizer, sample_segments, 0, sample_config)
        review_messages = build_llm_messages(review, sample_segments, 0, sample_config)

        marker = "ASR 转写结束。"
        assert main_messages[0] == review_messages[0]
        assert main_messages[1]["content"].split(marker, 1)[0] == (
            review_messages[1]["content"].split(marker, 1)[0]
        )
        assert recognizer.get_tools(sample_config) == review.get_tools(sample_config)
        assert "[[1,2]]" in review_messages[1]["content"]
        assert "[[0,2]]" in review_messages[1]["content"]

    def test_filter_matches_to_segment_ranges_rejects_covered_indices(self):
        from dd_clip_miner_llm.song_postprocess import _filter_matches_to_segment_ranges

        matches = [
            ContentMatch(
                content_type="song",
                title="mixed",
                segment_indices=[1, 2, 3, 8, 9],
                confidence=0.9,
            )
        ]

        filtered = _filter_matches_to_segment_ranges(matches, [(2, 3), (9, 9)])

        assert len(filtered) == 1
        assert filtered[0].segment_indices == [2, 3, 9]

    def test_full_transcript_merges_adjacent_same_title_results(self):
        from dd_clip_miner_llm.song_postprocess import _merge_adjacent_same_title_matches

        segments = [
            TranscriptSegment(start=0.0, end=10.0, text="a"),
            TranscriptSegment(start=12.0, end=20.0, text="b"),
            TranscriptSegment(start=45.0, end=55.0, text="c"),
        ]
        matches = [
            ContentMatch("song", "same", [0], 0.8),
            ContentMatch("song", "same", [1], 0.9),
            ContentMatch("song", "same", [2], 0.7),
        ]

        merged, events = _merge_adjacent_same_title_matches(
            segments,
            matches,
            max_gap_seconds=20.0,
        )

        assert [match.segment_indices for match in merged] == [[0, 1], [2]]
        assert merged[0].confidence == 0.9
        assert len(events) == 1

    def test_full_transcript_sanitizes_review_output(self):
        from dd_clip_miner_llm.song_postprocess import (
            _sanitize_full_transcript_review_results,
        )

        segments = [
            TranscriptSegment(start=float(i * 10), end=float(i * 10 + 8), text=str(i))
            for i in range(8)
        ]
        candidates = [ContentMatch("song", "known", [1, 2], 0.8)]
        reviewed = [
            ContentMatch("song", "known", [1, 2, 3], 0.9),
            ContentMatch("song", "new", [4, 5, 6], 0.8),
            ContentMatch("song", "对话片段", [4], 1.0),
            ContentMatch("speech", "looks named", [4], 1.0),
            ContentMatch("song", "outside", [7], 0.7),
        ]

        sanitized, events = _sanitize_full_transcript_review_results(
            segments,
            deep_merge(DEFAULT_CONFIG, {}),
            reviewed,
            candidates,
            [(4, 5)],
        )

        assert [(match.title, match.segment_indices) for match in sanitized] == [
            ("known", [1, 2, 3]),
            ("new", [4, 5]),
        ]
        assert {event["type"] for event in events} >= {
            "invalid_audit_result",
            "review_result_cropped_to_audit_targets",
            "review_result_outside_audit_targets",
        }

    def test_full_transcript_empty_array_does_not_fallback(self, tmp_path, monkeypatch):
        from dd_clip_miner_llm.song_postprocess import _recheck_uncovered_song_segments
        from dd_clip_miner_llm.recognizers import get_recognizer

        segments = [
            TranscriptSegment(start=float(i * 30), end=float((i + 1) * 30), text=str(i))
            for i in range(4)
        ]
        matches = [
            ContentMatch(
                content_type="song",
                title="covered",
                segment_indices=[0],
                confidence=0.9,
            )
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "llm": {
                "cache_friendly_prompt_layout": True,
                "compact_segment_ranges": True,
            },
            "song": {
                "padding": {"min_song_seconds": 0.0},
                "missed_recheck": {
                    "strategy": "full_transcript",
                    "fallback_strategy": "windowed_on_structural_failure",
                },
            },
        })

        def fake_identify_content(_segments, _config, _recognizer, debug_dir=None, **_kwargs):
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "llm_batch_000000.json").write_text(
                json.dumps({
                    "error": None,
                    "parse_valid": True,
                    "parsed_items": [],
                    "json_fix_rounds": [],
                    "tool_rounds": [{"finish_reason": "stop"}],
                    "usage": [],
                }),
                encoding="utf-8",
            )
            return []

        def fail_windowed(*_args, **_kwargs):
            raise AssertionError("windowed fallback must not run for a valid empty array")

        monkeypatch.setattr("dd_clip_miner_llm.llm.identify_content", fake_identify_content)
        monkeypatch.setattr(
            "dd_clip_miner_llm.song_postprocess.recheck._run_windowed_missed_recheck",
            fail_windowed,
        )

        result = _recheck_uncovered_song_segments(
            segments,
            config,
            get_recognizer("song"),
            matches,
            tmp_path,
        )

        audit = json.loads(
            (tmp_path / "missed_recheck" / "audit.json").read_text(encoding="utf-8")
        )
        assert result == matches
        assert audit["status"] == "success"
        assert audit["fallback_used"] is False
        assert audit["additional_match_count"] == 0

    @pytest.mark.parametrize(
        ("payload", "expected_reason"),
        [
            ({"error": "network", "parse_valid": False}, "api_error"),
            ({"error": None, "parse_valid": False}, "invalid_result_json"),
            ({
                "error": None,
                "parse_valid": True,
                "tool_rounds": [{"finish_reason": "length"}],
            }, "output_truncated"),
            ({
                "error": None,
                "parse_valid": True,
                "json_fix_rounds": [{"round": 1}],
            }, "json_repair"),
        ],
    )
    def test_full_transcript_structural_failure_falls_back(
        self,
        tmp_path,
        monkeypatch,
        payload,
        expected_reason,
    ):
        from dd_clip_miner_llm.song_postprocess import _recheck_uncovered_song_segments
        from dd_clip_miner_llm.recognizers import get_recognizer

        segments = [
            TranscriptSegment(start=0.0, end=30.0, text="covered"),
            TranscriptSegment(start=30.0, end=120.0, text="target"),
        ]
        matches = [
            ContentMatch(
                content_type="song",
                title="covered",
                segment_indices=[0],
                confidence=0.9,
            )
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "llm": {
                "cache_friendly_prompt_layout": True,
                "compact_segment_ranges": True,
            },
            "song": {
                "padding": {"min_song_seconds": 0.0},
                "missed_recheck": {
                    "strategy": "full_transcript",
                    "fallback_strategy": "windowed_on_structural_failure",
                },
            },
        })

        def fake_identify_content(_segments, _config, _recognizer, debug_dir=None, **_kwargs):
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_payload = {
                "parsed_items": [],
                "json_fix_rounds": [],
                "tool_rounds": [{"finish_reason": "stop"}],
                "usage": [],
                **payload,
            }
            (debug_dir / "llm_batch_000000.json").write_text(
                json.dumps(debug_payload),
                encoding="utf-8",
            )
            return []

        fallback_match = ContentMatch(
            content_type="song",
            title="fallback",
            segment_indices=[1],
            confidence=0.8,
        )

        def fake_windowed(*_args, **_kwargs):
            return [fallback_match], ["000001_000001/llm_batch_000000.json"]

        monkeypatch.setattr("dd_clip_miner_llm.llm.identify_content", fake_identify_content)
        monkeypatch.setattr(
            "dd_clip_miner_llm.song_postprocess.recheck._run_windowed_missed_recheck",
            fake_windowed,
        )

        result = _recheck_uncovered_song_segments(
            segments,
            config,
            get_recognizer("song"),
            matches,
            tmp_path,
        )

        audit = json.loads(
            (tmp_path / "missed_recheck" / "audit.json").read_text(encoding="utf-8")
        )
        assert result[-1].title == "fallback"
        assert audit["status"] == "fallback_success"
        assert audit["fallback_used"] is True
        assert expected_reason in audit["structural_failures"]

    def test_missed_recheck_fingerprint_changes_for_each_input(self):
        from dd_clip_miner_llm.song_postprocess import _missed_recheck_fingerprint

        segments = [TranscriptSegment(start=0.0, end=10.0, text="a")]
        matches = [
            ContentMatch(
                content_type="song",
                title="song",
                segment_indices=[0],
                confidence=0.9,
            )
        ]
        base = _missed_recheck_fingerprint(
            segments,
            DEFAULT_CONFIG,
            matches,
            [(0, 0)],
        )
        changed_transcript = _missed_recheck_fingerprint(
            [TranscriptSegment(start=0.0, end=10.0, text="b")],
            DEFAULT_CONFIG,
            matches,
            [(0, 0)],
        )
        changed_candidates = _missed_recheck_fingerprint(
            segments,
            DEFAULT_CONFIG,
            [
                ContentMatch(
                    content_type="song",
                    title="other",
                    segment_indices=[0],
                    confidence=0.9,
                )
            ],
            [(0, 0)],
        )
        changed_ranges = _missed_recheck_fingerprint(
            segments,
            DEFAULT_CONFIG,
            matches,
            [(1, 1)],
        )
        changed_config = _missed_recheck_fingerprint(
            segments,
            deep_merge(DEFAULT_CONFIG, {"llm": {"model": "other"}}),
            matches,
            [(0, 0)],
        )

        assert base["transcript"] != changed_transcript["transcript"]
        assert base["candidates"] != changed_candidates["candidates"]
        assert base["target_ranges"] != changed_ranges["target_ranges"]
        assert base["config"] != changed_config["config"]

    def test_usage_summary_counts_only_valid_debug_files(self, tmp_path):
        from dd_clip_miner_llm.profile_state import _write_usage_summary, _write_valid_debug_manifest

        profile_dir = tmp_path / "accuracy"
        song_dir = profile_dir / "song"
        song_dir.mkdir(parents=True)
        (song_dir / "llm_batch_000000.json").write_text(
            json.dumps({
                "phase": "main",
                "usage": [{
                    "prompt_cache_hit_tokens": 100,
                    "prompt_cache_miss_tokens": 50,
                    "completion_tokens": 10,
                }],
                "parse_valid": True,
            }),
            encoding="utf-8",
        )
        (song_dir / "active_debug_files.json").write_text(
            json.dumps(["llm_batch_000000.json"]),
            encoding="utf-8",
        )
        (song_dir / "llm_batch_000500.json").write_text(
            json.dumps({
                "phase": "main",
                "usage": [{
                    "prompt_cache_hit_tokens": 777,
                    "prompt_cache_miss_tokens": 777,
                    "completion_tokens": 777,
                }],
            }),
            encoding="utf-8",
        )
        stale_dir = song_dir / "review" / "before_missed_recheck" / "cluster_001"
        stale_dir.mkdir(parents=True)
        (stale_dir / "llm_batch_000000.json").write_text(
            json.dumps({
                "phase": "review_before",
                "usage": [{
                    "prompt_cache_hit_tokens": 999,
                    "prompt_cache_miss_tokens": 999,
                    "completion_tokens": 999,
                }],
                "parse_valid": True,
            }),
            encoding="utf-8",
        )
        stale_overlong = song_dir / "overlong_recheck" / "stale"
        stale_overlong.mkdir(parents=True)
        (stale_overlong / "llm_batch_000000.json").write_text(
            json.dumps({
                "phase": "overlong",
                "usage": [{
                    "prompt_cache_hit_tokens": 888,
                    "prompt_cache_miss_tokens": 888,
                    "completion_tokens": 888,
                }],
            }),
            encoding="utf-8",
        )
        _write_valid_debug_manifest(song_dir)

        summary = _write_usage_summary(profile_dir)

        assert summary["phases"]["main"]["calls"] == 1
        assert summary["phases"]["main"]["prompt_cache_hit_tokens"] == 100
        assert "review_before" not in summary["phases"]
        assert summary["totals"]["prompt_cache_hit_tokens"] == 100

    def test_profile_usage_totals_respects_valid_debug_manifest(self, tmp_path):
        from dd_clip_miner_llm.profile_state import _profile_usage_totals

        song_dir = tmp_path / "song"
        song_dir.mkdir()
        for name, miss in [("active.json", 10), ("stale.json", 999)]:
            (song_dir / name).write_text(
                json.dumps({
                    "usage": [{
                        "prompt_cache_hit_tokens": 20,
                        "prompt_cache_miss_tokens": miss,
                        "completion_tokens": 3,
                    }]
                }),
                encoding="utf-8",
            )
        (song_dir / "valid_debug_files.json").write_text(
            json.dumps(["active.json"]),
            encoding="utf-8",
        )

        totals = _profile_usage_totals(tmp_path)

        assert totals == {
            "prompt_cache_hit_tokens": 20,
            "prompt_cache_miss_tokens": 10,
            "completion_tokens": 3,
        }

    def test_valid_debug_manifest_skips_unchanged_review_clusters(self, tmp_path):
        from dd_clip_miner_llm.profile_state import _write_valid_debug_manifest

        review_dir = tmp_path / "review" / "after_missed_recheck"
        for index in (1, 2):
            cluster_dir = review_dir / f"cluster_{index:03d}"
            cluster_dir.mkdir(parents=True, exist_ok=True)
            (cluster_dir / "llm_batch_000000.json").write_text(
                json.dumps({"usage": []}),
                encoding="utf-8",
            )
            (cluster_dir / "active_debug_files.json").write_text(
                json.dumps(["llm_batch_000000.json"]),
                encoding="utf-8",
            )
        (review_dir / "summary.json").write_text(
            json.dumps({
                "cluster_count": 2,
                "clusters": [
                    {"cluster": 1, "resolution": "llm"},
                    {
                        "cluster": 2,
                        "resolution": "unchanged_pre_audit_cluster",
                    },
                ],
            }),
            encoding="utf-8",
        )

        _write_valid_debug_manifest(tmp_path)

        manifest = json.loads(
            (tmp_path / "valid_debug_files.json").read_text(encoding="utf-8")
        )
        assert manifest == [
            "review/after_missed_recheck/cluster_001/llm_batch_000000.json"
        ]

    def test_offset_recognizer_forwards_system_prompt(self):
        from dd_clip_miner_llm.song_postprocess import _OffsetRecognizer

        class Recognizer:
            name = "song"
            default_config = {}

            def build_system_prompt(self, config):
                return f"system:{config['value']}"

        offset = _OffsetRecognizer(Recognizer(), 10)

        assert offset.build_system_prompt({"value": "ok"}) == "system:ok"

    def test_cached_identify_matches_loads_valid_debug(self, tmp_path):
        from dd_clip_miner_llm.llm import (
            build_llm_messages,
            build_providers,
            build_request_debug_metadata,
        )
        from dd_clip_miner_llm.song_postprocess import _load_cached_identify_matches
        from dd_clip_miner_llm.recognizers import get_recognizer

        segments = [
            TranscriptSegment(start=0.0, end=1.0, text="a"),
            TranscriptSegment(start=1.0, end=2.0, text="b"),
            TranscriptSegment(start=2.0, end=3.0, text="c"),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "llm": {
                "api_key": "test-key",
                "use_tools": False,
            },
        })
        recognizer = get_recognizer("song")
        provider = build_providers(config)[0]
        messages = build_llm_messages(recognizer, segments, 0, config)
        metadata = build_request_debug_metadata(
            messages,
            config=config,
            provider=provider,
            recognizer=recognizer,
            segments=segments,
            batch_start=0,
            tools=recognizer.get_tools(config),
            debug_phase="overlong",
        )
        debug_dir = tmp_path / "debug"
        debug_dir.mkdir()
        (debug_dir / "llm_batch_000000.json").write_text(
            json.dumps({
                **metadata,
                "error": None,
                "parse_valid": True,
                "tool_rounds": [],
                "json_fix_rounds": [],
                "reasoning_followups": [],
                "parsed_items": [{
                    "content_type": "song",
                    "title": "天后",
                    "segment_indices": [1, 2],
                    "confidence": 0.9,
                }],
            }),
            encoding="utf-8",
        )

        matches = _load_cached_identify_matches(
            debug_dir,
            recognizer,
            config,
            segments,
            debug_phase="overlong",
        )

        assert matches is not None
        assert [match.title for match in matches] == ["天后"]
        changed_segments = [
            *segments[:2],
            TranscriptSegment(start=2.0, end=3.0, text="changed"),
        ]
        assert _load_cached_identify_matches(
            debug_dir,
            recognizer,
            config,
            changed_segments,
            debug_phase="overlong",
        ) is None

    def test_usage_summary_includes_multiple_content_types(self, tmp_path):
        from dd_clip_miner_llm.profile_state import _write_usage_summary

        for content_type, miss in (("song", 10), ("dialogue", 20)):
            content_dir = tmp_path / content_type
            content_dir.mkdir()
            (content_dir / "llm_batch_000000.json").write_text(
                json.dumps({
                    "phase": "main",
                    "usage": [{
                        "prompt_cache_hit_tokens": 100,
                        "prompt_cache_miss_tokens": miss,
                        "completion_tokens": 5,
                    }],
                }),
                encoding="utf-8",
            )
            (content_dir / "valid_debug_files.json").write_text(
                json.dumps(["llm_batch_000000.json"]),
                encoding="utf-8",
            )

        summary = _write_usage_summary(tmp_path)

        assert summary["totals"]["calls"] == 2
        assert summary["totals"]["prompt_cache_miss_tokens"] == 30
        assert set(summary["content_types"]) == {"song", "dialogue"}

    def test_valid_debug_manifest_includes_anchor_recheck(self, tmp_path):
        from dd_clip_miner_llm.profile_state import _write_valid_debug_manifest

        anchor_dir = tmp_path / "anchor_recheck"
        anchor_dir.mkdir(parents=True)
        debug_file = anchor_dir / "llm_batch_000000.json"
        debug_file.write_text(
            json.dumps({"parse_valid": True, "usage": []}),
            encoding="utf-8",
        )
        (anchor_dir / "active_debug_files.json").write_text(
            json.dumps([debug_file.name]),
            encoding="utf-8",
        )

        _write_valid_debug_manifest(tmp_path)

        manifest = json.loads(
            (tmp_path / "valid_debug_files.json").read_text(encoding="utf-8")
        )
        assert manifest == ["anchor_recheck/llm_batch_000000.json"]

    def test_review_scope_gate_requires_each_named_song(self):
        from scripts.review_scope_ab import _required_title_checks

        partial = _required_title_checks(["七月七日晴", "雨天", "天后"])
        complete = _required_title_checks([
            "七月七日晴",
            "说了再见",
            "雨天",
            "下雨天",
            "天后",
        ])

        assert partial["说了再见"] is False
        assert partial["下雨天"] is False
        assert all(complete.values())
        assert _required_title_checks([*complete, "天后"])["天后"] is False

    def test_review_scope_gate_accepts_traditional_title_variants(self):
        from scripts.review_scope_ab import _required_title_checks

        checks = _required_title_checks([
            "七月七日晴",
            "說了再見",
            "雨天",
            "下雨天",
            "天后",
        ])

        assert all(checks.values())


# ============ ffmpeg.py 测试 ============

    def test_overlong_song_recheck_replaces_when_second_pass_splits(self, tmp_path, monkeypatch):
        from dd_clip_miner_llm.song_postprocess import _recheck_overlong_song_matches
        from dd_clip_miner_llm.recognizers import get_recognizer

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
        stale_dir = tmp_path / "overlong_recheck" / "000000_000003"
        stale_dir.mkdir(parents=True)
        (stale_dir / "llm_batch_000000.json").write_text(
            json.dumps({
                "error": None,
                "parse_valid": True,
                "parsed_items": [{
                    "content_type": "song",
                    "title": "stale",
                    "segment_indices": [0, 1, 2, 3],
                    "confidence": 0.9,
                }],
            }),
            encoding="utf-8",
        )

        def fake_identify_content(_chunk, _config, _recognizer, debug_dir=None, **_kwargs):
            return [
                ContentMatch(content_type="song", title="part 1", segment_indices=[0, 1], confidence=0.9),
                ContentMatch(content_type="song", title="part 2", segment_indices=[2, 3], confidence=0.9),
            ]

        monkeypatch.setattr("dd_clip_miner_llm.llm.identify_content", fake_identify_content)

        result = _recheck_overlong_song_matches(
            segments,
            config,
            get_recognizer("song"),
            matches,
            tmp_path,
        )

        assert [match.title for match in result] == ["part 1", "part 2"]
        assert [match.segment_indices for match in result] == [[0, 1], [2, 3]]

    def test_overlong_song_recheck_keeps_first_pass_when_second_pass_still_overlong(self, tmp_path, monkeypatch):
        from dd_clip_miner_llm.song_postprocess import _recheck_overlong_song_matches

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

        def fake_identify_content(_chunk, _config, _recognizer, debug_dir=None, **_kwargs):
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
            "dd_clip_miner_llm.concat.runner.probe_many",
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
            "dd_clip_miner_llm.concat.runner.probe_many",
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

    def test_tail_window_repair_rejects_bogus_fixed_duration(self, tmp_path, monkeypatch):
        input_video = tmp_path / "bad.mp4"
        output = tmp_path / "concat.mp4"
        output.write_bytes(b"existing-final")

        def fake_run_command(args, timeout=3600, **_kwargs):
            Path(args[-1]).write_bytes(b"candidate")

        def fake_duration(path):
            return 1000.0 if Path(path).name == "fixed.mp4" else 100.0

        monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)
        monkeypatch.setattr(ffmpeg, "get_duration", fake_duration)
        monkeypatch.setattr(ffmpeg, "_get_stream_duration", lambda *_args: None)
        monkeypatch.setattr(ffmpeg, "_get_video_fps", lambda _video: 60.0)
        monkeypatch.setattr(ffmpeg, "_repair_video_filter_args", lambda _fps: [])
        monkeypatch.setattr(ffmpeg, "_has_audio_stream", lambda _video: True)
        monkeypatch.setattr(ffmpeg, "_targeted_repair_encode_candidates", lambda *_args: [["-c:v", "libx264"]])

        with pytest.raises(ffmpeg.FFmpegError, match="too long"):
            ffmpeg._concat_tail_window_repaired_bad_segments_copy(
                [input_video],
                output,
                "ffmpeg",
                "auto",
                320,
                100.0,
                bad_indexes=[0],
            )

        assert output.read_bytes() == b"existing-final"

    def test_output_duration_failure_detection(self):
        from dd_clip_miner_llm.concat.pipeline import _is_output_duration_failure

        assert _is_output_duration_failure("Concat output video stream duration is too short: 1")
        assert not _is_output_duration_failure("Invalid NAL unit size")

    def test_concat_falls_back_to_discard_corrupt_copy_after_direct_copy_fails(self, tmp_path, monkeypatch):
        from dd_clip_miner_llm.concat import pipeline as concat_pipeline

        calls = []
        copy_attempts = 0

        def fake_run_command(args, timeout=3600, **_kwargs):
            calls.append(args)
            nonlocal copy_attempts
            if "-f" in args and "concat" in args and "-c" in args and args[args.index("-c") + 1] == "copy":
                copy_attempts += 1
                if copy_attempts == 1:
                    raise ffmpeg.FFmpegError("copy failed")
            Path(args[-1]).write_bytes(b"ok")

        monkeypatch.setattr(ffmpeg, "require_binary", lambda name: name)
        monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)
        monkeypatch.setattr(concat_pipeline.MkvMergeStrategy, "execute", lambda *_args: False)
        monkeypatch.setattr(concat_pipeline.SelectiveNormalizeStrategy, "execute", lambda *_args: False)
        monkeypatch.setattr(
            ffmpeg,
            "_find_bad_h264_segments",
            lambda _videos, _bin, tail_seconds=None: [],
        )
        monkeypatch.setattr(ffmpeg, "_get_min_video_size", lambda _videos: (641, 361))
        monkeypatch.setattr(ffmpeg, "get_duration", lambda _path: 10.0)
        monkeypatch.setattr(ffmpeg, "_get_stream_duration", lambda *_args: 10.0)
        monkeypatch.setattr(ffmpeg, "_validate_audio_decodable", lambda *_args: None)

        output = tmp_path / "concat.mp4"
        ffmpeg.concat_videos(
            [tmp_path / "a.mp4", tmp_path / "b.mp4"],
            output,
            video_codec="auto",
            audio_bitrate_kbps=320,
        )

        discard_call = next(
            call for call in calls
            if "-fflags" in call and "+discardcorrupt" in call[call.index("-fflags") + 1]
        )
        assert discard_call[discard_call.index("-c") + 1] == "copy"
        assert "-async" not in discard_call
        assert not (tmp_path / "concat_list.txt").exists()

    def test_timestamp_audio_resync_uses_aresample_not_async(self, tmp_path, monkeypatch):
        calls = []

        def fake_run_command(args, timeout=3600, **_kwargs):
            calls.append(args)
            # Create the output file so shutil.move works
            Path(args[-1]).write_bytes(b"fake-output")

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
            "dd_clip_miner_llm.concat.runner.probe_many",
            lambda probe_paths: [
                VideoMeta(Path(path), durations[index], True, True, "h264", 640, 360, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo")
                for index, path in enumerate(probe_paths)
            ],
        )
        def fake_find_bad_h264(videos, _bin, tail_seconds=None):
            bad: list[int] = []
            for index, video in enumerate(videos):
                name = Path(video).name
                if name in {"part_1.mp4", "part_2.mp4", "part_3.mp4"}:
                    bad.append(index)
            return bad

        monkeypatch.setattr(ffmpeg, "_find_bad_h264_segments", fake_find_bad_h264)
        monkeypatch.setattr(concat_pipeline.MkvMergeStrategy, "execute", lambda *_args: False)

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

        monkeypatch.setattr("dd_clip_miner_llm.concat.strategies._concat_demuxer_full_reencode", fake_demuxer)

        def fake_filter_commands(_inputs, output_path, *_args, **_kwargs):
            return [["ffmpeg", "-y", str(output_path)]]

        def fake_run_command(args, timeout=3600, **_kwargs):
            Path(args[-1]).write_bytes(b"filter-candidate-bytes")

        monkeypatch.setattr(ffmpeg, "_concat_filter_commands", fake_filter_commands)
        monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)
        monkeypatch.setattr("dd_clip_miner_llm.concat.strategies._validate_output", lambda *_args, **_kwargs: None)

        assert concat_pipeline.FullReencodeStrategy().execute(context) is True

        assert seen_outputs[0] != output
        assert not seen_outputs[0].exists()
        assert not (tmp_path / "_concat_candidates").exists()
        assert output.read_bytes() == b"filter-candidate-bytes"
        logs = sorted((tmp_path / "concat_attempts").glob("*.log"))
        assert len(logs) == 1
        assert "candidate_00" in logs[0].stem

    def test_targeted_repair_candidate_does_not_overwrite_final_until_validated(self, tmp_path, monkeypatch):
        output = tmp_path / "concat.mp4"
        output.write_bytes(b"existing-final")
        inputs = [tmp_path / "good.mp4", tmp_path / "bad.mp4"]

        monkeypatch.setattr(ffmpeg, "_find_bad_h264_segments", lambda *_args, **_kwargs: [1])
        monkeypatch.setattr(ffmpeg, "_targeted_repair_encode_candidates", lambda *_args: [["-c:v", "libx264"]])
        monkeypatch.setattr(ffmpeg, "_get_video_fps", lambda _video: 60.0)
        monkeypatch.setattr(ffmpeg, "_repair_video_filter_args", lambda _fps: [])
        monkeypatch.setattr(ffmpeg, "_has_audio_stream", lambda _video: True)
        monkeypatch.setattr(ffmpeg, "_validate_audio_decodable", lambda *_args: None)
        monkeypatch.setattr(ffmpeg, "get_duration", lambda _path: 10.0)

        def fake_run_command(args, timeout=3600, **_kwargs):
            Path(args[-1]).write_bytes(b"candidate-bytes")

        monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)
        monkeypatch.setattr(
            ffmpeg,
            "_validate_concat_duration",
            lambda *_args: (_ for _ in ()).throw(ffmpeg.FFmpegError("Concat output duration is too long")),
        )

        with pytest.raises(ffmpeg.FFmpegError, match="Targeted concat repair failed"):
            ffmpeg._concat_reencoded_bad_segments_copy(
                inputs,
                output,
                "ffmpeg",
                "auto",
                320,
                20.0,
                bad_indexes=[1],
            )

        assert output.read_bytes() == b"existing-final"
        assert list(tmp_path.glob("_concat_repair_*")) == []

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

        monkeypatch.setattr("dd_clip_miner_llm.concat.helpers._normalize_to_profile", fake_normalize)
        monkeypatch.setattr("dd_clip_miner_llm.concat.helpers._concat_copy_with_list", lambda *_args, **_kwargs: None)

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

    def test_selective_normalize_can_force_cpu_for_corrupt_indexes(self, tmp_path, monkeypatch):
        from dd_clip_miner_llm.concat.models import TargetProfile, VideoMeta
        from dd_clip_miner_llm.concat.pipeline import _selective_normalize_concat

        inputs = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
        metas = [
            VideoMeta(inputs[0], None, True, True, "h264", 640, 360, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo"),
            VideoMeta(inputs[1], None, True, True, "h264", 640, 360, 60.0, "yuv420p", "1:1", "aac", 48000, 2, "stereo"),
        ]
        normalized = []

        def fake_normalize(source, output, *_args, prefer_cpu=False, **_kwargs):
            normalized.append((source, prefer_cpu))
            output.write_bytes(b"normalized")

        monkeypatch.setattr("dd_clip_miner_llm.concat.helpers._normalize_to_profile", fake_normalize)
        monkeypatch.setattr("dd_clip_miner_llm.concat.helpers._concat_copy_with_list", lambda *_args, **_kwargs: None)

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
            cpu_only_indexes={1},
        )

        assert normalized == [(inputs[1], True)]

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

    def test_targeted_repair_prefers_cpu_for_corrupt_streams(self, monkeypatch):
        monkeypatch.setattr(
            ffmpeg,
            "detect_video_encoders",
            lambda _ffmpeg_bin: {"h264_nvenc", "h264_qsv", "libx264"},
        )

        candidates = ffmpeg._targeted_repair_encode_candidates("ffmpeg", "auto")

        # 硬件编码优先，CPU 作为最终回退
        assert candidates[0][1] == "h264_nvenc"
        assert candidates[1][1] == "h264_qsv"
        assert candidates[-1] == ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"]


class TestBatch:
    def test_batch_concat_skips_generated_merged_outputs(self, tmp_path, monkeypatch):
        from dd_clip_miner_llm import batch

        source = tmp_path / "source"
        source.mkdir()
        first = source / "a_fix.mp4"
        second = source / "b_fix.mp4"
        generated_default = source / "merged_output.mp4"
        generated_dated = source / "merged_2026_06_07.mp4"
        for path in (first, second, generated_default, generated_dated):
            path.write_bytes(b"video")

        seen: list[Path] = []

        def fake_concat(video_paths, output, **_kwargs):
            seen.extend(video_paths)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"concat")

        monkeypatch.setattr(batch, "concat_videos", fake_concat)
        monkeypatch.setattr(batch, "run_pipeline", lambda *_args, **_kwargs: {"song": []})

        batch.run_batch(
            source,
            tmp_path / "results",
            tmp_path / "runs",
            {"output": {"concat_videos": True}},
        )

        assert seen == [first, second]

    def test_cleanup_concat_source_removes_concat_intermediates(self, tmp_path):
        from dd_clip_miner_llm.batch import _cleanup_concat_source

        concat_dir = tmp_path / "concat"
        concat_dir.mkdir()
        concat_file = concat_dir / "concat.mp4"
        concat_file.write_bytes(b"video")
        list_file = concat_dir / "concat_list.txt"
        list_file.write_text("file 'a.mp4'\n", encoding="utf-8")

        _cleanup_concat_source(concat_dir)

        assert concat_file.exists()
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

    def test_build_parser_profile(self):
        from dd_clip_miner_llm.cli import build_parser

        parser = build_parser()
        run_args = parser.parse_args(["run", "test.mp4", "--profile", "kv_optimized"])
        batch_args = parser.parse_args([
            "batch-run",
            "input/",
            "--result-root",
            "output/",
            "--profile",
            "accuracy",
        ])

        assert run_args.profile == "kv_optimized"
        assert batch_args.profile == "accuracy"

    def test_generated_config_enables_fixed_risk_strategy_for_kv_profile(self):
        import yaml

        from dd_clip_miner_llm.cli import _generate_config_yaml

        config = yaml.safe_load(_generate_config_yaml())

        missed = config["profiles"]["kv_optimized"]["song"]["missed_recheck"]
        review = config["profiles"]["kv_optimized"]["song"]["review"]
        pipeline = config["profiles"]["kv_optimized"]["song"]["pipeline"]
        assert pipeline["strategy"] == "risk_routed_v3"
        assert pipeline["stages"]["discovery"] == "precision"
        assert pipeline["stages"]["recall_audit"] == "uncovered_evidence"
        assert pipeline["stages"]["adjudication"] == "full_transcript"
        assert pipeline["continuation_overlap_segments"] == 50
        assert pipeline["anchor_boundary_expansion"] is False
        assert missed["enabled"] is True
        assert missed["min_uncovered_seconds"] == 10.0
        assert review["enabled"] is False
        assert config["llm"]["max_completion_tokens"] == 32768
        assert config["llm"]["final_tool_max_tokens"] == 32768
        assert config["llm"]["continuation_on_length"] is True
        assert config["llm"]["max_continuation_rounds"] == 8
        assert config["song"]["review"]["max_completion_tokens"] == 32768
        assert config["song"]["missed_recheck"]["max_completion_tokens"] == 32768
        assert config["song"]["padding"]["merge_gap_seconds"] == 40.0

    @pytest.mark.parametrize(
        "path",
        ["config.yaml", "config.example.yaml", "config.deepseek.example.yaml"],
    )
    def test_shipped_profile_configs_keep_only_shared_values_common(self, path):
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        common_llm = raw["llm"]
        common_song = raw["song"]

        for key in (
            "cache_friendly_prompt_layout",
            "compact_segment_ranges",
        ):
            assert key not in common_llm
        assert common_llm["max_completion_tokens"] == 32768
        assert common_llm["final_tool_max_tokens"] == 32768
        assert common_llm["continuation_on_length"] is True
        assert common_llm["max_continuation_rounds"] == 8
        assert "pipeline" not in common_song
        assert "naming" not in common_song
        assert "enabled" not in common_song["review"]
        assert "enabled" not in common_song["missed_recheck"]
        assert "strategy" not in common_song["missed_recheck"]

        accuracy = load_config(path, profile="accuracy")
        optimized = load_config(path, profile="kv_optimized")
        assert accuracy["llm"]["max_completion_tokens"] == 32768
        assert optimized["llm"]["max_completion_tokens"] == 32768
        assert accuracy["song"]["review"]["max_completion_tokens"] == 32768
        assert optimized["song"]["missed_recheck"]["max_completion_tokens"] == 32768
        assert accuracy["song"]["padding"]["merge_gap_seconds"] == 40.0
        assert optimized["song"]["padding"]["merge_gap_seconds"] == 40.0

    def test_adaptive_review_scope_prefers_local_for_few_clusters(self):
        from dd_clip_miner_llm.song_adaptive import resolve_review_transcript_scope

        config = {
            "_profile_name": "kv_optimized",
            "llm": {"cache_friendly_prompt_layout": True},
            "song": {
                "review": {
                    "transcript_scope": "adaptive",
                    "adaptive": {"mode": "heuristic"},
                }
            },
        }
        scope, reason, _details = resolve_review_transcript_scope(
            config,
            cluster_count=2,
            segment_count=3211,
        )
        assert scope == "local"
        assert reason.startswith("adaptive_clusters_le_")

    def test_adaptive_review_scope_prefers_full_for_many_clusters(self):
        from dd_clip_miner_llm.song_adaptive import resolve_review_transcript_scope

        config = {
            "_profile_name": "kv_optimized",
            "llm": {"cache_friendly_prompt_layout": True},
            "song": {
                "review": {
                    "transcript_scope": "adaptive",
                    "adaptive": {"mode": "heuristic"},
                }
            },
        }
        scope, reason, _details = resolve_review_transcript_scope(
            config,
            cluster_count=9,
            segment_count=2875,
        )
        assert scope == "full"
        assert "clusters_ge_6" in reason
        assert "segments_ge_2000" in reason

    def test_adaptive_review_scope_forces_local_on_accuracy_profile(self):
        from dd_clip_miner_llm.song_adaptive import resolve_review_transcript_scope

        config = {
            "_profile_name": "accuracy",
            "llm": {"cache_friendly_prompt_layout": False},
            "song": {"review": {"transcript_scope": "adaptive"}},
        }
        scope, reason, _details = resolve_review_transcript_scope(
            config,
            cluster_count=12,
            segment_count=3211,
        )
        assert scope == "local"
        assert reason == "accuracy_profile_forced_local"

    def test_adaptive_missed_recheck_prefers_windowed_for_many_targets(self):
        from dd_clip_miner_llm.song_adaptive import resolve_missed_recheck_strategy

        config = {
            "_profile_name": "kv_optimized",
            "llm": {"cache_friendly_prompt_layout": True},
            "song": {
                "missed_recheck": {
                    "strategy": "adaptive",
                    "adaptive": {"mode": "heuristic"},
                }
            },
        }
        strategy, reason, _details = resolve_missed_recheck_strategy(
            config,
            segment_count=2875,
            target_range_count=20,
        )
        assert strategy == "windowed"
        assert "target_ranges_ge_" in reason

    def test_adaptive_missed_recheck_prefers_full_for_kv_friendly_stream(self):
        from dd_clip_miner_llm.song_adaptive import resolve_missed_recheck_strategy

        config = {
            "_profile_name": "kv_optimized",
            "llm": {"cache_friendly_prompt_layout": True},
            "song": {
                "missed_recheck": {
                    "strategy": "adaptive",
                    "adaptive": {"mode": "heuristic"},
                }
            },
        }
        strategy, reason, _details = resolve_missed_recheck_strategy(
            config,
            segment_count=2875,
            target_range_count=14,
        )
        assert strategy == "full_transcript"
        assert reason == "adaptive_kv_friendly_full_audit"

    def test_adaptive_missed_recheck_forces_windowed_on_accuracy_profile(self):
        from dd_clip_miner_llm.song_adaptive import resolve_missed_recheck_strategy

        config = {
            "_profile_name": "accuracy",
            "llm": {"cache_friendly_prompt_layout": False},
            "song": {"missed_recheck": {"strategy": "adaptive"}},
        }
        strategy, reason, _details = resolve_missed_recheck_strategy(
            config,
            segment_count=2875,
            target_range_count=14,
        )
        assert strategy == "windowed"
        assert reason == "accuracy_profile_forced_windowed"

    def test_joint_adaptive_strategies_picks_minimum_total(self):
        from dd_clip_miner_llm.models import ContentMatch, TranscriptSegment
        from dd_clip_miner_llm.song_adaptive import resolve_song_adaptive_strategies

        segments = [
            TranscriptSegment(start=float(i), end=float(i + 1), text=f"歌词片段{i}" * 12)
            for i in range(2875)
        ]
        clusters = []
        for index in range(9):
            start = 100 + index * 280
            end = start + 479
            clusters.append([
                ContentMatch(
                    content_type="song",
                    title=f"未知歌曲：测试{index}",
                    artist="",
                    segment_indices=list(range(start, end)),
                    confidence=0.5,
                    tags=[],
                    description="",
                ),
            ])
        config = {
            "_profile_name": "kv_optimized",
            "llm": {"cache_friendly_prompt_layout": True},
            "song": {
                "review": {
                    "transcript_scope": "adaptive",
                    "context_segments": 10,
                    "max_window_segments": 500,
                    "adaptive": {"mode": "cost_estimate"},
                },
                "missed_recheck": {
                    "strategy": "adaptive",
                    "adaptive": {"mode": "cost_estimate"},
                },
            },
        }
        joint = resolve_song_adaptive_strategies(
            config,
            clusters=clusters,
            segments=segments,
            matches=[],
            target_ranges=[(10, 40), (500, 530)],
        )
        assert joint["resolution_mode"] == "joint_cost_estimate"
        assert len(joint["combinations"]) == 4
        min_combo = min(joint["combinations"], key=lambda item: item["total_usd"])
        assert joint["review_scope_resolved"] == min_combo["review_scope"]
        assert joint["missed_strategy_resolved"] == min_combo["missed_strategy"]
        assert joint["chosen_total_usd"] == min_combo["total_usd"]
        assert "main_cost_usd" in joint
        assert "overlong_cost_usd" in joint
        assert "review_before_cost_usd" in joint
        assert "review_after_cost_usd" in joint
        assert min_combo["main_cost_usd"] == joint["main_cost_usd"]

    def test_pipeline_joint_cost_includes_review_after_discount_for_full_transcript(self):
        from dd_clip_miner_llm.models import ContentMatch, TranscriptSegment
        from dd_clip_miner_llm.song_adaptive_cost import (
            estimate_review_after_cost,
            estimate_review_cost,
        )

        segments = [
            TranscriptSegment(start=float(i), end=float(i + 1), text=f"歌词片段{i}" * 12)
            for i in range(400)
        ]
        cluster = [
            ContentMatch(
                content_type="song",
                title="未知歌曲：测试",
                artist="",
                segment_indices=list(range(120, 180)),
                confidence=0.5,
                tags=[],
                description="",
            ),
        ]
        config = {
            "_profile_name": "kv_optimized",
            "llm": {"cache_friendly_prompt_layout": True},
            "song": {
                "review": {
                    "enabled": True,
                    "transcript_scope": "adaptive",
                    "context_segments": 10,
                    "max_window_segments": 500,
                    "adaptive": {"mode": "cost_estimate"},
                },
            },
        }
        before = estimate_review_cost(
            config,
            scope="local",
            clusters=[cluster],
            segments=segments,
        )
        after_windowed = estimate_review_after_cost(
            config,
            scope="local",
            missed_strategy="windowed",
            clusters=[cluster],
            segments=segments,
            target_ranges=[(120, 180)],
        )
        after_full = estimate_review_after_cost(
            config,
            scope="local",
            missed_strategy="full_transcript",
            clusters=[cluster],
            segments=segments,
            target_ranges=[(120, 180)],
        )
        assert after_windowed.total_usd == before.total_usd
        assert after_full.total_usd == 0.0

    def test_adaptive_review_cost_estimate_prefers_full_for_many_clusters(self):
        from dd_clip_miner_llm.models import ContentMatch, TranscriptSegment
        from dd_clip_miner_llm.song_adaptive import resolve_review_transcript_scope

        segments = [
            TranscriptSegment(start=float(i), end=float(i + 1), text=f"歌词片段{i}" * 12)
            for i in range(2875)
        ]
        clusters = []
        for index in range(9):
            start = 100 + index * 280
            end = start + 479
            clusters.append([
                ContentMatch(
                    content_type="song",
                    title=f"未知歌曲：测试{index}",
                    artist="",
                    segment_indices=list(range(start, end)),
                    confidence=0.5,
                    tags=[],
                    description="",
                ),
            ])
        config = {
            "_profile_name": "kv_optimized",
            "llm": {"cache_friendly_prompt_layout": True},
            "song": {
                "review": {
                    "transcript_scope": "adaptive",
                    "context_segments": 10,
                    "max_window_segments": 500,
                    "adaptive": {"mode": "cost_estimate"},
                }
            },
        }
        scope, reason, details = resolve_review_transcript_scope(
            config,
            clusters=clusters,
            segments=segments,
        )
        assert scope == "full"
        assert reason.startswith("adaptive_cost_")
        assert details["adaptive_mode"] == "cost_estimate"
        assert details["cost_estimate_full_usd"] < details["cost_estimate_local_usd"]

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

    def test_song_compact_segment_ranges_expand_to_indices(self):
        from dd_clip_miner_llm.recognizers import get_recognizer

        song = get_recognizer("song")
        matches = song.parse_response(
            [{
                "content_type": "song",
                "title": "Test Song",
                "segment_ranges": [[12, 14], [18, 18], [20, 19], ["bad", 22]],
                "confidence": 0.9,
            }],
            {},
        )

        assert len(matches) == 1
        assert matches[0].segment_indices == [12, 13, 14, 18]

    def test_song_risk_duration_is_soft_review_signal(self):
        from dd_clip_miner_llm.song_postprocess import score_song_match_risks

        segments = [
            TranscriptSegment(start=float(i * 10), end=float(i * 10 + 8), text="lyric")
            for i in range(10)
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {"pipeline": {"strategy": "risk_routed_v2"}}
        })
        match = ContentMatch("song", "Known", list(range(10)), 0.9)

        records, suspicious = score_song_match_risks(
            segments, config, [match], source="main",
        )

        assert records[0].action.startswith("review")
        assert "duration_below_soft_range" in records[0].reasons
        assert len(suspicious) == 1
        assert match.segment_indices == list(range(10))

    def test_unknown_song_does_not_trigger_review_by_title_alone(self):
        from dd_clip_miner_llm.song_postprocess import score_song_match_risks

        segments = [
            TranscriptSegment(float(i * 10), float(i * 10 + 9), "lyric")
            for i in range(20)
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {"pipeline": {"strategy": "risk_routed_v2"}}
        })
        match = ContentMatch("song", "未知歌曲：代表歌词", list(range(20)), 0.9)

        records, suspicious = score_song_match_risks(
            segments, config, [match], source="main",
        )

        assert records[0].action == "accept"
        assert "unknown_title" in records[0].reasons
        assert len(suspicious) == 0

    def test_previous_review_keys_are_loaded_from_audit(self, tmp_path):
        from dd_clip_miner_llm.song_postprocess.review import _load_previously_reviewed_keys

        root = tmp_path / "review" / "before_missed_recheck"
        root.mkdir(parents=True)
        match = ContentMatch("song", "Song", [1, 2], 0.8)
        (root / "cluster_001.json").write_text(
            json.dumps({"after": [match.to_dict()]}), encoding="utf-8",
        )

        assert _load_previously_reviewed_keys(tmp_path) == {("song", (1, 2))}

    def test_song_risk_boundary_repair_splits_time_gap(self):
        from dd_clip_miner_llm.song_postprocess import repair_song_boundaries

        segments = [
            TranscriptSegment(0.0, 5.0, "a"),
            TranscriptSegment(6.0, 10.0, "b"),
            TranscriptSegment(60.0, 65.0, "c"),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "pipeline": {"strategy": "risk_routed_v2"},
                "normalization": {
                    "chorus_aware_split": True,
                    "chorus_gap_seconds": 120.0,
                    "chorus_similarity_threshold": 0.3,
                    "chorus_context_segments": 1,
                },
            }
        })
        repaired, events = repair_song_boundaries(
            segments, config, [ContentMatch("song", "Song", [0, 1, 2], 0.8)],
        )

        # 50s gap > merge_gap_seconds (40s) and different lyrics → split
        assert [match.segment_indices for match in repaired] == [[0, 1], [2]]
        assert events[0]["type"] == "risk_boundary_split"

    def test_temporal_adjudicated_range_survives_asr_time_gap(self):
        """v2: temporal_adjudicated with 35s gap stays together (≤ merge_gap_seconds)."""
        from dd_clip_miner_llm.song_postprocess import (
            _normalize_song_matches,
            repair_song_boundaries,
        )

        segments = [
            TranscriptSegment(0.0, 5.0, "世界突然安静你也一样吗"),
            TranscriptSegment(40.0, 45.0, "今天天气真好出去走走吧"),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "pipeline": {"strategy": "risk_routed_v2"},
                "normalization": {
                    "chorus_aware_split": True,
                    "chorus_gap_seconds": 120.0,
                    "chorus_similarity_threshold": 0.3,
                    "chorus_context_segments": 1,
                },
            }
        })
        match = ContentMatch(
            "song", "Song", [0, 1], 0.9, tags=["temporal_adjudicated"]
        )

        repaired, boundary_events = repair_song_boundaries(
            segments, config, [match],
        )
        normalized, normalization_events, suspicious = _normalize_song_matches(
            segments, config, repaired,
        )

        # 35s gap ≤ merge_gap_seconds (40s) → stays together
        assert [item.segment_indices for item in normalized] == [[0, 1]]

    def test_temporal_adjudicated_chorus_keeps_together(self):
        """v2: temporal_adjudicated with similar lyrics and 65s gap stays together."""
        from dd_clip_miner_llm.song_postprocess import (
            _normalize_song_matches,
            repair_song_boundaries,
        )

        segments = [
            TranscriptSegment(0.0, 5.0, "世界突然安静 你也一样吗"),
            TranscriptSegment(70.0, 75.0, "世界突然安静 你也一样吗"),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "pipeline": {"strategy": "risk_routed_v2"},
                "normalization": {
                    "chorus_aware_split": True,
                    "chorus_gap_seconds": 120.0,
                    "chorus_similarity_threshold": 0.3,
                    "chorus_context_segments": 1,
                },
            }
        })
        match = ContentMatch(
            "song", "昨日青空", [0, 1], 0.9, tags=["temporal_adjudicated"]
        )

        repaired, boundary_events = repair_song_boundaries(
            segments, config, [match],
        )
        normalized, normalization_events, suspicious = _normalize_song_matches(
            segments, config, repaired,
        )

        # Similar lyrics + 65s gap (40-120s range) → chorus detected, stays together
        assert [item.segment_indices for item in normalized] == [[0, 1]]

    def test_temporal_adjudicated_different_lyrics_splits_in_40_120_range(self):
        """v2: temporal_adjudicated with different lyrics and 65s gap gets split."""
        from dd_clip_miner_llm.song_postprocess import (
            _normalize_song_matches,
            repair_song_boundaries,
        )

        segments = [
            TranscriptSegment(0.0, 5.0, "世界突然安静你也一样吗"),
            TranscriptSegment(70.0, 75.0, "今天天气真好出去走走吧"),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "pipeline": {"strategy": "risk_routed_v2"},
                "normalization": {
                    "chorus_aware_split": True,
                    "chorus_gap_seconds": 120.0,
                    "chorus_similarity_threshold": 0.3,
                    "chorus_context_segments": 1,
                },
            }
        })
        match = ContentMatch(
            "song", "Song", [0, 1], 0.9, tags=["temporal_adjudicated"]
        )

        repaired, boundary_events = repair_song_boundaries(
            segments, config, [match],
        )
        normalized, normalization_events, suspicious = _normalize_song_matches(
            segments, config, repaired,
        )

        # Different lyrics + 65s gap (40-120s range) → no chorus, split
        assert [item.segment_indices for item in normalized] == [[0], [1]]

    def test_search_rejects_title_only_echo(self):
        """搜索标题仅回显查询、摘要无歌词时必须拒绝。"""
        from dd_clip_miner_llm.song_postprocess.pipeline import (
            _parse_search_result_per_item,
            _verify_search_evidence,
        )

        query = "有些日子就得自己过"
        result = {
            "query": query,
            "results": [{"title": "有些日子就得自己过 lyrics", "snippet": "", "url": ""}],
            "lyrics_hints": [],
        }
        items = _parse_search_result_per_item(result)
        assert len(items) == 1
        # lyrics 后缀应被清理
        assert items[0]["title"] == "有些日子就得自己过"
        accepted, score, reason = _verify_search_evidence(query, items[0])
        assert not accepted
        assert reason == "no_evidence_text"

    def test_search_accepts_snippet_lyrics_evidence(self):
        """摘要或 lyrics_hints 与歌词锚点匹配时允许更新歌名。"""
        from dd_clip_miner_llm.song_postprocess.pipeline import (
            _parse_search_result_per_item,
            _verify_search_evidence,
        )

        query = "世界突然安静你也一样吗"
        result = {
            "query": query,
            "results": [{
                "title": "昨日青空 - 尤长靖 lyrics",
                "snippet": "世界突然安静 你也一样吗 青春有你初心",
                "url": "",
            }],
            "lyrics_hints": ["世界突然安静 你也一样吗"],
        }
        items = _parse_search_result_per_item(result)
        assert len(items) == 1
        # lyrics 后缀应被清理
        assert items[0]["title"] == "昨日青空 - 尤长靖"
        assert "artist" not in items[0]  # 无结构化字段
        accepted, score, reason = _verify_search_evidence(query, items[0])
        assert accepted
        assert score >= 0.15
        assert reason == "lyrics_evidence_found"

    def test_search_per_result_evidence_binding(self):
        """第二条含歌词不能证明第一条标题。"""
        from dd_clip_miner_llm.song_postprocess.pipeline import (
            _parse_search_result_per_item,
            _verify_search_evidence,
        )

        query = "世界突然安静你也一样吗"
        result = {
            "query": query,
            "results": [
                {"title": "无关歌曲 lyrics", "snippet": "随便什么内容", "url": ""},
                {"title": "昨日青空 - 尤长靖", "snippet": "世界突然安静 你也一样吗", "url": ""},
            ],
            "lyrics_hints": [],
        }
        items = _parse_search_result_per_item(result)
        assert len(items) == 2
        # 第一条：摘要与查询不匹配 → 拒绝
        accepted0, _, _ = _verify_search_evidence(query, items[0])
        assert not accepted0
        # 第二条：摘要与查询匹配 → 接受
        accepted1, score1, _ = _verify_search_evidence(query, items[1])
        assert accepted1
        assert score1 >= 0.15

    def test_search_lyrics_suffix_cleanup(self):
        """lyrics 后缀在各种格式下都能清理。"""
        from dd_clip_miner_llm.song_postprocess.pipeline import _parse_search_result_per_item

        cases = [
            ("昨日青空 lyrics", "昨日青空"),
            ("昨日青空 - 尤长靖 lyrics", "昨日青空 - 尤长靖"),
            ("昨日青空 (lyrics)", "昨日青空"),
            ("昨日青空 - 歌词", "昨日青空"),
            ("昨日青空 | Official Audio", "昨日青空"),
            ("昨日青空 MV", "昨日青空"),
        ]
        for raw, expected in cases:
            result = {"results": [{"title": raw, "snippet": "", "url": ""}]}
            items = _parse_search_result_per_item(result)
            assert len(items) == 1, f"Failed for: {raw}"
            assert items[0]["title"] == expected, f"Expected '{expected}', got '{items[0]['title']}' for input '{raw}'"

    def test_search_no_artist_guessing_from_hyphen(self):
        """无结构化字段时不得猜测歌手位置。"""
        from dd_clip_miner_llm.song_postprocess.pipeline import _parse_search_result_per_item

        # "歌名 - 歌手" 格式 → 不猜测，完整标题作为歌名
        result = {"results": [{"title": "昨日青空 - 尤长靖", "snippet": "", "url": ""}]}
        items = _parse_search_result_per_item(result)
        assert items[0]["title"] == "昨日青空 - 尤长靖"
        assert "artist" not in items[0]

        # "歌手 - 歌名" 格式 → 同样不猜测
        result2 = {"results": [{"title": "尤长靖 - 昨日青空", "snippet": "", "url": ""}]}
        items2 = _parse_search_result_per_item(result2)
        assert items2[0]["title"] == "尤长靖 - 昨日青空"
        assert "artist" not in items2[0]

    def test_search_preserves_segment_indices(self, tmp_path):
        """搜索阶段只修改 title/artist，不修改 segment_indices。"""
        from dd_clip_miner_llm.song_postprocess.pipeline import (
            SearchVerificationStage,
            SongPipelineContext,
        )

        segments = [
            TranscriptSegment(0.0, 5.0, "世界突然安静你也一样吗"),
            TranscriptSegment(5.0, 10.0, "青春有你初心"),
        ]
        original_indices = [0, 1]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "pipeline": {"strategy": "risk_routed_v2"},
                "search": {"enabled": False},
            }
        })
        match = ContentMatch("song", "未知歌曲：测试", original_indices, 0.7)
        ctx = SongPipelineContext(segments, config, None, tmp_path, [match])

        stage = SearchVerificationStage()
        stage.run(ctx)

        # 搜索关闭时不做任何修改
        assert ctx.matches[0].segment_indices == original_indices

    def test_risk_audit_records_both_thresholds(self):
        """风险审计记录实际 40 秒拆分阈值和独立的 20 秒风险阈值。"""
        from dd_clip_miner_llm.song_postprocess import repair_song_boundaries

        segments = [
            TranscriptSegment(0.0, 5.0, "a"),
            TranscriptSegment(6.0, 10.0, "b"),
            TranscriptSegment(60.0, 65.0, "c"),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "pipeline": {"strategy": "risk_routed_v2"},
                "normalization": {
                    "chorus_aware_split": True,
                    "chorus_gap_seconds": 120.0,
                    "chorus_similarity_threshold": 0.3,
                    "chorus_context_segments": 1,
                },
            }
        })
        _, events = repair_song_boundaries(
            segments, config, [ContentMatch("song", "Song", [0, 1, 2], 0.8)],
        )
        assert len(events) == 1
        assert events[0]["gap_threshold_seconds"] == 40.0
        assert events[0]["risk_boundary_gap_seconds"] == 20.0

    def test_anchor_recheck_forces_single_batch(self, tmp_path):
        """Anchor 补查默认关闭，不修改结果。"""
        from dd_clip_miner_llm.song_postprocess.pipeline import (
            AnchorMissedRecheckStage,
            SongPipelineContext,
        )

        segments = [
            TranscriptSegment(0.0, 5.0, "歌词第一句"),
            TranscriptSegment(5.0, 10.0, "歌词第二句"),
            TranscriptSegment(100.0, 105.0, "聊天内容"),
            TranscriptSegment(200.0, 205.0, "歌词第三句"),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "pipeline": {"strategy": "risk_routed_v2"},
                "missed_recheck": {"enabled": False},
            }
        })
        match = ContentMatch("song", "歌曲A", [0, 1], 0.9)
        ctx = SongPipelineContext(segments, config, None, tmp_path, [match])

        stage = AnchorMissedRecheckStage()
        stage.run(ctx)

        # 默认关闭，不做任何修改，stage_history 为空（直接 return）
        assert len(ctx.matches) == 1
        assert ctx.matches[0].title == "歌曲A"
        assert len(ctx.stage_history) == 0

    def test_v2_anchor_minimum_keeps_two_lines_or_ten_seconds(self):
        from dd_clip_miner_llm.song_postprocess.pipeline import (
            _anchor_has_minimum_evidence,
        )

        segments = [
            TranscriptSegment(0.0, 4.0, "第一句"),
            TranscriptSegment(4.0, 8.0, "第二句"),
            TranscriptSegment(10.0, 21.0, "合并的长歌词"),
            TranscriptSegment(22.0, 25.0, "单句感叹"),
        ]

        assert _anchor_has_minimum_evidence(
            segments, ContentMatch("song", "未知", [0, 1], 0.7)
        )
        assert _anchor_has_minimum_evidence(
            segments, ContentMatch("song", "未知", [2], 0.7)
        )
        assert not _anchor_has_minimum_evidence(
            segments, ContentMatch("song", "未知", [3], 0.7)
        )

    def test_v2_initial_stage_preserves_main_candidate_range(self, tmp_path):
        from dd_clip_miner_llm.song_postprocess.pipeline import (
            BoundaryRiskStage,
            SongPipelineContext,
        )

        segments = [
            TranscriptSegment(0.0, 5.0, "lyric before instrumental"),
            TranscriptSegment(40.0, 45.0, "lyric after instrumental"),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "pipeline": {"strategy": "risk_routed_v2"},
                "risk": {"boundary_gap_seconds": 8.0},
            }
        })
        context = SongPipelineContext(
            segments=segments,
            config=config,
            recognizer=None,
            llm_dir=tmp_path,
            matches=[ContentMatch("song", "Candidate", [0, 1], 0.8)],
        )

        BoundaryRiskStage(
            "initial", "main", preserve_candidate_ranges=True,
        ).run(context)

        assert [match.segment_indices for match in context.matches] == [[0, 1]]
        assert context.stage_history[0]["input_count"] == 1
        assert context.stage_history[0]["output_count"] == 1

    def test_search_query_alone_is_not_evidence(self, tmp_path):
        from dd_clip_miner_llm.song_postprocess import load_supported_search_titles

        payload = {
            "tool_calls_log": [{
                "arguments": {"title": "Guessed Song"},
                "result_preview": "{'query': 'Guessed Song lyrics', 'results': []}",
            }]
        }
        (tmp_path / "llm_batch_000000.json").write_text(
            json.dumps(payload), encoding="utf-8",
        )
        candidate = ContentMatch("song", "Guessed Song", [0], 0.8)

        assert load_supported_search_titles(tmp_path, [candidate]) == set()

    def test_search_result_text_supports_title(self, tmp_path):
        from dd_clip_miner_llm.song_postprocess import load_supported_search_titles

        payload = {
            "tool_calls_log": [{
                "arguments": {"title": "distinctive lyric"},
                "result_preview": "{'results': [{'title': 'Kingyo Hanabi lyrics', 'snippet': 'lyrics'}]}",
            }]
        }
        (tmp_path / "llm_batch_000000.json").write_text(
            json.dumps(payload), encoding="utf-8",
        )
        candidate = ContentMatch("song", "Kingyo Hanabi", [0], 0.8)

        assert load_supported_search_titles(tmp_path, [candidate]) == {"kingyo hanabi"}

    def test_missed_anchor_output_cannot_keep_large_range(self):
        from dd_clip_miner_llm.song_postprocess.recheck import _constrain_missed_anchor_matches

        match = ContentMatch("song", "Song", list(range(100)), 0.8)
        constrained = _constrain_missed_anchor_matches([match], max_anchor_segments=12)

        assert len(constrained[0].segment_indices) <= 12
        assert constrained[0].segment_indices != list(range(100))

    def test_missed_anchor_expands_only_within_contiguous_target(self):
        from dd_clip_miner_llm.song_postprocess import expand_song_anchors

        segments = [
            TranscriptSegment(0.0, 5.0, "a"),
            TranscriptSegment(6.0, 10.0, "b"),
            TranscriptSegment(11.0, 15.0, "c"),
            TranscriptSegment(30.0, 35.0, "chat after gap"),
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {
                "pipeline": {"strategy": "risk_routed_v2"},
                "risk": {"boundary_gap_seconds": 8.0},
                "missed_recheck": {"anchor_max_expansion_seconds": 420.0},
            }
        })
        anchors = [ContentMatch("song", "Song", [1], 0.8)]

        expanded, events = expand_song_anchors(
            segments, config, anchors, [(0, 3)], [],
        )

        assert expanded[0].segment_indices == [0, 1, 2]
        assert events[0]["expanded_ranges"] == [[0, 2]]

    def test_missed_anchor_does_not_expand_into_existing_coverage(self):
        from dd_clip_miner_llm.song_postprocess import expand_song_anchors

        segments = [
            TranscriptSegment(float(i * 6), float(i * 6 + 5), str(i))
            for i in range(5)
        ]
        config = deep_merge(DEFAULT_CONFIG, {
            "song": {"pipeline": {"strategy": "risk_routed_v2"}}
        })
        anchors = [ContentMatch("song", "New", [3], 0.8)]
        existing = [ContentMatch("song", "Existing", [0, 1, 2], 0.9)]

        expanded, _ = expand_song_anchors(
            segments, config, anchors, [(0, 4)], existing,
        )

        assert expanded[0].segment_indices == [3, 4]

    def test_risk_review_clusters_are_batched_with_bounded_span(self):
        from dd_clip_miner_llm.song_postprocess import _batch_risk_review_clusters

        clusters = [
            [ContentMatch("song", "A", [0], 0.8)],
            [ContentMatch("song", "B", [10], 0.8)],
            [ContentMatch("song", "C", [700], 0.8)],
        ]
        batched = _batch_risk_review_clusters(
            clusters, max_span_segments=500, max_candidates=6,
        )

        assert [[match.title for match in cluster] for cluster in batched] == [
            ["A", "B"],
            ["C"],
        ]

    def test_temporal_adjudication_restores_title_by_overlap(self):
        from dd_clip_miner_llm.song_postprocess import _restore_temporal_titles

        source = ContentMatch("song", "Original title", [10, 11, 12], 0.7)
        temporal = ContentMatch(
            "song", "未知歌曲：时序裁决", [9, 10, 11, 12, 13], 0.9
        )

        restored, events = _restore_temporal_titles([temporal], [source])

        assert restored[0].title == "Original title"
        assert restored[0].segment_indices == [9, 10, 11, 12, 13]
        assert "temporal_adjudicated" in restored[0].tags
        assert events[0]["type"] == "temporal_boundary_replacement"

    def test_temporal_adjudication_recovers_distinct_source_boundary(self):
        from dd_clip_miner_llm.song_postprocess import (
            _split_temporal_at_source_boundaries,
        )

        temporal = ContentMatch("song", "Unknown", list(range(10, 30)), 0.8)
        sources = [
            ContentMatch("song", "First anchor", list(range(10, 20)), 0.7),
            ContentMatch("song", "Second anchor", list(range(20, 30)), 0.7),
        ]

        refined, events = _split_temporal_at_source_boundaries(
            [temporal], sources,
        )

        assert [match.segment_indices for match in refined] == [
            list(range(10, 20)),
            list(range(20, 30)),
        ]
        assert events[0]["type"] == "temporal_source_boundary_split"

    def test_temporal_adjudication_does_not_split_same_source_identity(self):
        from dd_clip_miner_llm.song_postprocess import (
            _split_temporal_at_source_boundaries,
        )

        temporal = ContentMatch("song", "Unknown", list(range(10, 30)), 0.8)
        sources = [
            ContentMatch("song", "Same anchor", list(range(10, 20)), 0.7),
            ContentMatch("song", "Same anchor", list(range(20, 30)), 0.7),
        ]

        refined, events = _split_temporal_at_source_boundaries(
            [temporal], sources,
        )

        assert refined == [temporal]
        assert events == []

    def test_temporal_adjudication_preserves_unmatched_confident_source(self):
        from dd_clip_miner_llm.song_postprocess import _restore_temporal_titles

        source = ContentMatch("song", "Source", [20, 21], 0.8)

        restored, events = _restore_temporal_titles([], [source])

        assert restored == [source]
        assert events[0]["type"] == "temporal_source_preserved"

    def test_song_cache_friendly_prompt_requests_compact_ranges(self, sample_segments, sample_config):
        from dd_clip_miner_llm.recognizers import get_recognizer

        sample_config["llm"]["cache_friendly_prompt_layout"] = True
        prompt = get_recognizer("song").build_prompt(sample_segments, 0, sample_config)

        assert "segment_ranges" in prompt
        assert "不要输出 segment_indices" in prompt
        assert "从第一个 segment 到最后一个 segment 按时间顺序检查一遍" in prompt
        assert "不能只返回能确认歌名的歌曲" in prompt

    def test_song_accuracy_prompt_keeps_legacy_coverage_layout(self, sample_segments, sample_config):
        from dd_clip_miner_llm.recognizers import get_recognizer

        prompt = get_recognizer("song").build_prompt(sample_segments, 0, sample_config)

        assert "segment_indices" in prompt
        assert "从第一个 segment 到最后一个 segment 按时间顺序检查一遍" not in prompt

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


# ============ song_evaluation.py tests ============

class TestSongEvaluation:
    def test_profile_evaluation_treats_duration_as_fixture_risk(self, tmp_path):
        from dd_clip_miner_llm.song_evaluation import evaluate_song_profile

        profile = tmp_path / "kv_optimized"
        (profile / "song").mkdir(parents=True)
        matches = [{
            "title": "Song",
            "artist": "",
            "confidence": 0.9,
            "segment_indices": [0, 1],
        }]
        (profile / "song" / "matches.json").write_text(
            json.dumps(matches), encoding="utf-8",
        )
        (profile / "usage_summary.json").write_text(
            json.dumps({"totals": {
                "prompt_cache_hit_tokens": 900,
                "prompt_cache_miss_tokens": 100,
                "completion_tokens": 10,
            }}),
            encoding="utf-8",
        )
        transcript = [
            {"start": 0.0, "end": 10.0, "text": "a"},
            {"start": 11.0, "end": 20.0, "text": "b"},
        ]

        metrics = evaluate_song_profile(profile, transcript, matches)

        assert metrics["weak_anchor_recall"] == 1.0
        assert metrics["fixture_duration_risk_examples"][0]["classification"] == (
            "fixture_risk_example_not_global_rule"
        )
        assert metrics["cache_hit_ratio"] == 0.9

    def test_cost_estimator_is_invalid_above_15_percent_error(self, tmp_path):
        from dd_clip_miner_llm.song_evaluation import evaluate_song_profile

        profile = tmp_path / "kv_optimized"
        (profile / "song").mkdir(parents=True)
        (profile / "song" / "matches.json").write_text("[]", encoding="utf-8")
        (profile / "song" / "adaptive_strategies.json").write_text(
            json.dumps({"chosen_total_usd": 0.01}), encoding="utf-8",
        )
        (profile / "usage_summary.json").write_text(
            json.dumps({"totals": {"completion_tokens": 100000}}),
            encoding="utf-8",
        )

        metrics = evaluate_song_profile(profile, [], [])

        assert metrics["cost_estimator"]["valid_for_runtime_selection"] is False

    def test_profile_evaluation_counts_same_title_overlap(self, tmp_path):
        from dd_clip_miner_llm.song_evaluation import evaluate_song_profile

        profile = tmp_path / "kv_optimized"
        (profile / "song").mkdir(parents=True)
        (profile / "song" / "matches.json").write_text(json.dumps([
            {"title": "Song", "segment_indices": [1, 2, 3]},
            {"title": "Song", "segment_indices": [2, 3, 4]},
        ]), encoding="utf-8")

        metrics = evaluate_song_profile(profile, [], [])

        assert metrics["same_title_overlap_pairs"] == 1

    def test_profile_evaluation_reports_boundary_fragmentation_and_coalescing(self, tmp_path):
        from dd_clip_miner_llm.song_evaluation import evaluate_song_profile

        profile = tmp_path / "kv_optimized"
        (profile / "song").mkdir(parents=True)
        (profile / "song" / "matches.json").write_text(json.dumps([
            {"title": "Combined", "segment_indices": [0, 1, 2, 3]},
            {"title": "Fragment", "segment_indices": [10]},
            {"title": "Fragment", "segment_indices": [11]},
        ]), encoding="utf-8")
        weak = [
            {"title": "A", "segment_indices": [0, 1]},
            {"title": "B", "segment_indices": [2, 3]},
            {"title": "C", "segment_indices": [10, 11]},
        ]

        metrics = evaluate_song_profile(profile, [], weak)

        assert metrics["fragmented_weak_anchor_count"] == 1
        assert metrics["coalesced_weak_anchor_count"] == 1
        assert metrics["mixed_title_coalesced_count"] == 1


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
