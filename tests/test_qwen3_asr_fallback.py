"""Qwen3 ASR fallback — unit tests (offline, no GPU)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dd_clip_miner_llm.asr_fallback import (
    _build_qwen3_local_config,
    detect_qwen3_fallback_ranges,
    faster_whisper_fallback_config,
    is_qwen3_fallback_enabled,
    merge_replace_ranges,
    qwen3_fallback_config,
    transcribe_qwen3_with_fallback,
)
from dd_clip_miner_llm.asr_backends.funasr_backend import repair_qwen3_zero_duration_segments
from dd_clip_miner_llm.models import TranscriptSegment


class TestZeroDurationRepair:
    def test_merge_punctuation_into_previous_segment(self):
        segments = [
            TranscriptSegment(10.0, 12.5, "你好"),
            TranscriptSegment(12.5, 12.5, "。"),
        ]
        repaired, stats = repair_qwen3_zero_duration_segments(segments)
        assert len(repaired) == 1
        assert repaired[0].text == "你好。"
        assert repaired[0].start == 10.0
        assert repaired[0].end == 12.5
        assert stats["merged_count"] == 1
        assert all(seg.end > seg.start for seg in repaired)

    def test_merge_leading_zero_duration_into_next_segment(self):
        segments = [
            TranscriptSegment(0.0, 0.0, "嗯"),
            TranscriptSegment(1.0, 2.0, "开始"),
        ]
        repaired, stats = repair_qwen3_zero_duration_segments(segments)
        assert len(repaired) == 1
        assert repaired[0].text == "嗯开始"
        assert stats["merged_count"] == 1


class TestFallbackDetection:
    def test_sparse_long_segment_is_flagged(self):
        segments = [
            TranscriptSegment(0.0, 207.0, "嗯"),
        ]
        ranges = detect_qwen3_fallback_ranges(segments, {
            "max_segment_seconds": 15,
            "sparse_chars_per_sec": 1.0,
            "repeat_threshold": 3,
        })
        assert len(ranges) == 1
        assert "sparse_long_segment" in ranges[0]["reasons"]

    def test_repeated_segments_are_flagged(self):
        segments = [
            TranscriptSegment(0.0, 1.0, "一遍"),
            TranscriptSegment(1.0, 2.0, "一遍"),
            TranscriptSegment(2.0, 3.0, "一遍"),
        ]
        ranges = detect_qwen3_fallback_ranges(segments, {
            "max_segment_seconds": 15,
            "sparse_chars_per_sec": 1.0,
            "repeat_threshold": 3,
        })
        assert len(ranges) == 1
        assert "repeated_segment" in ranges[0]["reasons"]


class TestMergeReplaceRanges:
    def test_replace_segments_inside_fallback_range(self):
        primary = [
            TranscriptSegment(0.0, 5.0, "keep-before"),
            TranscriptSegment(10.0, 30.0, "bad"),
            TranscriptSegment(40.0, 45.0, "keep-after"),
        ]
        fallback_items = [{
            "padded_start": 8.0,
            "padded_end": 32.0,
            "segments": [
                {"start": 10.0, "end": 12.0, "text": "fixed-1"},
                {"start": 12.0, "end": 14.0, "text": "fixed-2"},
            ],
        }]
        merged = merge_replace_ranges(primary, fallback_items)
        texts = [seg.text for seg in merged]
        assert "keep-before" in texts
        assert "keep-after" in texts
        assert "bad" not in texts
        assert "fixed-1" in texts
        assert "fixed-2" in texts


class TestFallbackConfig:
    def test_reads_funasr_fallback_not_lyrics(self):
        config = {
            "mode": "local",
            "local": {
                "backend": "qwen3_asr",
                "funasr": {
                    "fallback": {
                        "enabled": True,
                        "chunk_seconds": 5,
                    },
                    "lyrics": {"enabled": False},
                },
            },
        }
        assert is_qwen3_fallback_enabled(config) is True
        assert qwen3_fallback_config(config)["chunk_seconds"] == 5

    def test_whisper_backend_does_not_enable_qwen3_fallback(self):
        config = {
            "mode": "local",
            "local": {
                "backend": "faster_whisper",
                "funasr": {"fallback": {"enabled": True}},
            },
        }
        assert is_qwen3_fallback_enabled(config) is False

    def test_qwen3_fallback_inherits_gpu_funasr_hub(self):
        config = {
            "mode": "local",
            "local": {
                "backend": "qwen3_asr",
                "funasr": {
                    "device": "auto",
                    "fallback": {
                        "enabled": True,
                        "chunk_seconds": 5,
                    },
                },
                "gpu": {
                    "funasr": {
                        "model": "Qwen/Qwen3-ASR-1.7B",
                        "hub": "ms",
                        "device": "cuda:0",
                        "dtype": "bf16",
                        "timestamp_chunk_seconds": 180,
                    },
                },
            },
        }

        local_cfg = _build_qwen3_local_config(config, chunk_seconds=5)

        assert local_cfg["funasr"]["model"] == "Qwen/Qwen3-ASR-1.7B"
        assert local_cfg["funasr"]["hub"] == "ms"
        assert local_cfg["funasr"]["device"] == "cuda:0"
        assert local_cfg["funasr"]["timestamp_chunk_seconds"] == 5


class TestTranscribeQwen3WithFallback:
    def test_skips_fallback_when_no_suspicious_ranges(self, tmp_path):
        source_wav = tmp_path / "source.wav"
        source_wav.write_bytes(b"wav")
        asr_dir = tmp_path / "02_asr"

        primary_segments = [TranscriptSegment(0.0, 2.0, "hello")]
        mock_backend = MagicMock()
        mock_backend.transcribe.return_value = primary_segments

        asr_config = {
            "mode": "local",
            "local": {
                "backend": "qwen3_asr",
                "funasr": {
                    "fallback": {
                        "enabled": True,
                        "chunk_seconds": 5,
                        "max_segment_seconds": 15,
                        "sparse_chars_per_sec": 1.0,
                        "repeat_threshold": 3,
                        "padding_seconds": 2.0,
                    },
                },
            },
        }

        with patch("dd_clip_miner_llm.asr_fallback.build_asr_backend", return_value=mock_backend), \
             patch("dd_clip_miner_llm.asr_fallback.get_duration", return_value=10.0):
            segments, meta = transcribe_qwen3_with_fallback(source_wav, asr_config, asr_dir)

        assert segments == primary_segments
        assert meta["fallback_range_count"] == 0
        assert (asr_dir / "fallback_ranges.json").exists()


class TestFasterWhisperFallbackConfig:
    def test_empty_when_backend_is_qwen3(self):
        cfg = {
            "mode": "local",
            "local": {
                "backend": "qwen3_asr",
                "faster_whisper": {"fallback": {"enabled": True, "merge_policy": "replace_ranges"}},
            },
        }
        assert faster_whisper_fallback_config(cfg) == {}

    def test_returns_fallback_when_backend_is_faster_whisper(self):
        fallback = {"enabled": True, "merge_policy": "replace_ranges"}
        cfg = {
            "mode": "local",
            "local": {
                "backend": "faster_whisper",
                "faster_whisper": {"fallback": fallback},
            },
        }
        assert faster_whisper_fallback_config(cfg) == fallback
