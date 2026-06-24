"""Qwen3-ASR lyrics mode — unit tests for FunASR backend postprocessing.

Covers:
  A. Lyric splitter (split_lyrics_text)
  B. Result conversion (funasr_result_to_segments with lyrics cfg)
  C. Backend behaviour (FunASRBackend with mocked AutoModel / model.generate)

All tests run offline — no model download or GPU required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from dd_clip_miner_llm.asr_backends.funasr_backend import (
    FunASRBackend,
    funasr_result_to_segments,
    repair_qwen3_zero_duration_segments,
    split_lyrics_text,
)
from dd_clip_miner_llm.ffmpeg.fsutil import safe_rmtree
from dd_clip_miner_llm.models import TranscriptSegment


# ---------------------------------------------------------------------------
# Helpers for Group C — inject a fake ``funasr`` package into sys.modules
# so that ``from funasr import AutoModel`` inside _load_model() resolves.
# ---------------------------------------------------------------------------

@pytest.fixture()
def _fake_funasr():
    """Temporarily install a mock ``funasr`` package in sys.modules."""
    fake = ModuleType("funasr")
    mock_automodel = MagicMock(name="AutoModel")
    fake.AutoModel = mock_automodel  # type: ignore[attr-defined]
    saved = sys.modules.get("funasr")
    sys.modules["funasr"] = fake
    try:
        yield mock_automodel
    finally:
        if saved is None:
            sys.modules.pop("funasr", None)
        else:
            sys.modules["funasr"] = saved


# ---------------------------------------------------------------------------
# Group A — Lyric splitter
# ---------------------------------------------------------------------------

class TestLyricSplitter:
    """split_lyrics_text: line-breaking for lyrics display."""

    def test_split_lyrics_text_chinese_without_punctuation(self):
        """Chinese text without punctuation splits at max_line_chars."""
        result = split_lyrics_text("拿脚底下的泥给种子盖个小家", max_line_chars=7)
        assert "\n" in result
        lines = result.split("\n")
        assert len(lines) > 1
        assert all(len(line) <= 7 for line in lines if line.strip())

    def test_split_lyrics_text_chinese_with_punctuation(self):
        """Chinese text with punctuation splits at sentence boundaries."""
        result = split_lyrics_text("你好。世界！")
        assert "\n" in result
        lines = result.split("\n")
        assert len(lines) >= 2

    def test_split_lyrics_text_english_soft_wrap(self):
        """English text wraps at word boundaries."""
        result = split_lyrics_text("Hello world this is a test", max_line_chars=15)
        assert "\n" in result
        lines = result.split("\n")
        # No line should exceed max_line_chars
        assert all(len(line) <= 15 for line in lines if line.strip())
        # Words should not be broken
        assert "Hello" in result
        assert "world" in result

    def test_split_lyrics_text_keeps_decimal_and_version_dot(self):
        """v2.5 and 3.14 don't split on the decimal dot."""
        result = split_lyrics_text("v2.5 is ok. 3.14 is pi.")
        assert "v2.5" in result
        assert "3.14" in result

    def test_split_lyrics_text_empty_input(self):
        """Empty string returns empty string."""
        assert split_lyrics_text("") == ""

    def test_split_lyrics_text_preserves_existing_newlines(self):
        """Existing newlines are preserved as-is."""
        result = split_lyrics_text("line1\nline2\nline3")
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result
        lines = result.split("\n")
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# Group B — Result conversion
# ---------------------------------------------------------------------------

class TestZeroDurationRepair:
    def test_postprocess_merges_trailing_punctuation_segment(self):
        result = [
            {
                "text": "你好。",
                "timestamp": [[0, 1000], [1000, 2000], [2000, 2000]],
            }
        ]
        segments = funasr_result_to_segments(result, "test.wav", cfg={"lyrics": {"enabled": False}})
        assert len(segments) == 1
        assert segments[0].text == "你好。"
        assert segments[0].end > segments[0].start


class TestResultConversion:
    """funasr_result_to_segments: routing by result format & cfg."""

    def test_qwen3_text_only_lyrics_enabled(self):
        """Qwen3 text+timestamp with lyrics.enabled=true → line-level segments."""
        result = [
            {
                "text": "你好。世界。",
                "timestamp": [[0, 1000], [1000, 2000], [2000, 3000], [3000, 4000]],
            }
        ]
        cfg = {"lyrics": {"enabled": True, "max_line_chars": 24}}
        segments = funasr_result_to_segments(result, "test.wav", cfg=cfg)
        assert len(segments) > 0
        # Each segment should have valid timing
        for seg in segments:
            assert seg.end >= seg.start
            assert seg.text.strip() != ""

    def test_qwen3_text_only_lyrics_disabled(self):
        """Qwen3 text+timestamp with lyrics.enabled=false → sentence-level segments."""
        result = [
            {
                "text": "你好。世界。",
                "timestamp": [[0, 1000], [1000, 2000], [2000, 3000], [3000, 4000]],
            }
        ]
        cfg = {"lyrics": {"enabled": False}}
        segments = funasr_result_to_segments(result, "test.wav", cfg=cfg)
        assert len(segments) > 0
        for seg in segments:
            assert seg.end >= seg.start
            assert seg.text.strip() != ""

    def test_sentence_info_priority_unchanged(self):
        """SenseVoice sentence_info still returns timestamped segments."""
        result = [
            {
                "sentence_info": [
                    {"start": 0, "end": 1000, "text": "hello"},
                    {"start": 1000, "end": 2000, "text": "world"},
                ]
            }
        ]
        segments = funasr_result_to_segments(result, "test.wav")
        assert len(segments) > 0
        assert any("hello" in seg.text for seg in segments)

    def test_timestamp_priority_unchanged(self):
        """Paraformer timestamp (3-elem tuples) still returns original segments."""
        result = [
            {"timestamp": [[0, 1000, "hello"], [1000, 2000, "world"]]}
        ]
        segments = funasr_result_to_segments(result, "test.wav")
        assert len(segments) >= 1
        # First segment should start around 0
        assert segments[0].start >= 0.0

    def test_plain_text_fallback_unchanged_without_cfg(self):
        """Plain text without cfg → single segment spanning duration."""
        result = [{"text": "hello world"}]
        with patch(
            "dd_clip_miner_llm.ffmpeg.get_duration",
            return_value=10.0,
        ):
            segments = funasr_result_to_segments(result, "test.wav")
        assert len(segments) == 1
        assert segments[0].text == "hello world"
        assert segments[0].end == pytest.approx(10.0)

    def test_qwen3_forced_aligner_seconds_are_normalized_to_segment_seconds(self):
        """qwen-asr forced aligner seconds are not divided as if they were milliseconds."""
        result = [
            {
                "text": "你好",
                "timestamp": [[0, 1], [1, 2]],
            }
        ]
        with patch("dd_clip_miner_llm.ffmpeg.get_duration", return_value=2.0):
            segments = funasr_result_to_segments(result, "test.wav", cfg={"lyrics": {"enabled": False}})
        assert segments[-1].end == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Group C — Backend behaviour with mocks
# ---------------------------------------------------------------------------

class TestBackendBehaviour:
    """FunASRBackend: model loading and generate_kwargs construction."""

    def test_funasr_backend_passes_dtype_to_automodel(self, _fake_funasr: MagicMock):
        """dtype from settings is forwarded to AutoModel constructor."""
        _fake_funasr.return_value = MagicMock()
        settings = {
            "funasr": {
                "model": "test-model",
                "device": "cpu",
                "dtype": "bf16",
            }
        }
        backend = FunASRBackend(settings)
        backend._load_model()
        _fake_funasr.assert_called_once()
        call_kwargs = _fake_funasr.call_args[1]
        assert call_kwargs["dtype"] == "bf16"

    def test_funasr_backend_passes_forced_aligner_to_automodel(self, _fake_funasr: MagicMock):
        """Qwen3 timestamp mode initializes AutoModel with forced aligner settings."""
        _fake_funasr.return_value = MagicMock()
        settings = {
            "funasr": {
                "model": "Qwen/Qwen3-ASR-1.7B",
                "device": "cpu",
                "generate_kwargs": {"return_time_stamps": True},
                "forced_aligner_kwargs": {"dtype": "bf16"},
                "max_inference_batch_size": 2,
            }
        }
        backend = FunASRBackend(settings)
        backend._load_model()
        call_kwargs = _fake_funasr.call_args[1]
        assert call_kwargs["forced_aligner"] == "Qwen/Qwen3-ForcedAligner-0.6B"
        assert call_kwargs["forced_aligner_kwargs"] == {"dtype": "bf16"}
        assert call_kwargs["max_inference_batch_size"] == 2

    def test_funasr_backend_omits_batch_size_when_none(self, _fake_funasr: MagicMock):
        """batch_size=None → not present in generate() call."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "ok", "timestamp": []}]
        _fake_funasr.return_value = mock_model

        settings = {"funasr": {"model": "test-model", "device": "cpu"}}
        backend = FunASRBackend(settings)
        backend._load_model()

        cfg = {"batch_size": None}
        backend._transcribe_chunk(Path("test.wav"), 0.0, cfg)

        call_kwargs = mock_model.generate.call_args[1]
        assert "batch_size" not in call_kwargs

    def test_funasr_backend_passes_batch_size_when_set(self, _fake_funasr: MagicMock):
        """batch_size=1 → present in generate() call."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "ok", "timestamp": []}]
        _fake_funasr.return_value = mock_model

        settings = {"funasr": {"model": "test-model", "device": "cpu"}}
        backend = FunASRBackend(settings)
        backend._load_model()

        cfg = {"batch_size": 1}
        backend._transcribe_chunk(Path("test.wav"), 0.0, cfg)

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["batch_size"] == 1

    def test_funasr_backend_generate_kwargs_override(self, _fake_funasr: MagicMock):
        """generate_kwargs from cfg are merged into generate() call."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "ok", "timestamp": []}]
        _fake_funasr.return_value = mock_model

        settings = {"funasr": {"model": "test-model", "device": "cpu"}}
        backend = FunASRBackend(settings)
        backend._load_model()

        cfg: dict = {
            "generate_kwargs": {"language": "Chinese"},
        }
        backend._transcribe_chunk(Path("test.wav"), 0.0, cfg)

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["language"] == "Chinese"

    def test_funasr_backend_passes_return_time_stamps_to_generate(self, _fake_funasr: MagicMock):
        """return_time_stamps from generate_kwargs is forwarded to generate()."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [
            {"text": "你好", "timestamp": [[0, 1000], [1000, 2000]]}
        ]
        _fake_funasr.return_value = mock_model

        backend = FunASRBackend({"funasr": {"model": "Qwen/Qwen3-ASR-1.7B", "device": "cpu"}})
        backend._load_model()
        backend._transcribe_chunk(
            Path("test.wav"),
            0.0,
            {"generate_kwargs": {"return_time_stamps": True}},
        )

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["return_time_stamps"] is True

    def test_funasr_backend_fails_when_requested_timestamps_are_missing(self, _fake_funasr: MagicMock):
        """Timestamp mode must not silently fall back to one long text segment."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [{"text": "plain text only"}]
        _fake_funasr.return_value = mock_model

        backend = FunASRBackend({"funasr": {"model": "Qwen/Qwen3-ASR-1.7B", "device": "cpu"}})
        backend._load_model()

        with pytest.raises(RuntimeError, match="timestamp output was requested"):
            backend._transcribe_chunk(
                Path("test.wav"),
                0.0,
                {"generate_kwargs": {"return_time_stamps": True}},
            )

    def test_funasr_chunk_mode_uses_asr_dir_and_cleans_on_success(self):
        """Chunk files live under 02_asr/funasr_chunks and are cleaned after success."""
        work_dir = Path.cwd() / ".tmp" / "test_funasr_chunk_success"
        safe_rmtree(work_dir)
        asr_dir = work_dir / "02_asr"
        backend = FunASRBackend({"funasr": {"timestamp_chunk_seconds": 60}}, runtime_context={"asr_dir": asr_dir})
        cut_paths: list[Path] = []

        def fake_cut_audio(_source: Path, target: Path, _start: float, _end: float) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("chunk", encoding="utf-8")
            cut_paths.append(target)

        with patch("dd_clip_miner_llm.ffmpeg.get_duration", return_value=120.0), \
             patch("dd_clip_miner_llm.ffmpeg.cut_audio", side_effect=fake_cut_audio), \
             patch.object(backend, "_transcribe_chunk", return_value=[TranscriptSegment(0.0, 1.0, "ok")]):
            segments = backend.transcribe(work_dir / "source.wav")

        assert len(segments) == 2
        assert cut_paths
        chunk_dir = asr_dir / "funasr_chunks" / "source"
        assert all(path.parent == chunk_dir for path in cut_paths)
        assert not chunk_dir.exists()
        safe_rmtree(work_dir)

    def test_funasr_chunk_mode_preserves_chunks_on_failure(self):
        """Failed chunk transcription keeps chunk audio for debugging."""
        work_dir = Path.cwd() / ".tmp" / "test_funasr_chunk_failure"
        safe_rmtree(work_dir)
        asr_dir = work_dir / "02_asr"
        backend = FunASRBackend({"funasr": {"timestamp_chunk_seconds": 60}}, runtime_context={"asr_dir": asr_dir})

        def fake_cut_audio(_source: Path, target: Path, _start: float, _end: float) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("chunk", encoding="utf-8")

        with patch("dd_clip_miner_llm.ffmpeg.get_duration", return_value=120.0), \
             patch("dd_clip_miner_llm.ffmpeg.cut_audio", side_effect=fake_cut_audio), \
             patch.object(backend, "_transcribe_chunk", side_effect=RuntimeError("boom")), \
             pytest.raises(RuntimeError, match="boom"):
            backend.transcribe(work_dir / "source.wav")

        assert (asr_dir / "funasr_chunks" / "source" / "chunk_0000.wav").exists()
        safe_rmtree(work_dir)
