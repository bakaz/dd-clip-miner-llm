from __future__ import annotations

from pathlib import Path

import pytest

from dd_clip_miner_llm.models import TranscriptSegment
from dd_clip_miner_llm.pipeline.utils import resolve_total_duration


def test_resolve_total_duration_flv_underreport(tmp_path, monkeypatch):
    input_path = tmp_path / "input.flv"
    source_wav = tmp_path / "source.wav"
    input_path.write_bytes(b"flv")
    source_wav.write_bytes(b"wav")

    def fake_duration(path):
        if Path(path) == input_path:
            return 8454.779
        if Path(path) == source_wav:
            return 20116.16
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr("dd_clip_miner_llm.ffmpeg.get_duration", fake_duration)

    segments = [TranscriptSegment(start=20094.0, end=20098.0, text="tail")]
    assert resolve_total_duration(input_path, source_wav, segments) == 20116.16


def test_resolve_total_duration_accurate_container(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp4"
    source_wav = tmp_path / "source.wav"
    input_path.write_bytes(b"mp4")
    source_wav.write_bytes(b"wav")
    monkeypatch.setattr("dd_clip_miner_llm.ffmpeg.get_duration", lambda _path: 3600.0)

    segments = [TranscriptSegment(start=3400.0, end=3500.0, text="speech")]
    assert resolve_total_duration(input_path, source_wav, segments) == 3600.0


def test_resolve_total_duration_empty_segments(tmp_path, monkeypatch):
    input_path = tmp_path / "input.flv"
    source_wav = tmp_path / "source.wav"
    input_path.write_bytes(b"flv")
    source_wav.write_bytes(b"wav")

    def fake_duration(path):
        if Path(path) == input_path:
            return 8454.779
        if Path(path) == source_wav:
            return 20116.16
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr("dd_clip_miner_llm.ffmpeg.get_duration", fake_duration)

    assert resolve_total_duration(input_path, source_wav, []) == 20116.16


def test_resolve_total_duration_logs_correction(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "input.flv"
    source_wav = tmp_path / "source.wav"
    input_path.write_bytes(b"flv")
    source_wav.write_bytes(b"wav")
    monkeypatch.setattr("dd_clip_miner_llm.ffmpeg.get_duration", lambda _path: 8454.779)

    segments = [TranscriptSegment(start=1.0, end=9000.0, text="speech")]
    assert resolve_total_duration(input_path, source_wav, segments) == 9000.0

    captured = capsys.readouterr()
    assert "total_duration corrected" in captured.out