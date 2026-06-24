"""Tests for the cut-copy workflow module."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from dd_clip_miner_llm.cut_copy import (
    _format_folder,
    _load_done_marker,
    _save_done_marker,
    load_cut_copy_config,
    run_batch_cut_copy,
    run_cut_copy,
    scan_pending_files,
    schedule_shutdown,
    verify_copy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(path: Path, overrides: dict | None = None) -> Path:
    """Write a minimal valid cut_copy config YAML and return its path."""
    cfg = {
        "source": {"path": str(path / "source")},
        "destination": {"path": str(path / "dest")},
        "processing": {"config_path": str(path / "pipeline_config.yaml")},
    }
    if overrides:
        for section, values in overrides.items():
            if isinstance(values, dict):
                cfg.setdefault(section, {}).update(values)
            else:
                cfg[section] = values
    config_file = path / "cut_copy.yaml"
    config_file.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
    return config_file


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadCutCopyConfig:
    def test_load_cut_copy_config_valid(self, tmp_path):
        """All required fields present → config loads with defaults filled."""
        config_file = _write_config(tmp_path)

        cfg = load_cut_copy_config(config_file)

        assert cfg["source"]["path"] == str(tmp_path / "source")
        assert cfg["destination"]["path"] == str(tmp_path / "dest")
        assert cfg["processing"]["config_path"] == str(tmp_path / "pipeline_config.yaml")
        # Defaults applied
        assert cfg["source"]["pattern"] == "*_fix.mp4"
        assert cfg["behavior"]["shutdown_after"] is True
        assert cfg["behavior"]["max_files"] == 0
        assert cfg["destination"]["folder_format"] == "{date}_{streamer}"

    def test_load_cut_copy_config_missing_required(self, tmp_path):
        """Missing source.path → ValueError."""
        cfg = {
            "source": {},
            "destination": {"path": "/tmp/dest"},
            "processing": {"config_path": "/tmp/proc.yaml"},
        }
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(yaml.dump(cfg), encoding="utf-8")

        with pytest.raises(ValueError, match="source\\.path"):
            load_cut_copy_config(config_file)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

class TestScanPendingFiles:
    def test_scan_pending_files_filters_correctly(self, tmp_path):
        """Only *_fix.mp4 files not in the done marker are returned."""
        source = tmp_path / "source"
        source.mkdir()

        # Matching files
        (source / "video1_fix.mp4").write_bytes(b"A" * 100)
        (source / "video2_fix.mp4").write_bytes(b"B" * 100)
        # Non-matching file
        (source / "video3.mp4").write_bytes(b"C" * 100)

        # Mark video1 as done
        marker = source / ".dd_clip_miner_cut_copy_done.json"
        marker.write_text(
            json.dumps({"processed": [{"source": str(source / "video1_fix.mp4")}]}),
            encoding="utf-8",
        )

        config = {
            "source": {
                "path": str(source),
                "pattern": "*_fix.mp4",
                "done_marker": ".dd_clip_miner_cut_copy_done.json",
            },
            "behavior": {"max_files": 0},
        }

        pending = scan_pending_files(config)

        names = [p.name for p in pending]
        assert "video2_fix.mp4" in names
        assert "video1_fix.mp4" not in names
        assert "video3.mp4" not in names

    def test_scan_pending_files_respects_max_files(self, tmp_path):
        """max_files=1 → only 1 file returned."""
        source = tmp_path / "source"
        source.mkdir()
        for i in range(5):
            (source / f"video{i}_fix.mp4").write_bytes(b"X" * 10)

        config = {
            "source": {
                "path": str(source),
                "pattern": "*_fix.mp4",
                "done_marker": ".dd_clip_miner_cut_copy_done.json",
            },
            "behavior": {"max_files": 1},
        }

        pending = scan_pending_files(config)

        assert len(pending) == 1


# ---------------------------------------------------------------------------
# Done-marker roundtrip
# ---------------------------------------------------------------------------

class TestDoneMarker:
    def test_done_marker_roundtrip(self, tmp_path):
        """Save and load done marker preserves data."""
        source = tmp_path / "source"
        source.mkdir()
        marker_name = ".dd_clip_miner_cut_copy_done.json"

        data = {"processed": [{"source": "/a/b.mp4", "processed_at": "2026-01-01"}]}
        _save_done_marker(source, marker_name, data)

        loaded = _load_done_marker(source, marker_name)

        assert loaded == data
        assert loaded["processed"][0]["source"] == "/a/b.mp4"


# ---------------------------------------------------------------------------
# Copy verification
# ---------------------------------------------------------------------------

class TestVerifyCopy:
    def test_verify_copy_passes_when_files_match(self, tmp_path):
        """Matching dirs → returns True."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        for name in ("a.mp4", "b.json"):
            (src / name).write_bytes(b"data" * 25)
            (dst / name).write_bytes(b"data" * 25)

        assert verify_copy(src, dst) is True

    def test_verify_copy_fails_when_size_mismatch(self, tmp_path):
        """Different sizes → RuntimeError."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "a.mp4").write_bytes(b"12345")
        (dst / "a.mp4").write_bytes(b"12")

        with pytest.raises(RuntimeError, match="Copy verification failed"):
            verify_copy(src, dst)


# ---------------------------------------------------------------------------
# Shutdown scheduling
# ---------------------------------------------------------------------------

class TestScheduleShutdown:
    def test_schedule_shutdown_windows(self, monkeypatch):
        """On win32, uses shutdown /s /t <delay>."""
        called_cmd: list[str] = []

        def fake_run(cmd, **kwargs):
            called_cmd.extend(cmd)

        monkeypatch.setattr(sys, "platform", "win32")
        with patch("dd_clip_miner_llm.cut_copy.subprocess.run", side_effect=fake_run):
            schedule_shutdown(120)

        assert called_cmd == ["shutdown", "/s", "/t", "120"]

    def test_schedule_shutdown_linux(self, monkeypatch):
        """On linux, uses shutdown -h +<minutes>."""
        called_cmd: list[str] = []

        def fake_run(cmd, **kwargs):
            called_cmd.extend(cmd)

        monkeypatch.setattr(sys, "platform", "linux")
        with patch("dd_clip_miner_llm.cut_copy.subprocess.run", side_effect=fake_run):
            schedule_shutdown(120)

        assert called_cmd == ["shutdown", "-h", "+2"]


# ---------------------------------------------------------------------------
# Dry-run & no-shutdown
# ---------------------------------------------------------------------------

class TestRunCutCopy:
    def _make_valid_env(self, tmp_path) -> Path:
        """Create source dir + config + pipeline config, return config path."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "video_fix.mp4").write_bytes(b"fake" * 100)

        proc_yaml = tmp_path / "pipeline_config.yaml"
        proc_yaml.write_text("{}", encoding="utf-8")

        return _write_config(tmp_path)

    def test_dry_run_does_not_call_pipeline(self, tmp_path):
        """--dry-run skips process_video entirely."""
        config_file = self._make_valid_env(tmp_path)

        with patch(
            "dd_clip_miner_llm.cut_copy.process_video",
        ) as mock_proc:
            rc = run_cut_copy(config_file, dry_run=True)

        assert rc == 0
        mock_proc.assert_not_called()

    def test_no_shutdown_flag_prevents_shutdown(self, tmp_path, monkeypatch):
        """--no-shutdown prevents schedule_shutdown from being called."""
        config_file = self._make_valid_env(tmp_path)

        with (
            patch("dd_clip_miner_llm.cut_copy.process_video", return_value=tmp_path / "work" / "video_fix"),
            patch("dd_clip_miner_llm.cut_copy.copy_to_destination", return_value=tmp_path / "dest" / "out"),
            patch("dd_clip_miner_llm.cut_copy.verify_copy", return_value=True),
            patch("dd_clip_miner_llm.cut_copy.schedule_shutdown") as mock_shutdown,
        ):
            rc = run_cut_copy(config_file, no_shutdown=True)

        assert rc == 0
        mock_shutdown.assert_not_called()


class TestRunBatchCutCopy:
    def _config(self, tmp_path) -> dict:
        return {
            "enabled": True,
            "destination": {"path": str(tmp_path / "dest"), "username": "", "password": ""},
            "processing": {"skip_on_failure": True},
            "behavior": {
                "delete_source_after_copy": True,
                "delete_work_dir": True,
                "shutdown_after": True,
                "shutdown_delay": 60,
                "log_file": str(tmp_path / "cut_copy.log"),
            },
        }

    def test_skips_successful_marker_records_from_previous_runs(self, tmp_path):
        config = self._config(tmp_path)
        result_dir = tmp_path / "old_result"
        result_dir.mkdir()
        video = tmp_path / "video_fix.mp4"
        video.write_bytes(b"video")
        runs = [{
            "video": str(video),
            "result_dir": str(result_dir),
            "status": "success",
        }]

        with (
            patch("dd_clip_miner_llm.cut_copy.copy_to_destination") as mock_copy,
            patch("dd_clip_miner_llm.cut_copy.delete_source_file") as mock_delete_source,
            patch("dd_clip_miner_llm.cut_copy.delete_directory") as mock_delete_dir,
            patch("dd_clip_miner_llm.cut_copy.schedule_shutdown") as mock_shutdown,
        ):
            rc = run_batch_cut_copy(config, runs)

        assert rc == 0
        mock_copy.assert_not_called()
        mock_delete_source.assert_not_called()
        mock_delete_dir.assert_not_called()
        mock_shutdown.assert_not_called()

    def test_processes_successful_runs_from_current_batch(self, tmp_path):
        config = self._config(tmp_path)
        result_dir = tmp_path / "new_result"
        result_dir.mkdir()
        (result_dir / "out.txt").write_text("ok", encoding="utf-8")
        video = tmp_path / "video_fix.mp4"
        video.write_bytes(b"video")
        runs = [{
            "video": str(video),
            "result_dir": str(result_dir),
            "status": "success",
            "processed_this_run": True,
        }]

        with (
            patch("dd_clip_miner_llm.cut_copy.copy_to_destination", return_value=tmp_path / "dest" / "new_result") as mock_copy,
            patch("dd_clip_miner_llm.cut_copy.verify_copy", return_value=True),
            patch("dd_clip_miner_llm.cut_copy.delete_source_file") as mock_delete_source,
            patch("dd_clip_miner_llm.cut_copy.delete_directory") as mock_delete_dir,
            patch("dd_clip_miner_llm.cut_copy.schedule_shutdown") as mock_shutdown,
        ):
            rc = run_batch_cut_copy(config, runs)

        assert rc == 0
        mock_copy.assert_called_once()
        mock_delete_source.assert_called_once_with(video)
        mock_delete_dir.assert_called_once_with(result_dir)
        mock_shutdown.assert_called_once_with(60)


# ---------------------------------------------------------------------------
# _format_folder
# ---------------------------------------------------------------------------

class TestFormatFolder:
    def test_format_folder_replaces_variables(self, tmp_path):
        """{date} and {streamer} placeholders are substituted."""
        video = tmp_path / "streamer_name" / "video_fix.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"")

        config: dict = {}
        result = _format_folder("{date}_{streamer}", video, config)

        from datetime import datetime

        expected_date = datetime.now().strftime("%y%m%d")
        assert result == f"{expected_date}_streamer_name"

    def test_format_folder_strips_invalid_chars(self, tmp_path):
        """Invalid path characters in template output are replaced with underscores."""
        video = tmp_path / "goodname" / "v.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"")

        # Template itself contains invalid path chars
        result = _format_folder("{streamer}|<>?*", video, {})

        assert "|" not in result
        assert "<" not in result
        assert ">" not in result
        assert "?" not in result
        assert "*" not in result
