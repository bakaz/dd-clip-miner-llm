"""Tests for cut_copy scheduled-task path probing."""

from __future__ import annotations

import json
from pathlib import Path

from dd_clip_miner_llm.cut_copy_task import (
    PathCheck,
    check_path_ready,
    path_checks_from_task,
    resolve_task_paths,
    wait_for_paths,
)
from dd_clip_miner_llm.cut_copy_task import TaskPaths


def _write_cut_copy_conf(path: Path, *, source: str, destination: str = "") -> Path:
    lines = [
        "source:",
        f"  path: {json.dumps(source, ensure_ascii=False)}",
        '  pattern: "*_fix.mp4"',
        "destination:",
        f"  path: {json.dumps(destination, ensure_ascii=False)}",
        "processing:",
        '  config_path: "config/local/main.yaml"',
        "behavior:",
        '  log_file: "cut_copy.log"',
    ]
    conf = path / "cut_copy.conf"
    conf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return conf


class TestCheckPathReady:
    def test_readable_succeeds_when_directory_lists(self, tmp_path: Path):
        (tmp_path / "child.txt").write_text("x", encoding="utf-8")
        ok, detail = check_path_ready(str(tmp_path), "readable")
        assert ok is True
        assert detail == "readable"

    def test_readable_fails_for_missing_directory(self, tmp_path: Path):
        ok, detail = check_path_ready(str(tmp_path / "missing"), "readable")
        assert ok is False
        assert detail == "not_a_directory"

    def test_writable_succeeds_when_probe_file_can_be_created(self, tmp_path: Path):
        ok, detail = check_path_ready(str(tmp_path), "writable")
        assert ok is True
        assert detail == "writable"

    def test_skip_always_ready(self):
        ok, detail = check_path_ready("", "skip")
        assert ok is True
        assert detail == "skipped"


class TestResolveTaskPaths:
    def test_resolves_relative_pipeline_config(self, tmp_path: Path):
        conf = _write_cut_copy_conf(
            tmp_path,
            source=r"\\nas\recordings\12345_StreamerName",
            destination=r"\\nas\result",
        )
        paths = resolve_task_paths(conf, project_root=tmp_path)
        assert paths.source_path == r"\\nas\recordings\12345_StreamerName"
        assert paths.destination_path == r"\\nas\result"
        assert paths.pipeline_config == str((tmp_path / "config/local/main.yaml").resolve())


class TestWaitForPaths:
    def test_returns_true_when_checks_pass_immediately(self, tmp_path: Path):
        messages: list[str] = []

        def _log(msg: str, level: str = "INFO") -> None:
            messages.append(f"[{level}] {msg}")

        checks = [PathCheck("source", str(tmp_path), "readable")]
        assert wait_for_paths(
            checks,
            wait_minutes=1,
            poll_seconds=1,
            log=_log,
        )
        assert any("ready on attempt 1" in item for item in messages)

    def test_returns_false_after_timeout(self, tmp_path: Path):
        checks = [PathCheck("source", str(tmp_path / "missing"), "readable")]

        assert not wait_for_paths(
            checks,
            wait_minutes=0,
            poll_seconds=1,
            log=lambda *_args: None,
        )


class TestPathChecksFromTask:
    def test_includes_destination_writable_check(self):
        paths = TaskPaths(
            source_path=r"\\nas\src",
            destination_path=r"\\nas\dst",
            pipeline_config="config/local/main.yaml",
            cut_copy_conf="config/local/cut_copy.conf",
        )
        checks = path_checks_from_task(paths)
        assert len(checks) == 2
        assert checks[0].mode == "readable"
        assert checks[1].mode == "writable"

    def test_omits_destination_when_empty(self):
        paths = TaskPaths(
            source_path=r"\\nas\src",
            destination_path="",
            pipeline_config="config/local/main.yaml",
            cut_copy_conf="config/local/cut_copy.conf",
        )
        checks = path_checks_from_task(paths)
        assert len(checks) == 1