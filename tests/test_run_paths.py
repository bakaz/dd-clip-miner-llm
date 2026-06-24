"""Tests for portable run path resolution."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dd_clip_miner_llm.run_paths import (
    as_run_relative,
    portable_run_dir,
    resolve_input_video,
    resolve_run_path,
)


def _write_run(root: Path) -> dict[str, Path]:
    run = root / "demo_fix"
    source = run / "00_input" / "input.mp4"
    video_dir = run / "03_clips" / "video" / "song"
    manifest = run / "manifest.json"
    for path in (source.parent, video_dir):
        path.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"video")
    manifest.write_text(
        json.dumps({"input_video": str(source), "total_duration": 90.0}),
        encoding="utf-8",
    )
    context = {
        "run_dir": str(root / "stale" / "demo_fix"),
        "manifest_path": str(root / "stale" / "demo_fix" / "manifest.json"),
        "input_video": str(root / "stale" / "demo_fix" / "00_input" / "input.mp4"),
    }
    context_path = video_dir / "merge_recut_context.json"
    context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
    return {"run": run, "source": source, "context": context_path, "video_dir": video_dir}


def test_portable_run_dir_prefers_03_clips_parent(tmp_path):
    fixture = _write_run(tmp_path)
    recorded = tmp_path / "stale" / "demo_fix"
    derived = portable_run_dir(fixture["context"].parent, recorded)
    assert derived == fixture["run"].resolve()


def test_resolve_run_path_supports_relative_and_absolute(tmp_path):
    fixture = _write_run(tmp_path)
    run_root = fixture["run"]
    relative = as_run_relative(fixture["source"], run_root)

    assert resolve_run_path(relative, run_root=run_root) == fixture["source"].resolve()
    assert resolve_run_path(
        fixture["source"],
        run_root=run_root,
        recorded_run_dir=tmp_path / "stale" / "demo_fix",
    ) == fixture["source"].resolve()


def test_resolve_input_video_from_stale_absolute_paths(tmp_path):
    fixture = _write_run(tmp_path)
    fixture["source"].unlink()
    recorded = tmp_path / "stale" / "demo_fix"
    stale_source = recorded / "00_input" / "input.mp4"
    stale_source.parent.mkdir(parents=True, exist_ok=True)
    stale_source.write_bytes(b"video")

    context = json.loads(fixture["context"].read_text(encoding="utf-8"))
    resolved = resolve_input_video(
        fixture["run"],
        context=context,
        recorded_run_dir=recorded,
    )
    assert resolved == stale_source.resolve()


def test_resolve_run_path_missing_file_returns_best_candidate(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    candidate = resolve_run_path("00_input/missing.mp4", run_root=run_root)
    assert candidate == run_root / "00_input" / "missing.mp4"