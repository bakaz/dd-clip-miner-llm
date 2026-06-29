"""Tests for the portable miner bundle installer."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dd_clip_miner_llm.portable_bundle import (
    PortableBundleError,
    _bundle_validation_script,
    install_portable_bundle,
    sync_portable_bats,
    validate_portable_bundle,
)

_BAT_COMMAND_MODULES = (
    "dd_clip_miner_llm.cli",
    "dd_clip_miner_llm.post_merge",
    "dd_clip_miner_llm.manual_cut_context",
    "dd_clip_miner_llm.cleanup_context",
)

_BAT_CLI_COMMANDS = (
    ("post-merge", ["post-merge", "--help"]),
    ("manual-cut-context", ["manual-cut-context", "--help"]),
    ("cleanup-source", ["cleanup-source", "--help"]),
)


def _import_bundle_modules(bundle_root: Path) -> str | None:
    result = subprocess.run(
        [sys.executable, "-c", _bundle_validation_script()],
        cwd=bundle_root,
        env={
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(bundle_root),
            "SystemRoot": __import__("os").environ.get("SystemRoot", r"C:\Windows"),
            "PATH": __import__("os").environ.get("PATH", ""),
        },
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return (result.stderr or result.stdout).strip()
    return None


def test_install_portable_bundle_copies_miner_package(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()

    bundle_root = install_portable_bundle(run_root)

    package_root = bundle_root / "dd_clip_miner_llm"
    assert package_root.is_dir()
    assert (package_root / "post_merge.py").is_file()
    assert (package_root / "errors.py").is_file()
    assert (package_root / "cleanup_context.py").is_file()
    assert (package_root / "ffmpeg" / "__init__.py").is_file()
    assert (package_root / "concat" / "models.py").is_file()
    assert (package_root / "assets" / "merge_mp4.bat").is_file()
    assert (package_root / "assets" / "cleanup_source.bat").is_file()
    assert (bundle_root / "VERSION").is_file()


def test_portable_bundle_imports_cli_for_bat_commands(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    bundle_root = install_portable_bundle(run_root)

    assert _import_bundle_modules(bundle_root) is None


def test_sync_portable_bats_updates_song_export_dirs(tmp_path):
    run_root = tmp_path / "run"
    song_dir = run_root / "03_clips" / "kv_optimized" / "video" / "song"
    song_dir.mkdir(parents=True)
    stale_bat = song_dir / "merge_mp4.bat"
    stale_bat.write_text("stale", encoding="utf-8")

    updated = sync_portable_bats(run_root)

    assert stale_bat in updated
    assert "collect_files" in stale_bat.read_text(encoding="utf-8")


def test_validate_portable_bundle_rejects_incomplete_bundle(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    bundle_root = install_portable_bundle(run_root)
    package_dir = bundle_root / "dd_clip_miner_llm"
    (package_dir / "post_merge.py").unlink()
    pycache = package_dir / "__pycache__"
    if pycache.is_dir():
        shutil.rmtree(pycache)

    with pytest.raises(PortableBundleError, match="validation failed"):
        validate_portable_bundle(bundle_root)


def test_refresh_portable_cli(tmp_path):
    run_root = tmp_path / "run"
    song_dir = run_root / "03_clips" / "video" / "song"
    song_dir.mkdir(parents=True)

    result = subprocess.run(
        [sys.executable, "-m", "dd_clip_miner_llm", "refresh-portable", str(run_root)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert (run_root / "_tools" / "miner" / "dd_clip_miner_llm" / "errors.py").is_file()
    assert "collect_files" in (song_dir / "merge_mp4.bat").read_text(encoding="utf-8")


def test_all_bat_cli_commands_start_from_portable_bundle(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    bundle_root = install_portable_bundle(run_root)
    env = {
        "PYTHONPATH": str(bundle_root),
        "PYTHONNOUSERSITE": "1",
        "SystemRoot": __import__("os").environ.get("SystemRoot", r"C:\Windows"),
        "PATH": __import__("os").environ.get("PATH", ""),
    }

    for label, args in _BAT_CLI_COMMANDS:
        result = subprocess.run(
            [sys.executable, "-m", "dd_clip_miner_llm", *args],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode == 0, f"{label} failed: {result.stderr or result.stdout}"