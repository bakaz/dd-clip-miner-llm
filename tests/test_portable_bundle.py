"""Tests for the portable miner bundle installer."""

from __future__ import annotations

from pathlib import Path

from dd_clip_miner_llm.portable_bundle import install_portable_bundle


def test_install_portable_bundle_copies_miner_package(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()

    bundle_root = install_portable_bundle(run_root)

    package_root = bundle_root / "dd_clip_miner_llm"
    assert package_root.is_dir()
    assert (package_root / "post_merge.py").is_file()
    assert (package_root / "ffmpeg" / "__init__.py").is_file()
    assert (package_root / "assets" / "merge_mp4.bat").is_file()
    assert (bundle_root / "VERSION").is_file()