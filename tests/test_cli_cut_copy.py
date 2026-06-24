"""Tests for batch-run cut_copy config resolution."""

from __future__ import annotations

from pathlib import Path

import yaml

from dd_clip_miner_llm.cli import _CUT_COPY_CONF_FROM_CONFIG, resolve_batch_cut_copy_conf


def _write_config(path: Path, *, enabled: bool, conf_path: str = "cut_copy.conf") -> Path:
    config_file = path / "config.yaml"
    config_file.write_text(
        yaml.dump({"cut_copy": {"enabled": enabled, "conf_path": conf_path}}, allow_unicode=True),
        encoding="utf-8",
    )
    (path / conf_path).write_text("enabled: true\nsource:\n  path: .\ndestination:\n  path: .\nprocessing:\n  config_path: config.yaml\n", encoding="utf-8")
    return config_file


def test_resolve_batch_cut_copy_conf_when_enabled_and_flag_omitted(tmp_path):
    config_file = _write_config(tmp_path, enabled=True)

    resolved = resolve_batch_cut_copy_conf(config_file, None)

    assert resolved == str(tmp_path / "cut_copy.conf")


def test_resolve_batch_cut_copy_conf_when_disabled_and_flag_omitted(tmp_path):
    config_file = _write_config(tmp_path, enabled=False)

    resolved = resolve_batch_cut_copy_conf(config_file, None)

    assert resolved is None


def test_resolve_batch_cut_copy_conf_flag_without_path_reads_config(tmp_path):
    config_file = _write_config(tmp_path, enabled=False, conf_path="my_cut_copy.conf")

    resolved = resolve_batch_cut_copy_conf(config_file, _CUT_COPY_CONF_FROM_CONFIG)

    assert resolved == str(tmp_path / "my_cut_copy.conf")


def test_resolve_batch_cut_copy_conf_explicit_path(tmp_path):
    config_file = _write_config(tmp_path, enabled=False)
    explicit = tmp_path / "explicit.conf"
    explicit.write_text("enabled: true\nsource:\n  path: .\ndestination:\n  path: .\nprocessing:\n  config_path: config.yaml\n", encoding="utf-8")

    resolved = resolve_batch_cut_copy_conf(config_file, str(explicit))

    assert resolved == str(explicit)