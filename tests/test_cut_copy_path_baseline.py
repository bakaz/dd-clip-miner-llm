"""Baseline characterization test for cut_copy config path resolution.

Given: config/example/cut_copy.yaml exists as standalone cut_copy config
When: load_cut_copy_config('config/example/cut_copy.yaml') is called
Then: required fields resolve to expected values
"""

from __future__ import annotations

from dd_clip_miner_llm.cut_copy import load_cut_copy_config


def test_load_cut_copy_example_path():
    """Explicit path to example config loads required fields."""
    cfg = load_cut_copy_config("config/example/cut_copy.yaml")
    assert cfg["source"]["path"] == "//ddtv-machine/Recordings"
    assert cfg["processing"]["config_path"] == "config/local/main.yaml"


def test_load_cut_copy_default_resolution_finds_example(tmp_path, monkeypatch):
    """No path argument → resolves to config/example/cut_copy.yaml when local is absent."""
    import yaml

    import dd_clip_miner_llm.cut_copy as cc_mod

    example_dir = tmp_path / "config" / "example"
    example_dir.mkdir(parents=True)
    example_cfg = {
        "source": {"path": "//ddtv-machine/Recordings"},
        "destination": {"path": "//nas/dd-clip-results"},
        "processing": {"config_path": "config/local/main.yaml"},
    }
    (example_dir / "cut_copy.yaml").write_text(
        yaml.dump(example_cfg, allow_unicode=True), encoding="utf-8"
    )

    missing_local = tmp_path / "config" / "local" / "cut_copy.yaml"
    monkeypatch.setattr(cc_mod, "_LOCAL_CONFIG_PATH", missing_local)
    monkeypatch.setattr(cc_mod, "_EXAMPLE_CONFIG_PATH", example_dir / "cut_copy.yaml")

    cfg = load_cut_copy_config()
    assert cfg["source"]["path"] == "//ddtv-machine/Recordings"
    assert cfg["processing"]["config_path"] == "config/local/main.yaml"


def test_load_cut_copy_default_resolution_finds_local(tmp_path, monkeypatch):
    """When config/local/cut_copy.yaml exists, it overrides example."""
    from pathlib import Path

    import yaml

    local_dir = tmp_path / "config" / "local"
    local_dir.mkdir(parents=True)
    example_dir = tmp_path / "config" / "example"
    example_dir.mkdir(parents=True)

    local_cfg = {
        "source": {"path": "/local/recordings"},
        "destination": {"path": "/local/dest"},
        "processing": {"config_path": "local_config.yaml"},
    }
    (local_dir / "cut_copy.yaml").write_text(
        yaml.dump(local_cfg, allow_unicode=True), encoding="utf-8"
    )

    example_cfg = {
        "source": {"path": "/example/recordings"},
        "destination": {"path": "/example/dest"},
        "processing": {"config_path": "example_config.yaml"},
    }
    (example_dir / "cut_copy.yaml").write_text(
        yaml.dump(example_cfg, allow_unicode=True), encoding="utf-8"
    )

    import dd_clip_miner_llm.cut_copy as cc_mod

    monkeypatch.setattr(cc_mod, "_LOCAL_CONFIG_PATH", local_dir / "cut_copy.yaml")
    monkeypatch.setattr(cc_mod, "_EXAMPLE_CONFIG_PATH", example_dir / "cut_copy.yaml")

    cfg = load_cut_copy_config()
    assert cfg["source"]["path"] == "/local/recordings"
    assert cfg["processing"]["config_path"] == "local_config.yaml"


def test_load_cut_copy_default_missing_both_raises(monkeypatch):
    """When neither config/local/ nor config/example/ cut_copy.yaml exists, raise FileNotFoundError."""
    from pathlib import Path

    import dd_clip_miner_llm.cut_copy as cc_mod

    monkeypatch.setattr(cc_mod, "_LOCAL_CONFIG_PATH", Path("/nonexistent/local/cut_copy.yaml"))
    monkeypatch.setattr(cc_mod, "_EXAMPLE_CONFIG_PATH", Path("/nonexistent/example/cut_copy.yaml"))

    import pytest

    with pytest.raises(FileNotFoundError, match="cut_copy\\.yaml"):
        load_cut_copy_config()


def test_load_cut_copy_explicit_path_still_works(tmp_path):
    """Explicit path argument bypasses default resolution."""
    from pathlib import Path

    import yaml

    cfg_data = {
        "source": {"path": "/explicit/path"},
        "destination": {"path": "/explicit/dest"},
        "processing": {"config_path": "explicit.yaml"},
    }
    cfg_file = tmp_path / "my_cut_copy.yaml"
    cfg_file.write_text(yaml.dump(cfg_data, allow_unicode=True), encoding="utf-8")

    cfg = load_cut_copy_config(str(cfg_file))
    assert cfg["source"]["path"] == "/explicit/path"
