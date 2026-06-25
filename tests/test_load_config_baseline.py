"""Baseline characterization: modular config loading with !include and profiles.

These tests MUST FAIL before modular load_config is implemented.
"""
from pathlib import Path

import pytest

from dd_clip_miner_llm.config import ConfigError, load_config, list_profile_names


# ── Happy path: modular config ────────────────────────────────────

def test_load_config_example_default():
    """Loading the modular main.yaml without a profile returns merged config."""
    cfg = load_config("config/example/main.yaml")
    assert cfg["asr"]["mode"] == "local"
    assert cfg["llm"]["providers"]["default"]["model"] == "gpt-4o"


def test_load_config_example_profile_kv_optimized():
    """Loading with profile='kv_optimized' applies the three-stage KV pipeline."""
    cfg = load_config("config/example/main.yaml", profile="kv_optimized")
    assert cfg["_profile_name"] == "kv_optimized"
    assert cfg["song"]["pipeline"]["strategy"] == "risk_routed_kv"


def test_load_config_example_profile_accuracy():
    """Loading with profile='accuracy' keeps legacy layout with review enabled."""
    cfg = load_config("config/example/main.yaml", profile="accuracy")
    assert cfg["_profile_name"] == "accuracy"
    assert cfg["song"]["review"]["enabled"] is True
    assert cfg["llm"]["cache_friendly_prompt_layout"] is False


def test_load_config_example_profile_kv_v2():
    """Loading with profile='kv_v2' applies KV-cache-friendly settings."""
    cfg = load_config("config/example/main.yaml", profile="kv_v2")
    assert cfg["_profile_name"] == "kv_v2"
    assert cfg["song"]["review"]["transcript_scope"] == "full"


# ── Profile listing ──────────────────────────────────────────────

def test_list_profile_names_from_loaded():
    """list_profile_names returns profiles from a loaded dict."""
    loaded = {
        "default_profile": "kv_optimized",
        "profiles": {"accuracy": {}, "kv_optimized": {}, "kv_v2": {}},
    }
    names = list_profile_names(loaded)
    assert names[0] == "kv_optimized"  # default first
    assert "accuracy" in names
    assert "kv_v2" in names


def test_list_profile_names_from_dir():
    """list_profile_names scans a profiles directory when no inline profiles."""
    # config/example/profiles/ has accuracy.yaml, kv_optimized.yaml, kv_v2.yaml
    names = list_profile_names(config_dir=Path("config/example"))
    assert "accuracy" in names
    assert "kv_optimized" in names
    assert "kv_v2" in names


# ── Old-format rejection ─────────────────────────────────────────

def test_old_format_single_file_rejected():
    """Old single-file configs at project root are rejected with migration guidance."""
    import tempfile

    # Create a temp file at the project root (CWD) to exercise old-format detection.
    # The file must be at the project root for _detect_old_format to trigger.
    cwd = Path.cwd()
    tmp_file = cwd / f"_tmp_old_format_test_{tempfile.gettempprefix()}.yaml"
    try:
        tmp_file.write_text("llm:\n  model: old\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="MIGRATION.md"):
            load_config(str(tmp_file))
    finally:
        tmp_file.unlink(missing_ok=True)


# ── Directory support ────────────────────────────────────────────

def test_directory_looks_for_config_yaml():
    """Passing a directory looks for config.yaml inside it."""
    cfg = load_config("config/example")
    assert cfg["asr"]["mode"] == "local"


def test_directory_without_config_yaml_raises(tmp_path: Path):
    """Empty directory raises ConfigError."""
    empty_dir = tmp_path / "empty_config"
    empty_dir.mkdir()
    with pytest.raises(ConfigError, match="does not contain config.yaml or main.yaml"):
        load_config(empty_dir)


# ── Temp-file configs still work (not flagged as old format) ─────

def test_temp_file_config_still_works(tmp_path: Path):
    """Configs created in temp directories are not rejected."""
    config_file = tmp_path / "my_config.yaml"
    config_file.write_text("asr:\n  mode: remote\n", encoding="utf-8")
    cfg = load_config(config_file)
    assert cfg["asr"]["mode"] == "remote"


def test_temp_file_with_profile(tmp_path: Path):
    """Temp configs with inline profiles still work."""
    config_file = tmp_path / "profiles.yaml"
    config_file.write_text("""
profiles:
  test_p:
    llm:
      model: test-model
""", encoding="utf-8")
    cfg = load_config(config_file, profile="test_p")
    assert cfg["_profile_name"] == "test_p"
    assert cfg["llm"]["model"] == "test-model"


# ── None path defaults to modular config ─────────────────────────

def test_load_config_none_falls_back():
    """load_config(None) auto-detects config/example/main.yaml."""
    cfg = load_config(None)
    assert "audio" in cfg
    assert "padding" in cfg
    # When config/example/main.yaml is loaded, profiles are applied by default
    assert "_profile_name" in cfg


# ── Profile not found ────────────────────────────────────────────

def test_unknown_profile_raises():
    """Requesting a non-existent profile raises ValueError."""
    with pytest.raises(ValueError, match="Unknown config profile"):
        load_config("config/example/main.yaml", profile="nonexistent")


def test_reserved_profile_all_rejected():
    """The 'all' profile name is reserved and raises."""
    with pytest.raises(ValueError, match="reserved CLI value"):
        load_config("config/example/main.yaml", profile="all")


# ── Malformed input ──────────────────────────────────────────────

def test_non_existent_file_raises():
    """Non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")
