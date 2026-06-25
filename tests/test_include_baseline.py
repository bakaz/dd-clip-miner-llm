"""Baseline characterization: existing config functionality must survive !include changes."""
from pathlib import Path

from dd_clip_miner_llm.config import load_config


def test_load_config_still_works():
    cfg = load_config("config/example/main.yaml", profile="kv_optimized")
    assert cfg["asr"]["mode"] == "local"
    assert cfg["_profile_name"] == "kv_optimized"
