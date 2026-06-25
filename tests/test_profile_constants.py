"""Test that profile name constants exist and match expected values."""
from __future__ import annotations

from dd_clip_miner_llm import config


def test_profile_constants_exist():
    assert config.PROFILE_KV_OPTIMIZED == "kv_optimized"
    assert config.PROFILE_KV_V2 == "kv_v2"
    assert config.PROFILE_ACCURACY == "accuracy"
