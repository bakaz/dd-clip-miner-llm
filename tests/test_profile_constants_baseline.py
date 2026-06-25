"""Baseline characterization: profile helpers unchanged after constant migration."""
from __future__ import annotations

from dd_clip_miner_llm.config import load_config, is_kv_v3, is_risk_routed_kv


def test_profile_helpers_unchanged():
    cfg_kv = load_config("config/example/main.yaml", profile="kv_optimized")
    assert cfg_kv["_profile_name"] == "kv_optimized"
    assert is_risk_routed_kv(cfg_kv) is True

    cfg_acc = load_config("config/example/main.yaml", profile="accuracy")
    assert cfg_acc["_profile_name"] == "accuracy"
    assert is_kv_v3(cfg_acc) is False
    assert is_risk_routed_kv(cfg_acc) is False
