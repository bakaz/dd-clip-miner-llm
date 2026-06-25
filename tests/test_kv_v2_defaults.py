"""Proof that DEFAULT_CONFIG supplies kv_v2 defaults so direct loads without
a profile don't cause KeyError in song_postprocess/song_kv/optimizations.py."""

from dd_clip_miner_llm.config import DEFAULT_CONFIG


def test_default_config_has_kv_v2():
    """Adding kv_v2 defaults to DEFAULT_CONFIG['song']."""
    kv2 = DEFAULT_CONFIG["song"]["kv_v2"]
    assert kv2["min_cluster_size_for_review"] == 2
    assert kv2["deletion_confidence_threshold"] == 0.75
    assert kv2["unknown_deletion_confidence_threshold"] == 0.6
