"""Baseline characterization: streamer dictionary path migration."""
from __future__ import annotations

from dd_clip_miner_llm.config import DEFAULT_CONFIG
from dd_clip_miner_llm.clip_naming import load_streamer_dictionary


def test_new_example_path_loads():
    """Given: config/example/streamer_dictionary.json exists
    When: loaded via load_streamer_dictionary
    Then: returns valid data with at least 1 entry"""
    data, entries = load_streamer_dictionary("config/example/streamer_dictionary.json")
    assert len(entries) >= 1


def test_default_config_dictionary_path():
    """Given: DEFAULT_CONFIG
    When: reading output.clip_naming.dictionary_path
    Then: points to config/local/streamer_dictionary.json"""
    path = DEFAULT_CONFIG["output"]["clip_naming"]["dictionary_path"]
    assert path == "config/local/streamer_dictionary.json"
