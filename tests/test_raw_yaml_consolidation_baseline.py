"""Baseline characterization tests for YAML loading consolidation.

Verifies that !include-aware YAML loading works in:
- _load_raw_yaml_config (cli.py) for profile enumeration
- resolve_batch_config (resolve_batch_config.py) for batch config resolution
- _load_yaml_with_includes (config.py) directly
"""

from __future__ import annotations

from pathlib import Path

from dd_clip_miner_llm.config import _load_yaml_with_includes
from dd_clip_miner_llm.cli import _load_raw_yaml_config
from dd_clip_miner_llm.resolve_batch_config import resolve_batch_config


def test_load_raw_yaml_config_supports_include():
    """Given: a modular config with !include tags
    When: _load_raw_yaml_config loads it
    Then: the included values are resolved (asr.mode == 'local')
    """
    cfg = _load_raw_yaml_config("config/example/main.yaml")
    assert cfg["asr"]["mode"] == "local"


def test_resolve_batch_config_with_modular():
    """Given: a modular config with !include tags
    When: resolve_batch_config resolves it with a dummy video
    Then: the result dict contains a 'config' key
    """
    result = resolve_batch_config("config/example/main.yaml", video=Path("dummy.mp4"))
    assert "config" in result


def test_load_yaml_with_includes_direct():
    """Given: a modular config with !include tags
    When: _load_yaml_with_includes loads it directly
    Then: asr.mode resolves correctly
    """
    cfg = _load_yaml_with_includes(Path("config/example/main.yaml"))
    assert cfg["asr"]["mode"] == "local"
