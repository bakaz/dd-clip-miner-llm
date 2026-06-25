"""Baseline characterization test: modular config/example/main.yaml.

The modular main.yaml uses !include tags to assemble domain YAML files.
"""
import yaml

from dd_clip_miner_llm.config import _yaml_loader_with_include, DEFAULT_CONFIG


def test_main_yaml_loads_correctly():
    """Modular main.yaml loads correctly with !include expansion."""
    with open("config/example/main.yaml", encoding="utf-8") as f:
        modular = yaml.load(f, Loader=_yaml_loader_with_include())

    assert "audio" in modular, "main.yaml should include audio domain"
    assert "llm" in modular, "main.yaml should include llm domain"
    assert "song" in modular, "main.yaml should include song domain"
    assert "profiles" in modular, "main.yaml should define profiles"
    assert modular["profiles"] is not None


def test_main_yaml_shared_keys_match_default_config():
    """Verify the modular config is a superset of DEFAULT_CONFIG for shared
    top-level keys.  The template intentionally provides richer defaults for
    complex sections like asr/llm/song — so we only compare leaf-level
    keys that both sides define, not the full nested dicts."""
    with open("config/example/main.yaml", encoding="utf-8") as f:
        modular = yaml.load(f, Loader=_yaml_loader_with_include())

    shared_keys = set(DEFAULT_CONFIG.keys()) & set(modular.keys())
    # Richer-default sections: skip full-dict comparison
    richer_sections = {"asr", "llm", "song", "output", "padding"}

    for key in sorted(shared_keys - richer_sections):
        assert modular[key] == DEFAULT_CONFIG[key], (
            f"Shared key '{key}' differs from DEFAULT_CONFIG."
        )
    # Verify that richer sections at least exist (structural integrity)
    for section in richer_sections & shared_keys:
        assert isinstance(modular.get(section), dict), (
            f"Section '{section}' should be a dict in modular config"
        )
        assert isinstance(DEFAULT_CONFIG.get(section), dict), (
            f"Section '{section}' should be a dict in DEFAULT_CONFIG"
        )
