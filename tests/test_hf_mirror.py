from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from dd_clip_miner_llm.config import apply_hf_mirror, load_config


@pytest.fixture(autouse=True)
def _clear_hf_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)


def test_apply_hf_mirror_sets_default_endpoint() -> None:
    endpoint = apply_hf_mirror({})
    assert endpoint == "https://hf-mirror.com"
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"


def test_apply_hf_mirror_respects_existing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://existing.example")
    endpoint = apply_hf_mirror({"asr": {"hf_endpoint": "https://hf-mirror.com"}})
    assert endpoint == "https://hf-mirror.com"
    assert os.environ["HF_ENDPOINT"] == "https://existing.example"


def test_apply_hf_mirror_disabled_with_null() -> None:
    endpoint = apply_hf_mirror({"asr": {"hf_endpoint": None}})
    assert endpoint is None
    assert "HF_ENDPOINT" not in os.environ


def test_load_config_applies_hf_mirror(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"asr": {"hf_endpoint": "https://hf-mirror.com"}}),
        encoding="utf-8",
    )
    load_config(config_path)
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"