"""Tests for !include custom YAML tag constructor.

These tests MUST FAIL before the implementation is in place.
"""
from pathlib import Path

import pytest
import yaml

from dd_clip_miner_llm.config import _yaml_loader_with_include


def test_include_tag_loads_yaml(tmp_path: Path):
    (tmp_path / "sub.yaml").write_text("inner: value\n", encoding="utf-8")
    (tmp_path / "main.yaml").write_text("base: !include sub.yaml\n", encoding="utf-8")
    with open(tmp_path / "main.yaml", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_yaml_loader_with_include())
    assert data == {"base": {"inner": "value"}}


def test_include_missing_file_raises(tmp_path: Path):
    (tmp_path / "main.yaml").write_text("base: !include missing.yaml\n", encoding="utf-8")
    with open(tmp_path / "main.yaml", encoding="utf-8") as f:
        with pytest.raises(FileNotFoundError):
            yaml.load(f, Loader=_yaml_loader_with_include())


def test_include_tag_loads_json(tmp_path: Path):
    (tmp_path / "data.json").write_text('{"key": [1, 2, 3]}\n', encoding="utf-8")
    (tmp_path / "main.yaml").write_text("base: !include data.json\n", encoding="utf-8")
    with open(tmp_path / "main.yaml", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_yaml_loader_with_include())
    assert data == {"base": {"key": [1, 2, 3]}}


def test_nested_include(tmp_path: Path):
    (tmp_path / "nested.yaml").write_text("deep: 42\n", encoding="utf-8")
    (tmp_path / "mid.yaml").write_text("inner: !include nested.yaml\n", encoding="utf-8")
    (tmp_path / "main.yaml").write_text("base: !include mid.yaml\n", encoding="utf-8")
    with open(tmp_path / "main.yaml", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_yaml_loader_with_include())
    assert data == {"base": {"inner": {"deep": 42}}}


def test_include_scalar_value(tmp_path: Path):
    # Include a file containing a scalar (not mapping/sequence)
    (tmp_path / "scalar.yaml").write_text("just a string\n", encoding="utf-8")
    (tmp_path / "main.yaml").write_text("base: !include scalar.yaml\n", encoding="utf-8")
    with open(tmp_path / "main.yaml", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_yaml_loader_with_include())
    assert data == {"base": "just a string"}


def test_non_scalar_node_raises_typeerror(tmp_path: Path):
    (tmp_path / "main.yaml").write_text("base: !include [sub.yaml, other.yaml]\n", encoding="utf-8")
    with open(tmp_path / "main.yaml", encoding="utf-8") as f:
        with pytest.raises(TypeError):
            yaml.load(f, Loader=_yaml_loader_with_include())
