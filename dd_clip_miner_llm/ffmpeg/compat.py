"""Resolve symbols through the ffmpeg package for test monkeypatch compatibility."""
from __future__ import annotations

from importlib import import_module
from typing import Any

_PKG = "dd_clip_miner_llm.ffmpeg"


def pkg_attr(name: str) -> Any:
    return getattr(import_module(_PKG), name)