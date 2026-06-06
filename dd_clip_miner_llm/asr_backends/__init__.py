from __future__ import annotations

from typing import Any

from .base import ASRBackend
from .faster_whisper import FasterWhisperBackend
from .funasr_backend import FunASRBackend


def build_asr_backend(settings: dict[str, Any]) -> ASRBackend:
    backend = str(settings.get("backend", "faster_whisper")).lower().replace("-", "_")
    if backend in {"whisper", "faster_whisper"}:
        return FasterWhisperBackend(settings)
    if backend in {"funasr", "fun_asr", "qwen3", "qwen3_asr"}:
        return FunASRBackend(settings)
    raise ValueError(f"Unsupported ASR backend: {settings.get('backend')}")


__all__ = ["ASRBackend", "FasterWhisperBackend", "FunASRBackend", "build_asr_backend"]
