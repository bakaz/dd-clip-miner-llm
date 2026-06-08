from __future__ import annotations

from typing import Any

from .base import ASRBackend
from .faster_whisper import FasterWhisperBackend
from .funasr_backend import FunASRBackend
from .mimo_asr_backend import MimoASRBackend


def build_asr_backend(settings: dict[str, Any]) -> ASRBackend:
    """构建 ASR 后端，支持新旧两种配置格式。

    新格式：
        asr:
          mode: local  # 或 remote
          local:
            backend: funasr
            funasr: {...}
          remote:
            provider: mimo
            base_url: ...

    旧格式（兼容）：
        asr:
          backend: funasr
          funasr: {...}
    """
    mode = str(settings.get("mode", "")).lower()

    # 新格式：mode = local / remote
    if mode == "local":
        local_cfg = settings.get("local", {})
        backend = str(local_cfg.get("backend", "faster_whisper")).lower().replace("-", "_")
        # 把 local 子配置扁平化传给后端
        flat = {**local_cfg}
        if backend in flat:
            flat = {**flat, **flat[backend]}
        return _build_local_backend(backend, flat)

    if mode == "remote":
        remote_cfg = settings.get("remote", {})
        provider = str(remote_cfg.get("provider", "mimo")).lower().replace("-", "_")
        # 把 remote 子配置作为 mimo_settings 传入
        flat = {"backend": provider, "mimo": remote_cfg}
        return _build_remote_backend(provider, flat)

    # 旧格式兼容：直接看 backend 字段
    backend = str(settings.get("backend", "faster_whisper")).lower().replace("-", "_")
    return _build_local_backend(backend, settings)


def _build_local_backend(backend: str, settings: dict[str, Any]) -> ASRBackend:
    if backend in {"whisper", "faster_whisper"}:
        return FasterWhisperBackend(settings)
    if backend in {"funasr", "fun_asr", "qwen3", "qwen3_asr", "sensevoice"}:
        return FunASRBackend(settings)
    raise ValueError(f"Unsupported local ASR backend: {backend}")


def _build_remote_backend(provider: str, settings: dict[str, Any]) -> ASRBackend:
    if provider in {"mimo", "mimo_asr", "mimo_v2", "mimo_v25", "xiaomi"}:
        return MimoASRBackend(settings)
    raise ValueError(f"Unsupported remote ASR provider: {provider}")


__all__ = ["ASRBackend", "FasterWhisperBackend", "FunASRBackend", "MimoASRBackend", "build_asr_backend"]
