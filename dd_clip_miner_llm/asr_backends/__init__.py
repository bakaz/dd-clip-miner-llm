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


def resolve_asr_model_name(settings: dict[str, Any]) -> str:
    """返回当前 ASR 配置对应的有效模型名（用于 manifest / 日志）。"""
    mode = str(settings.get("mode", "")).lower()
    if mode == "local":
        local_cfg = settings.get("local", {})
        if not isinstance(local_cfg, dict):
            local_cfg = {}
        backend = str(local_cfg.get("backend", "faster_whisper")).lower().replace("-", "_")
        if backend in {"funasr", "fun_asr", "qwen3", "qwen3_asr", "sensevoice"}:
            funasr = local_cfg.get("funasr", {})
            if isinstance(funasr, dict) and funasr.get("model"):
                return str(funasr["model"])
        if backend in {"whisper", "faster_whisper"}:
            fw = local_cfg.get("faster_whisper", {})
            if isinstance(fw, dict) and fw.get("model"):
                return str(fw["model"])
    if mode == "remote":
        remote = settings.get("remote", {})
        if isinstance(remote, dict):
            if remote.get("model"):
                return str(remote["model"])
            if remote.get("provider"):
                return str(remote["provider"])
    backend = str(settings.get("backend", "faster_whisper")).lower().replace("-", "_")
    if backend in {"funasr", "fun_asr", "qwen3", "qwen3_asr", "sensevoice"}:
        funasr = settings.get("funasr", {})
        if isinstance(funasr, dict) and funasr.get("model"):
            return str(funasr["model"])
    if backend in {"whisper", "faster_whisper"} and settings.get("model"):
        return str(settings["model"])
    return str(settings.get("model", "unknown"))


def apply_asr_model_override(settings: dict[str, Any], model_name: str) -> None:
    """将 CLI --asr-model 写入新旧两种 ASR 配置结构。"""
    settings["model"] = model_name
    mode = str(settings.get("mode", "")).lower()
    if mode == "local":
        local_cfg = settings.setdefault("local", {})
        if not isinstance(local_cfg, dict):
            return
        backend = str(local_cfg.get("backend", "faster_whisper")).lower().replace("-", "_")
        if backend in {"funasr", "fun_asr", "qwen3", "qwen3_asr", "sensevoice"}:
            local_cfg.setdefault("funasr", {})["model"] = model_name
        elif backend in {"whisper", "faster_whisper"}:
            local_cfg.setdefault("faster_whisper", {})["model"] = model_name
        return
    if mode == "remote":
        remote_cfg = settings.setdefault("remote", {})
        if isinstance(remote_cfg, dict):
            remote_cfg["model"] = model_name
        return
    backend = str(settings.get("backend", "faster_whisper")).lower().replace("-", "_")
    if backend in {"funasr", "fun_asr", "qwen3", "qwen3_asr", "sensevoice"}:
        settings.setdefault("funasr", {})["model"] = model_name


__all__ = [
    "ASRBackend",
    "FasterWhisperBackend",
    "FunASRBackend",
    "MimoASRBackend",
    "apply_asr_model_override",
    "build_asr_backend",
    "resolve_asr_model_name",
]
