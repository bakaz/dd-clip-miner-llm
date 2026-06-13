from __future__ import annotations

from copy import deepcopy
from typing import Any

from .base import ASRBackend
from .faster_whisper import FasterWhisperBackend
from .funasr_backend import FunASRBackend
from .mimo_asr_backend import MimoASRBackend

from ..config import deep_merge


def _is_gpu_available() -> bool:
    """Check CUDA through CTranslate2, the runtime used by faster-whisper."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _resolve_hardware_local_config(local_cfg: dict[str, Any]) -> dict[str, Any]:
    """If gpu/cpu sections present, auto-select and merge the appropriate one
    based on hardware detection. This supports user's 'auto' expectation:
    write 'device: auto' (or omit), have gpu:/cpu: sections, code diverts.
    """
    if not isinstance(local_cfg, dict):
        return local_cfg
    local_cfg = deepcopy(local_cfg)
    backend = str(local_cfg.get("backend", "faster_whisper")).lower().replace("-", "_")
    backend_cfg = local_cfg.get(backend, {})
    if not isinstance(backend_cfg, dict):
        backend_cfg = {}
    device = str(backend_cfg.get("device", local_cfg.get("device", "auto"))).lower()
    if device != "auto":
        return local_cfg

    hardware_cfg = local_cfg
    if "gpu" not in hardware_cfg and "cpu" not in hardware_cfg:
        hardware_cfg = backend_cfg
    if "gpu" not in hardware_cfg and "cpu" not in hardware_cfg:
        return local_cfg

    is_gpu = _is_gpu_available()
    hw_key = "gpu" if is_gpu else "cpu"
    if hw_key not in hardware_cfg:
        # fallback to the other if present
        other = "cpu" if hw_key == "gpu" else "gpu"
        if other in hardware_cfg:
            hw_key = other
        else:
            return local_cfg

    hw_section = hardware_cfg[hw_key]
    if not isinstance(hw_section, dict):
        return local_cfg

    if backend in hw_section and isinstance(hw_section[backend], dict):
        selected_backend_cfg = hw_section[backend]
    else:
        selected_backend_cfg = hw_section
    local_cfg[backend] = deep_merge(backend_cfg, selected_backend_cfg)

    # Preserve generic hardware-specific keys for legacy flat configurations.
    for k, v in hw_section.items():
        if k not in ["faster_whisper", "funasr"]:
            local_cfg[k] = v

    return local_cfg


def build_asr_backend(settings: dict[str, Any]) -> ASRBackend:
    """构建 ASR 后端，支持新旧两种配置格式，以及 gpu/cpu 硬件自动分流。

    新格式支持 gpu/cpu 分流（当存在 gpu: 或 cpu: 节时，基于硬件检测自动选择并合并）：
        asr:
          mode: local
          local:
            backend: faster_whisper
            faster_whisper:
              model: small
              device: auto   # "auto" + gpu/cpu 节触发自动分流
            gpu:
              faster_whisper:
                device: cuda
                compute_type: float16
            cpu:
              faster_whisper:
                device: cpu
                compute_type: int8
                ...

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
        if not isinstance(local_cfg, dict):
            local_cfg = {}
        local_cfg = _resolve_hardware_local_config(local_cfg)
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
    if "gpu" in settings or "cpu" in settings:
        settings = _resolve_hardware_local_config(settings)
    backend = str(settings.get("backend", "faster_whisper")).lower().replace("-", "_")
    flat = {**settings}
    if isinstance(flat.get(backend), dict):
        flat = {**flat, **flat[backend]}
    return _build_local_backend(backend, flat)


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
        local_cfg = _resolve_hardware_local_config(local_cfg)
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
    # 旧格式兼容 for gpu/cpu
    if "gpu" in settings or "cpu" in settings:
        settings = _resolve_hardware_local_config(settings)
    backend = str(settings.get("backend", "faster_whisper")).lower().replace("-", "_")
    if backend in {"funasr", "fun_asr", "qwen3", "qwen3_asr", "sensevoice"}:
        funasr = settings.get("funasr", {})
        if isinstance(funasr, dict) and funasr.get("model"):
            return str(funasr["model"])
    if backend in {"whisper", "faster_whisper"} and settings.get("model"):
        return str(settings["model"])
    return str(settings.get("model", "unknown"))


def apply_asr_model_override(settings: dict[str, Any], model_name: str) -> None:
    """将 CLI --asr-model 写入新旧两种 ASR 配置结构。
    也应用到 gpu/cpu 子部分，以便硬件自动选择时覆盖生效。
    """
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
        # 应用到 gpu/cpu 子部分
        for hw in ("gpu", "cpu"):
            if hw in local_cfg and isinstance(local_cfg[hw], dict):
                hw_cfg = local_cfg[hw]
                if backend in {"funasr", "fun_asr", "qwen3", "qwen3_asr", "sensevoice"}:
                    hw_cfg.setdefault("funasr", {})["model"] = model_name
                elif backend in {"whisper", "faster_whisper"}:
                    hw_cfg.setdefault("faster_whisper", {})["model"] = model_name
        return
    if mode == "remote":
        remote_cfg = settings.setdefault("remote", {})
        if isinstance(remote_cfg, dict):
            remote_cfg["model"] = model_name
        return
    backend = str(settings.get("backend", "faster_whisper")).lower().replace("-", "_")
    if backend in {"funasr", "fun_asr", "qwen3", "qwen3_asr", "sensevoice"}:
        settings.setdefault("funasr", {})["model"] = model_name
    # 旧格式 gpu/cpu
    for hw in ("gpu", "cpu"):
        if hw in settings and isinstance(settings[hw], dict):
            hw_cfg = settings[hw]
            if backend in {"funasr", "fun_asr", "qwen3", "qwen3_asr", "sensevoice"}:
                hw_cfg.setdefault("funasr", {})["model"] = model_name
            elif backend in {"whisper", "faster_whisper"}:
                hw_cfg.setdefault("faster_whisper", {})["model"] = model_name


__all__ = [
    "ASRBackend",
    "FasterWhisperBackend",
    "FunASRBackend",
    "MimoASRBackend",
    "apply_asr_model_override",
    "build_asr_backend",
    "resolve_asr_model_name",
    "_is_gpu_available",
    "_resolve_hardware_local_config",
]
