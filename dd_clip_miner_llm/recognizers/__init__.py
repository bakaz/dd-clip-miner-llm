"""识别器注册表

使用方式：
    from dd_clip_miner_llm.recognizers import get_recognizer, list_recognizers
    
    # 获取识别器
    song_recognizer = get_recognizer("song")
    
    # 列出所有可用识别器
    available = list_recognizers()
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from .base import BaseRecognizer


# 识别器注册表
_REGISTRY: dict[str, type[BaseRecognizer]] = {}


def register(recognizer_class: type[BaseRecognizer]) -> type[BaseRecognizer]:
    """注册识别器（可用作装饰器）
    
    使用方式：
        @register
        class MyRecognizer(BaseRecognizer):
            ...
    """
    instance = recognizer_class()
    _REGISTRY[instance.name] = recognizer_class
    return recognizer_class


def get_recognizer(name: str) -> BaseRecognizer | None:
    """获取识别器实例
    
    Args:
        name: 识别器名称（如 "song", "dialogue"）
        
    Returns:
        识别器实例，或 None 如果未找到
    """
    _ensure_discovered()
    cls = _REGISTRY.get(name)
    if cls is None:
        return None
    return cls()


def list_recognizers() -> list[str]:
    """列出所有已注册的识别器名称"""
    _ensure_discovered()
    return list(_REGISTRY.keys())


def get_all_recognizers() -> dict[str, BaseRecognizer]:
    """获取所有识别器实例"""
    _ensure_discovered()
    return {name: cls() for name, cls in _REGISTRY.items()}


# 自动发现
_discovered = False


def _autodiscover() -> None:
    """自动发现并注册所有识别器"""
    package_path = __path__
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        if module_name not in ("base", "__init__"):
            importlib.import_module(f".{module_name}", package=__name__)


def _ensure_discovered() -> None:
    """确保已执行自动发现"""
    global _discovered
    if not _discovered:
        _autodiscover()
        _discovered = True


# 模块导入时立即执行自动发现
_ensure_discovered()
