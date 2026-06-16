"""统一错误处理模块

定义项目的异常层次结构。

异常层次：
    DDClipMinerError (base)
    ├── LLMError
    │   └── StreamInterruptedError
    ├── ASRError
    └── FFmpegError
        └── AllConcatAttemptsFailed
"""
from __future__ import annotations


class DDClipMinerError(Exception):
    """项目基础异常类"""
    pass


class LLMError(DDClipMinerError):
    """LLM 相关错误"""
    pass


class ASRError(DDClipMinerError):
    """ASR 相关错误"""
    pass
