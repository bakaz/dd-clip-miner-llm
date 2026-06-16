"""错误分类模块

提供 LLM 调用相关错误的分类逻辑。
"""
from __future__ import annotations


def _classify_error(exc: Exception) -> tuple[bool, str]:
    """分类异常是否可重试，返回 (retryable, reason)。

    可重试：网络错误、超时、429、5xx。
    不可重试：401、403、确定的请求参数错误。
    """
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            PermissionDeniedError,
            RateLimitError,
            APIStatusError,
        )
    except ImportError:
        msg = str(exc).lower()
        if any(k in msg for k in ("timeout", "connection", "network")):
            return True, "network_error"
        return False, "unknown_error"

    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True, "network_error"
    if isinstance(exc, TimeoutError):
        return True, "timeout"
    if isinstance(exc, (ConnectionError, OSError)):
        return True, "network_error"
    if isinstance(exc, RateLimitError):
        return True, "rate_limited"
    if isinstance(exc, AuthenticationError):
        return False, "auth_error"
    if isinstance(exc, PermissionDeniedError):
        return False, "permission_denied"
    if isinstance(exc, BadRequestError):
        return False, "bad_request"
    if isinstance(exc, APIStatusError):
        code = exc.status_code
        if code == 429:
            return True, "rate_limited"
        if code >= 500:
            return True, f"server_error_{code}"
        if code in (401, 403):
            return False, f"http_{code}"
        if 400 <= code < 500:
            return False, f"client_error_{code}"
        return True, f"http_{code}"
    return False, "unknown_error"
