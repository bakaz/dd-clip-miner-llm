from __future__ import annotations

import shutil
from pathlib import Path

def safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def safe_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def short_error(exc: Exception, max_length: int = 500) -> str:
    text = str(exc).strip().replace("\r\n", "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 6:
        lines = [*lines[:3], "...", *lines[-2:]]
    summary = " | ".join(lines)
    if len(summary) > max_length:
        return summary[: max_length - 3] + "..."
    return summary
