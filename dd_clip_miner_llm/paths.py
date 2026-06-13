from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".flv", ".avi", ".webm", ".ts", ".m4v",
}


def resolve_existing_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Input video not found: {path}\n"
        "If this path contains Chinese/Japanese characters and was typed through a "
        "legacy terminal, put the path in a UTF-8 text file and pass the file path "
        "to batch-run, or run from PowerShell 7 / Windows Terminal."
    )


def should_stage_for_ffmpeg(path: Path) -> bool:
    text = str(path)
    return not text.isascii() or _is_unc_path(path)


def stage_input_for_ffmpeg(input_path: str | Path, staging_dir: str | Path) -> Path:
    source = resolve_existing_path(input_path)
    if not should_stage_for_ffmpeg(source):
        return source

    target_dir = Path(staging_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / _staged_name(source)

    if target.exists() and target.stat().st_size == source.stat().st_size:
        return target
    if target.exists():
        target.unlink()

    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def iter_video_files(root: str | Path, extensions: set[str] | None = None) -> list[Path]:
    base = Path(root).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"Input root not found: {base}")
    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in (extensions or VIDEO_EXTENSIONS)}
    return sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def safe_path_part(value: str, fallback: str = "item", max_length: int = 120) -> str:
    import re
    import unicodedata

    if not value:
        return fallback
    # Remove replacement chars (mojibake artifacts) and control / invalid fs chars
    value = value.replace("\ufffd", "")
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = re.sub(r'[<>:"/\\|?*\x00-\x1f\ufffd]', "_", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .")
    # Keep CJK, letters, digits, common safe punctuation; the above already protects fs
    return (normalized[:max_length] or fallback).strip(" .") or fallback


def _staged_name(path: Path) -> str:
    """Generate a stable staged filename for a video.

    Uses content-based hash (file size + head bytes) instead of path string.
    This ensures that *the same video content* always resolves to the same
    target name in 00_input/, even when the caller passes:
      - the original UNC path,
      - a previous staged copy,
      - or a different absolute path string for the identical file.

    This greatly improves reliability of ASR/audio reuse across
    re-runs, AB对照 (different profiles on same video), and cases where
    stage_input_for_ffmpeg is called with different input representations.
    """
    if not path.exists():
        # fallback to path hash if file not readable yet
        digest = hashlib.sha1(str(path.resolve()).encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
    else:
        size = path.stat().st_size
        with path.open("rb") as f:
            head = f.read(1024 * 1024)  # 1 MiB head is plenty for stable ID
        digest = hashlib.sha1(head + str(size).encode("ascii")).hexdigest()[:12]
    suffix = path.suffix if path.suffix.isascii() else ".mp4"
    return f"input_{digest}{suffix.lower()}"


def _is_unc_path(path: Path) -> bool:
    text = str(path)
    return text.startswith("\\\\") or path.drive.startswith("\\\\")
