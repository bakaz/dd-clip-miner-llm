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

    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .")
    return (normalized[:max_length] or fallback).strip(" .") or fallback


def _staged_name(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
    suffix = path.suffix if path.suffix.isascii() else ".mp4"
    return f"input_{digest}{suffix.lower()}"


def _is_unc_path(path: Path) -> bool:
    text = str(path)
    return text.startswith("\\\\") or path.drive.startswith("\\\\")
