"""Install a minimal dd-clip-miner-llm package for NAS post-merge/manual-cut."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_BUNDLE_ROOT_NAME = "_tools/miner"

_PORTABLE_FILES = (
    "__init__.py",
    "__main__.py",
    "cli.py",
    "post_merge.py",
    "manual_cut_context.py",
    "merger.py",
    "models.py",
    "config.py",
    "paths.py",
    "run_paths.py",
    "run_relocate.py",
    "clip_naming.py",
)

_PORTABLE_DIRS = (
    "ffmpeg",
    "assets",
)


def install_portable_bundle(run_root: str | Path) -> Path:
    """Copy the minimal miner package into ``<run_root>/_tools/miner``."""
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run root not found: {root}")

    bundle_root = root / _BUNDLE_ROOT_NAME
    package_dest = bundle_root / "dd_clip_miner_llm"
    if package_dest.exists():
        shutil.rmtree(package_dest)

    package_dest.mkdir(parents=True, exist_ok=True)
    for name in _PORTABLE_FILES:
        source = _PACKAGE_DIR / name
        if source.is_file():
            shutil.copy2(source, package_dest / name)

    for name in _PORTABLE_DIRS:
        source = _PACKAGE_DIR / name
        if source.is_dir():
            shutil.copytree(source, package_dest / name)

    version_path = bundle_root / "VERSION"
    version_path.write_text(_bundle_version_text(), encoding="utf-8")
    return bundle_root


def _bundle_version_text() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PACKAGE_DIR.parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"