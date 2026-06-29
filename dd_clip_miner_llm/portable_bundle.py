"""Install a minimal dd-clip-miner-llm package for NAS post-merge/manual-cut."""

from __future__ import annotations

import importlib.resources
import shutil
import subprocess
import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_BUNDLE_ROOT_NAME = "_tools/miner"

PORTABLE_BAT_FILES = (
    "merge_mp4.bat",
    "manual_cut.bat",
    "cleanup_source.bat",
    "_resolve_env.bat",
)

_PORTABLE_FILES = (
    "__init__.py",
    "__main__.py",
    "cli.py",
    "post_merge.py",
    "manual_cut_context.py",
    "cleanup_context.py",
    "merger.py",
    "models.py",
    "config.py",
    "paths.py",
    "run_paths.py",
    "run_relocate.py",
    "clip_naming.py",
    "errors.py",
)

_PORTABLE_DIRS = (
    "ffmpeg",
    "concat",
    "assets",
)

_BAT_COMMAND_MODULES = (
    "dd_clip_miner_llm.cli",
    "dd_clip_miner_llm.post_merge",
    "dd_clip_miner_llm.manual_cut_context",
    "dd_clip_miner_llm.cleanup_context",
)


class PortableBundleError(RuntimeError):
    """Raised when a portable bundle is incomplete or fails smoke validation."""


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

    validate_portable_bundle(bundle_root)
    sync_portable_bats(root)
    return bundle_root


def refresh_portable_bundle(run_root: str | Path) -> Path:
    """Reinstall, validate, and sync portable bats for an existing run directory."""
    return install_portable_bundle(run_root)


def _bundle_validation_script(bundle_root: Path) -> str:
    root_text = str(bundle_root.resolve()).replace("\\", "\\\\")
    imports = "\n".join(
        f'importlib.import_module("{name}")' for name in _BAT_COMMAND_MODULES
    )
    return (
        "import importlib, os, sys, sysconfig\n"
        f"root = {root_text!r}\n"
        "sys.path = [root]\n"
        "for key in ('stdlib', 'platstdlib'):\n"
        "    p = sysconfig.get_path(key)\n"
        "    if p and os.path.isdir(p) and p not in sys.path:\n"
        "        sys.path.append(p)\n"
        "if os.name == 'nt':\n"
        "    for sub in ('Lib', 'DLLs'):\n"
        "        p = os.path.join(sys.prefix, sub)\n"
        "        if os.path.isdir(p) and p not in sys.path:\n"
        "            sys.path.append(p)\n"
        f"{imports}\n"
    )


def validate_portable_bundle(bundle_root: str | Path) -> None:
    """Smoke-test that the bundle can import all bat entrypoint modules."""
    root = Path(bundle_root).resolve()
    package_root = root / "dd_clip_miner_llm"
    missing = [
        name
        for name in _PORTABLE_FILES
        if not (package_root / name).is_file()
    ]
    if missing:
        raise PortableBundleError(
            "Portable bundle validation failed: missing files: "
            + ", ".join(missing)
        )
    result = subprocess.run(
        [sys.executable, "-I", "-c", _bundle_validation_script(root)],
        env={
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
            "SystemRoot": __import__("os").environ.get("SystemRoot", r"C:\Windows"),
            "PATH": __import__("os").environ.get("PATH", ""),
        },
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise PortableBundleError(f"Portable bundle validation failed: {detail}")


def sync_portable_bats(run_root: str | Path) -> list[Path]:
    """Refresh drag-drop bat tools under ``03_clips/**/song``."""
    root = Path(run_root).resolve()
    asset_root = importlib.resources.files("dd_clip_miner_llm.assets")
    clips_root = root / "03_clips"
    if not clips_root.is_dir():
        return []

    updated: list[Path] = []
    for song_dir in clips_root.rglob("song"):
        if not song_dir.is_dir():
            continue
        for name in PORTABLE_BAT_FILES:
            template = asset_root.joinpath(name)
            target = song_dir / name
            target.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
            updated.append(target)
    return updated


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