"""Cut-copy workflow: scan DDTV output, run pipeline, copy results to SMB share.

Standalone module — no circular imports from other dd_clip_miner_llm modules.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_REQUIRED_TOP = ("source", "destination", "processing")
_REQUIRED_SOURCE = ("path",)
_REQUIRED_DEST = ("path",)
_REQUIRED_PROC = ("config_path",)

_LOCAL_CONFIG_PATH = Path("config/local/cut_copy.yaml")
_EXAMPLE_CONFIG_PATH = Path("config/example/cut_copy.yaml")


def _resolve_cut_copy_path(path: str | Path | None) -> Path:
    """Resolve the cut_copy config file path.

    Resolution order:
    1. If *path* is provided, use it directly.
    2. Try ``config/local/cut_copy.yaml``.
    3. Fall back to ``config/example/cut_copy.yaml``.
    4. Raise :class:`FileNotFoundError` with both paths listed if neither exists.
    """
    if path is not None:
        return Path(path)

    candidates = (_LOCAL_CONFIG_PATH, _EXAMPLE_CONFIG_PATH)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No cut_copy config found. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def load_cut_copy_config(path: str | Path | None = None) -> dict:
    """Load and validate a cut_copy YAML config file.

    If *path* is not provided, resolves to ``config/local/cut_copy.yaml``
    (with fallback to ``config/example/cut_copy.yaml``).

    Raises :class:`FileNotFoundError` if the config file cannot be found.
    Raises :class:`ValueError` on missing required fields.
    """
    p = _resolve_cut_copy_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must be a YAML mapping: {p}")

    missing: list[str] = []
    for section in _REQUIRED_TOP:
        if section not in cfg:
            missing.append(section)
            cfg.setdefault(section, {})

    for field in _REQUIRED_SOURCE:
        if not cfg["source"].get(field):
            missing.append(f"source.{field}")
    for field in _REQUIRED_DEST:
        if not cfg["destination"].get(field):
            missing.append(f"destination.{field}")
    for field in _REQUIRED_PROC:
        if not cfg["processing"].get(field):
            missing.append(f"processing.{field}")

    if missing:
        raise ValueError(
            "Missing required config fields: " + ", ".join(missing)
        )

    # Defaults for optional sections / fields
    cfg.setdefault("enabled", True)
    cfg.setdefault("behavior", {})
    cfg["behavior"].setdefault("shutdown_after", True)
    cfg["behavior"].setdefault("shutdown_delay", 60)
    cfg["behavior"].setdefault("delete_source_after_copy", True)
    cfg["behavior"].setdefault("delete_work_dir", True)
    cfg["behavior"].setdefault("log_file", "cut_copy.log")
    cfg["behavior"].setdefault("max_files", 0)
    cfg["behavior"].setdefault("max_runtime", 0)

    cfg["processing"].setdefault("skip_on_failure", True)

    cfg["source"].setdefault("pattern", "*_fix.mp4")
    cfg["source"].setdefault("done_marker", ".dd_clip_miner_cut_copy_done.json")

    cfg["destination"].setdefault("username", "")
    cfg["destination"].setdefault("password", "")
    cfg["destination"].setdefault("folder_format", "{date}_{streamer}")

    return cfg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(message: str, log_file: Path) -> None:
    """Print to console and append to *log_file*."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line)
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass  # don't crash on log-write failure


# ---------------------------------------------------------------------------
# Done-marker helpers
# ---------------------------------------------------------------------------

def _load_done_marker(source_path: Path, marker_name: str) -> dict:
    """Return the done-marker dict, or a fresh one if missing."""
    marker = source_path / marker_name
    if not marker.is_file():
        return {"processed": []}
    try:
        with marker.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return {"processed": []}
        data.setdefault("processed", [])
        return data
    except (json.JSONDecodeError, OSError):
        return {"processed": []}


def _save_done_marker(source_path: Path, marker_name: str, data: dict) -> None:
    """Persist *data* to the done-marker JSON file."""
    marker = source_path / marker_name
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise RuntimeError(f"Failed to write done marker {marker}: {exc}") from exc


def _mark_done(
    source_path: Path,
    marker_name: str,
    video: Path,
    result_dir: Path,
    dest_dir: Path,
) -> None:
    """Append an entry for *video* to the done-marker."""
    data = _load_done_marker(source_path, marker_name)
    entry = {
        "source": str(video),
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "result_dir": str(result_dir),
        "dest_dir": str(dest_dir),
    }
    data["processed"].append(entry)
    _save_done_marker(source_path, marker_name, data)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_pending_files(config: dict) -> list[Path]:
    """Return videos that still need processing (oldest first)."""
    source_path = Path(config["source"]["path"])
    pattern = config["source"].get("pattern", "*_fix.mp4")
    marker_name = config["source"].get(
        "done_marker", ".dd_clip_miner_cut_copy_done.json"
    )

    if not source_path.is_dir():
        _log(f"Source directory does not exist: {source_path}", Path(config.get("behavior", {}).get("log_file", "cut_copy.log")))
        return []

    all_files = sorted(source_path.glob(pattern), key=lambda p: p.stat().st_mtime)

    done = _load_done_marker(source_path, marker_name)
    done_set = {Path(e["source"]) for e in done.get("processed", []) if isinstance(e, dict)}

    pending = [f for f in all_files if f not in done_set]

    max_files = config["behavior"].get("max_files", 0)
    if max_files > 0:
        pending = pending[:max_files]

    return pending


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_video(
    video: Path,
    config: dict,
    work_base: Path,
) -> Path | None:
    """Run the dd-clip-miner-llm pipeline on *video*.

    Returns the result directory on success, ``None`` on failure.
    """
    proc = config["processing"]
    behavior = config["behavior"]
    log_file = Path(behavior["log_file"])

    work_dir = work_base / video.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        sys.executable,
        "-m",
        "dd_clip_miner_llm",
        "run",
        str(video),
        "--config",
        str(proc["config_path"]),
    ]

    _log(f"  CMD: {' '.join(cmd)}", log_file)

    proc_log = work_dir / "cut_copy_run.log"
    try:
        with proc_log.open("w", encoding="utf-8") as fh:
            result = subprocess.run(
                cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                cwd=str(work_base),
                timeout=None,
            )
        if result.returncode != 0:
            _log(
                f"  Pipeline exited with code {result.returncode}. "
                f"See {proc_log}",
                log_file,
            )
            return None
    except Exception as exc:
        _log(f"  Pipeline exception: {exc}", log_file)
        return None

    return work_dir


# ---------------------------------------------------------------------------
# SMB authentication
# ---------------------------------------------------------------------------

def _smb_auth(dest_path: str, username: str, password: str) -> None:
    """Authenticate to an SMB share via ``net use`` (Windows)."""
    if not username:
        return
    try:
        # Disconnect first to handle already-connected case
        subprocess.run(
            ["net", "use", dest_path, "/delete", "/y"],
            capture_output=True,
            timeout=30,
        )
    except Exception:
        pass

    cmd = ["net", "use", dest_path, f"/user:{username}", password]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"SMB auth failed (rc={result.returncode}): {result.stderr.strip()}"
            )
    except FileNotFoundError:
        # Not on Windows — ignore
        pass


# ---------------------------------------------------------------------------
# Copy & verify
# ---------------------------------------------------------------------------

def _format_folder(template: str, video: Path, config: dict) -> str:
    """Format *template* with date and streamer placeholders."""
    date_str = datetime.now().strftime("%y%m%d")
    # Use original video path from batch-run if available (for streamer extraction)
    src_video = config.get("_batch_video_path")
    if src_video is not None:
        src_video = Path(src_video)
    else:
        src_video = video
    # Try to extract streamer from parent folder name
    streamer = src_video.parent.name if src_video.parent.name else "unknown"
    folder = template.replace("{date}", date_str).replace("{streamer}", streamer)
    # Remove characters invalid in Windows paths
    folder = re.sub(r'[<>:"/\\|?*]', "_", folder)
    return folder


def copy_to_destination(
    result_dir: Path,
    dest_path: Path,
    config: dict,
) -> Path:
    """Copy *result_dir* to the configured SMB destination.

    Returns the destination path.
    """
    dest_cfg = config["destination"]
    folder_fmt = dest_cfg.get("folder_format", "{date}_{streamer}")

    # SMB auth if configured
    _smb_auth(
        str(dest_path),
        dest_cfg.get("username", ""),
        dest_cfg.get("password", ""),
    )

    # We need the original video path to format the folder.
    # Caller should pass it via config or we derive from result_dir name.
    # Use result_dir stem as a fallback "video" for folder formatting.
    fake_video = result_dir  # _format_folder only uses parent name
    subfolder = _format_folder(folder_fmt, fake_video, config)

    dest = dest_path / subfolder / result_dir.name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if result_dir.is_dir():
        shutil.copytree(str(result_dir), str(dest), dirs_exist_ok=True)
    else:
        shutil.copy2(str(result_dir), str(dest))

    return dest


def _finalize_copied_run(dest: Path, log_file: Path) -> None:
    try:
        from .run_relocate import relocate_run_artifacts

        relocate_run_artifacts(dest)
    except Exception as exc:
        _log(f"  Warning: failed to relocate run artifacts on destination: {exc}", log_file)

    try:
        from .portable_bundle import install_portable_bundle

        bundle_root = install_portable_bundle(dest)
        _log(f"  Installed portable miner bundle: {bundle_root}", log_file)
    except Exception as exc:
        _log(f"  Warning: failed to install portable miner bundle: {exc}", log_file)


def verify_copy(source: Path, dest: Path) -> bool:
    """Verify that *dest* matches *source* in file count and total size."""
    if source.is_dir():
        src_files = list(source.rglob("*"))
        dst_files = list(dest.rglob("*"))
        src_count = sum(1 for f in src_files if f.is_file())
        dst_count = sum(1 for f in dst_files if f.is_file())
        src_size = sum(f.stat().st_size for f in src_files if f.is_file())
        dst_size = sum(f.stat().st_size for f in dst_files if f.is_file())
    else:
        src_count = 1
        dst_count = 1
        src_size = source.stat().st_size
        dst_size = dest.stat().st_size

    if src_count != dst_count or src_size != dst_size:
        raise RuntimeError(
            f"Copy verification failed: "
            f"src={src_count} files / {src_size} bytes, "
            f"dst={dst_count} files / {dst_size} bytes"
        )
    return True


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def delete_directory(path: Path) -> None:
    """Recursively delete *path*."""
    try:
        shutil.rmtree(str(path), ignore_errors=True)
    except Exception:
        pass


def delete_source_file(video: Path) -> None:
    """Delete the source *video* file."""
    try:
        video.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def schedule_shutdown(delay: int) -> None:
    """Schedule a system shutdown after *delay* seconds."""
    if sys.platform == "win32":
        cmd = ["shutdown", "/s", "/t", str(delay)]
    else:
        minutes = max(1, delay // 60)
        cmd = ["shutdown", "-h", f"+{minutes}"]

    try:
        subprocess.run(cmd, capture_output=True, timeout=10)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cut_copy(
    conf_path: str | Path,
    *,
    dry_run: bool = False,
    no_shutdown: bool = False,
) -> int:
    """Run the cut-copy workflow.

    Returns 0 on success, 1 on failure.
    """
    config = load_cut_copy_config(conf_path)
    log_file = Path(config["behavior"]["log_file"])

    pending = scan_pending_files(config)
    if not pending:
        _log("No pending files found.", log_file)
        return 0

    _log(f"Found {len(pending)} pending file(s).", log_file)

    if dry_run:
        for v in pending:
            _log(f"  [dry-run] {v.name}", log_file)
        return 0

    work_base = Path("runs") / "cut_copy"
    work_base.mkdir(parents=True, exist_ok=True)

    source_path = Path(config["source"]["path"])
    marker_name = config["source"]["done_marker"]
    dest_path = Path(config["destination"]["path"])
    skip = config["processing"].get("skip_on_failure", True)

    processed: list[Path] = []
    start_time = time.monotonic()
    max_runtime = config["behavior"].get("max_runtime", 0)

    for video in pending:
        elapsed = time.monotonic() - start_time
        if max_runtime > 0 and elapsed > max_runtime:
            _log(f"Max runtime ({max_runtime}s) reached. Stopping.", log_file)
            break

        _log(f"Processing: {video.name}", log_file)
        result_dir = process_video(video, config, work_base)

        if result_dir is None:
            if skip:
                _log(f"  Skipped (failed): {video.name}", log_file)
                continue
            return 1

        # Copy to destination
        try:
            dest = copy_to_destination(result_dir, dest_path, config)
            verify_copy(result_dir, dest)
            _finalize_copied_run(dest, log_file)
            _log(f"  Copied to: {dest}", log_file)
        except Exception as exc:
            _log(f"  Copy failed: {exc}", log_file)
            if skip:
                continue
            return 1

        # Mark done
        try:
            _mark_done(source_path, marker_name, video, result_dir, dest)
        except Exception as exc:
            _log(f"  Warning: failed to mark done: {exc}", log_file)

        # Delete source
        if config["behavior"].get("delete_source_after_copy", True):
            delete_source_file(video)

        # Delete work dir
        if config["behavior"].get("delete_work_dir", True):
            delete_directory(result_dir)

        processed.append(video)

    _log(f"Done. Processed {len(processed)}/{len(pending)} file(s).", log_file)

    # Shutdown
    if (
        not no_shutdown
        and config["behavior"].get("shutdown_after", True)
        and processed
    ):
        delay = config["behavior"].get("shutdown_delay", 60)
        _log(f"Shutting down in {delay} seconds...", log_file)
        schedule_shutdown(delay)

    return 0


# ---------------------------------------------------------------------------
# Batch-run post-processing entry point
# ---------------------------------------------------------------------------

def run_batch_cut_copy(
    config: dict,
    runs: list[dict],
    *,
    no_shutdown: bool = False,
) -> int:
    """Run cut_copy post-processing after a batch-run completes.

    Iterates successful runs, copies results to SMB destination,
    verifies, optionally deletes source/work, and schedules shutdown.

    Parameters
    ----------
    config : dict
        Cut-copy config loaded by :func:`load_cut_copy_config`.
    runs : list[dict]
        Batch-run result records (from :func:`batch.run_batch`).
        Each dict must have at least ``video``, ``result_dir``, ``status``.
    no_shutdown : bool
        If *True*, skip shutdown even if config says to.

    Returns
    -------
    int
        0 on success, 1 on failure.
    """
    log_file = Path(config["behavior"]["log_file"])

    # Check enabled flag (default True for backward compat)
    if not config.get("enabled", True):
        _log("Cut-copy post-processing is disabled.", log_file)
        return 0

    dest_path = Path(config["destination"]["path"])
    skip = config["processing"].get("skip_on_failure", True)

    successful = [
        r for r in runs
        if r.get("status") == "success" and r.get("processed_this_run") is True
    ]
    if not successful:
        _log("No newly processed successful runs to post-process.", log_file)
        return 0

    _log(f"Cut-copy post-processing: {len(successful)} successful run(s).", log_file)

    processed: list[dict] = []
    for run in successful:
        video_path = Path(run["video"])
        result_dir = Path(run["result_dir"])

        if not result_dir.is_dir():
            _log(f"  [skip] Result dir not found: {result_dir}", log_file)
            continue

        _log(f"  Processing: {video_path.name}", log_file)

        # Store original video path for _format_folder to extract streamer name
        config["_batch_video_path"] = video_path

        # Copy to destination
        try:
            dest = copy_to_destination(result_dir, dest_path, config)
            verify_copy(result_dir, dest)
            _finalize_copied_run(dest, log_file)
            _log(f"    Copied to: {dest}", log_file)
        except Exception as exc:
            _log(f"    Copy failed: {exc}", log_file)
            if skip:
                continue
            return 1

        # Delete source file
        if config["behavior"].get("delete_source_after_copy", True):
            delete_source_file(video_path)

        # Delete work dir (result_dir)
        if config["behavior"].get("delete_work_dir", True):
            delete_directory(result_dir)

        processed.append(run)

    _log(
        f"Cut-copy done. Post-processed {len(processed)}/{len(successful)} run(s).",
        log_file,
    )

    # Shutdown
    if (
        not no_shutdown
        and config["behavior"].get("shutdown_after", True)
        and processed
    ):
        delay = config["behavior"].get("shutdown_delay", 60)
        _log(f"Shutting down in {delay} seconds...", log_file)
        schedule_shutdown(delay)

    return 0
