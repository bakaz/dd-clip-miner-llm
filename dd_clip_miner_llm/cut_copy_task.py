"""Scheduled-task launcher for cut_copy batch-run workflows.

Path readiness is checked entirely in Python (no PowerShell string passing).
Readiness means the share responds to real I/O, not merely ``Path.exists()``:

- *readable*: directory exists and ``os.scandir`` succeeds
- *writable*: directory exists and a probe file can be created and removed
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .cut_copy import load_cut_copy_config

_PROBE_NAME = ".dd_clip_miner_path_probe"
PathCheckMode = Literal["readable", "writable", "skip"]


@dataclass(frozen=True)
class PathCheck:
    label: str
    path: str
    mode: PathCheckMode


@dataclass(frozen=True)
class TaskPaths:
    source_path: str
    destination_path: str
    pipeline_config: str
    cut_copy_conf: str


def _resolve_against_base(path: str, base: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((base / candidate).resolve())


def resolve_task_paths(
    cut_copy_conf: str | Path,
    *,
    project_root: str | Path = ".",
    input_root: str = "",
) -> TaskPaths:
    """Load batch-run paths from a cut_copy workflow config."""
    conf_path = Path(cut_copy_conf).resolve()
    base = Path(project_root).resolve()
    cfg = load_cut_copy_config(conf_path)

    source_path = input_root.strip() or str(cfg.get("source", {}).get("path", "") or "")
    if not source_path:
        raise ValueError("Batch input root is empty. Set source.path or pass --input-root.")

    pipeline_config = str(cfg.get("processing", {}).get("config_path", "") or "")
    if not pipeline_config:
        raise ValueError("processing.config_path is empty in cut_copy conf.")
    pipeline_config = _resolve_against_base(pipeline_config, base)

    destination_path = str(cfg.get("destination", {}).get("path", "") or "")

    return TaskPaths(
        source_path=source_path,
        destination_path=destination_path,
        pipeline_config=pipeline_config,
        cut_copy_conf=str(conf_path),
    )


def path_checks_from_task(paths: TaskPaths) -> list[PathCheck]:
    checks = [PathCheck("source", paths.source_path, "readable")]
    if paths.destination_path.strip():
        checks.append(
            PathCheck("destination", paths.destination_path, "writable")
        )
    return checks


def check_path_ready(path: str, mode: PathCheckMode) -> tuple[bool, str]:
    """Return ``(ready, detail)`` after a functional SMB/UNC probe."""
    if mode == "skip" or not path.strip():
        return True, "skipped"

    target = Path(path)
    if not target.is_dir():
        return False, "not_a_directory"

    if mode == "readable":
        try:
            next(target.iterdir(), None)
        except OSError as exc:
            return False, f"unreadable: {exc}"
        return True, "readable"

    if mode == "writable":
        probe = target / _PROBE_NAME
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            if probe.exists():
                try:
                    probe.unlink()
                except OSError:
                    pass
            return False, f"not_writable: {exc}"
        return True, "writable"

    return False, f"unknown_mode:{mode}"


def wait_for_paths(
    checks: list[PathCheck],
    *,
    wait_minutes: int,
    poll_seconds: int,
    log: Callable[[str, str], None],
) -> bool:
    """Poll until all checks pass or the wait window expires."""
    deadline = time.monotonic() + max(0, wait_minutes) * 60
    attempt = 0

    while time.monotonic() < deadline:
        attempt += 1
        statuses: list[str] = []
        ready = True

        for check in checks:
            ok, detail = check_path_ready(check.path, check.mode)
            statuses.append(f"{check.label}={ok}({detail})")
            if not ok:
                ready = False

        if ready:
            log(
                f"Network paths ready on attempt {attempt}: {', '.join(statuses)}",
                "INFO",
            )
            return True

        remaining = max(0, int((deadline - time.monotonic()) / 60))
        log(
            "Waiting for network paths "
            f"(attempt {attempt}, ~{remaining} min left): {', '.join(statuses)}",
            "INFO",
        )
        time.sleep(max(1, poll_seconds))

    return False


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def make_file_logger(log_path: Path) -> Callable[[str, str], None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(message: str, level: str = "INFO") -> None:
        line = f"[{_utc_timestamp()}] [{level}] {message}"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line)

    return _log


def run_batch_run(paths: TaskPaths, *, project_root: Path) -> int:
    cmd = [
        sys.executable,
        "-m",
        "dd_clip_miner_llm",
        "batch-run",
        paths.source_path,
        "--result-root",
        "runs/batch",
        "--work-root",
        "runs/batch",
        "--config",
        paths.pipeline_config,
        "--cut-copy-conf",
        paths.cut_copy_conf,
    ]
    result = subprocess.run(cmd, cwd=str(project_root))
    return int(result.returncode)


def run_cut_copy_task(
    cut_copy_conf: str | Path,
    *,
    project_root: str | Path = ".",
    input_root: str = "",
    network_wait_minutes: int = 45,
    network_poll_seconds: int = 30,
    log_file: str | Path = "cut_copy_task.log",
) -> int:
    """Wait for SMB paths, then run batch-run. Returns process exit code."""
    root = Path(project_root).resolve()
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = root / log_path
    log = make_file_logger(log_path)

    log(
        "Launcher started "
        f"(user={os.environ.get('USERDOMAIN', '')}\\{os.environ.get('USERNAME', '')}, "
        f"pid={os.getpid()}, python={sys.executable})"
    )
    log(f"CutCopyConf={Path(cut_copy_conf).resolve()} ProjectRoot={root}")

    try:
        paths = resolve_task_paths(
            cut_copy_conf,
            project_root=root,
            input_root=input_root,
        )
    except (OSError, ValueError) as exc:
        log(f"Failed to resolve cut_copy task paths: {exc}", "ERROR")
        return 2

    checks = path_checks_from_task(paths)
    watch_summary = "; ".join(f"{c.label}|{c.path}" for c in checks)
    log(f"Pipeline config: {paths.pipeline_config}")
    log(f"Watching paths: {watch_summary}")

    if not wait_for_paths(
        checks,
        wait_minutes=network_wait_minutes,
        poll_seconds=network_poll_seconds,
        log=log,
    ):
        log(
            f"Network paths not ready within {network_wait_minutes} minutes; "
            "skipping this run.",
            "WARN",
        )
        return 0

    log(f"Starting batch-run: {' '.join([sys.executable, '-m', 'dd_clip_miner_llm', 'batch-run', paths.source_path, '...'])}")
    started = time.monotonic()
    exit_code = run_batch_run(paths, project_root=root)
    elapsed = int(time.monotonic() - started)

    if exit_code == 0:
        log(f"batch-run finished successfully in {elapsed}s")
    else:
        log(f"batch-run failed with exit code {exit_code} after {elapsed}s", "ERROR")
    return exit_code


def task_paths_to_json(paths: TaskPaths) -> dict[str, str]:
    return {
        "source_path": paths.source_path,
        "destination_path": paths.destination_path,
        "pipeline_config": paths.pipeline_config,
        "cut_copy_conf": paths.cut_copy_conf,
    }


def probe_paths_json(
    cut_copy_conf: str | Path,
    *,
    project_root: str | Path = ".",
    input_root: str = "",
) -> dict[str, object]:
    paths = resolve_task_paths(
        cut_copy_conf,
        project_root=project_root,
        input_root=input_root,
    )
    checks = path_checks_from_task(paths)
    statuses: dict[str, object] = {}
    all_ready = True
    for check in checks:
        ok, detail = check_path_ready(check.path, check.mode)
        statuses[check.label] = {"ready": ok, "detail": detail, "path": check.path}
        if not ok:
            all_ready = False
    return {
        **task_paths_to_json(paths),
        "all_ready": all_ready,
        "checks": statuses,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wait for cut_copy SMB paths and run batch-run (scheduled task)."
    )
    parser.add_argument("--conf", required=True, help="cut_copy.conf path")
    parser.add_argument("--project-root", default=".", help="Repository root")
    parser.add_argument("--input-root", default="", help="Override source.path")
    parser.add_argument(
        "--network-wait-minutes",
        type=int,
        default=45,
        help="Max minutes to wait for SMB paths (default: 45)",
    )
    parser.add_argument(
        "--network-poll-seconds",
        type=int,
        default=30,
        help="Seconds between readiness probes (default: 30)",
    )
    parser.add_argument(
        "--log-file",
        default="cut_copy_task.log",
        help="Launcher log file (relative to project root unless absolute)",
    )
    parser.add_argument(
        "--resolve-json",
        action="store_true",
        help="Print resolved task paths as JSON and exit",
    )
    parser.add_argument(
        "--resolve-json-file",
        default="",
        help="Write resolved task paths JSON to this UTF-8 file and exit",
    )
    parser.add_argument(
        "--probe-json",
        action="store_true",
        help="Probe path readiness and print JSON (exit 1 if not all ready)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve()

    if args.resolve_json or args.resolve_json_file:
        try:
            paths = resolve_task_paths(
                args.conf,
                project_root=root,
                input_root=args.input_root,
            )
            payload = task_paths_to_json(paths)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.resolve_json_file:
            out = Path(args.resolve_json_file)
            out.write_text(text + "\n", encoding="utf-8")
        else:
            sys.stdout.write(text + "\n")
        return 0

    if args.probe_json:
        try:
            payload = probe_paths_json(
                args.conf,
                project_root=root,
                input_root=args.input_root,
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return 0 if payload.get("all_ready") else 1

    return run_cut_copy_task(
        args.conf,
        project_root=root,
        input_root=args.input_root,
        network_wait_minutes=args.network_wait_minutes,
        network_poll_seconds=args.network_poll_seconds,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())