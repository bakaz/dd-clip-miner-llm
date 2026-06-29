"""Cleanup run-local source videos and sus/ folder from export context JSON."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO

from .post_merge import PostMergeError, _load_json_object, _source_video_and_duration
from .run_paths import path_belongs_to_run, portable_run_dir, recorded_run_dir_from_context

CleanupKind = Literal["source_video", "concat_video", "sus_dir"]


class CleanupContextError(RuntimeError):
    """Raised when a cleanup request cannot be fulfilled."""


@dataclass(frozen=True)
class CleanupTarget:
    kind: CleanupKind
    path: Path
    label: str


def cleanup_from_context(
    context_path: str | Path,
    *,
    dry_run: bool = False,
    yes: bool = False,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> dict[str, Any]:
    context_file = Path(context_path)
    stdin = input_stream or sys.stdin
    stdout = output_stream or sys.stdout

    try:
        context = _load_json_object(context_file)
        recorded_run_dir = recorded_run_dir_from_context(context)
        run_dir = portable_run_dir(context_file.parent, recorded_run_dir)
        _total_duration, input_video = _source_video_and_duration(
            context,
            run_dir,
            recorded_run_dir=recorded_run_dir,
        )
    except PostMergeError as exc:
        raise CleanupContextError(str(exc)) from exc

    targets, skipped, warnings = _collect_cleanup_targets(
        run_dir,
        context_file,
        input_video,
    )

    if not targets:
        return {
            "deleted_files": [],
            "deleted_dirs": [],
            "skipped": skipped,
            "warnings": warnings or ["Nothing to delete: all targets are absent or outside run root."],
            "dry_run": dry_run,
            "run_dir": str(run_dir),
        }

    if not yes:
        _print_plan(stdout, targets, run_dir=run_dir, warnings=warnings)
        if not _confirm(stdin, stdout, "Proceed with cleanup? [y/N]: "):
            raise CleanupContextError("Cleanup cancelled by user.")

    deleted_files: list[str] = []
    deleted_dirs: list[str] = []

    for target in targets:
        if target.kind in {"source_video", "concat_video"}:
            if dry_run:
                deleted_files.append(str(target.path))
                continue
            target.path.unlink()
            deleted_files.append(str(target.path))
            _maybe_remove_empty_parent(target.path.parent)
        elif target.kind == "sus_dir":
            if dry_run:
                deleted_dirs.append(str(target.path))
                continue
            shutil.rmtree(target.path)
            deleted_dirs.append(str(target.path))

    return {
        "deleted_files": deleted_files,
        "deleted_dirs": deleted_dirs,
        "skipped": skipped,
        "warnings": warnings,
        "dry_run": dry_run,
        "run_dir": str(run_dir),
    }


def _collect_cleanup_targets(
    run_dir: Path,
    context_file: Path,
    input_video: Path,
) -> tuple[list[CleanupTarget], list[str], list[str]]:
    run_root = run_dir.resolve()
    targets: list[CleanupTarget] = []
    skipped: list[str] = []
    warnings: list[str] = []

    resolved_input = input_video.resolve()
    if path_belongs_to_run(resolved_input, run_root) and resolved_input.is_file():
        targets.append(CleanupTarget("source_video", resolved_input, "input video"))
    elif resolved_input.is_file():
        skipped.append("source_video")
        warnings.append(
            "Skipped input video outside run root: "
            f"{resolved_input} (run_root={run_root})"
        )
    else:
        skipped.append("source_video")
        warnings.append(f"Input video not found, skipped: {resolved_input}")

    concat_video = run_root / "concat" / "concat.mp4"
    if concat_video.is_file():
        targets.append(CleanupTarget("concat_video", concat_video.resolve(), "concat video"))

    sus_dir = context_file.parent / "sus"
    if sus_dir.is_dir():
        targets.append(CleanupTarget("sus_dir", sus_dir.resolve(), "sus folder"))
    else:
        skipped.append("sus_dir")

    return targets, skipped, warnings


def _maybe_remove_empty_parent(directory: Path) -> None:
    try:
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    except OSError:
        pass


def _print(stream: TextIO, message: str) -> None:
    stream.write(f"{message}\n")
    stream.flush()


def _print_plan(
    stream: TextIO,
    targets: list[CleanupTarget],
    *,
    run_dir: Path,
    warnings: list[str],
) -> None:
    _print(stream, f"Run root: {run_dir}")
    _print(stream, "The following will be deleted:")
    for target in targets:
        _print(stream, f"  - {target.label}: {target.path}")
    for warning in warnings:
        _print(stream, f"Warning: {warning}")


def _confirm(stdin: TextIO, stdout: TextIO, prompt: str) -> bool:
    _print(stdout, prompt)
    try:
        answer = stdin.readline()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().lower() in {"y", "yes"}