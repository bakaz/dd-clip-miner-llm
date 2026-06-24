"""Rewrite run artifact paths after cut_copy so merge/manual bat work on destination."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .run_paths import as_run_relative, resolve_input_video, resolve_run_path

_CONTEXT_FILES = ("merge_recut_context.json", "manual_cut_context.json")


def relocate_run_artifacts(run_root: str | Path) -> None:
    """Normalize manifest/context/report paths relative to *run_root*."""
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run root not found: {root}")

    recorded_run_dir = _recorded_run_dir_from_tree(root)
    input_video = resolve_input_video(root, recorded_run_dir=recorded_run_dir)
    total_duration = _resolve_total_duration(root, input_video)

    for manifest_path in sorted(root.glob("manifest*.json")):
        _relocate_manifest(manifest_path, root, input_video, total_duration)

    for songs_path in root.rglob("songs.json"):
        if songs_path.is_file():
            _relocate_songs_json(songs_path, root, recorded_run_dir=recorded_run_dir)

    for context_name in _CONTEXT_FILES:
        for context_path in root.rglob(context_name):
            if context_path.is_file():
                _relocate_context_json(
                    context_path,
                    root,
                    input_video=input_video,
                    total_duration=total_duration,
                    recorded_run_dir=recorded_run_dir,
                )


def _recorded_run_dir_from_tree(run_root: Path) -> Path | None:
    for context_name in _CONTEXT_FILES:
        for context_path in run_root.rglob(context_name):
            try:
                context = _read_json_object(context_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            value = context.get("run_dir")
            if value and str(value).strip() not in (".", ""):
                return Path(str(value))
    return None


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _read_json_list(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return data


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_total_duration(run_root: Path, input_video: Path) -> float | None:
    for manifest_path in sorted(run_root.glob("manifest*.json")):
        try:
            manifest = _read_json_object(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        value = manifest.get("total_duration")
        if value is not None:
            return float(value)

    try:
        from .ffmpeg import get_duration

        return float(get_duration(input_video))
    except Exception:
        return None


def _relocate_manifest(
    manifest_path: Path,
    run_root: Path,
    input_video: Path,
    total_duration: float | None,
) -> None:
    manifest = _read_json_object(manifest_path)
    manifest["input_video"] = as_run_relative(input_video, run_root)
    if total_duration is not None:
        manifest["total_duration"] = total_duration
    _write_json(manifest_path, manifest)


def _relocate_songs_json(
    songs_path: Path,
    run_root: Path,
    *,
    recorded_run_dir: Path | None,
) -> None:
    songs = _read_json_list(songs_path)
    changed = False
    for item in songs:
        if not isinstance(item, dict):
            continue
        for key in ("audio_path", "video_path"):
            value = item.get(key)
            if not value:
                continue
            resolved = resolve_run_path(value, run_root=run_root, recorded_run_dir=recorded_run_dir)
            if not resolved.exists():
                continue
            relative = as_run_relative(resolved, run_root)
            if item.get(key) != relative:
                item[key] = relative
                changed = True
    if changed:
        _write_json(songs_path, songs)


def _relocate_context_json(
    context_path: Path,
    run_root: Path,
    *,
    input_video: Path,
    total_duration: float | None,
    recorded_run_dir: Path | None,
) -> None:
    context = _read_json_object(context_path)
    profile = str(context.get("profile") or "").strip()
    content_type = str(context.get("content_type") or "song")

    if profile:
        reports_path = run_root / "04_reports" / profile / content_type / f"{content_type}s.json"
        llm_dir = run_root / "02_asr" / "llm" / profile / content_type
        manifest_candidates = [
            run_root / f"manifest.{profile}.json",
            run_root / "manifest.json",
        ]
    else:
        reports_path = run_root / "04_reports" / content_type / f"{content_type}s.json"
        llm_dir = run_root / "02_asr" / "llm" / content_type
        manifest_candidates = [run_root / "manifest.json"]

    manifest_path = next((path for path in manifest_candidates if path.is_file()), None)
    transcript_path = run_root / "02_asr" / "transcript.json"
    matches_path = llm_dir / "matches.json"

    context["run_dir"] = "."
    context["input_video"] = as_run_relative(input_video, run_root)
    if manifest_path is not None:
        context["manifest_path"] = as_run_relative(manifest_path, run_root)
    if reports_path.is_file():
        context["reports_path"] = as_run_relative(reports_path, run_root)
    if llm_dir.is_dir():
        context["llm_dir"] = as_run_relative(llm_dir, run_root)
    if matches_path.is_file():
        context["matches_path"] = as_run_relative(matches_path, run_root)
    if transcript_path.is_file():
        context["transcript_path"] = as_run_relative(transcript_path, run_root)
    if total_duration is not None:
        context["total_duration"] = total_duration

    for key in ("python_executable", "project_root"):
        context.pop(key, None)

    _write_json(context_path, context)