"""Portable path resolution for run artifacts (relative + absolute)."""

from __future__ import annotations

from pathlib import Path

RUN_SUBDIRS = ("00_input", "01_audio", "02_asr", "03_clips", "04_reports", "05_manual")


def as_run_relative(path: Path, run_root: Path) -> str:
    """Return *path* relative to *run_root* when possible."""
    try:
        return str(path.resolve().relative_to(run_root.resolve()))
    except ValueError:
        return str(path)


def derive_run_dir_from_context_dir(context_dir: Path) -> Path | None:
    """Walk upward from a clip export folder to the run root."""
    for parent in (Path(context_dir), *Path(context_dir).parents):
        if parent.name == "03_clips":
            return parent.parent
    return None


def recorded_run_dir_from_context(context: dict) -> Path | None:
    value = context.get("run_dir")
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == ".":
        return None
    return Path(text)


def portable_run_dir(context_dir: Path, recorded_run_dir: Path | None = None) -> Path:
    """Prefer run root derived from ``03_clips`` over recorded JSON paths."""
    derived = derive_run_dir_from_context_dir(context_dir)
    if derived is not None:
        return derived.resolve()
    if recorded_run_dir is not None:
        return Path(recorded_run_dir).resolve()
    return Path(context_dir).resolve()


def path_belongs_to_run(path: Path, run_root: Path) -> bool:
    try:
        path.resolve().relative_to(run_root.resolve())
        return True
    except ValueError:
        return False


def run_relative_suffix(path_value: str | Path) -> str | None:
    normalized = str(path_value).replace("\\", "/")
    for anchor in RUN_SUBDIRS:
        token = f"{anchor}/"
        index = normalized.find(token)
        if index >= 0:
            return normalized[index:]
    return None


def _unique_file_by_name(run_root: Path, name: str) -> Path | None:
    if not name:
        return None
    matches = [hit for hit in run_root.rglob(name) if hit.is_file()]
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_input_video_candidates(run_root: Path) -> list[Path]:
    input_dir = run_root / "00_input"
    if not input_dir.is_dir():
        return []
    candidates = sorted(input_dir.glob("input*.mp4"))
    if candidates:
        return candidates
    return sorted(input_dir.glob("*.mp4"))


def resolve_run_path(
    value: str | Path,
    *,
    run_root: Path,
    recorded_run_dir: Path | None = None,
) -> Path:
    """Resolve a stored path against *run_root*, accepting relative or absolute values."""
    run_root = run_root.resolve()
    raw = Path(str(value))
    candidates: list[Path] = []

    if not raw.is_absolute():
        candidates.append(run_root / raw)
    else:
        if raw.exists() and path_belongs_to_run(raw, run_root):
            candidates.append(raw.resolve())
        suffix = run_relative_suffix(raw)
        if suffix is not None:
            candidates.append(run_root / Path(suffix))
        if recorded_run_dir is not None:
            recorded = Path(recorded_run_dir)
            try:
                relative = raw.resolve().relative_to(recorded.resolve())
                candidates.append(run_root / relative)
            except ValueError:
                pass
            if raw.exists():
                candidates.append(raw.resolve())
            try:
                relative = raw.relative_to(recorded)
                recorded_candidate = recorded / relative
                if recorded_candidate.exists():
                    candidates.append(recorded_candidate.resolve())
            except ValueError:
                pass

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()

    if candidates:
        return candidates[0]
    return raw if raw.is_absolute() else run_root / raw


def resolve_input_video(
    run_root: Path,
    *,
    manifest: dict | None = None,
    context: dict | None = None,
    recorded_run_dir: Path | None = None,
) -> Path:
    """Locate the source input video under *run_root*."""
    value = None
    if manifest:
        value = manifest.get("input_video")
    if not value and context:
        value = context.get("input_video")
    if value:
        candidate = resolve_run_path(
            value,
            run_root=run_root,
            recorded_run_dir=recorded_run_dir,
        )
        if candidate.exists():
            return candidate.resolve()

    for candidate in _resolve_input_video_candidates(run_root):
        return candidate.resolve()

    if recorded_run_dir is not None:
        recorded = Path(recorded_run_dir)
        for candidate in _resolve_input_video_candidates(recorded):
            return candidate.resolve()

    raise FileNotFoundError(f"Could not resolve input video under: {run_root}")