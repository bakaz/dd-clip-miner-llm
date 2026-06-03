from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_STREAMER = "StreamerName"
DEFAULT_DATE = "250101"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".m4v", ".webm", ".avi", ".flv"}
INVALID_FILENAME_CHARS = r'<>:"/\|?*'


@dataclass(frozen=True)
class RenamePlan:
    source: Path
    target: Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename dragged clip files to: 【主播】歌名-歌手-YYMMDD.ext",
    )
    parser.add_argument("paths", nargs="+", help="Files or directories to rename")
    parser.add_argument(
        "--streamer",
        default=os.environ.get("CLIP_RENAMER_STREAMER", DEFAULT_STREAMER),
        help=f"Streamer name inside 【】. Default: {DEFAULT_STREAMER}",
    )
    parser.add_argument(
        "--date",
        default=os.environ.get("CLIP_RENAMER_DATE", DEFAULT_DATE),
        help="Date suffix in YYMMDD. Use 'mtime' to use each file's modified date.",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan directories recursively")
    parser.add_argument("--dry-run", action="store_true", help="Print planned renames without changing files")
    args = parser.parse_args()

    files = collect_files([Path(p) for p in args.paths], recursive=args.recursive)
    if not files:
        print("No video files found.")
        return 1

    plans = [
        build_rename_plan(path, streamer=args.streamer, date_setting=args.date)
        for path in files
    ]
    plans = [plan for plan in plans if plan is not None]

    if not plans:
        print("No files need renaming.")
        return 0

    for plan in plans:
        print(f"{plan.source.name} -> {plan.target.name}")
        if not args.dry_run:
            plan.source.rename(plan.target)

    print(f"Done. Renamed {len(plans)} file(s).")
    return 0


def collect_files(paths: list[Path], recursive: bool = False) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(path)
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            files.extend(
                item for item in iterator
                if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
            )
        else:
            print(f"[skip] Not found or unsupported: {path}")
    return sorted(files)


def build_rename_plan(path: Path, streamer: str, date_setting: str) -> RenamePlan | None:
    title, artist = parse_title_artist(path.stem)
    if not title:
        print(f"[skip] Could not parse title: {path.name}")
        return None

    date_text = clip_date(path, date_setting)
    streamer = clean_part(streamer)
    title = clean_part(title)
    artist = clean_part(artist)

    if artist:
        new_stem = f"【{streamer}】{title}-{artist}-{date_text}"
    else:
        new_stem = f"【{streamer}】{title}-{date_text}"

    target = unique_target(path.with_name(f"{new_stem}{path.suffix.lower()}"))
    if path.resolve() == target.resolve():
        return None
    return RenamePlan(source=path, target=target)


def parse_title_artist(stem: str) -> tuple[str, str]:
    text = strip_existing_prefix_and_date(stem)
    text = re.sub(r"^\s*\d{1,4}\s*[-_—]+\s*", "", text).strip()

    spaced_parts = [part.strip() for part in re.split(r"\s+-\s+", text) if part.strip()]
    if len(spaced_parts) >= 2:
        return spaced_parts[0], spaced_parts[1]

    plain_parts = [part.strip() for part in text.rsplit("-", 1)]
    if len(plain_parts) == 2 and all(plain_parts):
        return plain_parts[0], plain_parts[1]

    return text.strip(), ""


def strip_existing_prefix_and_date(stem: str) -> str:
    text = re.sub(r"^【[^】]+】", "", stem).strip()
    text = re.sub(r"[-_ ]\d{6}$", "", text).strip()
    return text


def clean_part(value: str) -> str:
    cleaned = "".join("_" if ch in INVALID_FILENAME_CHARS else ch for ch in value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-_")
    return cleaned


def clip_date(path: Path, date_setting: str) -> str:
    setting = (date_setting or "").strip().lower()
    if setting in {"mtime", "auto", ""}:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%y%m%d")
    if not re.fullmatch(r"\d{6}", setting):
        raise ValueError(f"Date must be YYMMDD or 'mtime': {date_setting}")
    return setting


def unique_target(target: Path) -> Path:
    if not target.exists():
        return target

    index = 2
    while True:
        candidate = target.with_name(f"{target.stem}_{index}{target.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


if __name__ == "__main__":
    raise SystemExit(main())
