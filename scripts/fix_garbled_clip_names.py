#!/usr/bin/env python3
"""
Fix garbled clip filenames (mojibake with � / replacement chars) produced by
older clip_naming + song LLM artist data in a batch result tree.

Usage (after a run or to clean previous bad exports):
  python scripts/fix_garbled_clip_names.py results
  python scripts/fix_garbled_clip_names.py results/2026_06_08/...

It uses:
- clip_naming.json (authoritative good streamer + date)
- 02_asr/llm/song/matches.json or 04_reports/song/songs.json (good titles)
- Rebuilds stems with the improved safe/clean logic (【streamer】title-date)
- Renames matching files in 03_clips/audio/song and 03_clips/video/song
  that contain replacement chars or obvious garbled "���c��Aya��" prefixes.

Safe: only renames when it can compute a clearly better stem; skips on conflict.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

REPLACEMENT = "\ufffd"
GARBLED_PREFIX_RE = re.compile(r"[\ufffd?]{2,}c[\ufffd?]{2,}Aya[\ufffd?]{2,}", re.IGNORECASE)

def safe_path_part(value: str, fallback: str = "item", max_length: int = 120) -> str:
    if not value:
        return fallback
    value = value.replace(REPLACEMENT, "")
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = re.sub(r'[<>:"/\\|?*\x00-\x1f\ufffd]', "_", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .")
    return (normalized[:max_length] or fallback).strip(" .") or fallback

def _clean_name_part(value: str) -> str:
    if not value:
        return "untitled"
    value = value.replace(REPLACEMENT, "").strip()
    cleaned = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-_")
    return cleaned or "untitled"

def load_clip_naming(result_dir: Path) -> dict[str, Any] | None:
    p = result_dir / "clip_naming.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def load_song_titles(result_dir: Path) -> dict[int, dict[str, str]]:
    """Return {index: {"title": , "artist": }} from best available source."""
    matches_p = result_dir / "02_asr" / "llm" / "song" / "matches.json"
    songs_p = result_dir / "04_reports" / "song" / "songs.json"
    data: list[dict] = []
    if matches_p.exists():
        try:
            data = json.loads(matches_p.read_text(encoding="utf-8"))
        except Exception:
            data = []
    if not data and songs_p.exists():
        try:
            data = json.loads(songs_p.read_text(encoding="utf-8"))
        except Exception:
            data = []
    out: dict[int, dict[str, str]] = {}
    for item in data:
        idx = item.get("index")
        if idx is None:
            continue
        try:
            idx = int(idx)
        except Exception:
            continue
        title = _clean_name_part(item.get("title", "") or "")
        artist = _clean_name_part(item.get("artist", "") or "")
        out[idx] = {"title": title, "artist": artist}
    return out

def build_good_stem(idx: int, title: str, artist: str, streamer: str, date: str) -> str:
    t = _clean_name_part(title)
    a = _clean_name_part(artist)
    s = _clean_name_part(streamer)
    d = date or ""
    if a and a != "untitled":
        raw = f"【{s}】{t}-{a}-{d}"
    else:
        raw = f"【{s}】{t}-{d}"
    stem = safe_path_part(raw, fallback=f"clip_{idx:03d}")
    stem = stem.replace(REPLACEMENT, "")
    return stem or f"clip_{idx:03d}"

def looks_garbled(name: str) -> bool:
    if REPLACEMENT in name:
        return True
    if GARBLED_PREFIX_RE.search(name):
        return True
    # Also catch cases where streamer part looks broken even if no � visible in this env
    if "Aya" in name and ("�" in name or name.count("_") > 4 and "untitled" not in name.lower()):
        return True
    return False

def fix_dir(result_dir: Path, dry_run: bool = False) -> int:
    cn = load_clip_naming(result_dir)
    if not cn:
        return 0
    streamer = cn.get("streamer") or "Streamer"
    date = cn.get("date") or ""
    titles = load_song_titles(result_dir)
    if not titles:
        return 0

    renamed = 0
    for content in ("audio", "video"):
        clips_dir = result_dir / "03_clips" / content / "song"
        if not clips_dir.exists():
            continue
        for f in list(clips_dir.iterdir()):
            if not f.is_file():
                continue
            if not looks_garbled(f.name):
                continue
            # Try to guess index from the bad name or from existing patterns like 003-...
            # Best: look for a match in titles by scanning possible indices 0..999
            stem = f.stem
            ext = f.suffix
            chosen_idx = None
            for idx in titles:
                # Heuristic: the bad name often still contains fragments of the title
                t = titles[idx]["title"]
                if t and t != "untitled" and t[:4] in stem:
                    chosen_idx = idx
                    break
            if chosen_idx is None:
                # fallback: try to parse leading digits if present in bad name
                m = re.match(r"(\d{2,3})", stem)
                if m:
                    cand = int(m.group(1))
                    if cand in titles:
                        chosen_idx = cand
            if chosen_idx is None:
                continue
            info = titles[chosen_idx]
            new_stem = build_good_stem(chosen_idx, info["title"], info["artist"], streamer, date)
            new_name = new_stem + ext
            if new_name == f.name:
                continue
            target = f.with_name(new_name)
            if target.exists():
                # avoid overwrite
                continue
            print(f"  rename: {f.name} -> {new_name}")
            if not dry_run:
                try:
                    f.rename(target)
                    renamed += 1
                except OSError as e:
                    print(f"    failed: {e}")
    return renamed

def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: python scripts/fix_garbled_clip_names.py <result_root_or_dir> [--dry-run]")
        return 2
    root = Path(argv[0])
    dry = "--dry-run" in argv or "-n" in argv
    total = 0
    if (root / "clip_naming.json").exists():
        total += fix_dir(root, dry_run=dry)
    else:
        for d in root.rglob("clip_naming.json"):
            total += fix_dir(d.parent, dry_run=dry)
    print(f"Renamed {total} files." + (" (dry-run)" if dry else ""))
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
