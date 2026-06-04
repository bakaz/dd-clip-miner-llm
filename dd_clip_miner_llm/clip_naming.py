"""切片导出命名：JSON 词典匹配主播，路径严格解析 YYMMDD，输出【主播】歌名-歌手-YYMMDD。"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .models import ContentResult
from .paths import safe_path_part

YYMMDD_TOKEN_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
YYYYMMDD_SEP_RE = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")


@dataclass(frozen=True, slots=True)
class ClipDictionaryEntry:
    streamer: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ClipNamingProfile:
    streamer: str
    date: str
    matched_alias: str = ""
    score: float = 0.0
    source: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "streamer": self.streamer,
            "date": self.date,
            "matched_alias": self.matched_alias,
            "score": round(self.score, 4),
            "source": self.source,
        }


def get_clip_naming_settings(config: dict[str, Any]) -> dict[str, Any]:
    output = config.get("output", {})
    settings = output.get("clip_naming", {})
    if not isinstance(settings, dict):
        settings = {}
    apply_to = settings.get("apply_to")
    if isinstance(apply_to, str):
        apply_to = [item.strip() for item in apply_to.split(",") if item.strip()]
    elif not isinstance(apply_to, list):
        apply_to = ["song"]
    return {
        "enabled": bool(settings.get("enabled", False)),
        "dictionary_path": str(settings.get("dictionary_path") or "").strip(),
        "default_streamer": str(settings.get("default_streamer") or "StreamerName").strip(),
        "min_score": float(settings.get("min_score", 0.65)),
        "apply_to": apply_to,
    }


def load_clip_dictionary(path: str | Path) -> tuple[dict[str, Any], list[ClipDictionaryEntry]]:
    dictionary_path = Path(path)
    if not dictionary_path.exists():
        raise FileNotFoundError(f"Clip dictionary not found: {dictionary_path}")

    data = json.loads(dictionary_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Clip dictionary JSON must be an object")

    raw_entries = data.get("entries") or data.get("streamers") or []
    if not isinstance(raw_entries, list):
        raise ValueError("Clip dictionary 'entries' must be a list")

    entries: list[ClipDictionaryEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        streamer = str(item.get("streamer") or item.get("name") or "").strip()
        if not streamer:
            continue
        aliases = _split_aliases(item.get("aliases") or item.get("keywords") or item.get("match") or [])
        if streamer not in aliases:
            aliases = [streamer, *aliases]
        entries.append(ClipDictionaryEntry(streamer=streamer, aliases=tuple(aliases)))
    return data, entries


def is_valid_yymmdd(value: str) -> bool:
    if not re.fullmatch(r"\d{6}", value):
        return False
    month = int(value[2:4])
    day = int(value[4:6])
    return 1 <= month <= 12 and 1 <= day <= 31


def extract_yymmdd_from_texts(texts: list[str]) -> str | None:
    """从路径片段中严格提取 YYMMDD（不支持词典或 mtime）。"""
    for text in texts:
        found = _extract_yymmdd_from_text(text)
        if found:
            return found
    return None


def resolve_clip_naming_profile(
    input_path: str | Path,
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    extra_texts: list[str] | None = None,
) -> ClipNamingProfile | None:
    settings = get_clip_naming_settings(config)
    if not settings["enabled"]:
        return None

    search_texts = _collect_search_texts(input_path, extra_texts)
    date_value = extract_yymmdd_from_texts(search_texts)
    if date_value is None:
        print("[warn] clip_naming: no YYMMDD found in input path, using legacy filenames")
        return None

    dictionary_path = settings["dictionary_path"]
    if not dictionary_path:
        return ClipNamingProfile(
            streamer=settings["default_streamer"] or "StreamerName",
            date=date_value,
            source="config_default",
        )

    resolved_path = _resolve_dictionary_path(dictionary_path, config_path)
    dictionary_meta, entries = load_clip_dictionary(resolved_path)
    default_streamer = str(
        dictionary_meta.get("default_streamer") or settings["default_streamer"]
    ).strip()
    min_score = float(dictionary_meta.get("min_score", settings["min_score"]))

    best_entry: ClipDictionaryEntry | None = None
    best_alias = ""
    best_score = 0.0

    for entry in entries:
        for alias in entry.aliases:
            for text in search_texts:
                score = text_similarity(alias, text)
                if score > best_score:
                    best_score = score
                    best_entry = entry
                    best_alias = alias

    if best_entry is not None and best_score >= min_score:
        return ClipNamingProfile(
            streamer=best_entry.streamer,
            date=date_value,
            matched_alias=best_alias,
            score=best_score,
            source="dictionary",
        )

    return ClipNamingProfile(
        streamer=default_streamer or "StreamerName",
        date=date_value,
        source="fallback",
    )


def resolve_export_stem(
    result: ContentResult,
    config: dict[str, Any],
    content_type: str,
    naming_profile: ClipNamingProfile | None,
    *,
    legacy_safe_filename: Any,
) -> str:
    if naming_profile is not None and should_apply_clip_naming(content_type, config):
        return build_clip_export_stem(result, naming_profile)
    name_bits = [f"{result.index:03d}", result.title]
    if result.artist:
        name_bits.append(result.artist)
    return legacy_safe_filename("-".join(name_bits))


def build_clip_export_stem(
    result: ContentResult,
    profile: ClipNamingProfile,
) -> str:
    title = _clean_name_part(result.title)
    artist = _clean_name_part(result.artist)
    streamer = _clean_name_part(profile.streamer)
    date_text = profile.date

    if artist:
        raw = f"【{streamer}】{title}-{artist}-{date_text}"
    else:
        raw = f"【{streamer}】{title}-{date_text}"
    return safe_path_part(raw, fallback=f"clip_{result.index:03d}")


def should_apply_clip_naming(content_type: str, config: dict[str, Any]) -> bool:
    settings = get_clip_naming_settings(config)
    if not settings["enabled"]:
        return False
    apply_to = settings["apply_to"]
    if not apply_to:
        return True
    return content_type in apply_to


def text_similarity(left: str, right: str) -> float:
    left_norm = normalize_match_text(left)
    right_norm = normalize_match_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        shorter = min(len(left_norm), len(right_norm))
        longer = max(len(left_norm), len(right_norm))
        if longer and shorter / longer >= 0.45:
            return max(0.92, SequenceMatcher(None, left_norm, right_norm).ratio())
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def normalize_match_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    text = re.sub(r"[\s\-_·・，。、！!？?（）()\[\]【】「」『』《》\"'“”‘’:：;；]+", "", text)
    return text


def _extract_yymmdd_from_text(text: str) -> str | None:
    for match in YYYYMMDD_SEP_RE.finditer(text):
        yymmdd = f"{match.group(1)[2:]}{match.group(2)}{match.group(3)}"
        if is_valid_yymmdd(yymmdd):
            return yymmdd

    for match in YYMMDD_TOKEN_RE.finditer(text):
        token = match.group(1)
        if is_valid_yymmdd(token):
            return token
    return None


def _collect_search_texts(input_path: str | Path, extra_texts: list[str] | None) -> list[str]:
    path = Path(input_path)
    parts: list[str] = [path.stem, str(path.parent), str(path)]
    for parent in path.parents:
        parts.append(parent.name)
        if parent == parent.parent:
            break
    if extra_texts:
        parts.extend(extra_texts)
    seen: set[str] = set()
    texts: list[str] = []
    for part in parts:
        cleaned = part.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            texts.append(cleaned)
    return texts


def _resolve_dictionary_path(path: str, config_path: Path | None) -> Path:
    dictionary_path = Path(path)
    if dictionary_path.is_absolute():
        return dictionary_path
    if config_path is not None:
        candidate = config_path / dictionary_path
        if candidate.exists():
            return candidate
    return Path.cwd() / dictionary_path


def _split_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"[|;/\n]+", text)
    return [part.strip() for part in parts if part.strip()]


def _clean_name_part(value: str) -> str:
    cleaned = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-_")
    return cleaned or "untitled"