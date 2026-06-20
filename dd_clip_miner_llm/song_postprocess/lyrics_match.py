"""Lyrics matching module for unknown song identification.

Provides post-processing capabilities to match unknown songs against
known lyrics databases, improving song identification rates.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..models import ContentMatch, TranscriptSegment
from .normalize import (
    _clone_match_with_indices,
    _indices_to_ranges,
    _segment_range_duration_seconds,
)


# ─── Constants ──────────────────────────────────────────────────

# Minimum overlap coefficient for lyrics matching
_LYRICS_MATCH_THRESHOLD = 0.3

# Maximum number of lyrics searches per unknown song
_MAX_LYRICS_SEARCHES = 3

# Minimum ASR text length to attempt lyrics matching
_MIN_ASR_TEXT_LENGTH = 10


# ─── Public API ─────────────────────────────────────────────────


def _apply_lyrics_matching(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    matches: list[ContentMatch],
    llm_dir: Path,
) -> tuple[list[ContentMatch], list[dict[str, Any]]]:
    """Apply lyrics matching to unknown songs.

    For each unknown song match, extract representative lyrics snippets
    and attempt to match against known lyrics. If a strong match is found,
    update the title and artist.

    Args:
        segments: Full ASR transcript segments
        config: Configuration dict
        matches: Current song matches
        llm_dir: Directory for LLM debug output

    Returns:
        (updated_matches, events)
    """
    events: list[dict[str, Any]] = []
    updated: list[ContentMatch] = []

    for match in matches:
        if not match.title.strip().startswith("未知歌曲"):
            updated.append(match)
            continue

        # Extract lyrics snippet for matching
        lyrics_snippet = _extract_lyrics_snippet(segments, match)
        if not lyrics_snippet or len(lyrics_snippet) < _MIN_ASR_TEXT_LENGTH:
            updated.append(match)
            continue

        # Attempt to find matching lyrics
        match_result = _search_lyrics_database(lyrics_snippet, config)
        if match_result:
            # Update match with identified song
            updated_match = ContentMatch(
                content_type=match.content_type,
                title=match_result.get("title", match.title),
                segment_indices=match.segment_indices,
                confidence=min(match.confidence + 0.1, 1.0),  # Boost confidence
                tags=[*match.tags, "lyrics_matched"],
                description=match.description,
                artist=match_result.get("artist", match.artist),
                lyrics_snippet=lyrics_snippet,
            )
            updated.append(updated_match)
            events.append({
                "type": "lyrics_match_found",
                "original_title": match.title,
                "matched_title": match_result.get("title"),
                "matched_artist": match_result.get("artist"),
                "confidence_boost": 0.1,
                "lyrics_snippet": lyrics_snippet[:100],
            })
        else:
            # Keep as unknown but store lyrics snippet
            updated_match = _clone_match_with_indices(match, match.segment_indices)
            if not updated_match.lyrics_snippet:
                updated_match = ContentMatch(
                    content_type=match.content_type,
                    title=match.title,
                    segment_indices=match.segment_indices,
                    confidence=match.confidence,
                    tags=match.tags,
                    description=match.description,
                    artist=match.artist,
                    lyrics_snippet=lyrics_snippet,
                )
            updated.append(updated_match)

    return updated, events


# ─── Internal helpers ───────────────────────────────────────────


def _extract_lyrics_snippet(
    segments: list[TranscriptSegment],
    match: ContentMatch,
) -> str:
    """Extract a representative lyrics snippet from a song match.

    Selects the most distinctive lines from the ASR transcript
    for lyrics matching purposes.
    """
    valid_indices = sorted({
        i for i in match.segment_indices
        if 0 <= i < len(segments)
    })
    if not valid_indices:
        return ""

    # Collect all non-empty texts
    texts = [
        segments[i].text.strip()
        for i in valid_indices
        if segments[i].text.strip()
    ]
    if not texts:
        return ""

    # Select most distinctive lines (longer lines tend to be more distinctive)
    scored_lines = []
    for text in texts:
        # Score based on length and content
        score = len(text)
        # Bonus for lines with CJK characters (likely lyrics)
        cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        if cjk_count > 0:
            score += cjk_count * 0.5
        # Bonus for lines with punctuation (likely complete phrases)
        if re.search(r"[，。！？、；：]", text):
            score += 5
        scored_lines.append((score, text))

    # Sort by score and take top lines
    scored_lines.sort(reverse=True)
    snippet_lines = [text for _, text in scored_lines[:5]]

    return " ".join(snippet_lines)


def _search_lyrics_database(
    query: str,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Search lyrics database for matching songs.

    This is a placeholder implementation. In production, this would
    connect to a lyrics API or local database.

    Args:
        query: Lyrics snippet to search for
        config: Configuration dict

    Returns:
        Match result dict with title/artist, or None if no match
    """
    # Check if lyrics search is enabled
    search_config = config.get("song", {}).get("search", {})
    if not search_config.get("enabled", False):
        return None

    # In a real implementation, this would:
    # 1. Call a lyrics API (e.g., QQ Music, NetEase Music)
    # 2. Or search a local lyrics database
    # 3. Return the best match if confidence is high enough

    # For now, return None (no match)
    # This can be extended with actual lyrics search logic
    return None


def _compute_lyrics_similarity(
    text_a: str,
    text_b: str,
) -> float:
    """Compute similarity between two lyrics texts.

    Uses overlap coefficient of character bigrams for CJK text,
    or word overlap for non-CJK text.
    """
    def _tokenize(text: str) -> set[str]:
        tokens: set[str] = set()
        cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        if cjk_count > len(text) * 0.3:
            # CJK text: use character bigrams
            clean = re.sub(r"[^\u4e00-\u9fff\w]", "", text)
            for i in range(len(clean) - 1):
                tokens.add(clean[i:i + 2].lower())
        else:
            # Non-CJK text: use words
            tokens.update(w.lower() for w in text.split() if w.isalpha())
        return tokens

    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))


# ─── Lyrics search tool integration ────────────────────────────


def _build_lyrics_search_tool() -> dict[str, Any]:
    """Build a tool definition for lyrics search.

    This can be used with LLM tool calling to search for lyrics.
    """
    return {
        "type": "function",
        "function": {
            "name": "search_lyrics",
            "description": "搜索歌词以确认歌名。输入应为最有辨识度的歌词片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "歌词片段，至少包含一句完整歌词",
                    },
                },
                "required": ["query"],
            },
        },
    }
