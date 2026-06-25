"""KV v3 optimization helpers.

Isolated functions that implement specific optimizations:
- High-confidence known title preservation
- Opening humming detection
- Cluster-size review skip threshold
- Opening segment review handling
"""
from __future__ import annotations

from typing import Any

from ....config import PROFILE_KV_V2
from ....models import ContentMatch, TranscriptSegment
from ....song_postprocess.normalize import (
    _clone_match_with_indices,
    _indices_to_ranges,
    _is_invalid_audit_title,
    _match_key,
    _matches_overlap,
    _segment_range_duration_seconds,
)


# ─── Constants ──────────────────────────────────────────────────

# Deletion threshold: candidates below this confidence are rejected
# (old kv_v2 used 0.70; new kv_v2 uses 0.75 for known songs, 0.60 for unknown songs)
KV_V3_DELETION_CONFIDENCE_THRESHOLD = 0.75
KV_V3_UNKNOWN_DELETION_CONFIDENCE_THRESHOLD = 0.60

# Minimum cluster size (number of matches) to trigger LLM review.
# Clusters smaller than this are resolved locally, saving ~40% review calls.
KV_V3_MIN_CLUSTER_SIZE_FOR_REVIEW = 2

# Opening detection: segments within this many seconds of transcript start
# are considered potential opening humming
KV_V3_OPENING_WINDOW_SECONDS = 45.0

# Minimum duration (seconds) for a segment to be considered valid singing
KV_V3_MIN_SINGING_DURATION_SECONDS = 10.0


# ─── High-confidence known title preservation ──────────────────


def _preserve_high_confidence_known_title(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    matches: list[ContentMatch],
    candidate_matches: list[ContentMatch],
) -> tuple[list[ContentMatch], list[dict[str, Any]]]:
    """Protect high-confidence known titles from review deletion.

    If a candidate has:
    - confidence >= 0.80
    - a known title (not "未知歌曲")
    - valid ASR evidence (>= 2 non-empty text segments or >= 10s duration)
    then it is retained even if review would have deleted it.

    This fixes the "唯一" regression where a correctly identified song
    was degraded to unknown during review.
    """
    final_coverage = {
        index
        for match in matches
        for index in match.segment_indices
    }
    existing_keys = {_match_key(match) for match in matches}
    events: list[dict[str, Any]] = []
    retained: list[ContentMatch] = []

    for candidate in candidate_matches:
        valid_indices = sorted({
            index for index in candidate.segment_indices
            if 0 <= index < len(segments)
        })
        event = {
            "title": candidate.title,
            "ranges": _indices_to_ranges(valid_indices),
            "confidence": candidate.confidence,
        }
        if not valid_indices:
            events.append({**event, "action": "rejected", "reason": "invalid_range"})
            continue

        normalized_candidate = _clone_match_with_indices(candidate, valid_indices)
        if _match_key(normalized_candidate) in existing_keys:
            events.append({**event, "action": "rejected", "reason": "exact_duplicate"})
            continue

        # Check if it's a valid song candidate
        if (
            candidate.content_type.strip().casefold() not in {"song", "music"}
            or _is_invalid_audit_title(candidate.title)
        ):
            events.append({**event, "action": "rejected", "reason": "invalid_song_candidate"})
            continue

        # High-confidence known title protection
        is_known_title = not candidate.title.strip().startswith("未知歌曲")
        is_high_confidence = float(candidate.confidence) >= 0.80
        has_evidence = _candidate_has_song_evidence(segments, normalized_candidate)

        if is_known_title and is_high_confidence and has_evidence:
            # Check overlap with existing matches
            overlap = len(set(valid_indices) & final_coverage)
            coverage_ratio = overlap / max(1, len(valid_indices))
            if coverage_ratio >= 0.5:
                events.append({
                    **event,
                    "action": "rejected",
                    "reason": "already_covered",
                    "coverage_ratio": round(coverage_ratio, 3),
                })
                continue
            if overlap > 0:
                uncovered = [idx for idx in valid_indices if idx not in final_coverage]
                if not uncovered:
                    events.append({
                        **event,
                        "action": "rejected",
                        "reason": "partial_overlap_fully_covered",
                    })
                    continue
                trimmed = _clone_match_with_indices(candidate, uncovered)
                if not _candidate_has_song_evidence(segments, trimmed):
                    events.append({
                        **event,
                        "action": "rejected",
                        "reason": "weak_trimmed_evidence",
                    })
                    continue
                retained.append(trimmed)
                final_coverage.update(uncovered)
                existing_keys.add(_match_key(trimmed))
                events.append({
                    **event,
                    "action": "retained_trimmed",
                    "reason": "high_confidence_known_title",
                })
                continue
            retained.append(normalized_candidate)
            final_coverage.update(valid_indices)
            existing_keys.add(_match_key(normalized_candidate))
            events.append({
                **event,
                "action": "retained",
                "reason": "high_confidence_known_title",
            })
            continue

        # For other candidates, apply deletion threshold
        # Known songs: no threshold, only check ASR evidence
        # Unknown songs: use lower threshold to preserve detection rate
        is_unknown = candidate.title.strip().startswith("未知歌曲")
        if is_unknown:
            if float(candidate.confidence) < KV_V3_UNKNOWN_DELETION_CONFIDENCE_THRESHOLD:
                events.append({**event, "action": "rejected", "reason": "low_confidence"})
                continue
        if not has_evidence:
            events.append({**event, "action": "rejected", "reason": "weak_asr_evidence"})
            continue

        # Standard overlap check
        overlap = len(set(valid_indices) & final_coverage)
        coverage_ratio = overlap / max(1, len(valid_indices))
        if coverage_ratio >= 0.5:
            events.append({
                **event,
                "action": "rejected",
                "reason": "already_covered",
                "coverage_ratio": round(coverage_ratio, 3),
            })
            continue
        if overlap > 0:
            uncovered = [idx for idx in valid_indices if idx not in final_coverage]
            if not uncovered:
                events.append({
                    **event,
                    "action": "rejected",
                    "reason": "partial_overlap_fully_covered",
                })
                continue
            trimmed = _clone_match_with_indices(candidate, uncovered)
            if not _candidate_has_song_evidence(segments, trimmed):
                events.append({
                    **event,
                    "action": "rejected",
                    "reason": "weak_trimmed_evidence",
                })
                continue
            retained.append(trimmed)
            final_coverage.update(uncovered)
            existing_keys.add(_match_key(trimmed))
            events.append({
                **event,
                "action": "trimmed",
                "reason": "partial_overlap_trimmed",
            })
            continue
        retained.append(normalized_candidate)
        final_coverage.update(valid_indices)
        existing_keys.add(_match_key(normalized_candidate))
        events.append({**event, "action": "retained", "reason": "standard_candidate"})

    return retained, events


def _candidate_has_song_evidence(
    segments: list[TranscriptSegment],
    match: ContentMatch,
) -> bool:
    """Check if a candidate has sufficient ASR evidence of singing."""
    valid_indices = sorted({i for i in match.segment_indices if 0 <= i < len(segments)})
    if not valid_indices:
        return False
    non_empty_texts = [
        segments[index].text.strip()
        for index in valid_indices
        if segments[index].text.strip()
    ]
    if len(non_empty_texts) >= 2:
        return True
    return _segment_range_duration_seconds(
        segments, min(valid_indices), max(valid_indices),
    ) >= KV_V3_MIN_SINGING_DURATION_SECONDS


# ─── Opening humming detection ─────────────────────────────────


def _detect_opening_humming(
    segments: list[TranscriptSegment],
    matches: list[ContentMatch],
) -> list[dict[str, Any]]:
    """Detect potential opening humming segments that may have been missed.

    Opening humming typically occurs in the first ~45 seconds of a stream
    and may consist of short, repeated vocal sounds without clear lyrics.
    """
    if not segments:
        return []

    events: list[dict[str, Any]] = []
    covered_indices = {idx for m in matches for idx in m.segment_indices}

    # Find segments in the opening window that are not covered
    opening_segments = []
    for i, seg in enumerate(segments):
        if float(seg.start) > KV_V3_OPENING_WINDOW_SECONDS:
            break
        if i not in covered_indices:
            opening_segments.append(i)

    if not opening_segments:
        return events

    # Check for humming-like patterns: short lines, repeated characters,
    # vocal markers
    humming_groups: list[list[int]] = []
    current_group: list[int] = []

    for idx in opening_segments:
        text = segments[idx].text.strip()
        if not text:
            if current_group:
                humming_groups.append(current_group)
                current_group = []
            continue

        is_humming = _is_humming_like_text(text)
        if is_humming:
            current_group.append(idx)
        else:
            if current_group:
                humming_groups.append(current_group)
                current_group = []

    if current_group:
        humming_groups.append(current_group)

    for group in humming_groups:
        if len(group) >= 2:
            duration = _segment_range_duration_seconds(
                segments, min(group), max(group)
            )
            if duration >= 5.0:  # At least 5 seconds of humming
                events.append({
                    "type": "opening_humming_detected",
                    "segment_ranges": _indices_to_ranges(group),
                    "duration_seconds": round(duration, 1),
                    "segment_count": len(group),
                })

    return events


def _is_humming_like_text(text: str) -> bool:
    """Check if text looks like humming/vocal sounds."""
    import re

    text_lower = text.lower().strip()
    if not text_lower:
        return False

    # Common humming patterns
    humming_patterns = [
        r"^[啦嗯啊哦呜呀哈嗯嗯]+$",
        r"^(la|na|ha|woo|yeah|oh|ah|um)+([\s,，.。]+(la|na|ha|woo|yeah|oh|ah|um)+)*[\s,，.。]*$",
        r"^(哼|嗯|啊|哦|呜|呀|哈)+[\s,，.。]*$",
    ]
    for pattern in humming_patterns:
        if re.match(pattern, text_lower):
            return True

    # Check for repeated characters (like "啦啦啦" or "lalala")
    if _has_repeated_vocal_pattern(text_lower):
        return True

    # Short text with at least one CJK vocalization character (啊, 嗯, 哦, 呀, 哈)
    # but not enough CJK content to be real lyrics
    cjk_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if len(text_lower) <= 8 and 1 <= cjk_chars <= 3:
        return True

    return False


def _has_repeated_vocal_pattern(text: str) -> bool:
    """Check for repeated vocal patterns like 啦啦啦 or lalala."""
    # Check for 3+ consecutive same characters
    for i in range(len(text) - 2):
        if text[i] == text[i + 1] == text[i + 2] and text[i].isalpha():
            return True

    # Check for alternating patterns like "lalala"
    if len(text) >= 6:
        for pattern_len in [1, 2]:
            pattern = text[:pattern_len]
            is_repeating = True
            for i in range(pattern_len, len(text), pattern_len):
                if text[i:i + pattern_len] != pattern:
                    is_repeating = False
                    break
            if is_repeating and len(text) >= pattern_len * 3:
                return True

    return False


# ─── Cluster-size review skip ──────────────────────────────────


def _should_skip_review(
    cluster: list[ContentMatch],
    config: dict[str, Any],
) -> bool:
    """Determine if a review cluster can be skipped (resolved locally).

    Small clusters (1 match) are resolved locally without LLM review,
    saving significant cost (~40% reduction in review calls).
    """
    min_cluster_size = int(
        config.get("song", {})
        .get(PROFILE_KV_V2, {})
        .get("min_cluster_size_for_review", KV_V3_MIN_CLUSTER_SIZE_FOR_REVIEW)
    )
    return len(cluster) < min_cluster_size


def _review_opening_segments(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    matches: list[ContentMatch],
    opening_events: list[dict[str, Any]],
) -> list[ContentMatch]:
    """Special handling for opening segment review.

    If opening humming was detected, ensure those segments are either
    covered by existing matches or explicitly marked as unknown songs.
    """
    if not opening_events:
        return matches

    # Collect all opening humming ranges
    opening_indices: set[int] = set()
    for event in opening_events:
        if event.get("type") == "opening_humming_detected":
            for range_pair in event.get("segment_ranges", []):
                if isinstance(range_pair, list) and len(range_pair) == 2:
                    start, end = range_pair
                    opening_indices.update(range(start, end + 1))

    if not opening_indices:
        return matches

    # Check if opening segments are already covered
    covered_indices = {idx for m in matches for idx in m.segment_indices}
    uncovered_opening = opening_indices - covered_indices

    if not uncovered_opening:
        return matches

    # Create an unknown song match for uncovered opening humming
    opening_match = ContentMatch(
        content_type="song",
        title="未知歌曲：开场哼唱",
        segment_indices=sorted(uncovered_opening),
        confidence=0.6,
        tags=["opening_humming"],
        description="Stream opening humming detected",
    )

    # Check if this overlaps with any existing match
    for existing in matches:
        if _matches_overlap(opening_match, existing):
            # Merge with existing match if it's also unknown
            if existing.title.strip().startswith("未知歌曲"):
                combined = _clone_match_with_indices(
                    existing,
                    sorted(set(existing.segment_indices) | uncovered_opening),
                )
                return [
                    combined if m is existing else m
                    for m in matches
                ]
            # Otherwise, keep existing match and skip opening
            return matches

    return sorted(
        [*matches, opening_match],
        key=lambda m: min(m.segment_indices),
    )
