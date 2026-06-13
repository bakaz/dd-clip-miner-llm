from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import get_padding_config, is_risk_routed, song_pipeline_strategy
from ..models import ContentMatch, TranscriptSegment
from .normalize import (
    _clone_match_with_indices,
    _indices_to_ranges,
    _match_key,
    _matches_overlap,
    _segment_range_duration_seconds,
    _split_indices_by_time_gap_for_recheck,
)


DEFAULT_RISK_CONFIG: dict[str, Any] = {
    "duration_weight": 1.0,
    "boundary_expansion_weight": 2.0,
    "overlap_weight": 2.0,
    "evidence_weight": 2.0,
    "review_threshold": 0.55,
    "reject_threshold": 0.90,
    "soft_min_seconds": 150.0,
    "soft_max_seconds": 360.0,
    "boundary_gap_seconds": 20.0,
    "low_confidence": 0.65,
}


def get_song_risk_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_RISK_CONFIG)
    raw = config.get("song", {}).get("risk", {})
    if isinstance(raw, dict):
        merged.update(raw)
    return merged


def is_unknown_song_title(title: str) -> bool:
    normalized = title.strip().casefold()
    return not normalized or normalized.startswith(("未知歌曲", "unknown song", "unknown"))


@dataclass(slots=True)
class SongRiskRecord:
    title: str
    artist: str
    segment_ranges: list[list[int]]
    duration_seconds: float
    score: float
    features: dict[str, float]
    reasons: list[str]
    action: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def repair_song_boundaries(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    matches: list[ContentMatch],
) -> tuple[list[ContentMatch], list[dict[str, Any]]]:
    """Split candidates at meaningful ASR time gaps without rejecting by duration."""
    if not is_risk_routed(config):
        return matches, []
    gap_seconds = max(
        0.0, float(get_song_risk_config(config).get("boundary_gap_seconds", 20.0))
    )
    merge_gap_seconds = float(get_padding_config(config, "song").get("merge_gap_seconds", 40.0))
    repaired: list[ContentMatch] = []
    events: list[dict[str, Any]] = []
    for match in matches:
        valid = sorted({index for index in match.segment_indices if 0 <= index < len(segments)})
        if is_risk_routed(config):
            from ..config import get_song_normalization_config
            norm_cfg = get_song_normalization_config(config)
            if norm_cfg.get("chorus_aware_split", False):
                from .normalize import _v2_chorus_aware_split
                groups = _v2_chorus_aware_split(
                    segments, valid, merge_gap_seconds,
                    chorus_gap=float(norm_cfg.get("chorus_gap_seconds", 120.0)),
                    similarity_threshold=float(norm_cfg.get("chorus_similarity_threshold", 0.3)),
                    context_segments=int(norm_cfg.get("chorus_context_segments", 3)),
                )
            else:
                groups = _split_indices_by_time_gap_for_recheck(
                    segments, valid, merge_gap_seconds,
                )
        else:
            if "temporal_adjudicated" in match.tags:
                groups = [valid] if valid else []
            else:
                groups = _split_indices_by_time_gap_for_recheck(
                    segments, valid, gap_seconds,
                )
        if len(groups) > 1:
            events.append({
                "type": "risk_boundary_split",
                "title": match.title,
                "before": _indices_to_ranges(valid),
                "after": [_indices_to_ranges(group) for group in groups],
                "gap_threshold_seconds": merge_gap_seconds,
                "risk_boundary_gap_seconds": gap_seconds,
            })
        repaired.extend(_clone_match_with_indices(match, group) for group in groups if group)
    repaired.sort(key=lambda item: min(item.segment_indices))
    return repaired, events


def expand_song_anchors(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    anchors: list[ContentMatch],
    target_ranges: list[tuple[int, int]],
    existing_matches: list[ContentMatch],
) -> tuple[list[ContentMatch], list[dict[str, Any]]]:
    """Build provisional boundaries around short anchors inside uncovered ranges."""
    if not anchors or not segments:
        return anchors, []
    risk = get_song_risk_config(config)
    recheck = config.get("song", {}).get("missed_recheck", {})
    gap_threshold = max(0.0, float(risk.get("boundary_gap_seconds", 20.0)))
    max_span = max(
        1.0, float(recheck.get("anchor_max_expansion_seconds", 420.0))
    )
    allowed = {
        index
        for start, end in target_ranges
        for index in range(max(0, start), min(len(segments) - 1, end) + 1)
    }
    blocked = {
        index
        for match in existing_matches
        for index in match.segment_indices
        if 0 <= index < len(segments)
    }
    seeds = {
        index: owner
        for owner, match in enumerate(anchors)
        for index in match.segment_indices
        if 0 <= index < len(segments)
    }
    claimed: set[int] = set()
    expanded: list[ContentMatch] = []
    events: list[dict[str, Any]] = []
    for owner, match in enumerate(anchors):
        valid = sorted({
            index for index in match.segment_indices
            if index in allowed and index not in blocked
        })
        groups = _split_indices_by_time_gap_for_recheck(segments, valid, gap_threshold)
        for group in groups:
            left, right = min(group), max(group)
            while left > 0:
                candidate = left - 1
                other_seed = seeds.get(candidate)
                if (
                    candidate not in allowed
                    or candidate in blocked
                    or candidate in claimed
                    or (other_seed is not None and other_seed != owner)
                    or float(segments[left].start) - float(segments[candidate].end) > gap_threshold
                    or float(segments[right].end) - float(segments[candidate].start) > max_span
                ):
                    break
                left = candidate
            while right + 1 < len(segments):
                candidate = right + 1
                other_seed = seeds.get(candidate)
                if (
                    candidate not in allowed
                    or candidate in blocked
                    or candidate in claimed
                    or (other_seed is not None and other_seed != owner)
                    or float(segments[candidate].start) - float(segments[right].end) > gap_threshold
                    or float(segments[candidate].end) - float(segments[left].start) > max_span
                ):
                    break
                right = candidate
            indices = list(range(left, right + 1))
            claimed.update(indices)
            expanded.append(_clone_match_with_indices(match, indices))
            events.append({
                "type": "anchor_boundary_expansion",
                "title": match.title,
                "anchor_ranges": _indices_to_ranges(group),
                "expanded_ranges": _indices_to_ranges(indices),
                "duration_seconds": round(
                    _segment_range_duration_seconds(segments, left, right), 3
                ),
                "max_expansion_seconds": max_span,
            })
    expanded.sort(key=lambda item: min(item.segment_indices))
    return expanded, events


def score_song_match_risks(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    matches: list[ContentMatch],
    *,
    source: str,
    structural_issue: bool = False,
) -> tuple[list[SongRiskRecord], set[tuple[str, tuple[int, ...]]]]:
    settings = get_song_risk_config(config)
    soft_min = max(0.0, float(settings.get("soft_min_seconds", 150.0)))
    soft_max = max(0.0, float(settings.get("soft_max_seconds", 360.0)))
    gap_threshold = max(0.1, float(settings.get("boundary_gap_seconds", 20.0)))
    low_confidence = float(settings.get("low_confidence", 0.65))
    review_threshold = float(settings.get("review_threshold", 0.55))
    reject_threshold = float(settings.get("reject_threshold", 0.90))
    weights = {
        "duration": max(0.0, float(settings.get("duration_weight", 1.0))),
        "boundary_expansion": max(0.0, float(settings.get("boundary_expansion_weight", 2.0))),
        "overlap": max(0.0, float(settings.get("overlap_weight", 2.0))),
        "evidence": max(0.0, float(settings.get("evidence_weight", 2.0))),
    }
    records: list[SongRiskRecord] = []
    suspicious: set[tuple[str, tuple[int, ...]]] = set()

    for match in matches:
        indices = sorted({index for index in match.segment_indices if 0 <= index < len(segments)})
        if not indices:
            continue
        start, end = min(indices), max(indices)
        duration = _segment_range_duration_seconds(segments, start, end)
        duration_risk = 0.0
        reasons: list[str] = []
        duration_outlier = False
        if soft_min > 0 and duration < soft_min:
            duration_risk = min(1.0, (soft_min - duration) / soft_min)
            duration_outlier = True
            reasons.append("duration_below_soft_range")
        if soft_max > 0 and duration > soft_max:
            duration_risk = max(duration_risk, min(1.0, (duration - soft_max) / soft_max + 0.5))
            duration_outlier = True
            reasons.append("duration_above_soft_range")

        speech_seconds = sum(
            max(0.0, float(segments[index].end) - float(segments[index].start))
            for index in indices
        )
        density_risk = 1.0 - min(1.0, speech_seconds / duration) if duration > 0 else 1.0
        max_gap = max(
            (
                max(0.0, float(segments[right].start) - float(segments[left].end))
                for left, right in zip(indices, indices[1:])
            ),
            default=0.0,
        )
        boundary_risk = max(density_risk, min(1.0, max_gap / gap_threshold))
        if boundary_risk >= 0.5:
            reasons.append("sparse_or_expanded_boundary")

        overlap_risk = 0.0
        distant_same_title = False
        for other in matches:
            if other is match or not other.segment_indices:
                continue
            same_title = other.title.strip().casefold() == match.title.strip().casefold()
            if not same_title and _matches_overlap(match, other):
                overlap_risk = 1.0
                reasons.append("different_title_overlap")
                break
            if same_title:
                left_end = min(max(indices), max(other.segment_indices))
                right_start = max(min(indices), min(other.segment_indices))
                if right_start > left_end:
                    gap = float(segments[right_start].start) - float(segments[left_end].end)
                    if gap > gap_threshold:
                        distant_same_title = True
                        overlap_risk = max(overlap_risk, min(1.0, gap / max(soft_max, 1.0)))
                        if "distant_same_title" not in reasons:
                            reasons.append("distant_same_title")

        evidence_risk = max(0.0, min(1.0, (low_confidence - match.confidence) / max(low_confidence, 0.01)))
        unknown_title = is_unknown_song_title(match.title)
        if unknown_title:
            evidence_risk = max(evidence_risk, 0.35)
            reasons.append("unknown_title")
        elif not match.lyrics_snippet.strip():
            evidence_risk = max(evidence_risk, 0.25)
            reasons.append("title_without_lyric_evidence")
        if structural_issue or source in {"json_repair", "truncated"}:
            evidence_risk = 1.0
            reasons.append("structural_output_issue")
        if source == "missed_recheck":
            evidence_risk = max(evidence_risk, 0.4)
            reasons.append("missed_recheck_candidate")

        features = {
            "duration": round(duration_risk, 6),
            "boundary_expansion": round(boundary_risk, 6),
            "overlap": round(overlap_risk, 6),
            "evidence": round(evidence_risk, 6),
        }
        total_weight = sum(weights.values()) or 1.0
        score = sum(features[name] * weights[name] for name in weights) / total_weight
        score = max(0.0, min(1.0, score))
        requires_review = (
            duration_outlier
            or distant_same_title
            or boundary_risk >= 0.75
            or score >= review_threshold
            or overlap_risk > 0
        )
        if requires_review:
            suspicious.add(_match_key(match))
        action = "review" if requires_review else "accept"
        if score >= reject_threshold:
            action = "review_requires_evidence"
            if "high_risk_requires_evidence" not in reasons:
                reasons.append("high_risk_requires_evidence")
        records.append(SongRiskRecord(
            title=match.title,
            artist=match.artist,
            segment_ranges=_indices_to_ranges(indices),
            duration_seconds=round(duration, 3),
            score=round(score, 6),
            features=features,
            reasons=reasons,
            action=action,
            source=source,
        ))
    return records, suspicious


def write_risk_audit(
    path: Path,
    *,
    strategy: str,
    source: str,
    records: list[SongRiskRecord],
    boundary_events: list[dict[str, Any]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "strategy": strategy,
        "source": source,
        "candidate_count": len(records),
        "review_count": sum(record.action != "accept" for record in records),
        "boundary_events": boundary_events or [],
        "candidates": [record.to_dict() for record in records],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def load_supported_search_titles(
    debug_root: Path,
    candidates: list[ContentMatch],
) -> set[str]:
    """Return titles supported by search result text, never by the query alone."""
    known = {
        match.title.strip().casefold(): match
        for match in candidates
        if not is_unknown_song_title(match.title)
    }
    supported: set[str] = set()
    for path in debug_root.rglob("llm_batch_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for call in payload.get("tool_calls_log", []):
            summary = call.get("result_summary")
            preview = str(call.get("result_preview", ""))
            if isinstance(summary, dict):
                evidence_text = json.dumps({
                    "results": summary.get("results", []),
                    "lyrics_hints": summary.get("lyrics_hints", []),
                }, ensure_ascii=False)
                folded = evidence_text.casefold()
                for title, match in known.items():
                    artist = match.artist.strip().casefold()
                    if title and title in folded and (not artist or artist in folded):
                        supported.add(title)
                continue
            try:
                parsed = json.loads(preview)
            except (TypeError, json.JSONDecodeError):
                try:
                    parsed = ast.literal_eval(preview)
                except (SyntaxError, ValueError):
                    parsed = None
            evidence_text = preview
            if isinstance(parsed, dict):
                evidence_text = json.dumps({
                    "results": parsed.get("results", []),
                    "lyrics_hints": parsed.get("lyrics_hints", []),
                }, ensure_ascii=False)
            else:
                evidence_text = re.sub(
                    r"['\"]query['\"]\s*:\s*['\"][^'\"]*['\"]\s*,?",
                    "",
                    evidence_text,
                    flags=re.IGNORECASE,
                )
            folded = evidence_text.casefold()
            for title, match in known.items():
                artist = match.artist.strip().casefold()
                if title and title in folded and (not artist or artist in folded):
                    supported.add(title)
    return supported
