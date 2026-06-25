"""Accuracy (legacy) song pipeline.

High-precision pipeline with review + missed_recheck.
Used by accuracy and kv_v2 profiles.
"""
from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any

from ...config import PROFILE_ACCURACY, PROFILE_KV_V2
from ...models import ContentMatch, TranscriptSegment


def _match_coverage(matches: list[ContentMatch]) -> int:
    """Count unique segments covered by matches."""
    return len({idx for m in matches for idx in m.segment_indices})


def _indices_to_ranges(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    sorted_indices = sorted(set(indices))
    ranges: list[list[int]] = []
    start = prev = sorted_indices[0]
    for index in sorted_indices[1:]:
        if index == prev + 1:
            prev = index
            continue
        ranges.append([start, prev])
        start = prev = index
    ranges.append([start, prev])
    return ranges


def _match_key(match: ContentMatch) -> tuple[str, tuple[int, ...]]:
    return (match.title.strip().casefold(), tuple(sorted(match.segment_indices)))


def _match_summary(match: ContentMatch) -> dict[str, Any]:
    indices = sorted(set(match.segment_indices))
    return {
        "title": match.title,
        "content_type": match.content_type,
        "confidence": match.confidence,
        "ranges": _indices_to_ranges(indices),
    }


def _stage_audit(
    stage: str,
    before_matches: list[ContentMatch],
    after_matches: list[ContentMatch],
) -> dict[str, Any]:
    before_keys = {_match_key(match): match for match in before_matches}
    after_keys = {_match_key(match): match for match in after_matches}
    added_keys = [key for key in after_keys if key not in before_keys]
    removed_keys = [key for key in before_keys if key not in after_keys]
    input_coverage = _match_coverage(before_matches)
    output_coverage = _match_coverage(after_matches)
    return {
        "stage": stage,
        "input_count": len(before_matches),
        "output_count": len(after_matches),
        "input_coverage_segments": input_coverage,
        "output_coverage_segments": output_coverage,
        "coverage_delta": output_coverage - input_coverage,
        "added_count": len(added_keys),
        "removed_count": len(removed_keys),
        "added_matches": [_match_summary(after_keys[key]) for key in added_keys],
        "removed_matches": [_match_summary(before_keys[key]) for key in removed_keys],
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _usage_total_tokens(summary: dict[str, Any] | None) -> int | None:
    if not summary:
        return None
    totals = summary.get("totals")
    if not isinstance(totals, dict):
        return None
    return (
        int(totals.get("prompt_cache_hit_tokens") or 0)
        + int(totals.get("prompt_cache_miss_tokens") or 0)
        + int(totals.get("completion_tokens") or 0)
    )


def _build_survival_audit(
    stages: list[dict[str, Any]],
    llm_dir: Path,
    *,
    is_kv_v2: bool,
) -> None:
    """Write survival audit to llm_dir/survival_audit.json."""
    review_after_summary = _load_json(
        llm_dir / "review" / "after_missed_recheck" / "summary.json"
    )
    deletion_events = []
    if review_after_summary:
        raw_events = review_after_summary.get("kv_v2_review_deletion_events", [])
        if isinstance(raw_events, list):
            deletion_events = [event for event in raw_events if isinstance(event, dict)]
    deletion_action_counts: dict[str, int] = {}
    for event in deletion_events:
        action = str(event.get("action", "unknown"))
        deletion_action_counts[action] = deletion_action_counts.get(action, 0) + 1

    audit = {
        "stages": stages,
        "total_stages": len(stages),
        "review_after_deletion_event_counts": deletion_action_counts,
        "review_after_deletion_events": deletion_events,
    }
    if is_kv_v2:
        audit["profile"] = PROFILE_KV_V2
        kv_usage = _load_json(llm_dir.parent / "usage_summary.json")
        accuracy_usage = _load_json(llm_dir.parent.parent / PROFILE_ACCURACY / "usage_summary.json")
        kv_tokens = _usage_total_tokens(kv_usage)
        accuracy_tokens = _usage_total_tokens(accuracy_usage)
        if kv_tokens is not None and accuracy_tokens:
            audit["cost_comparison"] = {
                "kv_v2_total_tokens": kv_tokens,
                "accuracy_total_tokens": accuracy_tokens,
                "kv_v2_to_accuracy_token_ratio": round(kv_tokens / accuracy_tokens, 4),
            }
    (llm_dir / "survival_audit.json").write_text(
        _json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    llm_dir: Path,
    *,
    matches: list[ContentMatch],
) -> list[ContentMatch]:
    """Run accuracy (legacy) pipeline: review + recheck."""
    from ...song_postprocess import (
        _recheck_overlong_song_matches,
        _recheck_uncovered_song_segments,
        _review_song_matches,
    )

    is_kv_v2 = config.get("_profile_name") == PROFILE_KV_V2
    stages: list[dict[str, Any]] = []

    def _log_stage(
        stage: str,
        before_matches: list[ContentMatch],
        after_matches: list[ContentMatch],
    ) -> None:
        print(
            f"  Song legacy stage {stage}: "
            f"input {len(before_matches)} match(es), "
            f"output {len(after_matches)} match(es)",
            flush=True,
        )
        stages.append(_stage_audit(stage, before_matches, after_matches))

    _log_stage("main_identify", [], matches)

    before_matches = list(matches)
    if len(matches) > 0:
        matches = _review_song_matches(
            segments, config, recognizer, matches, llm_dir,
            phase="before_missed_recheck",
        )
    _log_stage("review_before", before_matches, matches)

    before_matches = list(matches)
    matches = _recheck_overlong_song_matches(
        segments, config, recognizer, matches, llm_dir,
    )
    _log_stage("overlong_recheck", before_matches, matches)

    before_matches = list(matches)
    matches = _recheck_uncovered_song_segments(
        segments, config, recognizer, matches, llm_dir,
    )
    _log_stage("missed_recheck", before_matches, matches)

    before_matches = list(matches)
    matches = _review_song_matches(
        segments, config, recognizer, matches, llm_dir,
        phase="after_missed_recheck",
    )
    _log_stage("review_after", before_matches, matches)

    if is_kv_v2:
        _build_survival_audit(stages, llm_dir, is_kv_v2=True)

    return matches
