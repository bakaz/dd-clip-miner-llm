from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_PRICING = {
    "input_cache_hit_per_1m": 0.0028,
    "input_cache_miss_per_1m": 0.14,
    "output_per_1m": 0.28,
}
PROTECTED_TITLES = ("无法拥有的人要好好道别", "电台情歌", "Kingyo Hanabi")
FIXTURE_OVERLONG_TITLES = ("只要有你", "囚鸟", "so簡単には")


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _known_title(title: str) -> bool:
    return not title.strip().casefold().startswith(("未知歌曲", "unknown"))


def _match_duration(match: dict[str, Any], transcript: list[dict[str, Any]]) -> float:
    indices = sorted({int(index) for index in match.get("segment_indices", [])})
    indices = [index for index in indices if 0 <= index < len(transcript)]
    if not indices:
        return 0.0
    return max(0.0, float(transcript[indices[-1]]["end"]) - float(transcript[indices[0]]["start"]))


def _overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_indices = set(left.get("segment_indices", []))
    right_indices = set(right.get("segment_indices", []))
    if not left_indices:
        return 0.0
    return len(left_indices & right_indices) / len(left_indices)


def _indices_to_ranges(indices: list[int]) -> list[list[int]]:
    values = sorted(set(indices))
    if not values:
        return []
    ranges: list[list[int]] = []
    start = previous = values[0]
    for index in values[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append([start, previous])
        start = previous = index
    ranges.append([start, previous])
    return ranges


def _usage_cost(usage: dict[str, Any], pricing: dict[str, float]) -> float:
    totals = usage.get("totals", {})
    return (
        float(totals.get("prompt_cache_hit_tokens") or 0)
        * pricing["input_cache_hit_per_1m"]
        + float(totals.get("prompt_cache_miss_tokens") or 0)
        * pricing["input_cache_miss_per_1m"]
        + float(totals.get("completion_tokens") or 0)
        * pricing["output_per_1m"]
    ) / 1_000_000


def evaluate_song_profile(
    profile_dir: Path,
    transcript: list[dict[str, Any]],
    weak_anchors: list[dict[str, Any]],
    *,
    pricing: dict[str, float] | None = None,
) -> dict[str, Any]:
    pricing = pricing or DEFAULT_PRICING
    matches = _load_json(profile_dir / "song" / "matches.json", [])
    usage = _load_json(profile_dir / "usage_summary.json", {})
    exact_keys: set[tuple[str, tuple[int, ...]]] = set()
    duplicates = 0
    conflicts = 0
    same_title_overlaps = 0
    for index, match in enumerate(matches):
        key = (
            str(match.get("title", "")).strip().casefold(),
            tuple(sorted(set(match.get("segment_indices", [])))),
        )
        if key in exact_keys:
            duplicates += 1
        exact_keys.add(key)
        for other in matches[index + 1:]:
            overlap = set(match.get("segment_indices", [])) & set(
                other.get("segment_indices", [])
            )
            if not overlap:
                continue
            if (
                str(match.get("title", "")).strip().casefold()
                == str(other.get("title", "")).strip().casefold()
            ):
                same_title_overlaps += 1
            else:
                conflicts += 1

    anchor_hits = 0
    title_agreements = 0
    for anchor in weak_anchors:
        best = max(matches, key=lambda item: _overlap_ratio(anchor, item), default=None)
        ratio = _overlap_ratio(anchor, best) if best else 0.0
        if ratio >= 0.5:
            anchor_hits += 1
            if (
                str(anchor.get("title", "")).strip().casefold()
                == str(best.get("title", "")).strip().casefold()
            ):
                title_agreements += 1

    weak_coverage = {
        index
        for anchor in weak_anchors
        for index in anchor.get("segment_indices", [])
    }
    predicted_coverage = {
        index
        for match in matches
        for index in match.get("segment_indices", [])
    }
    coverage_intersection = weak_coverage & predicted_coverage
    temporal_precision = (
        len(coverage_intersection) / len(predicted_coverage)
        if predicted_coverage else None
    )
    temporal_recall = (
        len(coverage_intersection) / len(weak_coverage)
        if weak_coverage else None
    )
    fragmented_anchors = 0
    for anchor in weak_anchors:
        anchor_indices = set(anchor.get("segment_indices", []))
        overlapping_parts = sum(
            bool(anchor_indices & set(match.get("segment_indices", [])))
            for match in matches
        )
        if overlapping_parts > 1:
            fragmented_anchors += 1
    coalesced_anchors = 0
    mixed_title_coalesced = 0
    for match in matches:
        overlapping_anchors = [
            anchor for anchor in weak_anchors
            if set(anchor.get("segment_indices", []))
            & set(match.get("segment_indices", []))
        ]
        if len(overlapping_anchors) > 1:
            coalesced_anchors += 1
            if len({
                str(anchor.get("title", "")).strip().casefold()
                for anchor in overlapping_anchors
            }) > 1:
                mixed_title_coalesced += 1

    outliers = []
    for match in matches:
        duration = _match_duration(match, transcript)
        if duration < 150.0 or duration > 360.0:
            outliers.append({
                "title": match.get("title", ""),
                "duration_seconds": round(duration, 3),
                "segment_ranges": _indices_to_ranges(match.get("segment_indices", [])),
                "classification": "fixture_risk_example_not_global_rule",
            })

    totals = usage.get("totals", {})
    input_tokens = int(totals.get("prompt_tokens") or 0)
    if not input_tokens:
        input_tokens = int(totals.get("prompt_cache_hit_tokens") or 0) + int(
            totals.get("prompt_cache_miss_tokens") or 0
        )
    hit_ratio = (
        int(totals.get("prompt_cache_hit_tokens") or 0) / input_tokens
        if input_tokens else None
    )
    result = {
        "match_count": len(matches),
        "known_title_count": sum(_known_title(str(match.get("title", ""))) for match in matches),
        "unknown_title_count": sum(not _known_title(str(match.get("title", ""))) for match in matches),
        "exact_duplicate_count": duplicates,
        "different_title_overlap_pairs": conflicts,
        "same_title_overlap_pairs": same_title_overlaps,
        "weak_anchor_count": len(weak_anchors),
        "weak_anchor_hit_count": anchor_hits,
        "weak_anchor_recall": anchor_hits / len(weak_anchors) if weak_anchors else None,
        "weak_anchor_title_agreement_count": title_agreements,
        "temporal_coverage_precision": temporal_precision,
        "temporal_coverage_recall": temporal_recall,
        "fragmented_weak_anchor_count": fragmented_anchors,
        "coalesced_weak_anchor_count": coalesced_anchors,
        "mixed_title_coalesced_count": mixed_title_coalesced,
        "fixture_duration_risk_examples": outliers,
        "cache_hit_ratio": hit_ratio,
        "actual_cost_usd": round(_usage_cost(usage, pricing), 6),
        "protected_title_presence": {
            title: any(
                str(match.get("title", "")).strip().casefold() == title.casefold()
                for match in matches
            )
            for title in PROTECTED_TITLES
        },
        "fixture_overlong_regressions": [
            {
                "title": match.get("title", ""),
                "duration_seconds": round(_match_duration(match, transcript), 3),
            }
            for match in matches
            if str(match.get("title", "")).strip().casefold()
            in {title.casefold() for title in FIXTURE_OVERLONG_TITLES}
            and _match_duration(match, transcript) > 360.0
        ],
    }
    adaptive = _load_json(profile_dir / "song" / "adaptive_strategies.json", {})
    estimate = adaptive.get("chosen_total_usd")
    if estimate not in (None, 0):
        error_ratio = abs(result["actual_cost_usd"] - float(estimate)) / float(estimate)
        result["cost_estimator"] = {
            "estimated_usd": float(estimate),
            "actual_usd": result["actual_cost_usd"],
            "error_ratio": round(error_ratio, 6),
            "valid_for_runtime_selection": error_ratio <= 0.15,
        }
    return result


def evaluate_song_run(
    run_dir: Path,
    *,
    baseline_profile: str = "accuracy",
    pricing: dict[str, float] | None = None,
) -> dict[str, Any]:
    transcript = _load_json(run_dir / "02_asr" / "transcript.json", [])
    profiles_root = run_dir / "02_asr" / "llm"
    baseline_matches = _load_json(
        profiles_root / baseline_profile / "song" / "matches.json", []
    )
    weak_anchors = [
        match for match in baseline_matches
        if float(match.get("confidence", 0.0)) >= 0.6
        and len(match.get("segment_indices", [])) > 0
    ]
    profiles = {
        path.name: evaluate_song_profile(
            path, transcript, weak_anchors, pricing=pricing,
        )
        for path in profiles_root.iterdir()
        if path.is_dir() and (path / "song" / "matches.json").is_file()
    }
    baseline = profiles.get(baseline_profile, {})
    baseline_cost = float(baseline.get("actual_cost_usd") or 0.0)
    for name, metrics in profiles.items():
        if name == baseline_profile:
            continue
        metrics["gates"] = {
            "cost_at_most_90_percent_of_accuracy": (
                baseline_cost > 0
                and float(metrics.get("actual_cost_usd") or 0.0) <= baseline_cost * 0.9
            ),
            "temporal_coverage_precision_at_least_80_percent": (
                metrics.get("temporal_coverage_precision") is not None
                and float(metrics["temporal_coverage_precision"]) >= 0.80
            ),
            "temporal_coverage_recall_at_least_90_percent": (
                metrics.get("temporal_coverage_recall") is not None
                and float(metrics["temporal_coverage_recall"]) >= 0.90
            ),
            "no_fixture_overlong_regressions": not metrics["fixture_overlong_regressions"],
            "no_exact_duplicates": metrics["exact_duplicate_count"] == 0,
            "no_different_title_overlap": metrics["different_title_overlap_pairs"] == 0,
            "no_same_title_overlap": metrics["same_title_overlap_pairs"] == 0,
        }
        metrics["passes_all_gates"] = all(metrics["gates"].values())
    return {
        "run_dir": str(run_dir),
        "baseline_profile": baseline_profile,
        "weak_anchor_policy": "accuracy confidence>=0.6 intervals; temporal coverage is weak truth, titles are informational",
        "duration_policy": "150-360s is fixture-only negative/risk evidence, not a runtime validity rule",
        "profiles": profiles,
    }
