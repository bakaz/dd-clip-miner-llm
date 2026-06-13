"""Adaptive strategy resolution for song review and missed recheck."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import get_llm_config, get_song_recheck_config, get_song_review_config
from .models import ContentMatch, TranscriptSegment

ADAPTIVE_STRATEGIES_FILENAME = "adaptive_strategies.json"

REVIEW_SCOPES = frozenset({"local", "full", "adaptive"})
MISSED_STRATEGIES = frozenset({"windowed", "full_transcript", "adaptive"})
KV_PROFILE = "kv_optimized"
ADAPTIVE_MODES = frozenset({"heuristic", "cost_estimate"})

DEFAULT_REVIEW_ADAPTIVE: dict[str, int] = {
    "local_max_clusters": 3,
    "full_min_clusters": 6,
    "full_min_segments": 2000,
}

DEFAULT_MISSED_ADAPTIVE: dict[str, int] = {
    "full_transcript_max_segments": 3500,
    "windowed_min_target_ranges": 19,
}

DEFAULT_COST_MARGIN_RATIO = 0.05


def _kv_cache_friendly_enabled(config: dict[str, Any]) -> bool:
    return bool(get_llm_config(config).get("cache_friendly_prompt_layout", False))


def _adaptive_enabled(config: dict[str, Any]) -> bool:
    return (
        str(config.get("_profile_name") or "") == KV_PROFILE
        and _kv_cache_friendly_enabled(config)
    )


def _adaptive_settings(
    parent: dict[str, Any],
    defaults: dict[str, int],
) -> dict[str, int]:
    raw = parent.get("adaptive", {})
    if not isinstance(raw, dict):
        raw = {}
    return {key: int(raw.get(key, value)) for key, value in defaults.items()}


def _adaptive_mode(parent: dict[str, Any]) -> str:
    raw = parent.get("adaptive", {})
    if not isinstance(raw, dict):
        raw = {}
    mode = str(raw.get("mode", "cost_estimate")).strip().lower()
    if mode not in ADAPTIVE_MODES:
        mode = "cost_estimate"
    return mode


def _cost_margin_ratio(parent: dict[str, Any]) -> float:
    raw = parent.get("adaptive", {})
    if not isinstance(raw, dict):
        return DEFAULT_COST_MARGIN_RATIO
    try:
        return float(raw.get("cost_margin_ratio", DEFAULT_COST_MARGIN_RATIO))
    except (TypeError, ValueError):
        return DEFAULT_COST_MARGIN_RATIO


def _empty_details() -> dict[str, Any]:
    return {}


def _cost_details(
    *,
    adaptive_mode: str,
    local_cost: Any | None = None,
    full_cost: Any | None = None,
    windowed_cost: Any | None = None,
    chosen: str,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "adaptive_mode": adaptive_mode,
        "cost_estimate_chosen": chosen,
    }
    if local_cost is not None:
        details["cost_estimate_local_usd"] = round(local_cost.total_usd, 6)
        details["cost_estimate_local"] = local_cost.to_dict()
    if full_cost is not None:
        details["cost_estimate_full_usd"] = round(full_cost.total_usd, 6)
        details["cost_estimate_full"] = full_cost.to_dict()
    if windowed_cost is not None:
        details["cost_estimate_windowed_usd"] = round(windowed_cost.total_usd, 6)
        details["cost_estimate_windowed"] = windowed_cost.to_dict()
    return details


def _choose_by_cost(
    *,
    cheaper: str,
    conservative: str,
    cheap_cost: float,
    other_cost: float,
    margin_ratio: float,
) -> str:
    baseline = max(cheap_cost, other_cost, 1e-9)
    if abs(cheap_cost - other_cost) / baseline < margin_ratio:
        return conservative
    return cheaper


def adaptive_strategies_path(llm_dir: Path) -> Path:
    return llm_dir / ADAPTIVE_STRATEGIES_FILENAME


def load_adaptive_strategies_cache(llm_dir: Path) -> dict[str, Any] | None:
    path = adaptive_strategies_path(llm_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def save_adaptive_strategies_cache(llm_dir: Path, payload: dict[str, Any]) -> Path:
    path = adaptive_strategies_path(llm_dir)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _joint_cost_margin(config: dict[str, Any]) -> float:
    review_config = get_song_review_config(config)
    recheck_config = get_song_recheck_config(config)
    return max(
        _cost_margin_ratio(review_config),
        _cost_margin_ratio(recheck_config),
    )


def _review_options(config: dict[str, Any]) -> list[str]:
    requested = str(
        get_song_review_config(config).get("transcript_scope", "local")
    ).strip().lower()
    if requested == "adaptive" and _adaptive_enabled(config):
        return ["local", "full"]
    if requested in {"local", "full"}:
        return [requested]
    return ["local"]


def _missed_options(config: dict[str, Any]) -> list[str]:
    requested = str(
        get_song_recheck_config(config).get("strategy", "windowed")
    ).strip().lower()
    if requested == "adaptive" and _adaptive_enabled(config):
        return ["windowed", "full_transcript"]
    if requested in {"windowed", "full_transcript"}:
        return [requested]
    return ["windowed"]


def _uses_joint_cost_estimate(config: dict[str, Any]) -> bool:
    if not _adaptive_enabled(config):
        return False
    review_config = get_song_review_config(config)
    recheck_config = get_song_recheck_config(config)
    review_adaptive = (
        str(review_config.get("transcript_scope", "local")).strip().lower() == "adaptive"
    )
    missed_adaptive = (
        str(recheck_config.get("strategy", "windowed")).strip().lower() == "adaptive"
    )
    if not review_adaptive and not missed_adaptive:
        return False
    if review_adaptive and _adaptive_mode(review_config) != "cost_estimate":
        return False
    if missed_adaptive and _adaptive_mode(recheck_config) != "cost_estimate":
        return False
    return True


def _conservative_joint_choice(
    review_options: list[str],
    missed_options: list[str],
) -> tuple[str, str]:
    review = "local" if "local" in review_options else review_options[0]
    missed = (
        "windowed" if "windowed" in missed_options else missed_options[0]
    )
    return review, missed


def resolve_song_adaptive_strategies(
    config: dict[str, Any],
    *,
    clusters: list[list[ContentMatch]],
    segments: list[TranscriptSegment],
    matches: list[ContentMatch],
    target_ranges: list[tuple[int, int]],
    recognizer: Any | None = None,
) -> dict[str, Any]:
    """Pick review scope and missed strategy via joint pipeline cost minimization."""
    from .song_adaptive_cost import (
        estimate_main_cost,
        estimate_missed_cost,
        estimate_overlong_cost,
        estimate_review_after_cost,
        estimate_review_cost,
    )

    review_config = get_song_review_config(config)
    recheck_config = get_song_recheck_config(config)
    review_requested = str(review_config.get("transcript_scope", "local")).strip().lower()
    missed_requested = str(recheck_config.get("strategy", "windowed")).strip().lower()

    if str(config.get("_profile_name") or "") == "accuracy":
        return {
            "resolution_mode": "accuracy_profile_fixed",
            "review_scope_requested": review_requested,
            "missed_strategy_requested": missed_requested,
            "review_scope_resolved": "local",
            "missed_strategy_resolved": "windowed",
            "reason": "accuracy_profile_forced_local_windowed",
            "combinations": [],
        }

    review_options = _review_options(config)
    missed_options = _missed_options(config)

    if not _uses_joint_cost_estimate(config) or not clusters:
        review_scope, review_reason, review_details = resolve_review_transcript_scope(
            config,
            clusters=clusters,
            segments=segments,
            recognizer=recognizer,
        )
        missed_strategy, missed_reason, missed_details = resolve_missed_recheck_strategy(
            config,
            segments=segments,
            matches=matches,
            target_ranges=target_ranges,
            recognizer=recognizer,
        )
        return {
            "resolution_mode": review_details.get("adaptive_mode", "configured"),
            "review_scope_requested": review_requested,
            "missed_strategy_requested": missed_requested,
            "review_scope_resolved": review_scope,
            "missed_strategy_resolved": missed_strategy,
            "reason": f"review:{review_reason}; missed:{missed_reason}",
            "review_details": review_details,
            "missed_details": missed_details,
            "combinations": [],
        }

    main_cost = estimate_main_cost(
        config,
        segments=segments,
        recognizer=recognizer,
    )
    overlong_cost = estimate_overlong_cost(
        config,
        segments=segments,
        matches=matches,
        recognizer=recognizer,
    )
    review_costs = {
        scope: estimate_review_cost(
            config,
            scope=scope,
            clusters=clusters,
            segments=segments,
            recognizer=recognizer,
        )
        for scope in ("local", "full")
    }
    review_after_costs: dict[tuple[str, str], Any] = {}
    missed_costs = {
        strategy: estimate_missed_cost(
            config,
            strategy=strategy,
            segments=segments,
            matches=matches,
            target_ranges=target_ranges,
            recognizer=recognizer,
            include_fallback_penalty=(strategy == "full_transcript"),
        )
        for strategy in ("windowed", "full_transcript")
    }

    combinations: list[dict[str, Any]] = []
    for review_scope in review_options:
        for missed_strategy in missed_options:
            review_after_key = (review_scope, missed_strategy)
            if review_after_key not in review_after_costs:
                review_after_costs[review_after_key] = estimate_review_after_cost(
                    config,
                    scope=review_scope,
                    missed_strategy=missed_strategy,
                    clusters=clusters,
                    segments=segments,
                    target_ranges=target_ranges,
                    recognizer=recognizer,
                )
            review_after_cost = review_after_costs[review_after_key]
            review_before_cost = review_costs[review_scope]
            missed_cost = missed_costs[missed_strategy]
            total_usd = (
                main_cost.total_usd
                + review_before_cost.total_usd
                + overlong_cost.total_usd
                + missed_cost.total_usd
                + review_after_cost.total_usd
            )
            combinations.append({
                "review_scope": review_scope,
                "missed_strategy": missed_strategy,
                "total_usd": round(total_usd, 6),
                "main_cost_usd": round(main_cost.total_usd, 6),
                "review_before_cost_usd": round(review_before_cost.total_usd, 6),
                "review_after_cost_usd": round(review_after_cost.total_usd, 6),
                "overlong_cost_usd": round(overlong_cost.total_usd, 6),
                "missed_cost_usd": round(missed_cost.total_usd, 6),
                "review_cost_usd": round(
                    review_before_cost.total_usd + review_after_cost.total_usd,
                    6,
                ),
            })

    margin = _joint_cost_margin(config)
    best = min(combinations, key=lambda item: item["total_usd"])
    conservative_review, conservative_missed = _conservative_joint_choice(
        review_options,
        missed_options,
    )
    conservative = next(
        item
        for item in combinations
        if item["review_scope"] == conservative_review
        and item["missed_strategy"] == conservative_missed
    )
    baseline = max(best["total_usd"], conservative["total_usd"], 1e-9)
    chosen = (
        conservative
        if abs(best["total_usd"] - conservative["total_usd"]) / baseline < margin
        else best
    )

    return {
        "resolution_mode": "joint_cost_estimate",
        "review_scope_requested": review_requested,
        "missed_strategy_requested": missed_requested,
        "review_scope_resolved": chosen["review_scope"],
        "missed_strategy_resolved": chosen["missed_strategy"],
        "reason": (
            "adaptive_joint_cost_"
            f"{chosen['review_scope']}_{chosen['missed_strategy']}_"
            f"est_{chosen['total_usd']:.4f}_usd"
        ),
        "chosen_total_usd": chosen["total_usd"],
        "pipeline_cost_usd": chosen["total_usd"],
        "main_cost_usd": chosen["main_cost_usd"],
        "overlong_cost_usd": chosen["overlong_cost_usd"],
        "review_before_cost_usd": chosen["review_before_cost_usd"],
        "review_after_cost_usd": chosen["review_after_cost_usd"],
        "missed_cost_usd": chosen["missed_cost_usd"],
        "combinations": combinations,
        "main_details": {
            "adaptive_mode": "joint_cost_estimate",
            "cost_estimate": main_cost.to_dict(),
        },
        "overlong_details": {
            "adaptive_mode": "joint_cost_estimate",
            "cost_estimate": overlong_cost.to_dict(),
        },
        "review_details": {
            "adaptive_mode": "joint_cost_estimate",
            "cost_estimate_chosen": chosen["review_scope"],
            "cost_estimate_local_usd": round(review_costs["local"].total_usd, 6),
            "cost_estimate_full_usd": round(review_costs["full"].total_usd, 6),
            "cost_estimate_local": review_costs["local"].to_dict(),
            "cost_estimate_full": review_costs["full"].to_dict(),
        },
        "missed_details": {
            "adaptive_mode": "joint_cost_estimate",
            "cost_estimate_chosen": chosen["missed_strategy"],
            "cost_estimate_windowed_usd": round(
                missed_costs["windowed"].total_usd,
                6,
            ),
            "cost_estimate_full_usd": round(
                missed_costs["full_transcript"].total_usd,
                6,
            ),
            "cost_estimate_windowed": missed_costs["windowed"].to_dict(),
            "cost_estimate_full": missed_costs["full_transcript"].to_dict(),
        },
    }


def ensure_song_adaptive_strategies(
    llm_dir: Path,
    config: dict[str, Any],
    *,
    clusters: list[list[ContentMatch]],
    segments: list[TranscriptSegment],
    matches: list[ContentMatch],
    target_ranges: list[tuple[int, int]],
    recognizer: Any | None = None,
) -> dict[str, Any]:
    cached = load_adaptive_strategies_cache(llm_dir)
    if cached is not None:
        return cached
    payload = resolve_song_adaptive_strategies(
        config,
        clusters=clusters,
        segments=segments,
        matches=matches,
        target_ranges=target_ranges,
        recognizer=recognizer,
    )
    save_adaptive_strategies_cache(llm_dir, payload)
    return payload


def _resolve_review_heuristic(
    config: dict[str, Any],
    *,
    cluster_count: int,
    segment_count: int,
) -> tuple[str, str]:
    review_config = get_song_review_config(config)
    adaptive = _adaptive_settings(review_config, DEFAULT_REVIEW_ADAPTIVE)
    if cluster_count <= adaptive["local_max_clusters"]:
        return "local", f"adaptive_clusters_le_{adaptive['local_max_clusters']}"

    if (
        cluster_count >= adaptive["full_min_clusters"]
        and segment_count >= adaptive["full_min_segments"]
    ):
        return "full", (
            f"adaptive_clusters_ge_{adaptive['full_min_clusters']}_"
            f"segments_ge_{adaptive['full_min_segments']}"
        )

    return "local", "adaptive_mid_cluster_band"


def _resolve_review_cost_estimate(
    config: dict[str, Any],
    *,
    clusters: list[list[ContentMatch]],
    segments: list[TranscriptSegment],
    recognizer: Any | None,
) -> tuple[str, str, dict[str, Any]]:
    from .song_adaptive_cost import estimate_review_cost

    review_config = get_song_review_config(config)
    margin = _cost_margin_ratio(review_config)
    local_cost = estimate_review_cost(
        config,
        scope="local",
        clusters=clusters,
        segments=segments,
        recognizer=recognizer,
    )
    full_cost = estimate_review_cost(
        config,
        scope="full",
        clusters=clusters,
        segments=segments,
        recognizer=recognizer,
    )
    if full_cost.total_usd < local_cost.total_usd:
        chosen = _choose_by_cost(
            cheaper="full",
            conservative="local",
            cheap_cost=full_cost.total_usd,
            other_cost=local_cost.total_usd,
            margin_ratio=margin,
        )
        reason = (
            f"adaptive_cost_{chosen}_cheaper_est_"
            f"{full_cost.total_usd:.4f}_vs_{local_cost.total_usd:.4f}_usd"
        )
    else:
        chosen = _choose_by_cost(
            cheaper="local",
            conservative="local",
            cheap_cost=local_cost.total_usd,
            other_cost=full_cost.total_usd,
            margin_ratio=margin,
        )
        reason = (
            f"adaptive_cost_{chosen}_cheaper_est_"
            f"{local_cost.total_usd:.4f}_vs_{full_cost.total_usd:.4f}_usd"
        )
    return chosen, reason, _cost_details(
        adaptive_mode="cost_estimate",
        local_cost=local_cost,
        full_cost=full_cost,
        chosen=chosen,
    )


def resolve_review_transcript_scope(
    config: dict[str, Any],
    *,
    cluster_count: int | None = None,
    segment_count: int | None = None,
    clusters: list[list[ContentMatch]] | None = None,
    segments: list[TranscriptSegment] | None = None,
    recognizer: Any | None = None,
) -> tuple[str, str, dict[str, Any]]:
    review_config = get_song_review_config(config)
    requested = str(review_config.get("transcript_scope", "local")).strip().lower()
    if requested not in REVIEW_SCOPES:
        requested = "local"

    resolved_cluster_count = (
        len(clusters) if clusters is not None else int(cluster_count or 0)
    )
    resolved_segment_count = (
        len(segments) if segments is not None else int(segment_count or 0)
    )

    if str(config.get("_profile_name") or "") == "accuracy":
        return "local", "accuracy_profile_forced_local", _empty_details()

    if requested in {"local", "full"}:
        return requested, f"configured_{requested}", _empty_details()

    if not _adaptive_enabled(config):
        return "local", "non_kv_profile_fixed_local", _empty_details()

    mode = _adaptive_mode(review_config)
    if (
        mode == "cost_estimate"
        and clusters is not None
        and segments is not None
        and clusters
    ):
        return _resolve_review_cost_estimate(
            config,
            clusters=clusters,
            segments=segments,
            recognizer=recognizer,
        )

    scope, reason = _resolve_review_heuristic(
        config,
        cluster_count=resolved_cluster_count,
        segment_count=resolved_segment_count,
    )
    return scope, reason, _cost_details(
        adaptive_mode="heuristic",
        chosen=scope,
    )


def _resolve_missed_heuristic(
    config: dict[str, Any],
    *,
    segment_count: int,
    target_range_count: int,
) -> tuple[str, str]:
    recheck_config = get_song_recheck_config(config)
    adaptive = _adaptive_settings(recheck_config, DEFAULT_MISSED_ADAPTIVE)
    if segment_count > adaptive["full_transcript_max_segments"]:
        return (
            "windowed",
            f"adaptive_segments_gt_{adaptive['full_transcript_max_segments']}",
        )

    if target_range_count >= adaptive["windowed_min_target_ranges"]:
        return (
            "windowed",
            f"adaptive_target_ranges_ge_{adaptive['windowed_min_target_ranges']}",
        )

    return "full_transcript", "adaptive_kv_friendly_full_audit"


def _resolve_missed_cost_estimate(
    config: dict[str, Any],
    *,
    segments: list[TranscriptSegment],
    matches: list[ContentMatch],
    target_ranges: list[tuple[int, int]],
    recognizer: Any | None,
) -> tuple[str, str, dict[str, Any]]:
    from .song_adaptive_cost import estimate_missed_cost

    recheck_config = get_song_recheck_config(config)
    margin = _cost_margin_ratio(recheck_config)
    windowed_cost = estimate_missed_cost(
        config,
        strategy="windowed",
        segments=segments,
        matches=matches,
        target_ranges=target_ranges,
        recognizer=recognizer,
        include_fallback_penalty=False,
    )
    full_cost = estimate_missed_cost(
        config,
        strategy="full_transcript",
        segments=segments,
        matches=matches,
        target_ranges=target_ranges,
        recognizer=recognizer,
        include_fallback_penalty=True,
    )
    if full_cost.total_usd < windowed_cost.total_usd:
        chosen = _choose_by_cost(
            cheaper="full_transcript",
            conservative="windowed",
            cheap_cost=full_cost.total_usd,
            other_cost=windowed_cost.total_usd,
            margin_ratio=margin,
        )
        reason = (
            f"adaptive_cost_{chosen}_cheaper_est_"
            f"{full_cost.total_usd:.4f}_vs_{windowed_cost.total_usd:.4f}_usd"
        )
    else:
        chosen = _choose_by_cost(
            cheaper="windowed",
            conservative="windowed",
            cheap_cost=windowed_cost.total_usd,
            other_cost=full_cost.total_usd,
            margin_ratio=margin,
        )
        reason = (
            f"adaptive_cost_{chosen}_cheaper_est_"
            f"{windowed_cost.total_usd:.4f}_vs_{full_cost.total_usd:.4f}_usd"
        )
    return chosen, reason, _cost_details(
        adaptive_mode="cost_estimate",
        full_cost=full_cost,
        windowed_cost=windowed_cost,
        chosen=chosen,
    )


def resolve_missed_recheck_strategy(
    config: dict[str, Any],
    *,
    segment_count: int | None = None,
    target_range_count: int | None = None,
    segments: list[TranscriptSegment] | None = None,
    matches: list[ContentMatch] | None = None,
    target_ranges: list[tuple[int, int]] | None = None,
    recognizer: Any | None = None,
) -> tuple[str, str, dict[str, Any]]:
    recheck_config = get_song_recheck_config(config)
    requested = str(recheck_config.get("strategy", "windowed")).strip().lower()
    if requested not in MISSED_STRATEGIES:
        requested = "windowed"

    resolved_segment_count = (
        len(segments) if segments is not None else int(segment_count or 0)
    )
    resolved_target_count = (
        len(target_ranges)
        if target_ranges is not None
        else int(target_range_count or 0)
    )

    if str(config.get("_profile_name") or "") == "accuracy":
        return "windowed", "accuracy_profile_forced_windowed", _empty_details()

    if requested in {"windowed", "full_transcript"}:
        return requested, f"configured_{requested}", _empty_details()

    if not _adaptive_enabled(config):
        return "windowed", "non_kv_profile_fixed_windowed", _empty_details()

    mode = _adaptive_mode(recheck_config)
    if (
        mode == "cost_estimate"
        and segments is not None
        and matches is not None
        and target_ranges is not None
        and target_ranges
    ):
        return _resolve_missed_cost_estimate(
            config,
            segments=segments,
            matches=matches,
            target_ranges=target_ranges,
            recognizer=recognizer,
        )

    strategy, reason = _resolve_missed_heuristic(
        config,
        segment_count=resolved_segment_count,
        target_range_count=resolved_target_count,
    )
    return strategy, reason, _cost_details(
        adaptive_mode="heuristic",
        chosen=strategy,
    )