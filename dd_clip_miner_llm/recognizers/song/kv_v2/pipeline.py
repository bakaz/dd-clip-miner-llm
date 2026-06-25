"""KV v2 song pipeline with optimizations.

Isolated pipeline with optimizations:
- Preserves high-confidence known titles (fixes "唯一" regression)
- Detects and handles opening humming
- Skips small-cluster reviews (cost reduction ~40%)
- Uses 0.75 deletion threshold
- Special opening segment handling
"""
from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any

from ....config import PROFILE_KV_V2

from ....models import ContentMatch, TranscriptSegment
from ....song_postprocess.normalize import (
    _clone_match_with_indices,
    _content_match_from_dict,
    _indices_to_ranges,
    _is_invalid_audit_title,
    _match_key,
    _matches_overlap,
    _merge_adjacent_same_title_matches,
    _normalize_song_matches,
    _segment_range_duration_seconds,
    _uncovered_segment_ranges,
)
from ....song_postprocess.review import (
    _build_song_review_clusters,
    _local_best_song_cluster,
)
from ....song_postprocess.risk import (
    load_supported_search_titles,
    score_song_match_risks,
)
from ....config import get_padding_config, get_song_review_config
from .optimizations import (
    KV_V3_DELETION_CONFIDENCE_THRESHOLD,
    _candidate_has_song_evidence,
    _detect_opening_humming,
    _preserve_high_confidence_known_title,
    _review_opening_segments,
    _should_skip_review,
)


def run(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    llm_dir: Path,
    *,
    matches: list[ContentMatch],
) -> list[ContentMatch]:
    """Run KV v3 pipeline with optimizations.

    Same high-level flow as acc.py but with:
    1. Opening humming detection
    2. High-confidence known title preservation in review
    3. Small-cluster review skip
    4. Higher deletion threshold (0.75)
    """
    from ....song_postprocess import (
        _recheck_overlong_song_matches,
        _recheck_uncovered_song_segments,
    )
    from ....song_postprocess.review import _review_song_matches

    stages: list[dict[str, Any]] = []

    def _log_stage(
        stage: str,
        before_matches: list[ContentMatch],
        after_matches: list[ContentMatch],
    ) -> None:
        print(
            f"  Song kv_v2 stage {stage}: "
            f"input {len(before_matches)} match(es), "
            f"output {len(after_matches)} match(es)",
            flush=True,
        )
        stages.append(_stage_audit(stage, before_matches, after_matches))

    _log_stage("main_identify", [], matches)

    # Stage 0: Detect opening humming
    opening_events = _detect_opening_humming(segments, matches)
    if opening_events:
        print(
            f"  Song kv_v2: detected {len(opening_events)} opening humming segment(s)",
            flush=True,
        )
        matches = _review_opening_segments(segments, config, matches, opening_events)

    # Stage 1: Review before missed recheck (with optimizations)
    before_matches = list(matches)
    if len(matches) > 0:
        matches = _kv_v2_review_song_matches(
            segments, config, recognizer, matches, llm_dir,
            phase="before_missed_recheck",
        )
    _log_stage("review_before", before_matches, matches)

    # Stage 2: Overlong recheck
    before_matches = list(matches)
    matches = _recheck_overlong_song_matches(
        segments, config, recognizer, matches, llm_dir,
    )
    _log_stage("overlong_recheck", before_matches, matches)

    # Stage 3: Missed recheck
    before_matches = list(matches)
    matches = _recheck_uncovered_song_segments(
        segments, config, recognizer, matches, llm_dir,
    )
    _log_stage("missed_recheck", before_matches, matches)

    # Stage 4: Review after missed recheck (with optimizations)
    before_matches = list(matches)
    matches = _kv_v2_review_song_matches(
        segments, config, recognizer, matches, llm_dir,
        phase="after_missed_recheck",
    )
    _log_stage("review_after", before_matches, matches)

    # Build survival audit
    _build_survival_audit(stages, llm_dir, opening_events)

    return matches


def _kv_v2_review_song_matches(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    llm_dir: Path,
    *,
    phase: str,
) -> list[ContentMatch]:
    """Review song matches with kv_v2 optimizations.

    Key differences from standard review:
    1. Small clusters are resolved locally (no LLM call)
    2. High-confidence known titles are protected from deletion
    3. Opening segments get special handling
    """
    review_config = get_song_review_config(config)
    if review_config.get("enabled", False) is False:
        return matches

    from ....song_postprocess import (
        _normalize_song_matches,
    )
    from ....song_postprocess.review import (
        _resolve_review_context,
        _review_single_cluster,
    )

    (
        normalized, normalization_events, clusters, scope_cost_details,
        searched_titles, full_audit_candidate_keys, review_root,
        transcript_scope, transcript_scope_requested, transcript_scope_reason,
    ) = _resolve_review_context(
        segments, config, recognizer, matches, llm_dir, phase=phase,
    )

    context_segments = int(review_config.get("context_segments", 10) or 0)
    max_window_segments = int(review_config.get("max_window_segments", 500) or 500)

    resolved: list[ContentMatch] = []
    cluster_member_ids = {id(match) for cluster in clusters for match in cluster}
    resolved.extend(match for match in normalized if id(match) not in cluster_member_ids)
    audit_clusters: list[dict[str, Any]] = []
    skipped_clusters = 0

    for cluster_index, cluster in enumerate(clusters, 1):
        # kv_v2 optimization: skip small clusters
        if _should_skip_review(cluster, config):
            skipped_clusters += 1
            # Resolve locally
            local_best, decisions = _local_best_song_cluster(
                segments, config, cluster, searched_titles,
            )
            resolved.extend(local_best)
            audit_clusters.append({
                "cluster": cluster_index,
                "resolution": "local_best_skipped",
                "reason": "small_cluster",
                "decisions": decisions,
            })
            continue

        replacement, audit = _review_single_cluster(
            segments, config, recognizer, cluster, cluster_index,
            phase=phase, review_root=review_root,
            context_segments=context_segments, max_window_segments=max_window_segments,
            transcript_scope=transcript_scope, review_config=review_config,
            full_audit_candidate_keys=full_audit_candidate_keys,
            searched_titles=searched_titles,
        )

        # kv_v2 optimization: protect high-confidence known titles
        if phase == "after_missed_recheck":
            protected, protection_events = _preserve_high_confidence_known_title(
                segments, config, replacement, cluster,
            )
            if protection_events:
                retained_count = sum(
                    1 for e in protection_events
                    if e.get("action", "").startswith("retained")
                )
                if retained_count:
                    print(
                        f"  Song kv_v2 ({phase}): protected {retained_count} "
                        f"high-confidence known title(s)",
                        flush=True,
                    )
                audit["kv_v2_protection_events"] = protection_events
                replacement = protected

        resolved.extend(replacement)
        audit_clusters.append(audit)

    if skipped_clusters:
        print(
            f"  Song kv_v2 ({phase}): skipped {skipped_clusters} small cluster(s)",
            flush=True,
        )

    # Final normalization
    final_matches, final_events, _ = _normalize_song_matches(segments, config, resolved)

    # Handle residual conflicts locally
    residual_clusters = _build_song_review_clusters(final_matches, set())
    if residual_clusters:
        residual_ids = {id(match) for cluster in residual_clusters for match in cluster}
        conflict_free = [match for match in final_matches if id(match) not in residual_ids]
        for cluster in residual_clusters:
            replacement, decisions = _local_best_song_cluster(
                segments, config, cluster, searched_titles,
            )
            conflict_free.extend(replacement)
            final_events.append({"type": "residual_conflict_local_best", "decisions": decisions})
        final_matches = sorted(conflict_free, key=lambda match: min(match.segment_indices))

    # Write summary
    summary = {
        "phase": phase,
        "input_count": len(matches),
        "normalized_count": len(normalized),
        "cluster_count": len(clusters),
        "skipped_clusters": skipped_clusters,
        "output_count": len(final_matches),
        "transcript_scope_requested": transcript_scope_requested,
        "transcript_scope_resolved": transcript_scope,
        "transcript_scope_reason": transcript_scope_reason,
        **scope_cost_details,
        "normalization_events": normalization_events,
        "final_normalization_events": final_events,
        "clusters": audit_clusters,
        "kv_v2_optimizations": {
            "deletion_threshold": KV_V3_DELETION_CONFIDENCE_THRESHOLD,
            "min_cluster_size_for_review": 2,
        },
    }
    (review_root / "summary.json").write_text(
        _json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return final_matches


def _build_survival_audit(
    stages: list[dict[str, Any]],
    llm_dir: Path,
    opening_events: list[dict[str, Any]],
) -> None:
    """Write survival audit to llm_dir/survival_audit.json."""
    audit = {
        "stages": stages,
        "total_stages": len(stages),
        "profile": PROFILE_KV_V2,
        "kv_v2_optimizations": {
            "deletion_threshold": KV_V3_DELETION_CONFIDENCE_THRESHOLD,
            "opening_humming_events": opening_events,
        },
    }
    (llm_dir / "survival_audit.json").write_text(
        _json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _stage_audit(
    stage: str,
    before_matches: list[ContentMatch],
    after_matches: list[ContentMatch],
) -> dict[str, Any]:
    """Build audit record for a pipeline stage."""
    before_keys = {_match_key(m): m for m in before_matches}
    after_keys = {_match_key(m): m for m in after_matches}
    added_keys = [k for k in after_keys if k not in before_keys]
    removed_keys = [k for k in before_keys if k not in after_keys]
    input_coverage = len({idx for m in before_matches for idx in m.segment_indices})
    output_coverage = len({idx for m in after_matches for idx in m.segment_indices})
    return {
        "stage": stage,
        "input_count": len(before_matches),
        "output_count": len(after_matches),
        "input_coverage_segments": input_coverage,
        "output_coverage_segments": output_coverage,
        "coverage_delta": output_coverage - input_coverage,
        "added_count": len(added_keys),
        "removed_count": len(removed_keys),
    }
