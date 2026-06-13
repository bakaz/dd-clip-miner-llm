"""Song post-processing pipeline: normalization, review, temporal adjudication, risk scoring.

Submodules:
- normalize.py: same-title merge, chorus-aware split, general normalization
- review.py: LLM-based conflict resolution (local/full scope)
- recheck.py: missed segment recheck (windowed/full_transcript/anchor)
- temporal.py: temporal adjudication (full-transcript boundary correction)
- risk.py: risk scoring, boundary repair, anchor expansion
- pipeline.py: V2 risk-routed pipeline orchestration

All public symbols are re-exported here for backward compatibility.
"""
from .normalize import (
    _clone_match_with_indices,
    _content_match_from_dict,
    _expand_segment_range,
    _filter_matches_to_segment_range,
    _filter_matches_to_segment_ranges,
    _filter_short_segment_ranges,
    _group_segment_ranges,
    _indices_to_ranges,
    _is_invalid_audit_title,
    _match_groups_over_max_song_seconds,
    _match_key,
    _matches_overlap,
    _merge_adjacent_same_title_matches,
    _normalize_song_matches,
    _segment_range_duration_seconds,
    _split_indices_by_time_gap_for_recheck,
    _split_segment_ranges,
    _uncovered_segment_ranges,
)
from .review import (
    _OffsetRecognizer,
    _SongCoverageAuditRecognizer,
    _SongFullReviewRecognizer,
    _SongReviewRecognizer,
    _build_song_review_clusters,
    _batch_risk_review_clusters,
    _local_best_song_cluster,
    _review_loses_known_title,
    _review_song_matches,
    _sanitize_full_transcript_review_results,
)
from .recheck import (
    _active_debug_files,
    _cache_reuse_count,
    _finalize_windowed_missed_recheck_matches,
    _llm_debug_has_structural_issue,
    _llm_debug_structural_failures,
    _load_cached_identify_matches,
    _load_missed_recheck_audit,
    _load_searched_titles,
    _missed_recheck_fingerprint,
    _recheck_overlong_song_matches,
    _recheck_uncovered_song_segments,
    _review_debug_succeeded,
    _run_windowed_missed_recheck,
)
from .pipeline import (
    SongPipelineContext,
    SongPipelineStage,
    run_risk_routed_v2_pipeline,
)
from .risk import (
    SongRiskRecord,
    get_song_risk_config,
    expand_song_anchors,
    load_supported_search_titles,
    repair_song_boundaries,
    score_song_match_risks,
)
from ..config import is_risk_routed, is_risk_routed_v2, is_risk_routed_v3, song_pipeline_strategy
from .v3 import run_risk_routed_v3_pipeline
from .temporal import (
    _SongTemporalAdjudicationRecognizer,
    _restore_temporal_titles,
    _split_temporal_at_source_boundaries,
    run_temporal_adjudication,
)

__all__ = [
    # normalize
    "_clone_match_with_indices",
    "_content_match_from_dict",
    "_expand_segment_range",
    "_filter_matches_to_segment_range",
    "_filter_matches_to_segment_ranges",
    "_filter_short_segment_ranges",
    "_group_segment_ranges",
    "_indices_to_ranges",
    "_is_invalid_audit_title",
    "_match_groups_over_max_song_seconds",
    "_match_key",
    "_matches_overlap",
    "_merge_adjacent_same_title_matches",
    "_normalize_song_matches",
    "_segment_range_duration_seconds",
    "_split_indices_by_time_gap_for_recheck",
    "_split_segment_ranges",
    "_uncovered_segment_ranges",
    # review
    "_OffsetRecognizer",
    "_SongCoverageAuditRecognizer",
    "_SongFullReviewRecognizer",
    "_SongReviewRecognizer",
    "_build_song_review_clusters",
    "_batch_risk_review_clusters",
    "_local_best_song_cluster",
    "_review_loses_known_title",
    "_review_song_matches",
    "_sanitize_full_transcript_review_results",
    # recheck
    "_active_debug_files",
    "_cache_reuse_count",
    "_finalize_windowed_missed_recheck_matches",
    "_llm_debug_has_structural_issue",
    "_llm_debug_structural_failures",
    "_load_cached_identify_matches",
    "_load_missed_recheck_audit",
    "_load_searched_titles",
    "_missed_recheck_fingerprint",
    "_recheck_overlong_song_matches",
    "_recheck_uncovered_song_segments",
    "_review_debug_succeeded",
    "_run_windowed_missed_recheck",
    # V2 pipeline and risk routing
    "SongPipelineContext",
    "SongPipelineStage",
    "SongRiskRecord",
    "get_song_risk_config",
    "expand_song_anchors",
    "is_risk_routed_v2",
    "is_risk_routed_v3",
    "is_risk_routed",
    "load_supported_search_titles",
    "repair_song_boundaries",
    "run_risk_routed_v2_pipeline",
    "run_risk_routed_v3_pipeline",
    "score_song_match_risks",
    "song_pipeline_strategy",
    "_SongTemporalAdjudicationRecognizer",
    "_restore_temporal_titles",
    "_split_temporal_at_source_boundaries",
    "run_temporal_adjudication",
]
