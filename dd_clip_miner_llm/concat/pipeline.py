"""Concat pipeline public API (backward-compatible re-exports)."""
from __future__ import annotations

from .models import (
    AttemptRecord,
    ConcatAttempt,
    ConcatContext,
    HealthInfo,
    ProblemProfile,
    TargetProfile,
    VideoMeta,
)
from .health import (
    _build_health_profile,
    _get_annexb_bsf,
    _has_video_bitstream_corruption,
    _safe_transmux_to_ts,
)
from .helpers import (
    _attempt,
    _attempt_concat_filter,
    _attempt_full_reencode,
    _attempt_tail_repair,
    _boundary_indexes_for_duration,
    _candidate_output_path,
    _cleanup_empty_dir,
    _commit_candidate_output,
    _concat_copy_with_list,
    _concat_demuxer_full_reencode,
    _corrupt_duration_ratio,
    _corrupt_duration_seconds,
    _duration_failure_boundary_indexes,
    _extract_failed_video_duration,
    _format_failures,
    _has_video_bitstream_failure,
    _is_output_duration_failure,
    _latest_failed_video_duration,
    _normalize_to_profile,
    _remux_concat_audio_reencode,
    _remux_concat_copy,
    _remux_inputs,
    _run_filter_command,
    _safe_attempt_name,
    _selective_normalize_concat,
    _target_size,
    _targeted_repair,
    _unique_path,
    _validate_output,
    _write_concat_list,
)
from .probe import probe_many, probe_one
from .runner import ConcatPipeline, concat_videos_smart
from .strategies import (
    DirectCopyStrategy,
    DiscardCorruptCopyStrategy,
    FullReencodeStrategy,
    MkvMergeStrategy,
    SelectiveNormalizeStrategy,
    Strategy,
    TargetedRepairStrategy,
)

__all__ = [
    "AttemptRecord",
    "ConcatAttempt",
    "ConcatContext",
    "ConcatPipeline",
    "HealthInfo",
    "ProblemProfile",
    "Strategy",
    "TargetProfile",
    "VideoMeta",
    "concat_videos_smart",
    "DirectCopyStrategy",
    "DiscardCorruptCopyStrategy",
    "TargetedRepairStrategy",
    "MkvMergeStrategy",
    "SelectiveNormalizeStrategy",
    "FullReencodeStrategy",
]
