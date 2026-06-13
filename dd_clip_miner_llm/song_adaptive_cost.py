"""Local cost estimation for adaptive song review and missed recheck."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import get_llm_config, get_song_recheck_config, get_song_review_config
from .models import ContentMatch, TranscriptSegment

DEFAULT_PRICING: dict[str, float] = {
    "input_cache_hit_per_1m": 0.0028,
    "input_cache_miss_per_1m": 0.14,
    "output_per_1m": 0.28,
}

DEFAULT_COMPLETION_TOKENS: dict[str, int] = {
    "main": 720,
    "review": 420,
    "overlong": 360,
    "missed_full": 1100,
    "missed_window": 360,
}

DEFAULT_TOKEN_CHARS_RATIO = 0.62


@dataclass(frozen=True)
class CostBreakdown:
    total_usd: float
    input_hit_tokens: int
    input_miss_tokens: int
    output_tokens: int
    call_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _adaptive_cost_settings(parent: dict[str, Any]) -> dict[str, Any]:
    raw = parent.get("adaptive", {})
    if not isinstance(raw, dict):
        raw = {}

    pricing_raw = raw.get("pricing", {})
    if not isinstance(pricing_raw, dict):
        pricing_raw = {}
    pricing = {
        key: float(pricing_raw.get(key, value))
        for key, value in DEFAULT_PRICING.items()
    }

    completion_raw = raw.get("estimated_completion_tokens", {})
    if not isinstance(completion_raw, dict):
        completion_raw = {}
    completion = {
        key: int(completion_raw.get(key, value))
        for key, value in DEFAULT_COMPLETION_TOKENS.items()
    }

    return {
        "token_chars_ratio": float(raw.get("token_chars_ratio", DEFAULT_TOKEN_CHARS_RATIO)),
        "pricing": pricing,
        "estimated_completion_tokens": completion,
        "fallback_penalty": float(raw.get("fallback_penalty", 1.0)),
        "full_transcript_max_segments": int(
            raw.get("full_transcript_max_segments", 3500)
        ),
        "windowed_min_target_ranges": int(
            raw.get("windowed_min_target_ranges", 19)
        ),
    }


def chars_to_tokens(chars: int, *, ratio: float) -> int:
    if chars <= 0:
        return 0
    return max(1, int(chars * ratio))


def message_total_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(message.get("content", ""))) for message in messages)


def split_kv_cache_parts(messages: list[dict[str, Any]]) -> tuple[int, int]:
    """Return (shared_prefix_chars, per_call_suffix_chars) for KV-friendly layout."""
    system_chars = 0
    user_content = ""
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role == "system":
            system_chars += len(content)
        elif role == "user" and not user_content:
            user_content = content

    marker_start = "ASR 转写开始：\n"
    marker_end = "\nASR 转写结束。\n\n"
    if marker_start in user_content and marker_end in user_content:
        _, rest = user_content.split(marker_start, 1)
        transcript, suffix = rest.split(marker_end, 1)
        return system_chars + len(transcript), len(suffix)

    return 0, system_chars + len(user_content)


def apply_pricing(
    *,
    input_hit_tokens: int,
    input_miss_tokens: int,
    output_tokens: int,
    pricing: dict[str, float],
) -> float:
    hit_cost = input_hit_tokens * pricing["input_cache_hit_per_1m"] / 1_000_000
    miss_cost = input_miss_tokens * pricing["input_cache_miss_per_1m"] / 1_000_000
    output_cost = output_tokens * pricing["output_per_1m"] / 1_000_000
    return hit_cost + miss_cost + output_cost


def _build_cost_breakdown(
    *,
    input_hit_tokens: int,
    input_miss_tokens: int,
    output_tokens: int,
    call_count: int,
    pricing: dict[str, float],
) -> CostBreakdown:
    return CostBreakdown(
        total_usd=apply_pricing(
            input_hit_tokens=input_hit_tokens,
            input_miss_tokens=input_miss_tokens,
            output_tokens=output_tokens,
            pricing=pricing,
        ),
        input_hit_tokens=input_hit_tokens,
        input_miss_tokens=input_miss_tokens,
        output_tokens=output_tokens,
        call_count=call_count,
    )


def _api_rounds(config_section: dict[str, Any], llm_config: dict[str, Any]) -> int:
    tool_rounds = int(
        config_section.get("max_tool_rounds", llm_config.get("max_tool_rounds", 0)) or 0
    )
    return max(1, 1 + tool_rounds)


def _default_recognizer(recognizer: Any | None) -> Any:
    if recognizer is not None:
        return recognizer
    from .recognizers.song import SongRecognizer

    return SongRecognizer()


def _pipeline_cost_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Pricing/token defaults shared across song pipeline phases."""
    review_config = get_song_review_config(config)
    return _adaptive_cost_settings(review_config)


def _main_batches(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
) -> list[tuple[int, list[TranscriptSegment]]]:
    batch_size_value = get_llm_config(config).get("batch_size")
    if batch_size_value in (None, "", 0, "0"):
        return [(0, segments)]
    batch_size = int(batch_size_value)
    return [
        (batch_start, segments[batch_start : batch_start + batch_size])
        for batch_start in range(0, len(segments), batch_size)
    ]


def estimate_main_cost(
    config: dict[str, Any],
    *,
    segments: list[TranscriptSegment],
    recognizer: Any | None = None,
) -> CostBreakdown:
    from .llm import build_llm_messages

    if not segments:
        return _build_cost_breakdown(
            input_hit_tokens=0,
            input_miss_tokens=0,
            output_tokens=0,
            call_count=0,
            pricing=_pipeline_cost_settings(config)["pricing"],
        )

    settings = _pipeline_cost_settings(config)
    ratio = settings["token_chars_ratio"]
    pricing = settings["pricing"]
    completion_tokens = settings["estimated_completion_tokens"]["main"]
    kv_layout = bool(get_llm_config(config).get("cache_friendly_prompt_layout", False))
    api_rounds = _api_rounds({}, get_llm_config(config))
    base_recognizer = _default_recognizer(recognizer)

    miss_tokens = 0
    hit_tokens = 0
    batch_count = 0
    for batch_start, batch_segments in _main_batches(segments, config):
        messages = build_llm_messages(
            base_recognizer,
            batch_segments,
            batch_start,
            config,
        )
        if kv_layout:
            shared_chars, suffix_chars = split_kv_cache_parts(messages)
            hit_tokens += chars_to_tokens(shared_chars, ratio=ratio)
            miss_tokens += chars_to_tokens(suffix_chars, ratio=ratio)
        else:
            miss_tokens += chars_to_tokens(message_total_chars(messages), ratio=ratio)
        batch_count += 1

    miss_tokens *= api_rounds
    hit_tokens *= api_rounds
    return _build_cost_breakdown(
        input_hit_tokens=hit_tokens,
        input_miss_tokens=miss_tokens,
        output_tokens=completion_tokens * batch_count * api_rounds,
        call_count=batch_count * api_rounds,
        pricing=pricing,
    )


def _iter_overlong_recheck_windows(
    segments: list[TranscriptSegment],
    matches: list[ContentMatch],
    config: dict[str, Any],
) -> list[tuple[int, int]]:
    from .config import get_llm_config, get_padding_config, get_song_recheck_config, get_song_review_config
    from .song_postprocess import (
        _segment_range_duration_seconds,
        _split_indices_by_time_gap_for_recheck,
    )

    if not segments or not matches:
        return []

    recheck_config = get_song_recheck_config(config)
    if recheck_config.get("enabled", True) is False:
        return []

    padding_config = get_padding_config(config, "song")
    max_song_seconds = float(padding_config.get("max_song_seconds", 360.0))
    merge_gap_seconds = float(padding_config.get("merge_gap_seconds", 40.0))
    if max_song_seconds <= 0:
        return []

    windows: list[tuple[int, int]] = []
    for match in matches:
        valid_indices = sorted(
            {index for index in match.segment_indices if 0 <= index < len(segments)}
        )
        groups = _split_indices_by_time_gap_for_recheck(
            segments,
            valid_indices,
            merge_gap_seconds,
        )
        for group in groups:
            group_start = min(group)
            group_end = max(group)
            if (
                _segment_range_duration_seconds(segments, group_start, group_end)
                > max_song_seconds
            ):
                windows.append((group_start, group_end))
    return windows


def estimate_overlong_cost(
    config: dict[str, Any],
    *,
    segments: list[TranscriptSegment],
    matches: list[ContentMatch],
    recognizer: Any | None = None,
) -> CostBreakdown:
    from .llm import build_llm_messages
    from .song_postprocess import _expand_segment_range

    settings = _pipeline_cost_settings(config)
    ratio = settings["token_chars_ratio"]
    pricing = settings["pricing"]
    completion_tokens = settings["estimated_completion_tokens"]["overlong"]
    recheck_config = get_song_recheck_config(config)
    context_segments = int(recheck_config.get("context_segments", 10) or 0)
    api_rounds = _api_rounds(recheck_config, get_llm_config(config))
    base_recognizer = _default_recognizer(recognizer)

    windows = _iter_overlong_recheck_windows(segments, matches, config)
    if not windows:
        return _build_cost_breakdown(
            input_hit_tokens=0,
            input_miss_tokens=0,
            output_tokens=0,
            call_count=0,
            pricing=pricing,
        )

    miss_tokens = 0
    for group_start, group_end in windows:
        context_start, context_end = _expand_segment_range(
            group_start,
            group_end,
            len(segments),
            context_segments,
        )
        chunk = segments[context_start : context_end + 1]
        offset_recognizer = _OffsetRecognizer(base_recognizer, context_start)
        messages = build_llm_messages(offset_recognizer, chunk, 0, config)
        miss_tokens += chars_to_tokens(message_total_chars(messages), ratio=ratio)

    miss_tokens *= api_rounds
    return _build_cost_breakdown(
        input_hit_tokens=0,
        input_miss_tokens=miss_tokens,
        output_tokens=completion_tokens * len(windows) * api_rounds,
        call_count=len(windows) * api_rounds,
        pricing=pricing,
    )


class _OffsetRecognizer:
    def __init__(self, recognizer: Any, offset: int) -> None:
        self._recognizer = recognizer
        self._offset = offset

    @property
    def name(self) -> str:
        return self._recognizer.name

    @property
    def default_config(self) -> dict[str, Any]:
        return getattr(self._recognizer, "default_config", {})

    def transcript_index_start(self, batch_start: int) -> int:
        return self._offset + batch_start

    def build_prompt(
        self,
        segments: list[TranscriptSegment],
        batch_start: int,
        config: dict[str, Any],
    ) -> str:
        return self._recognizer.build_prompt(
            segments,
            self._offset + batch_start,
            config,
        )

    def build_system_prompt(self, config: dict[str, Any]) -> str | None:
        return self._recognizer.build_system_prompt(config)

    def parse_response(
        self,
        items: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[ContentMatch]:
        return self._recognizer.parse_response(items, config)

    def get_tools(self, config: dict[str, Any]) -> Any:
        return self._recognizer.get_tools(config)


def _cluster_target_range(cluster: list[ContentMatch]) -> tuple[int, int]:
    target_start = min(min(match.segment_indices) for match in cluster)
    target_end = max(max(match.segment_indices) for match in cluster)
    return target_start, target_end


def _cluster_likely_preaudited(
    cluster: list[ContentMatch],
    target_ranges: list[tuple[int, int]],
) -> bool:
    if not target_ranges:
        return False
    target_start, target_end = _cluster_target_range(cluster)
    return any(
        target_start >= range_start and target_end <= range_end
        for range_start, range_end in target_ranges
    )


def estimate_review_after_cost(
    config: dict[str, Any],
    *,
    scope: str,
    missed_strategy: str,
    clusters: list[list[ContentMatch]],
    segments: list[TranscriptSegment],
    target_ranges: list[tuple[int, int]] | None = None,
    recognizer: Any | None = None,
) -> CostBreakdown:
    review_config = get_song_review_config(config)
    if review_config.get("enabled", False) is False:
        return estimate_review_cost(
            config,
            scope=scope,
            clusters=[],
            segments=segments,
            recognizer=recognizer,
        )

    review_clusters = clusters
    if missed_strategy == "full_transcript" and target_ranges:
        review_clusters = [
            cluster
            for cluster in clusters
            if not _cluster_likely_preaudited(cluster, target_ranges)
        ]
    return estimate_review_cost(
        config,
        scope=scope,
        clusters=review_clusters,
        segments=segments,
        recognizer=recognizer,
    )


def estimate_review_cost(
    config: dict[str, Any],
    *,
    scope: str,
    clusters: list[list[ContentMatch]],
    segments: list[TranscriptSegment],
    recognizer: Any | None = None,
) -> CostBreakdown:
    from .llm import build_llm_messages
    from .song_postprocess import (
        _SongFullReviewRecognizer,
        _SongReviewRecognizer,
        _expand_segment_range,
    )

    review_config = get_song_review_config(config)
    settings = _adaptive_cost_settings(review_config)
    ratio = settings["token_chars_ratio"]
    pricing = settings["pricing"]
    completion_tokens = settings["estimated_completion_tokens"]["review"]
    context_segments = int(review_config.get("context_segments", 10) or 0)
    max_window_segments = int(review_config.get("max_window_segments", 500) or 500)
    kv_layout = bool(get_llm_config(config).get("cache_friendly_prompt_layout", False))
    base_recognizer = _default_recognizer(recognizer)

    reviewable_clusters: list[tuple[list[ContentMatch], int, int, int, int]] = []
    for cluster in clusters:
        target_start = min(min(match.segment_indices) for match in cluster)
        target_end = max(max(match.segment_indices) for match in cluster)
        if target_end - target_start + 1 > max_window_segments:
            continue
        context_start, context_end = _expand_segment_range(
            target_start,
            target_end,
            len(segments),
            context_segments,
        )
        reviewable_clusters.append(
            (cluster, target_start, target_end, context_start, context_end)
        )

    if not reviewable_clusters:
        return _build_cost_breakdown(
            input_hit_tokens=0,
            input_miss_tokens=0,
            output_tokens=0,
            call_count=0,
            pricing=pricing,
        )

    miss_tokens = 0
    hit_tokens = 0
    api_rounds = _api_rounds(review_config, get_llm_config(config))

    for cluster, target_start, target_end, context_start, context_end in reviewable_clusters:
        if scope == "full":
            review_recognizer = _SongFullReviewRecognizer(
                base_recognizer,
                cluster,
                target_start=target_start,
                target_end=target_end,
                allowed_start=context_start,
                allowed_end=context_end,
            )
            messages = build_llm_messages(
                review_recognizer,
                segments,
                0,
                config,
            )
        else:
            review_recognizer = _SongReviewRecognizer(base_recognizer, cluster)
            chunk = segments[context_start : context_end + 1]
            messages = build_llm_messages(
                review_recognizer,
                chunk,
                0,
                config,
            )

        if kv_layout and scope == "full":
            shared_chars, suffix_chars = split_kv_cache_parts(messages)
            hit_tokens += chars_to_tokens(shared_chars, ratio=ratio)
            miss_tokens += chars_to_tokens(suffix_chars, ratio=ratio)
        else:
            miss_tokens += chars_to_tokens(message_total_chars(messages), ratio=ratio)

    miss_tokens *= api_rounds
    hit_tokens *= api_rounds
    output_tokens = completion_tokens * len(reviewable_clusters) * api_rounds
    return _build_cost_breakdown(
        input_hit_tokens=hit_tokens,
        input_miss_tokens=miss_tokens,
        output_tokens=output_tokens,
        call_count=len(reviewable_clusters) * api_rounds,
        pricing=pricing,
    )


def estimate_missed_cost(
    config: dict[str, Any],
    *,
    strategy: str,
    segments: list[TranscriptSegment],
    matches: list[ContentMatch],
    target_ranges: list[tuple[int, int]],
    recognizer: Any | None = None,
    include_fallback_penalty: bool = False,
) -> CostBreakdown:
    from .llm import build_llm_messages
    from .song_postprocess import (
        _OffsetRecognizer,
        _SongCoverageAuditRecognizer,
        _expand_segment_range,
        _group_segment_ranges,
    )

    recheck_config = get_song_recheck_config(config)
    settings = _adaptive_cost_settings(recheck_config)
    ratio = settings["token_chars_ratio"]
    pricing = settings["pricing"]
    completion = settings["estimated_completion_tokens"]
    context_segments = int(recheck_config.get("context_segments", 10) or 0)
    batch_size_value = recheck_config.get(
        "batch_size",
        get_llm_config(config).get("batch_size") or 500,
    )
    batch_size = int(batch_size_value or 500)
    base_recognizer = _default_recognizer(recognizer)

    if not target_ranges:
        return _build_cost_breakdown(
            input_hit_tokens=0,
            input_miss_tokens=0,
            output_tokens=0,
            call_count=0,
            pricing=pricing,
        )

    kv_layout = bool(get_llm_config(config).get("cache_friendly_prompt_layout", False))
    api_rounds = _api_rounds(recheck_config, get_llm_config(config))
    miss_tokens = 0
    hit_tokens = 0
    call_count = 0
    output_tokens = 0

    if strategy == "full_transcript":
        audit_recognizer = _SongCoverageAuditRecognizer(
            base_recognizer,
            target_ranges,
            matches,
        )
        messages = build_llm_messages(audit_recognizer, segments, 0, config)
        if kv_layout:
            shared_chars, suffix_chars = split_kv_cache_parts(messages)
            hit_tokens += chars_to_tokens(shared_chars, ratio=ratio)
            miss_tokens += chars_to_tokens(suffix_chars, ratio=ratio)
        else:
            miss_tokens += chars_to_tokens(message_total_chars(messages), ratio=ratio)
        call_count = api_rounds
        output_tokens = completion["missed_full"] * api_rounds

        breakdown = _build_cost_breakdown(
            input_hit_tokens=hit_tokens,
            input_miss_tokens=miss_tokens,
            output_tokens=output_tokens,
            call_count=call_count,
            pricing=pricing,
        )
        if include_fallback_penalty:
            risk = len(segments) > settings["full_transcript_max_segments"]
            if risk:
                windowed = estimate_missed_cost(
                    config,
                    strategy="windowed",
                    segments=segments,
                    matches=matches,
                    target_ranges=target_ranges,
                    recognizer=recognizer,
                    include_fallback_penalty=False,
                )
                penalty = settings["fallback_penalty"]
                return CostBreakdown(
                    total_usd=breakdown.total_usd + windowed.total_usd * penalty,
                    input_hit_tokens=breakdown.input_hit_tokens,
                    input_miss_tokens=breakdown.input_miss_tokens,
                    output_tokens=breakdown.output_tokens,
                    call_count=breakdown.call_count,
                )
        return breakdown

    range_groups = _group_segment_ranges(target_ranges, batch_size)
    for target_group in range_groups:
        start = target_group[0][0]
        end = target_group[-1][1]
        context_start, context_end = _expand_segment_range(
            start,
            end,
            len(segments),
            context_segments,
        )
        chunk = segments[context_start : context_end + 1]
        offset_recognizer = _OffsetRecognizer(base_recognizer, context_start)
        messages = build_llm_messages(offset_recognizer, chunk, 0, config)
        miss_tokens += chars_to_tokens(message_total_chars(messages), ratio=ratio)
        call_count += api_rounds

    output_tokens = completion["missed_window"] * call_count
    return _build_cost_breakdown(
        input_hit_tokens=hit_tokens,
        input_miss_tokens=miss_tokens,
        output_tokens=output_tokens,
        call_count=call_count,
        pricing=pricing,
    )
