"""Tests for kv_v3 song pipeline optimizations (now in kv_v2)."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from dd_clip_miner_llm.config import DEFAULT_CONFIG
from dd_clip_miner_llm.models import ContentMatch, TranscriptSegment
from dd_clip_miner_llm.recognizers.song.kv_v2.optimizations import (
    KV_V3_DELETION_CONFIDENCE_THRESHOLD,
    KV_V3_UNKNOWN_DELETION_CONFIDENCE_THRESHOLD,
    KV_V3_MIN_CLUSTER_SIZE_FOR_REVIEW,
    KV_V3_OPENING_WINDOW_SECONDS,
    _candidate_has_song_evidence,
    _detect_opening_humming,
    _is_humming_like_text,
    _preserve_high_confidence_known_title,
    _review_opening_segments,
    _should_skip_review,
    _has_repeated_vocal_pattern,
)


# ─── Helpers ────────────────────────────────────────────────────


def _segments(
    count: int = 20,
    *,
    start_offset: float = 0.0,
    step: float = 3.0,
    text: str = "歌词文本",
) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            start=start_offset + index * step,
            end=start_offset + index * step + 2.0,
            text=f"{text} {index}",
        )
        for index in range(count)
    ]


def _match(
    title: str,
    indices: list[int],
    *,
    confidence: float = 0.85,
    content_type: str = "song",
) -> ContentMatch:
    return ContentMatch(
        content_type=content_type,
        title=title,
        segment_indices=indices,
        confidence=confidence,
        tags=[],
        description="",
        artist="",
        lyrics_snippet="",
    )


def _config() -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    config["_profile_name"] = "kv_v2"
    config["llm"]["cache_friendly_prompt_layout"] = True
    config["llm"]["compact_segment_ranges"] = True
    config["song"]["review"]["enabled"] = True
    config["song"]["review"]["transcript_scope"] = "full"
    config["song"]["missed_recheck"]["enabled"] = True
    config["song"]["missed_recheck"]["strategy"] = "full_transcript"
    return config


# ─── Pipeline detection ────────────────────────────────────────


def test_kv_v3_pipeline_detection() -> None:
    """kv_v2 profile should be detected correctly (now uses kv_v3 pipeline)."""
    from dd_clip_miner_llm.recognizers.song import _detect_pipeline

    config = _config()
    assert _detect_pipeline(config) == "kv_v2"


def test_kv_v2_pipeline_detection_unchanged() -> None:
    """kv_v2 profile should still be detected correctly."""
    from dd_clip_miner_llm.recognizers.song import _detect_pipeline

    config = deepcopy(DEFAULT_CONFIG)
    config["_profile_name"] = "kv_v2"
    assert _detect_pipeline(config) == "kv_v2"


def test_acc_pipeline_detection_unchanged() -> None:
    """Default profile should still use acc pipeline."""
    from dd_clip_miner_llm.recognizers.song import _detect_pipeline

    config = deepcopy(DEFAULT_CONFIG)
    assert _detect_pipeline(config) == "acc"


# ─── High-confidence known title preservation ──────────────────


def test_preserve_high_confidence_known_title() -> None:
    """High-confidence known titles should be preserved."""
    segments = _segments(20)
    config = _config()

    # Existing matches (from main identify)
    existing = [_match("歌曲A", [0, 1, 2, 3, 4], confidence=0.9)]

    # Candidate that should be preserved (known title, high confidence)
    candidates = [_match("唯一", [5, 6, 7, 8, 9], confidence=0.85)]

    retained, events = _preserve_high_confidence_known_title(
        segments, config, existing, candidates,
    )

    assert len(retained) == 1
    assert retained[0].title == "唯一"
    assert any(e.get("action") == "retained" for e in events)


def test_preserve_high_confidence_known_title_rejects_low_confidence() -> None:
    """Known songs without ASR evidence should be rejected."""
    segments = _segments(20)
    config = _config()

    existing = [_match("歌曲A", [0, 1, 2, 3, 4], confidence=0.9)]
    # Known song with no evidence (empty segments)
    empty_segs = [TranscriptSegment(start=i*3, end=i*3+2, text="") for i in range(20)]
    candidates = [_match("测试歌曲", [5, 6, 7], confidence=0.6)]

    retained, events = _preserve_high_confidence_known_title(
        empty_segs, config, existing, candidates,
    )

    assert len(retained) == 0
    assert any(e.get("action") == "rejected" for e in events)


def test_preserve_high_confidence_known_title_skips_unknown() -> None:
    """Unknown songs with high confidence should use standard threshold."""
    segments = _segments(20)
    config = _config()

    existing = [_match("歌曲A", [0, 1, 2, 3, 4], confidence=0.9)]
    candidates = [_match("未知歌曲：歌词片段", [5, 6, 7, 8, 9], confidence=0.85)]

    retained, events = _preserve_high_confidence_known_title(
        segments, config, existing, candidates,
    )

    # Unknown songs should pass with 0.75 threshold
    assert len(retained) == 1
    assert retained[0].title == "未知歌曲：歌词片段"


def test_preserve_high_confidence_known_title_handles_overlap() -> None:
    """Overlapping candidates should be trimmed or rejected."""
    segments = _segments(20)
    config = _config()

    existing = [_match("歌曲A", [0, 1, 2, 3, 4], confidence=0.9)]
    candidates = [_match("唯一", [3, 4, 5, 6, 7], confidence=0.85)]

    retained, events = _preserve_high_confidence_known_title(
        segments, config, existing, candidates,
    )

    # Should be trimmed to uncovered segments
    assert len(retained) == 1
    assert retained[0].segment_indices == [5, 6, 7]
    assert any(e.get("action") == "retained_trimmed" for e in events)


# ─── Opening humming detection ─────────────────────────────────


def test_detect_opening_humming() -> None:
    """Opening humming should be detected in first 45 seconds."""
    # Create segments with humming in the first 45 seconds
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="啦啦啦"),
        TranscriptSegment(start=2.0, end=4.0, text="啦啦啦"),
        TranscriptSegment(start=4.0, end=6.0, text="嗯嗯嗯"),
        TranscriptSegment(start=6.0, end=8.0, text="啊啊啊"),
        TranscriptSegment(start=10.0, end=12.0, text="正常歌词"),
        TranscriptSegment(start=50.0, end=52.0, text="后面的内容"),
    ]
    matches = [_match("歌曲A", [4, 5], confidence=0.9)]

    events = _detect_opening_humming(segments, matches)

    assert len(events) > 0
    assert events[0]["type"] == "opening_humming_detected"
    assert events[0]["duration_seconds"] >= 5.0


def test_detect_opening_humming_no_humming() -> None:
    """No humming detected when opening has normal content."""
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="大家好欢迎来到直播间"),
        TranscriptSegment(start=2.0, end=4.0, text="今天我们来聊聊天"),
        TranscriptSegment(start=5.0, end=7.0, text="开始唱歌了"),
    ]
    matches = []

    events = _detect_opening_humming(segments, matches)

    assert len(events) == 0


def test_detect_opening_humming_already_covered() -> None:
    """No humming event for already covered segments."""
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="啦啦啦"),
        TranscriptSegment(start=2.0, end=4.0, text="啦啦啦"),
        TranscriptSegment(start=4.0, end=6.0, text="嗯嗯嗯"),
    ]
    matches = [_match("开场歌曲", [0, 1, 2], confidence=0.8)]

    events = _detect_opening_humming(segments, matches)

    assert len(events) == 0


# ─── Humming text detection ────────────────────────────────────


def test_is_humming_like_text_chinese() -> None:
    """Chinese humming patterns should be detected."""
    assert _is_humming_like_text("啦啦啦") is True
    assert _is_humming_like_text("嗯嗯嗯") is True
    assert _is_humming_like_text("啊啊啊") is True
    assert _is_humming_like_text("哼哼哼") is True


def test_is_humming_like_text_english() -> None:
    """English humming patterns should be detected."""
    assert _is_humming_like_text("lalala") is True
    assert _is_humming_like_text("nanana") is True
    assert _is_humming_like_text("yeah yeah") is True


def test_is_humming_like_text_not_humming() -> None:
    """Normal lyrics should not be detected as humming."""
    assert _is_humming_like_text("今天天气真好") is False
    assert _is_humming_like_text("我终于鼓起勇气") is False
    assert _is_humming_like_text("这是一句很长的歌词文本") is False


def test_has_repeated_vocal_pattern() -> None:
    """Repeated vocal patterns should be detected."""
    assert _has_repeated_vocal_pattern("啦啦啦") is True
    assert _has_repeated_vocal_pattern("lalala") is True
    assert _has_repeated_vocal_pattern("nanana") is True
    assert _has_repeated_vocal_pattern("正常文本") is False


# ─── Cluster-size review skip ──────────────────────────────────


def test_should_skip_review_single_match() -> None:
    """Single-match clusters should be skipped."""
    config = _config()
    cluster = [_match("歌曲A", [0, 1, 2])]

    assert _should_skip_review(cluster, config) is True


def test_should_skip_review_multiple_matches() -> None:
    """Multi-match clusters should not be skipped."""
    config = _config()
    cluster = [
        _match("歌曲A", [0, 1, 2]),
        _match("歌曲B", [1, 2, 3]),
    ]

    assert _should_skip_review(cluster, config) is False


def test_should_skip_review_custom_threshold() -> None:
    """Custom threshold should be respected."""
    config = _config()
    config["song"]["kv_v2"] = {"min_cluster_size_for_review": 3}

    cluster = [
        _match("歌曲A", [0, 1, 2]),
        _match("歌曲B", [1, 2, 3]),
    ]

    assert _should_skip_review(cluster, config) is True


# ─── Review opening segments ───────────────────────────────────


def test_review_opening_segments_adds_unknown() -> None:
    """Uncovered opening humming should be added as unknown song."""
    segments = _segments(10, start_offset=0.0, step=2.0)
    matches = [_match("歌曲A", [5, 6, 7, 8, 9], confidence=0.9)]

    opening_events = [{
        "type": "opening_humming_detected",
        "segment_ranges": [[0, 3]],
        "duration_seconds": 8.0,
    }]

    result = _review_opening_segments(segments, _config(), matches, opening_events)

    # Should have added an unknown song for opening
    assert len(result) == 2
    assert result[0].title == "未知歌曲：开场哼唱"
    assert result[0].segment_indices == [0, 1, 2, 3]


def test_review_opening_segments_already_covered() -> None:
    """Covered opening segments should not be modified."""
    segments = _segments(10, start_offset=0.0, step=2.0)
    matches = [_match("开场歌曲", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], confidence=0.9)]

    opening_events = [{
        "type": "opening_humming_detected",
        "segment_ranges": [[0, 3]],
        "duration_seconds": 8.0,
    }]

    result = _review_opening_segments(segments, _config(), matches, opening_events)

    # Should not change anything
    assert len(result) == 1
    assert result[0].title == "开场歌曲"


def test_review_opening_segments_no_events() -> None:
    """No events should return matches unchanged."""
    segments = _segments(10)
    matches = [_match("歌曲A", [0, 1, 2])]

    result = _review_opening_segments(segments, _config(), matches, [])

    assert result == matches


# ─── Candidate evidence check ──────────────────────────────────


def test_candidate_has_song_evidence_multiple_texts() -> None:
    """Multiple non-empty texts should be evidence."""
    segments = _segments(10)
    match = _match("歌曲A", [0, 1, 2])

    assert _candidate_has_song_evidence(segments, match) is True


def test_candidate_has_song_evidence_long_duration() -> None:
    """Long duration should be evidence even with few texts."""
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="歌词"),
        TranscriptSegment(start=2.0, end=15.0, text=""),
    ]
    match = _match("歌曲A", [0, 1])

    assert _candidate_has_song_evidence(segments, match) is True


def test_candidate_has_song_evidence_short_empty() -> None:
    """Short empty segments should not be evidence."""
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text=""),
        TranscriptSegment(start=2.0, end=4.0, text=""),
    ]
    match = _match("歌曲A", [0, 1])

    assert _candidate_has_song_evidence(segments, match) is False


# ─── Constants validation ──────────────────────────────────────


def test_deletion_threshold_higher_than_kv_v2() -> None:
    """kv_v3 deletion threshold should be higher than kv_v2's 0.70."""
    assert KV_V3_DELETION_CONFIDENCE_THRESHOLD > 0.70


def test_unknown_deletion_threshold_lower_than_known() -> None:
    """Unknown songs should have a lower deletion threshold than known songs."""
    assert KV_V3_UNKNOWN_DELETION_CONFIDENCE_THRESHOLD < KV_V3_DELETION_CONFIDENCE_THRESHOLD


def test_unknown_song_preserved_with_lower_threshold() -> None:
    """Unknown songs with confidence 0.60-0.65 should be preserved."""
    segments = _segments(20)
    config = _config()

    existing = [_match("歌曲A", [0, 1, 2, 3, 4], confidence=0.9)]
    # Unknown song with confidence 0.62 - below known threshold (0.75) but above unknown threshold (0.60)
    candidates = [_match("未知歌曲：测试歌词片段", [5, 6, 7, 8, 9], confidence=0.62)]

    retained, events = _preserve_high_confidence_known_title(
        segments, config, existing, candidates,
    )

    assert len(retained) == 1
    assert retained[0].title == "未知歌曲：测试歌词片段"
    assert any(e.get("action") == "retained" for e in events)


def test_known_song_filtered_at_same_threshold() -> None:
    """Known songs without ASR evidence should be filtered."""
    config = _config()

    existing = [_match("歌曲A", [0, 1, 2, 3, 4], confidence=0.9)]
    # Known song with no evidence: empty text AND short duration (< 10s)
    empty_segs = [TranscriptSegment(start=i*0.5, end=i*0.5+0.3, text="") for i in range(20)]
    candidates = [_match("测试歌曲", [5, 6], confidence=0.62)]

    retained, events = _preserve_high_confidence_known_title(
        empty_segs, config, existing, candidates,
    )

    assert len(retained) == 0
    assert any(e.get("action") == "rejected" and e.get("reason") == "weak_asr_evidence" for e in events)


def test_min_cluster_size_for_review() -> None:
    """Minimum cluster size should be at least 2."""
    assert KV_V3_MIN_CLUSTER_SIZE_FOR_REVIEW >= 2


def test_opening_window_seconds() -> None:
    """Opening window should be reasonable (30-60 seconds)."""
    assert 30.0 <= KV_V3_OPENING_WINDOW_SECONDS <= 60.0


# ─── Lyrics matching ───────────────────────────────────────────


def test_apply_lyrics_matching_disabled() -> None:
    """Lyrics matching should be skipped when search is disabled."""
    from dd_clip_miner_llm.song_postprocess.lyrics_match import _apply_lyrics_matching

    segments = _segments(10)
    config = _config()
    config["song"]["search"]["enabled"] = False
    matches = [_match("未知歌曲：歌词片段", [0, 1, 2], confidence=0.7)]

    updated, events = _apply_lyrics_matching(segments, config, matches, None)

    assert len(updated) == 1
    assert updated[0].title == "未知歌曲：歌词片段"
    assert len(events) == 0


def test_apply_lyrics_matching_known_song_unchanged() -> None:
    """Known songs should not be modified by lyrics matching."""
    from dd_clip_miner_llm.song_postprocess.lyrics_match import _apply_lyrics_matching

    segments = _segments(10)
    config = _config()
    config["song"]["search"]["enabled"] = True
    matches = [_match("晴天", [0, 1, 2], confidence=0.9)]

    updated, events = _apply_lyrics_matching(segments, config, matches, None)

    assert len(updated) == 1
    assert updated[0].title == "晴天"
    assert len(events) == 0


def test_extract_lyrics_snippet() -> None:
    """Lyrics snippet extraction should work correctly."""
    from dd_clip_miner_llm.song_postprocess.lyrics_match import _extract_lyrics_snippet

    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="我终于鼓起勇气"),
        TranscriptSegment(start=2.0, end=4.0, text="唱出这首歌"),
        TranscriptSegment(start=4.0, end=6.0, text="啦啦啦"),
    ]
    match = _match("未知歌曲", [0, 1, 2])

    snippet = _extract_lyrics_snippet(segments, match)

    assert len(snippet) > 0
    assert "勇气" in snippet or "歌" in snippet


def test_compute_lyrics_similarity() -> None:
    """Lyrics similarity computation should work."""
    from dd_clip_miner_llm.song_postprocess.lyrics_match import _compute_lyrics_similarity

    text_a = "我终于鼓起勇气唱出这首歌"
    text_b = "鼓起勇气唱出心中的歌"

    similarity = _compute_lyrics_similarity(text_a, text_b)

    assert 0.0 <= similarity <= 1.0
    assert similarity > 0.0  # Should have some overlap


def test_compute_lyrics_similarity_no_overlap() -> None:
    """Completely different texts should have zero similarity."""
    from dd_clip_miner_llm.song_postprocess.lyrics_match import _compute_lyrics_similarity

    text_a = "今天天气真好"
    text_b = "我终于鼓起勇气"

    similarity = _compute_lyrics_similarity(text_a, text_b)

    assert similarity == 0.0


# ─── Integration tests ─────────────────────────────────────────


def test_kv_v3_config_isolation() -> None:
    """kv_v2 config should not affect other profiles."""
    config_v2 = deepcopy(DEFAULT_CONFIG)
    config_v2["_profile_name"] = "kv_v2"

    config_acc = deepcopy(DEFAULT_CONFIG)
    config_acc["_profile_name"] = "accuracy"

    # accuracy should not have kv_v2 settings
    assert "kv_v2" not in config_acc.get("song", {})


def test_kv_v3_profile_in_cli_config() -> None:
    """kv_v2 profile should be in generated config."""
    from dd_clip_miner_llm.cli import _generate_config_yaml

    yaml_content = _generate_config_yaml()

    assert "kv_v2:" in yaml_content
    assert "min_cluster_size_for_review: 2" in yaml_content
    assert "deletion_confidence_threshold: 0.75" in yaml_content


def test_kv_v3_import_isolation() -> None:
    """kv_v2 package should import correctly (now contains kv_v3 optimizations)."""
    from dd_clip_miner_llm.recognizers.song.kv_v2 import run
    from dd_clip_miner_llm.recognizers.song.kv_v2.optimizations import (
        _detect_opening_humming,
        _preserve_high_confidence_known_title,
        _should_skip_review,
    )

    assert callable(run)
    assert callable(_detect_opening_humming)
    assert callable(_preserve_high_confidence_known_title)
    assert callable(_should_skip_review)
