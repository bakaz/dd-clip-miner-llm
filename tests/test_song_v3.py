from __future__ import annotations

import re
from copy import deepcopy
from types import SimpleNamespace

from dd_clip_miner_llm.config import DEFAULT_CONFIG
from dd_clip_miner_llm.llm import LLMProvider, batch_debug_is_reusable, build_llm_messages
from dd_clip_miner_llm.models import TranscriptSegment
from dd_clip_miner_llm.recognizers.song import SongRecognizer
import dd_clip_miner_llm.song_postprocess.song_kv.runner as kv_runner
from dd_clip_miner_llm.song_postprocess.song_kv import (
    _PrecisionDiscoveryRecognizer,
    _RecallAuditRecognizer,
    _SegmentationAdjudicationRecognizer,
    _KVStageRunner,
    _candidate_explosion,
    _continuation_for_discovery,
    _validate_adjudication,
    _validate_discovery,
)


def _config() -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    config["llm"]["cache_friendly_prompt_layout"] = True
    config["song"]["pipeline"]["strategy"] = "risk_routed_kv"
    return config


def _segments(
    count: int = 20,
    *,
    start_offset: float = 0.0,
    step: float = 3.0,
) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            start=start_offset + index * step,
            end=start_offset + index * step + 2.0,
            text=f"文本 {index}",
        )
        for index in range(count)
    ]


def _long_timeline_segments(count: int = 20) -> list[TranscriptSegment]:
    return _segments(count, step=500.0)


def _total_duration_seconds(segments: list[TranscriptSegment]) -> float:
    return max((float(segment.end) for segment in segments), default=0.0)


def test_song_prompts_use_index_only_transcript_lines() -> None:
    config = _config()
    segments = _segments()
    recognizers = [
        SongRecognizer(),
        _PrecisionDiscoveryRecognizer(),
        _RecallAuditRecognizer([{"target_id": "U001", "segment_range": [0, 9]}]),
        _SegmentationAdjudicationRecognizer(
            [{
                "candidate_id": "P001",
                "segment_ranges": [[2, 8]],
                "confidence": 0.8,
                "anchor_text": "歌词",
            }],
            True,
        ),
    ]

    for recognizer in recognizers:
        messages = build_llm_messages(recognizer, segments, 0, config)
        content = messages[-1]["content"]
        assert "[0] 文本 0" in content
        assert re.search(r"\(\d+\.\ds-\d+\.\ds\)", content) is None


def test_kv_stages_share_system_and_full_asr_prefix() -> None:
    config = _config()
    segments = _segments()
    recognizers = [
        _PrecisionDiscoveryRecognizer(),
        _RecallAuditRecognizer([{"target_id": "U001", "segment_range": [0, 9]}]),
        _SegmentationAdjudicationRecognizer(
            [{
                "candidate_id": "P001",
                "segment_ranges": [[2, 8]],
                "confidence": 0.8,
                "anchor_text": "歌词",
            }],
            True,
        ),
    ]
    messages = [build_llm_messages(item, segments, 0, config) for item in recognizers]
    assert all(item[0] == messages[0][0] for item in messages)
    prefixes = [item[1]["content"].split("ASR 转写结束。\n\n", 1)[0] for item in messages]
    assert prefixes[0] == prefixes[1] == prefixes[2]
    assert all(item.get_tools(config) is None for item in recognizers)
    discovery_task = messages[0][1]["content"].split("ASR 转写结束。\n\n", 1)[1]
    assert "最后一个 segment index" in discovery_task
    assert "complete_through_segment" in discovery_task
    assert "只返回一个 JSON object" in discovery_task
    assert "JSON 数组" not in discovery_task


def test_coordinate_validator_detects_mixed_mode() -> None:
    segments = _long_timeline_segments()
    payload = {
        "candidates": [
            {"segment_ranges": [[2, 5]], "confidence": 0.8, "anchor_text": "前半段"},
            {"segment_ranges": [[8658, 8757]], "confidence": 0.7, "anchor_text": "后半段"},
        ],
        "scan_complete": True,
        "complete_through_segment": len(segments) - 1,
    }

    valid, reason, diagnostics = _validate_discovery(
        payload, len(segments), _total_duration_seconds(segments),
    )
    assert valid is False
    assert reason == "mixed_coordinate_mode"
    assert diagnostics["coordinate_mode"] == "mixed_coordinate_mode"
    assert diagnostics["seconds_like_range_count"] == 1
    assert diagnostics["example_seconds_like_ranges"] == [[8658, 8757]]


def test_coordinate_validator_detects_seconds_drift() -> None:
    segments = _long_timeline_segments()
    payload = {
        "candidates": [
            {"segment_ranges": [[8658, 8757]], "confidence": 0.7, "anchor_text": "后半段"},
        ],
        "scan_complete": True,
        "complete_through_segment": len(segments) - 1,
    }

    valid, reason, diagnostics = _validate_discovery(
        payload, len(segments), _total_duration_seconds(segments),
    )
    assert valid is False
    assert reason == "seconds_coordinate_drift"
    assert diagnostics["coordinate_mode"] == "seconds_coordinate_drift"
    assert diagnostics["seconds_like_range_count"] == 1


def test_batch_cache_rejects_mixed_coordinate_debug() -> None:
    metadata = {"request_fingerprint": "same"}
    payload = {
        **metadata,
        "error": None,
        "parse_valid": True,
        "json_fix_rounds": [],
        "reasoning_followups": [],
        "tool_rounds": [],
        "protocol_valid": False,
        "coordinate_mode": "mixed_coordinate_mode",
    }
    assert batch_debug_is_reusable(payload, expected_metadata=metadata) is False


def test_adjudication_requires_exactly_once_id_coverage() -> None:
    config = _config()
    valid = {
        "decisions": [
            {
                "candidate_ids": ["P001"],
                "action": "accept",
                "segment_ranges": [[1, 5]],
                "confidence": 0.8,
            },
            {
                "candidate_ids": ["R001"],
                "action": "reject",
                "segment_ranges": [],
                "confidence": 0.6,
            },
        ],
        "additions": [],
        "adjudication_complete": True,
    }
    segments = _segments()
    assert _validate_adjudication(valid, ["P001", "R001"], segments, config)[:2] == (True, None)

    missing = deepcopy(valid)
    missing["decisions"] = missing["decisions"][:1]
    assert _validate_adjudication(missing, ["P001", "R001"], segments, config)[0] is False

    duplicate = deepcopy(valid)
    duplicate["decisions"].append(deepcopy(duplicate["decisions"][0]))
    assert _validate_adjudication(duplicate, ["P001", "R001"], segments, config)[0] is False

    unknown = deepcopy(valid)
    unknown["decisions"][1]["candidate_ids"] = ["X001"]
    assert _validate_adjudication(unknown, ["P001", "R001"], segments, config)[0] is False


def test_kv_protocol_explosion_rejects_fragment_storm() -> None:
    config = _config()
    segments = _segments(3211)
    fragments = [
        {"segment_ranges": [[index, index]], "confidence": 0.5, "anchor_text": "啊"}
        for index in range(1398)
    ]
    assert _candidate_explosion(fragments, segments, config) is True


def test_kv_protocol_guard_allows_normal_candidate_set() -> None:
    config = _config()
    segments = _segments(3211)
    candidates = [
        {"segment_ranges": [[index * 100, index * 100 + 30]], "confidence": 0.8, "anchor_text": "歌词"}
        for index in range(20)
    ]
    assert _candidate_explosion(candidates, segments, config) is False


def test_discovery_requires_exact_last_segment() -> None:
    segments = _segments()
    payload = {
        "candidates": [],
        "scan_complete": True,
        "complete_through_segment": 18,
    }
    assert _validate_discovery(
        payload, 20, _total_duration_seconds(segments),
    )[:2] == (False, "discovery_incomplete_coverage")
    payload["complete_through_segment"] = 19
    assert _validate_discovery(
        payload, 20, _total_duration_seconds(segments),
    )[:2] == (True, None)


def test_final_discovery_requires_explicit_evidence() -> None:
    config = _config()
    segments = _segments()
    payload = {
        "decisions": [],
        "additions": [{
            "segment_ranges": [[2, 8]],
            "evidence_ranges": [[2, 2]],
            "confidence": 0.8,
            "anchor_text": "普通聊天",
            "final_discovery": True,
        }],
        "adjudication_complete": True,
    }
    valid, reason, _ = _validate_adjudication(payload, [], segments, config)
    assert valid is False
    assert reason == "adjudication_addition_without_evidence"


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content

    def model_dump(self) -> dict:
        return {"content": self.content}


class _FakeUsage:
    def model_dump(self) -> dict:
        return {
            "prompt_cache_hit_tokens": 100,
            "prompt_cache_miss_tokens": 10,
            "completion_tokens": 20,
        }


def _response(content: str, finish_reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=_FakeMessage(content),
            finish_reason=finish_reason,
        )],
        usage=_FakeUsage(),
        model="fake",
    )


def test_discovery_continuation_merges_complete_candidates(monkeypatch, tmp_path) -> None:
    config = _config()
    provider = LLMProvider(api_key="test", model="fake", max_completion_tokens=32768)
    responses = iter([
        _response(
            '{"candidates":[{"segment_ranges":[[2,5]],"confidence":0.8,'
            '"anchor_text":"第一段"},',
            "length",
        ),
        _response(
            '{"candidates":[{"segment_ranges":[[10,15]],"confidence":0.9,'
            '"anchor_text":"第二段"}],"scan_complete":true,'
            '"complete_through_segment":19}',
            "stop",
        ),
    ])
    calls = []
    monkeypatch.setattr(kv_runner, "build_providers", lambda _: [provider])
    monkeypatch.setattr(kv_runner, "_build_openai_clients", lambda _: {"test": object()})

    def fake_call(*args, **kwargs):
        calls.append(args[2])
        return next(responses)

    monkeypatch.setattr(kv_runner, "call_llm", fake_call)
    runner = _KVStageRunner(_segments(), config)
    payload, debug = runner.run(
        _PrecisionDiscoveryRecognizer(),
        tmp_path,
        validate=lambda value: _validate_discovery(value, 20, _total_duration_seconds(_segments())),
        partial_field="candidates",
        continuation_instruction=lambda items: _continuation_for_discovery(items, 20, 50),
    )

    assert payload is not None
    assert len(payload["candidates"]) == 2
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0]
    assert calls[0][1]["content"].split("ASR 转写结束。\n\n", 1)[0] == calls[1][1]["content"].split("ASR 转写结束。\n\n", 1)[0]
    assert debug["finish_reason"] == "stop"
    assert debug["parse_valid"] is True
    assert len(debug["usage"]) == 2


def test_discovery_stop_with_incomplete_coverage_continues_remaining_range(monkeypatch, tmp_path) -> None:
    config = _config()
    provider = LLMProvider(api_key="test", model="fake", max_completion_tokens=32768)
    responses = iter([
        _response(
            '{"candidates":[{"segment_ranges":[[2,5]],"confidence":0.8,'
            '"anchor_text":"歌曲"}],"scan_complete":true,'
            '"complete_through_segment":18}',
            "stop",
        ),
        _response(
            '{"candidates":[],"scan_complete":true,'
            '"complete_through_segment":19}',
            "stop",
        ),
    ])
    calls = []
    monkeypatch.setattr(kv_runner, "build_providers", lambda _: [provider])
    monkeypatch.setattr(kv_runner, "_build_openai_clients", lambda _: {"test": object()})

    def fake_call(*args, **kwargs):
        calls.append(args[2])
        return next(responses)

    monkeypatch.setattr(kv_runner, "call_llm", fake_call)
    payload, debug = _KVStageRunner(_segments(), config).run(
        _PrecisionDiscoveryRecognizer(),
        tmp_path,
        validate=lambda value: _validate_discovery(value, 20, _total_duration_seconds(_segments())),
        partial_field="candidates",
        continuation_instruction=lambda items: _continuation_for_discovery(items, 20, 50),
    )

    assert payload is not None
    assert payload["complete_through_segment"] == 19
    assert len(payload["candidates"]) == 1
    assert len(calls) == 2
    assert "[19,19]" in calls[1][-1]["content"]
    assert debug["continuation_rounds"][0]["reason"] == "discovery_incomplete_coverage"


def test_discovery_continues_previous_stop_coverage_failure(monkeypatch, tmp_path) -> None:
    config = _config()
    config["llm"]["continuation_on_length"] = False
    provider = LLMProvider(api_key="test", model="fake", max_completion_tokens=32768)
    first_response = _response(
        '{"candidates":[],"scan_complete":true,"complete_through_segment":18}',
        "stop",
    )
    calls = 0
    monkeypatch.setattr(kv_runner, "build_providers", lambda _: [provider])
    monkeypatch.setattr(kv_runner, "_build_openai_clients", lambda _: {"test": object()})

    def first_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return first_response

    monkeypatch.setattr(kv_runner, "call_llm", first_call)
    runner = _KVStageRunner(_segments(), config)
    failed, _ = runner.run(
        _PrecisionDiscoveryRecognizer(),
        tmp_path,
        validate=lambda value: _validate_discovery(value, 20, _total_duration_seconds(_segments())),
        partial_field="candidates",
        continuation_instruction=lambda items: _continuation_for_discovery(items, 20, 50),
    )
    assert failed is None
    assert calls == 1

    continuation_calls = []

    def continuation_call(*args, **kwargs):
        continuation_calls.append(args[2])
        return _response(
            '{"candidates":[],"scan_complete":true,'
            '"complete_through_segment":19}',
            "stop",
        )

    monkeypatch.setattr(kv_runner, "call_llm", continuation_call)
    relaxed_config = _config()
    relaxed_runner = _KVStageRunner(_segments(), relaxed_config)
    recovered, debug = relaxed_runner.run(
        _PrecisionDiscoveryRecognizer(),
        tmp_path,
        validate=lambda value: _validate_discovery(value, 20, _total_duration_seconds(_segments())),
        partial_field="candidates",
        continuation_instruction=lambda items: _continuation_for_discovery(items, 20, 50),
    )
    assert recovered is not None
    assert recovered["complete_through_segment"] == 19
    assert debug["continued_from_cached_incomplete_response"] is True
    assert len(continuation_calls) == 1
    assert "[19,19]" in continuation_calls[0][-1]["content"]
    assert len(debug["usage"]) == 2


def test_kv_runner_retries_provider_then_falls_back_on_coordinate_drift(monkeypatch, tmp_path) -> None:
    config = _config()
    segments = _long_timeline_segments()
    provider_a = LLMProvider(
        name="a",
        api_key="a",
        model="provider-a",
        max_completion_tokens=32768,
        result_retries=1,
        retry_backoff_seconds=[0],
        retry_jitter_ratio=0.0,
    )
    provider_b = LLMProvider(
        name="b",
        api_key="b",
        model="provider-b",
        max_completion_tokens=32768,
        result_retries=0,
        retry_backoff_seconds=[0],
        retry_jitter_ratio=0.0,
    )
    responses = {
        "provider-a": iter([
            _response(
                '{"candidates":[{"segment_ranges":[[2,5]],"confidence":0.8,"anchor_text":"前半段"},'
                '{"segment_ranges":[[8658,8757]],"confidence":0.7,"anchor_text":"后半段"}],'
                '"scan_complete":true,"complete_through_segment":19}',
                "stop",
            ),
            _response(
                '{"candidates":[{"segment_ranges":[[3,6]],"confidence":0.8,"anchor_text":"还是坏的"},'
                '{"segment_ranges":[[8700,8757]],"confidence":0.7,"anchor_text":"还是秒数"}],'
                '"scan_complete":true,"complete_through_segment":19}',
                "stop",
            ),
        ]),
        "provider-b": iter([
            _response(
                '{"candidates":[{"segment_ranges":[[1,4]],"confidence":0.91,"anchor_text":"前半段歌词"},'
                '{"segment_ranges":[[10,13]],"confidence":0.88,"anchor_text":"后半段副歌"}],'
                '"scan_complete":true,"complete_through_segment":19}',
                "stop",
            ),
        ]),
    }
    calls: list[str] = []
    monkeypatch.setattr(kv_runner, "build_providers", lambda _: [provider_a, provider_b])
    monkeypatch.setattr(kv_runner, "_build_openai_clients", lambda _: {
        "a": object(),
        "b": object(),
    })

    def fake_call(_client, provider, *_args, **_kwargs):
        calls.append(provider.model)
        return next(responses[provider.model])

    monkeypatch.setattr(kv_runner, "call_llm", fake_call)
    payload, debug = _KVStageRunner(segments, config).run(
        _PrecisionDiscoveryRecognizer(),
        tmp_path,
        validate=lambda value: _validate_discovery(
            value, len(segments), _total_duration_seconds(segments),
        ),
        partial_field="candidates",
        continuation_instruction=lambda items: _continuation_for_discovery(items, len(segments), 50),
    )

    assert payload is not None
    assert calls == ["provider-a", "provider-a", "provider-b"]
    assert [item["segment_ranges"] for item in payload["candidates"]] == [
        [[1, 4]],
        [[10, 13]],
    ]
    assert debug["provider"]["model"] == "provider-b"
    assert debug["provider_attempts"][0]["error"] == "mixed_coordinate_mode"


def test_kv_runner_does_not_result_retry_transport_exception(monkeypatch, tmp_path) -> None:
    config = _config()
    segments = _long_timeline_segments()
    provider_a = LLMProvider(
        name="a",
        api_key="a",
        model="provider-a",
        max_completion_tokens=32768,
        result_retries=2,
        retry_backoff_seconds=[0],
        retry_jitter_ratio=0.0,
    )
    provider_b = LLMProvider(
        name="b",
        api_key="b",
        model="provider-b",
        max_completion_tokens=32768,
        result_retries=0,
        retry_backoff_seconds=[0],
        retry_jitter_ratio=0.0,
    )
    calls: list[str] = []
    monkeypatch.setattr(kv_runner, "build_providers", lambda _: [provider_a, provider_b])
    monkeypatch.setattr(kv_runner, "_build_openai_clients", lambda _: {
        "a": object(),
        "b": object(),
    })

    def fake_call(_client, provider, *_args, **_kwargs):
        calls.append(provider.model)
        if provider.model == "provider-a":
            raise TimeoutError("stuck provider")
        return _response(
            '{"candidates":[{"segment_ranges":[[1,4]],"confidence":0.91,"anchor_text":"前半段歌词"}],'
            '"scan_complete":true,"complete_through_segment":19}',
            "stop",
        )

    monkeypatch.setattr(kv_runner, "call_llm", fake_call)
    payload, debug = _KVStageRunner(segments, config).run(
        _PrecisionDiscoveryRecognizer(),
        tmp_path,
        validate=lambda value: _validate_discovery(
            value, len(segments), _total_duration_seconds(segments),
        ),
        partial_field="candidates",
        continuation_instruction=lambda items: _continuation_for_discovery(items, len(segments), 50),
    )

    assert payload is not None
    assert calls == ["provider-a", "provider-b"]
    assert debug["provider"]["model"] == "provider-b"
