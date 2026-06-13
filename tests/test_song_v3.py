from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from dd_clip_miner_llm.config import DEFAULT_CONFIG
from dd_clip_miner_llm.llm import build_llm_messages
from dd_clip_miner_llm.llm import LLMProvider
from dd_clip_miner_llm.models import TranscriptSegment
from dd_clip_miner_llm.song_postprocess.v3 import (
    _PrecisionDiscoveryRecognizer,
    _RecallAuditRecognizer,
    _SegmentationAdjudicationRecognizer,
    _V3StageRunner,
    _candidate_explosion,
    _continuation_for_discovery,
    _validate_discovery,
    _validate_adjudication,
)


def _config() -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    config["llm"]["cache_friendly_prompt_layout"] = True
    config["song"]["pipeline"]["strategy"] = "risk_routed_v3"
    return config


def _segments(count: int = 20) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(start=float(index * 3), end=float(index * 3 + 2), text=f"文本 {index}")
        for index in range(count)
    ]


def test_v3_stages_share_system_and_full_asr_prefix() -> None:
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
    prefixes = [item[1]["content"].split("ASR 转写结束。", 1)[0] for item in messages]
    assert prefixes[0] == prefixes[1] == prefixes[2]
    assert all(item.get_tools(config) is None for item in recognizers)


def test_recall_prompt_cannot_modify_precision_candidates() -> None:
    prompt = _RecallAuditRecognizer(
        [{"target_id": "U001", "segment_range": [10, 19]}]
    ).task_instructions(_config())
    assert "第一轮已经确定的歌曲不能修改" in prompt
    assert "只返回短 evidence_ranges" in prompt
    assert "不要推测整首歌边界" in prompt


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
    assert _validate_adjudication(valid, ["P001", "R001"], segments, config) == (True, None)

    missing = deepcopy(valid)
    missing["decisions"] = missing["decisions"][:1]
    assert _validate_adjudication(missing, ["P001", "R001"], segments, config)[0] is False

    duplicate = deepcopy(valid)
    duplicate["decisions"].append(deepcopy(duplicate["decisions"][0]))
    assert _validate_adjudication(duplicate, ["P001", "R001"], segments, config)[0] is False

    unknown = deepcopy(valid)
    unknown["decisions"][1]["candidate_ids"] = ["X001"]
    assert _validate_adjudication(unknown, ["P001", "R001"], segments, config)[0] is False


def test_v3_protocol_explosion_rejects_fragment_storm() -> None:
    config = _config()
    segments = _segments(3211)
    fragments = [
        {"segment_ranges": [[index, index]], "confidence": 0.5, "anchor_text": "啊"}
        for index in range(1398)
    ]
    assert _candidate_explosion(fragments, segments, config) is True


def test_v3_protocol_guard_allows_normal_candidate_set() -> None:
    config = _config()
    segments = _segments(3211)
    candidates = [
        {"segment_ranges": [[index * 100, index * 100 + 30]], "confidence": 0.8, "anchor_text": "歌词"}
        for index in range(20)
    ]
    assert _candidate_explosion(candidates, segments, config) is False


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
    valid, reason = _validate_adjudication(payload, [], segments, config)
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
    import dd_clip_miner_llm.song_postprocess.v3 as v3

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
    monkeypatch.setattr(v3, "build_providers", lambda _: [provider])
    monkeypatch.setattr(v3, "_build_openai_clients", lambda _: {"test": object()})

    def fake_call(*args, **kwargs):
        calls.append(args[2])
        return next(responses)

    monkeypatch.setattr(v3, "call_llm", fake_call)
    runner = _V3StageRunner(_segments(), config)
    payload, debug = runner.run(
        _PrecisionDiscoveryRecognizer(),
        tmp_path,
        validate=lambda value: _validate_discovery(value, 20),
        partial_field="candidates",
        continuation_instruction=lambda items: _continuation_for_discovery(items, 20, 50),
    )

    assert payload is not None
    assert len(payload["candidates"]) == 2
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0]
    assert calls[0][1]["content"].split("ASR 转写结束。", 1)[0] == calls[1][1]["content"].split("ASR 转写结束。", 1)[0]
    assert debug["finish_reason"] == "stop"
    assert debug["parse_valid"] is True
    assert len(debug["usage"]) == 2
