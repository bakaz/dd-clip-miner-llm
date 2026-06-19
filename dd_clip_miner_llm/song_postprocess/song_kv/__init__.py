from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...models import ContentMatch, TranscriptSegment
from ..pipeline import BoundaryRiskStage, FinalAdjudicationStage, SearchVerificationStage, SongPipelineContext
from .orchestrator import (
    _apply_adjudication,
    _assign_discovery_ids,
    _build_recall_targets,
    _candidate_matches,
    _continuation_for_adjudication,
    _continuation_for_discovery,
    _continuation_for_recall,
    _sanitize_recall_anchors,
)
from .recognizers import (
    _PrecisionDiscoveryRecognizer,
    _RecallAuditRecognizer,
    _SegmentationAdjudicationRecognizer,
)
from .runner import _KVStageRunner
from .validation import _candidate_explosion, _validate_adjudication, _validate_discovery, _validate_recall


def run_risk_routed_kv_pipeline(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    llm_dir: Path,
) -> list[ContentMatch]:
    """Run the strict three-round KV song segmentation protocol."""
    kv_dir = llm_dir / "kv"
    kv_dir.mkdir(parents=True, exist_ok=True)
    runner = _KVStageRunner(segments, config)
    overlap = int(
        config.get("song", {}).get("pipeline", {}).get(
            "continuation_overlap_segments", 50
        ) or 50
    )
    total_duration_seconds = max((float(segment.end) for segment in segments), default=0.0)
    history: list[dict[str, Any]] = []

    discovery_recognizer = _PrecisionDiscoveryRecognizer()
    discovery_payload, discovery_debug = runner.run(
        discovery_recognizer,
        kv_dir / "discovery",
        validate=lambda value: _validate_discovery(
            value, len(segments), total_duration_seconds,
        ),
        partial_field="candidates",
        continuation_instruction=lambda items: _continuation_for_discovery(
            items, len(segments), overlap
        ),
    )
    if discovery_payload is None:
        audit = {
            "strategy": "risk_routed_kv",
            "status": "discovery_structural_failure",
            "stages": [{"stage": "precision_discovery", "status": "failed", "error": discovery_debug.get("error")}],
            "final_count": 0,
        }
        (kv_dir / "pipeline.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(
            "V3 precision discovery failed: "
            f"{discovery_debug.get('error') or 'invalid protocol'}"
        )

    discovery = _assign_discovery_ids(discovery_payload, len(segments))
    history.append({"stage": "precision_discovery", "status": "complete", "candidate_count": len(discovery)})
    (llm_dir / "initial_matches.json").write_text(
        json.dumps([match.to_dict() for match in _candidate_matches(discovery, "precision_discovery")], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    targets = _build_recall_targets(segments, discovery, config)
    recall_recognizer = _RecallAuditRecognizer(targets)
    recall_payload, recall_debug = runner.run(
        recall_recognizer,
        kv_dir / "recall_audit",
        validate=lambda value: _validate_recall(
            value, targets, len(segments), total_duration_seconds,
        ),
        partial_field="anchors",
        continuation_instruction=lambda items: _continuation_for_recall(items, targets),
    )
    recall_failed = recall_payload is None
    recall = _sanitize_recall_anchors(recall_payload or {"anchors": []}, targets, len(segments))
    history.append({
        "stage": "recall_audit",
        "status": "recall_incomplete" if recall_failed else "complete",
        "target_count": len(targets),
        "anchor_count": len(recall),
        "error": recall_debug.get("error") if recall_failed else None,
    })

    combined = [*discovery, *recall]
    candidate_ids = [item["candidate_id"] for item in combined]
    adjudication_recognizer = _SegmentationAdjudicationRecognizer(
        combined,
        bool(config.get("song", {}).get("pipeline", {}).get("allow_final_discovery", True)),
    )
    adjudication_payload, adjudication_debug = runner.run(
        adjudication_recognizer,
        kv_dir / "adjudication",
        validate=lambda value: _validate_adjudication(
            value, candidate_ids, segments, config
        ),
        partial_field="decisions",
        continuation_instruction=lambda items: _continuation_for_adjudication(items, candidate_ids),
    )
    if adjudication_payload is None:
        matches = [
            *_candidate_matches(discovery, "precision_discovery"),
            *_candidate_matches(recall, "unadjudicated_recall_anchor"),
        ]
        adjudication_status = "adjudication_incomplete"
    else:
        matches = _apply_adjudication(adjudication_payload, combined, segments, config)
        adjudication_status = "complete"
    history.append({
        "stage": "segmentation_adjudication",
        "status": adjudication_status,
        "input_count": len(combined),
        "output_count": len(matches),
        "error": adjudication_debug.get("error") if adjudication_payload is None else None,
    })

    context = SongPipelineContext(segments, config, recognizer, llm_dir, matches)
    BoundaryRiskStage("kv_final", "kv_adjudication").run(context)
    FinalAdjudicationStage().run(context)

    # 搜索验证命名（在所有分段和冲突裁决完成之后）
    from ...config import get_song_search_config
    if get_song_search_config(config).get("enabled", False):
        SearchVerificationStage().run(context)

    history.extend(context.stage_history)

    audit = {
        "strategy": "risk_routed_kv",
        "status": adjudication_status if not recall_failed else "recall_incomplete",
        "stages": history,
        "discovery_candidates": discovery,
        "recall_targets": targets,
        "recall_anchors": recall,
        "final_count": len(context.matches),
        "anchor_boundary_expansion": False,
        "search_enabled": get_song_search_config(config).get("enabled", False),
    }
    (kv_dir / "pipeline.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 写入 merge_events 供 sus 文件夹导出
    if context.merge_events:
        (llm_dir / "merge_events.json").write_text(
            json.dumps(context.merge_events, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return context.matches


__all__ = [
    "_PrecisionDiscoveryRecognizer",
    "_RecallAuditRecognizer",
    "_SegmentationAdjudicationRecognizer",
    "_candidate_explosion",
    "_validate_adjudication",
    "run_risk_routed_kv_pipeline",
]
