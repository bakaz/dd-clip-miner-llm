"""KV song pipeline (risk_routed_kv).

Three-stage pipeline: Precision Discovery → Recall Audit → Segmentation Adjudication.
Uses KV cache optimization with cache_friendly_prompt_layout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ...models import ContentMatch, TranscriptSegment


def run(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    llm_dir: Path,
    *,
    matches: list[ContentMatch],
) -> list[ContentMatch]:
    """Run KV pipeline: three-stage risk-routed pipeline."""
    from ...song_postprocess.song_kv import run_risk_routed_kv_pipeline
    return run_risk_routed_kv_pipeline(segments, config, recognizer, llm_dir)
