"""KV v2 song pipeline.

Cache-optimized pipeline with initial_tool_choice="none" and content_validator.
Same post-processing as accuracy pipeline (review + recheck).
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
    """Run KV v2 pipeline: same as accuracy but with cache-optimized LLM strategy."""
    from .acc import run as run_acc
    return run_acc(segments, config, recognizer, llm_dir, matches=matches)
