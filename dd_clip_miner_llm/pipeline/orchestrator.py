from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from ..models import TranscriptSegment
from ..asr_backends import resolve_asr_model_name
from ..profile_state import (
    _format_usage_summary_console,
    _write_profile_comparison,
    _write_profile_state,
    _write_usage_summary,
)
from .utils import _print_summary, _save_progress


def _write_manifest_and_summary(
    out: Path,
    config: dict[str, Any],
    input_path: Path,
    total_duration: float,
    segments: list[TranscriptSegment],
    all_results: dict[str, list],
    llm_base_dir: Path,
    asr_dir: Path,
    profile_name: str,
    profile_enabled: bool,
    profile_state_path: Path,
    config_fingerprint: str,
    transcript_fingerprint: str,
    asr_inference_mode: str,
) -> None:
    """Write manifest, usage summary, and profile state."""
    _save_progress(out, input_path, "export")
    _print_summary(all_results)

    usage_summary = _write_usage_summary(llm_base_dir)
    usage_console = _format_usage_summary_console(usage_summary)
    if usage_console:
        print(usage_console)

    manifest = {
        "input_video": str(input_path),
        "profile": config.get("_profile_name"),
        "total_duration": total_duration,
        "segment_count": len(segments),
        "content_types": {ct: len(results) for ct, results in all_results.items()},
        "config": {
            "asr_model": resolve_asr_model_name(config.get("asr", {})),
            "asr_inference_mode": asr_inference_mode,
            "llm_model": config.get("llm", {}).get("model", "unknown"),
        },
        "llm_usage": usage_summary,
    }
    manifest_path = out / (f"manifest.{profile_name}.json" if profile_enabled else "manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if profile_enabled:
        _write_profile_state(
            profile_state_path,
            input_path=input_path,
            config=config,
            config_fingerprint=config_fingerprint,
            transcript_fingerprint=transcript_fingerprint,
            status="complete",
        )
        _write_profile_comparison(asr_dir / "llm")
    _save_progress(out, input_path, "done")
