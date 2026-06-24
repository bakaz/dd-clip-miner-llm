from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from .asr_backends import build_asr_backend, resolve_faster_whisper_mode_settings
from .asr_backends.funasr_backend import repair_qwen3_zero_duration_segments
from .ffmpeg import cut_audio, get_duration
from .models import TranscriptSegment

_QWEN3_BACKENDS = {"qwen3", "qwen3_asr"}


def faster_whisper_fallback_config(asr_config: dict[str, Any]) -> dict[str, Any]:
    local_cfg = asr_config.get("local", {}) if str(asr_config.get("mode", "")).lower() == "local" else asr_config
    if not isinstance(local_cfg, dict):
        return {}
    backend = str(local_cfg.get("backend", "faster_whisper")).lower().replace("-", "_")
    if backend not in {"faster_whisper", "whisper"}:
        return {}
    fw = local_cfg.get("faster_whisper", {})
    return fw.get("fallback", {}) if isinstance(fw, dict) and isinstance(fw.get("fallback"), dict) else {}


def is_faster_whisper_fallback_enabled(asr_config: dict[str, Any]) -> bool:
    fallback = faster_whisper_fallback_config(asr_config)
    return bool(fallback) and bool(fallback.get("enabled", True))


def detect_transcript_gaps(
    segments: list[TranscriptSegment],
    total_duration: float,
    min_gap_seconds: float,
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    previous_end = 0.0
    previous_index = -1
    for index, segment in enumerate(sorted(segments, key=lambda item: item.start)):
        start = max(0.0, float(segment.start))
        end = max(start, float(segment.end))
        if start - previous_end >= min_gap_seconds:
            gaps.append({
                "index": len(gaps),
                "start": previous_end,
                "end": start,
                "duration": start - previous_end,
                "after_segment_index": previous_index,
                "before_segment_index": index,
            })
        if end > previous_end:
            previous_end = end
            previous_index = index
    if total_duration - previous_end >= min_gap_seconds:
        gaps.append({
            "index": len(gaps),
            "start": previous_end,
            "end": total_duration,
            "duration": total_duration - previous_end,
            "after_segment_index": previous_index,
            "before_segment_index": None,
        })
    return gaps


def fallback_audio_ranges(
    gaps: list[dict[str, Any]],
    total_duration: float,
    padding_seconds: float,
) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    for gap in gaps:
        start = max(0.0, float(gap["start"]) - padding_seconds)
        end = min(total_duration, float(gap["end"]) + padding_seconds)
        if end <= start:
            continue
        ranges.append({
            **gap,
            "padded_start": start,
            "padded_end": end,
            "padded_duration": end - start,
        })
    return ranges


def merge_fill_gaps(
    primary_segments: list[TranscriptSegment],
    fallback_items: list[dict[str, Any]],
) -> list[TranscriptSegment]:
    additions: list[TranscriptSegment] = []
    for item in fallback_items:
        gap_start = float(item["start"])
        gap_end = float(item["end"])
        for segment_data in item.get("segments", []):
            start = float(segment_data["start"])
            end = float(segment_data["end"])
            midpoint = (start + end) / 2.0
            if not (gap_start <= midpoint <= gap_end):
                continue
            clipped_start = max(gap_start, start)
            clipped_end = min(gap_end, end)
            text = str(segment_data.get("text", "")).strip()
            if text and clipped_end > clipped_start:
                additions.append(TranscriptSegment(clipped_start, clipped_end, text))
    return sorted([*primary_segments, *additions], key=lambda segment: (segment.start, segment.end))


def transcribe_with_fallback(
    source_wav: Path,
    asr_config: dict[str, Any],
    asr_dir: Path,
    cleanup_fallback_audio: bool = True,
) -> tuple[list[TranscriptSegment], dict[str, Any]]:
    fallback = faster_whisper_fallback_config(asr_config)
    primary_mode = str(fallback.get("primary_mode") or "batch")
    fallback_mode = str(fallback.get("fallback_mode") or "standard")
    min_gap_seconds = float(fallback.get("min_gap_seconds", 4.0))
    padding_seconds = float(fallback.get("padding_seconds", 2.0))
    max_workers = max(1, int(fallback.get("max_workers", 1)))
    merge_policy = str(fallback.get("merge_policy") or "fill_gaps")
    if merge_policy != "fill_gaps":
        raise ValueError(f"Unsupported ASR fallback merge_policy: {merge_policy}")

    primary_settings = resolve_faster_whisper_mode_settings(asr_config, primary_mode)
    primary_backend = build_asr_backend(primary_settings)
    primary_segments = primary_backend.transcribe(source_wav)
    total_duration = get_duration(source_wav)
    gaps = detect_transcript_gaps(primary_segments, total_duration, min_gap_seconds)
    ranges = fallback_audio_ranges(gaps, total_duration, padding_seconds)

    asr_dir.mkdir(parents=True, exist_ok=True)
    (asr_dir / "transcript_primary.json").write_text(
        json.dumps([segment.to_dict() for segment in primary_segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (asr_dir / "fallback_ranges.json").write_text(
        json.dumps(ranges, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fallback_segments = _run_fallback_ranges(
        source_wav,
        asr_config,
        fallback_mode,
        ranges,
        asr_dir / "fallback_audio",
        max_workers,
    )
    (asr_dir / "fallback_segments.json").write_text(
        json.dumps(fallback_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Cleanup fallback audio files (they're no longer needed after merging)
    if cleanup_fallback_audio:
        fallback_audio_dir = asr_dir / "fallback_audio"
        if fallback_audio_dir.exists():
            try:
                shutil.rmtree(fallback_audio_dir)
                print(f"  Cleaned up fallback audio: {fallback_audio_dir}")
            except Exception as exc:
                print(f"  Warning: failed to cleanup fallback audio: {exc}")

    merged = merge_fill_gaps(primary_segments, fallback_segments)
    metadata = {
        "primary_mode": primary_mode,
        "fallback_mode": fallback_mode,
        "min_gap_seconds": min_gap_seconds,
        "padding_seconds": padding_seconds,
        "max_workers": max_workers,
        "merge_policy": merge_policy,
        "primary_segment_count": len(primary_segments),
        "fallback_range_count": len(ranges),
        "fallback_segment_count": sum(len(item.get("segments", [])) for item in fallback_segments),
        "merged_segment_count": len(merged),
    }
    return merged, metadata


def _run_fallback_ranges(
    source_wav: Path,
    asr_config: dict[str, Any],
    fallback_mode: str,
    ranges: list[dict[str, Any]],
    audio_dir: Path,
    max_workers: int,
) -> list[dict[str, Any]]:
    audio_dir.mkdir(parents=True, exist_ok=True)
    if max_workers == 1:
        backend = build_asr_backend(resolve_faster_whisper_mode_settings(asr_config, fallback_mode))
        return [
            _run_one_fallback_range(source_wav, backend, audio_dir, item)
            for item in ranges
        ]

    def run_parallel(item: dict[str, Any]) -> dict[str, Any]:
        backend = build_asr_backend(resolve_faster_whisper_mode_settings(asr_config, fallback_mode))
        return _run_one_fallback_range(source_wav, backend, audio_dir, item)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_parallel, item) for item in ranges]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: int(item["index"]))


def _run_one_fallback_range(
    source_wav: Path,
    backend: Any,
    audio_dir: Path,
    item: dict[str, Any],
) -> dict[str, Any]:
    index = int(item["index"])
    padded_start = float(item["padded_start"])
    padded_end = float(item["padded_end"])
    audio_path = audio_dir / f"range_{index:04d}_{int(padded_start * 1000)}_{int(padded_end * 1000)}.wav"
    if not audio_path.exists():
        cut_audio(source_wav, audio_path, padded_start, padded_end)
    local_segments = backend.transcribe(audio_path)
    shifted_segments = [
        TranscriptSegment(
            start=padded_start + segment.start,
            end=padded_start + segment.end,
            text=segment.text,
        ).to_dict()
        for segment in local_segments
    ]
    return {
        **item,
        "audio_path": str(audio_path),
        "segments": shifted_segments,
    }


def _local_asr_config(asr_config: dict[str, Any]) -> dict[str, Any]:
    if str(asr_config.get("mode", "")).lower() == "local":
        local_cfg = asr_config.get("local", {})
        return local_cfg if isinstance(local_cfg, dict) else {}
    return asr_config if isinstance(asr_config, dict) else {}


def qwen3_fallback_config(asr_config: dict[str, Any]) -> dict[str, Any]:
    local_cfg = _local_asr_config(asr_config)
    backend = str(local_cfg.get("backend", "")).lower().replace("-", "_")
    if backend not in _QWEN3_BACKENDS:
        return {}
    funasr = local_cfg.get("funasr", {})
    if not isinstance(funasr, dict):
        return {}
    fallback = funasr.get("fallback", {})
    return fallback if isinstance(fallback, dict) else {}


def is_qwen3_fallback_enabled(asr_config: dict[str, Any]) -> bool:
    fallback = qwen3_fallback_config(asr_config)
    return bool(fallback) and bool(fallback.get("enabled", False))


def detect_qwen3_fallback_ranges(
    segments: list[TranscriptSegment],
    fallback_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    max_segment_seconds = float(fallback_cfg.get("max_segment_seconds", 15.0))
    sparse_chars_per_sec = float(fallback_cfg.get("sparse_chars_per_sec", 1.0))
    repeat_threshold = max(2, int(fallback_cfg.get("repeat_threshold", 3)))

    sorted_segments = sorted(segments, key=lambda item: (item.start, item.end))
    reasons_by_index: dict[int, set[str]] = {}

    for index, segment in enumerate(sorted_segments):
        duration = max(0.0, float(segment.end) - float(segment.start))
        text = segment.text.strip()
        if duration > max_segment_seconds and text and len(text) / duration < sparse_chars_per_sec:
            reasons_by_index.setdefault(index, set()).add("sparse_long_segment")

    index = 0
    while index < len(sorted_segments):
        text = sorted_segments[index].text.strip()
        if not text:
            index += 1
            continue
        run_start = index
        run_length = 1
        while (
            index + run_length < len(sorted_segments)
            and sorted_segments[index + run_length].text.strip() == text
        ):
            run_length += 1
        if run_length >= repeat_threshold:
            for run_index in range(run_start, run_start + run_length):
                reasons_by_index.setdefault(run_index, set()).add("repeated_segment")
        index += run_length if run_length > 1 else 1

    flagged_indices = sorted(reasons_by_index)
    if not flagged_indices:
        return []

    ranges: list[dict[str, Any]] = []
    group_indices = [flagged_indices[0]]
    for index in flagged_indices[1:]:
        if index == group_indices[-1] + 1:
            group_indices.append(index)
            continue
        selected = [sorted_segments[i] for i in group_indices]
        ranges.append({
            "index": len(ranges),
            "start": min(float(item.start) for item in selected),
            "end": max(float(item.end) for item in selected),
            "duration": max(float(item.end) for item in selected) - min(float(item.start) for item in selected),
            "reasons": sorted({
                reason
                for group_index in group_indices
                for reason in reasons_by_index[group_index]
            }),
            "segment_indices": list(group_indices),
        })
        group_indices = [index]

    selected = [sorted_segments[i] for i in group_indices]
    ranges.append({
        "index": len(ranges),
        "start": min(float(item.start) for item in selected),
        "end": max(float(item.end) for item in selected),
        "duration": max(float(item.end) for item in selected) - min(float(item.start) for item in selected),
        "reasons": sorted({
            reason
            for group_index in group_indices
            for reason in reasons_by_index[group_index]
        }),
        "segment_indices": list(group_indices),
    })
    return ranges


def pad_suspicious_ranges(
    ranges: list[dict[str, Any]],
    total_duration: float,
    padding_seconds: float,
) -> list[dict[str, Any]]:
    padded: list[dict[str, Any]] = []
    for item in ranges:
        start = max(0.0, float(item["start"]) - padding_seconds)
        end = min(total_duration, float(item["end"]) + padding_seconds)
        if end <= start:
            continue
        padded.append({
            **item,
            "padded_start": start,
            "padded_end": end,
            "padded_duration": end - start,
        })
    return padded


def merge_replace_ranges(
    primary_segments: list[TranscriptSegment],
    fallback_items: list[dict[str, Any]],
) -> list[TranscriptSegment]:
    replace_ranges = [
        (float(item["padded_start"]), float(item["padded_end"]))
        for item in fallback_items
    ]

    kept: list[TranscriptSegment] = []
    for segment in primary_segments:
        midpoint = (float(segment.start) + float(segment.end)) / 2.0
        if any(start <= midpoint <= end for start, end in replace_ranges):
            continue
        kept.append(segment)

    additions: list[TranscriptSegment] = []
    for item in fallback_items:
        for segment_data in item.get("segments", []):
            start = float(segment_data["start"])
            end = float(segment_data["end"])
            text = str(segment_data.get("text", "")).strip()
            if text and end > start:
                additions.append(TranscriptSegment(start, end, text))

    return sorted([*kept, *additions], key=lambda segment: (segment.start, segment.end))


def _build_qwen3_local_config(
    asr_config: dict[str, Any],
    *,
    chunk_seconds: int | None = None,
) -> dict[str, Any]:
    local_cfg = deepcopy(_local_asr_config(asr_config))
    funasr_cfg = deepcopy(local_cfg.get("funasr", {}))
    if not isinstance(funasr_cfg, dict):
        funasr_cfg = {}
    if chunk_seconds is not None:
        funasr_cfg["timestamp_chunk_seconds"] = int(chunk_seconds)
    local_cfg["funasr"] = funasr_cfg
    return local_cfg


def _run_qwen3_fallback_ranges(
    source_wav: Path,
    asr_config: dict[str, Any],
    ranges: list[dict[str, Any]],
    audio_dir: Path,
    max_workers: int,
    *,
    chunk_seconds: int,
    asr_dir: Path,
) -> list[dict[str, Any]]:
    audio_dir.mkdir(parents=True, exist_ok=True)
    local_cfg = _build_qwen3_local_config(asr_config, chunk_seconds=chunk_seconds)
    backend = build_asr_backend(
        {"mode": "local", "local": local_cfg},
        runtime_context={"asr_dir": asr_dir},
    )

    def run_one(item: dict[str, Any]) -> dict[str, Any]:
        result = _run_one_fallback_range(source_wav, backend, audio_dir, item)
        repaired_segments = [
            segment.to_dict()
            for segment in repair_qwen3_zero_duration_segments([
                TranscriptSegment(
                    float(segment_data["start"]),
                    float(segment_data["end"]),
                    str(segment_data.get("text", "")),
                )
                for segment_data in result.get("segments", [])
            ])[0]
        ]
        result["segments"] = repaired_segments
        return result

    if max_workers == 1:
        return [run_one(item) for item in ranges]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, item) for item in ranges]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: int(item["index"]))


def transcribe_qwen3_with_fallback(
    source_wav: Path,
    asr_config: dict[str, Any],
    asr_dir: Path,
    cleanup_fallback_audio: bool = True,
) -> tuple[list[TranscriptSegment], dict[str, Any]]:
    fallback_cfg = qwen3_fallback_config(asr_config)
    chunk_seconds = int(fallback_cfg.get("chunk_seconds", 5))
    padding_seconds = float(fallback_cfg.get("padding_seconds", 2.0))
    max_workers = max(1, int(fallback_cfg.get("max_workers", 1)))
    merge_policy = str(fallback_cfg.get("merge_policy") or "replace_ranges")
    if merge_policy != "replace_ranges":
        raise ValueError(f"Unsupported Qwen3 fallback merge_policy: {merge_policy}")

    primary_backend = build_asr_backend(asr_config, runtime_context={"asr_dir": asr_dir})
    primary_segments = primary_backend.transcribe(source_wav)
    primary_segments, repair_stats = repair_qwen3_zero_duration_segments(primary_segments)

    total_duration = get_duration(source_wav)
    suspicious = detect_qwen3_fallback_ranges(primary_segments, fallback_cfg)
    ranges = pad_suspicious_ranges(suspicious, total_duration, padding_seconds)

    asr_dir.mkdir(parents=True, exist_ok=True)
    (asr_dir / "transcript_primary.json").write_text(
        json.dumps([segment.to_dict() for segment in primary_segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (asr_dir / "fallback_ranges.json").write_text(
        json.dumps(ranges, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not ranges:
        metadata = {
            "backend": "qwen3_asr",
            "chunk_seconds": chunk_seconds,
            "padding_seconds": padding_seconds,
            "max_workers": max_workers,
            "merge_policy": merge_policy,
            "primary_segment_count": len(primary_segments),
            "fallback_range_count": 0,
            "fallback_segment_count": 0,
            "merged_segment_count": len(primary_segments),
            "zero_duration_repair": repair_stats,
        }
        return primary_segments, metadata

    fallback_segments = _run_qwen3_fallback_ranges(
        source_wav,
        asr_config,
        ranges,
        asr_dir / "fallback_audio",
        max_workers,
        chunk_seconds=chunk_seconds,
        asr_dir=asr_dir,
    )
    (asr_dir / "fallback_segments.json").write_text(
        json.dumps(fallback_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if cleanup_fallback_audio:
        fallback_audio_dir = asr_dir / "fallback_audio"
        if fallback_audio_dir.exists():
            try:
                shutil.rmtree(fallback_audio_dir)
                print(f"  Cleaned up fallback audio: {fallback_audio_dir}")
            except Exception as exc:
                print(f"  Warning: failed to cleanup fallback audio: {exc}")

    merged = merge_replace_ranges(primary_segments, fallback_segments)
    metadata = {
        "backend": "qwen3_asr",
        "chunk_seconds": chunk_seconds,
        "padding_seconds": padding_seconds,
        "max_workers": max_workers,
        "merge_policy": merge_policy,
        "primary_segment_count": len(primary_segments),
        "fallback_range_count": len(ranges),
        "fallback_segment_count": sum(len(item.get("segments", [])) for item in fallback_segments),
        "merged_segment_count": len(merged),
        "zero_duration_repair": repair_stats,
    }
    return merged, metadata
