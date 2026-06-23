#!/usr/bin/env python3
"""ASR benchmark: compare turbo batch vs standard vs batch+fallback.

Usage:
    python scripts/benchmark_asr.py path/to/audio.wav
    python scripts/benchmark_asr.py path/to/audio.wav -o results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dd_clip_miner_llm.asr_backends import build_asr_backend


def detect_garbage_start(segments: list[dict], threshold: int = 10) -> float | None:
    """Detect when garbage/repeated text starts.

    Returns the start time of the first garbage segment, or None if no garbage found.
    Looks for a window of `threshold` consecutive segments where most texts are identical.
    """
    if len(segments) < threshold:
        return None

    for i in range(threshold, len(segments)):
        recent = segments[max(0, i - threshold) : i]
        texts = [s.get("text", "").strip() for s in recent]
        non_empty = [t for t in texts if t]
        if len(non_empty) >= threshold // 2 and len(set(non_empty)) <= 2:
            return segments[max(0, i - threshold)].get("start", 0)

    return None


def run_benchmark(audio_path: str, config: dict, name: str) -> dict:
    """Run ASR benchmark with given config."""
    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print(f"{'=' * 60}")

    start_time = time.time()

    # Build backend and run ASR
    backend = build_asr_backend(config)
    segments = backend.transcribe(audio_path)

    elapsed = time.time() - start_time

    # Convert to dicts for analysis
    seg_dicts = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    print(f"  [{name}] transcription complete: {len(seg_dicts)} segments in {elapsed:.1f}s")

    # Detect garbage
    garbage_start = detect_garbage_start(seg_dicts)

    # Calculate stats
    total_duration = seg_dicts[-1]["end"] if seg_dicts else 0
    throughput = len(seg_dicts) / elapsed if elapsed > 0 else 0

    result = {
        "name": name,
        "elapsed_seconds": round(elapsed, 2),
        "total_segments": len(seg_dicts),
        "total_duration": round(total_duration, 2),
        "garbage_start": garbage_start,
        "throughput": round(throughput, 2),
        "segments": seg_dicts,
    }

    print(f"\n  {name} Results:")
    print(f"    Time:       {elapsed:.1f}s")
    print(f"    Segments:   {len(seg_dicts)}")
    print(f"    Duration:   {total_duration:.1f}s")
    print(f"    Throughput: {throughput:.2f} seg/s")
    if garbage_start is not None:
        print(f"    WARNING: Garbage text detected at {garbage_start:.1f}s")
    else:
        print(f"    Garbage:    None")

    return result


def run_benchmark_with_fallback(
    audio_path: str,
    config_batch: dict,
    config_standard: dict,
) -> dict:
    """Run turbo batch, detect garbage, and fallback to standard if needed."""
    name = "turbo_batch_with_fallback"
    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print(f"{'=' * 60}")

    start_time = time.time()

    # First run batch
    print("  [1/2] Running turbo batch...")
    backend = build_asr_backend(config_batch)
    segments = backend.transcribe(audio_path)
    seg_dicts = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    print(f"  [1/2] Batch complete: {len(seg_dicts)} segments")

    garbage_start = detect_garbage_start(seg_dicts)

    if garbage_start is not None:
        print(f"  Garbage detected at {garbage_start:.1f}s, running standard fallback...")
        print("  [2/2] Running turbo standard fallback...")
        backend_std = build_asr_backend(config_standard)
        segments_std = backend_std.transcribe(audio_path)
        seg_dicts_std = [
            {"start": s.start, "end": s.end, "text": s.text} for s in segments_std
        ]
        print(f"  [2/2] Standard complete: {len(seg_dicts_std)} segments")

        # Merge: batch before garbage point, standard after
        merged = [s for s in seg_dicts if s["start"] < garbage_start]
        merged.extend(s for s in seg_dicts_std if s["start"] >= garbage_start)
        seg_dicts = merged
        print(f"  Merged result: {len(seg_dicts)} segments")
    else:
        print("  No garbage detected, using batch results directly")

    elapsed = time.time() - start_time
    total_duration = seg_dicts[-1]["end"] if seg_dicts else 0
    throughput = len(seg_dicts) / elapsed if elapsed > 0 else 0

    result = {
        "name": name,
        "elapsed_seconds": round(elapsed, 2),
        "total_segments": len(seg_dicts),
        "total_duration": round(total_duration, 2),
        "garbage_start": garbage_start,
        "throughput": round(throughput, 2),
        "segments": seg_dicts,
    }

    print(f"\n  {name} Results:")
    print(f"    Time:       {elapsed:.1f}s")
    print(f"    Segments:   {len(seg_dicts)}")
    print(f"    Duration:   {total_duration:.1f}s")
    print(f"    Throughput: {throughput:.2f} seg/s")
    if garbage_start is not None:
        print(f"    Garbage:    Detected at {garbage_start:.1f}s (fallback used)")
    else:
        print(f"    Garbage:    None")

    return result


def save_transcript(path: Path, segments: list[dict]) -> None:
    """Save transcript segments to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'=' * 80}")
    print("COMPARISON TABLE")
    print(f"{'=' * 80}")
    header = f"{'Config':<30} {'Time':>8} {'Segments':>10} {'Throughput':>12} {'Garbage':>12}"
    print(header)
    print(f"{'-' * 30} {'-' * 8} {'-' * 10} {'-' * 12} {'-' * 12}")
    for r in results:
        garbage = f"{r['garbage_start']:.0f}s" if r["garbage_start"] is not None else "None"
        print(
            f"{r['name']:<30} "
            f"{r['elapsed_seconds']:>8.1f} "
            f"{r['total_segments']:>10} "
            f"{r['throughput']:>12.2f} "
            f"{garbage:>12}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR Benchmark: compare turbo batch vs standard vs batch+fallback")
    parser.add_argument("audio", help="Path to audio file (WAV)")
    parser.add_argument(
        "-o", "--output", default="asr_benchmark.json", help="Output results JSON (default: asr_benchmark.json)"
    )
    args = parser.parse_args()

    audio_path = args.audio
    if not Path(audio_path).is_file():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)

    # Common ASR settings
    common = {
        "device": "auto",
        "compute_type": "default",
        "cpu_threads": 8,
        "num_workers": 1,
        "language": None,
        "beam_size": 5,
        "initial_prompt": None,
        "word_timestamps": True,
        "split_on_word_gaps": True,
        "word_gap_seconds": 2.0,
        "max_segment_seconds": 15.0,
        "vad_filter": False,
    }

    # Config 1: turbo batch (batched mode REQUIRES vad_filter=True)
    config_batch = {
        "backend": "faster_whisper",
        "model": "turbo",
        "inference_mode": "batched",
        "batch_size": 8,
        **common,
        "vad_filter": True,  # Required for batched inference
    }

    # Config 2: turbo standard
    config_standard = {
        "backend": "faster_whisper",
        "model": "turbo",
        "inference_mode": "standard",
        **common,
    }

    results: list[dict] = []

    # Benchmark 1: turbo batch
    results.append(run_benchmark(audio_path, config_batch, "turbo_batch"))

    # Benchmark 2: turbo standard
    results.append(run_benchmark(audio_path, config_standard, "turbo_standard"))

    # Benchmark 3: turbo batch with fallback to standard
    results.append(run_benchmark_with_fallback(audio_path, config_batch, config_standard))

    # Save per-config transcripts
    for r in results:
        transcript_path = output_path.parent / f"{output_path.stem}_{r['name']}.json"
        save_transcript(transcript_path, r["segments"])
        print(f"  Transcript saved: {transcript_path}")

    # Save combined results (without individual segments to keep it manageable)
    summary = [
        {k: v for k, v in r.items() if k != "segments"}
        for r in results
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print comparison
    print_comparison_table(results)


if __name__ == "__main__":
    main()
