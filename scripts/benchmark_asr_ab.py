#!/usr/bin/env python3
"""Full-length ASR A/B: faster_whisper turbo+fallback vs Qwen3+fallback."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dd_clip_miner_llm.asr_fallback import transcribe_qwen3_with_fallback, transcribe_with_fallback
from dd_clip_miner_llm.config import load_config
from dd_clip_miner_llm.ffmpeg import get_duration
from dd_clip_miner_llm.models import TranscriptSegment


def detect_garbage_start(segments: list[TranscriptSegment], threshold: int = 10) -> float | None:
    if len(segments) < threshold:
        return None
    seg_dicts = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    for index in range(threshold, len(seg_dicts)):
        recent = seg_dicts[max(0, index - threshold) : index]
        texts = [item.get("text", "").strip() for item in recent]
        non_empty = [text for text in texts if text]
        if len(non_empty) >= threshold // 2 and len(set(non_empty)) <= 2:
            return float(seg_dicts[max(0, index - threshold)]["start"])
    return None


def count_gaps(segments: list[TranscriptSegment], total_duration: float, min_gap_seconds: float) -> int:
    previous_end = 0.0
    gap_count = 0
    for segment in sorted(segments, key=lambda item: item.start):
        start = max(0.0, float(segment.start))
        end = max(start, float(segment.end))
        if start - previous_end >= min_gap_seconds:
            gap_count += 1
        if end > previous_end:
            previous_end = end
    if total_duration - previous_end >= min_gap_seconds:
        gap_count += 1
    return gap_count


def analyze_segments(
    segments: list[TranscriptSegment],
    total_duration: float,
    *,
    min_gap_seconds: float = 4.0,
) -> dict:
    zero_duration = sum(1 for segment in segments if segment.end <= segment.start)
    max_segment = max(
        (max(0.0, float(segment.end) - float(segment.start)) for segment in segments),
        default=0.0,
    )
    covered_end = max((float(segment.end) for segment in segments), default=0.0)
    coverage = covered_end / total_duration if total_duration > 0 else 0.0
    return {
        "total_segments": len(segments),
        "zero_duration_segments": zero_duration,
        "max_segment_seconds": round(max_segment, 2),
        "coverage_ratio": round(coverage, 4),
        "covered_end_seconds": round(covered_end, 2),
        "gap_gt_seconds": min_gap_seconds,
        "gap_count": count_gaps(segments, total_duration, min_gap_seconds),
        "garbage_start": detect_garbage_start(segments),
    }


def run_side(
    name: str,
    audio_path: Path,
    asr_config: dict,
    work_dir: Path,
    runner,
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    asr_dir = work_dir / "02_asr"
    asr_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Running: {name}")
    print(f"{'=' * 70}")

    started = time.time()
    segments, meta = runner(audio_path, asr_config, asr_dir)
    elapsed = time.time() - started

    total_duration = get_duration(audio_path)
    stats = analyze_segments(segments, total_duration)
    transcript_path = work_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps([segment.to_dict() for segment in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = {
        "name": name,
        "elapsed_seconds": round(elapsed, 2),
        "total_duration_seconds": round(total_duration, 2),
        "meta": meta,
        **stats,
    }
    print(
        f"  done: {stats['total_segments']} segments in {elapsed:.1f}s | "
        f"coverage={stats['coverage_ratio']:.1%} | "
        f"zero_duration={stats['zero_duration_segments']} | "
        f"max_seg={stats['max_segment_seconds']:.1f}s | "
        f"gap>{stats['gap_gt_seconds']}s={stats['gap_count']}"
    )
    if stats["garbage_start"] is not None:
        print(f"  WARNING: garbage loop detected at {stats['garbage_start']:.1f}s")
    return result


def with_qwen3_max_workers(asr_config: dict, max_workers: int) -> dict:
    cfg = deepcopy(asr_config)
    local = cfg.setdefault("local", {})
    funasr = local.setdefault("funasr", {})
    if not isinstance(funasr, dict):
        funasr = {}
        local["funasr"] = funasr
    funasr["max_workers"] = int(max_workers)
    fallback = funasr.setdefault("fallback", {})
    if isinstance(fallback, dict):
        fallback["max_workers"] = int(max_workers)
    return cfg


def write_report(output_dir: Path, results: list[dict], *, merge_existing: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "report.json"
    if merge_existing and summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                by_name = {item["name"]: item for item in existing if isinstance(item, dict) and "name" in item}
                for item in results:
                    by_name[item["name"]] = item
                results = list(by_name.values())
        except Exception:
            pass
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# ASR A/B Report",
        "",
        "| Config | Time (s) | Segments | Coverage | Zero-dur | Max seg (s) | gap>4s | Garbage |",
        "|--------|----------|----------|----------|----------|-------------|--------|---------|",
    ]
    for item in results:
        garbage = "None" if item["garbage_start"] is None else f"{item['garbage_start']:.0f}s"
        lines.append(
            f"| {item['name']} | {item['elapsed_seconds']:.1f} | {item['total_segments']} | "
            f"{item['coverage_ratio']:.1%} | {item['zero_duration_segments']} | "
            f"{item['max_segment_seconds']:.1f} | {item['gap_count']} | {garbage} |"
        )
    lines.extend(["", f"Saved to `{summary_path}`."])
    report_md = output_dir / "report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {report_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR A/B benchmark")
    parser.add_argument("audio", type=Path, help="Path to source.wav")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("-o", "--output-dir", type=Path, default=Path(".tmp/qwen3_ab"))
    parser.add_argument(
        "--only",
        choices=["fw", "qwen3", "both"],
        default="both",
        help="Run only one side (default: both)",
    )
    parser.add_argument(
        "--qwen3-worker-ab",
        type=str,
        default="",
        help="Comma-separated max_workers values for Qwen3-only comparison, e.g. 1,2",
    )
    args = parser.parse_args()

    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        print(f"Error: audio not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(str(args.config))
    asr_config = config["asr"]
    results: list[dict] = []

    if args.qwen3_worker_ab:
        worker_values = [int(item.strip()) for item in args.qwen3_worker_ab.split(",") if item.strip()]
        if not worker_values:
            print("Error: --qwen3-worker-ab requires at least one value", file=sys.stderr)
            sys.exit(1)
        print(
            f"Qwen3 worker A/B: serial execution order = {worker_values} "
            "(one full run at a time)",
            flush=True,
        )
        for index, max_workers in enumerate(worker_values, start=1):
            print(
                f"\n>>> Serial step {index}/{len(worker_values)}: max_workers={max_workers}",
                flush=True,
            )
            side_config = with_qwen3_max_workers(asr_config, max_workers)
            results.append(run_side(
                f"qwen3_chunk180_fallback_w{max_workers}",
                audio_path,
                side_config,
                args.output_dir / f"qwen3_workers_{max_workers}",
                transcribe_qwen3_with_fallback,
            ))
        write_report(args.output_dir, results, merge_existing=len(worker_values) == 1)
        return

    if args.only in {"fw", "both"}:
        results.append(run_side(
            "fw_turbo_batch_fallback",
            audio_path,
            asr_config,
            args.output_dir / "fw_turbo_batch_fallback",
            transcribe_with_fallback,
        ))

    if args.only in {"qwen3", "both"}:
        results.append(run_side(
            "qwen3_chunk180_fallback",
            audio_path,
            asr_config,
            args.output_dir / "qwen3_chunk180_fallback",
            transcribe_qwen3_with_fallback,
        ))

    write_report(args.output_dir, results)


if __name__ == "__main__":
    main()