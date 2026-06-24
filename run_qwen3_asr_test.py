"""Qwen3-ASR local smoke test (requires GPU + funasr extra).

Usage:
    python run_qwen3_asr_test.py path/to/audio.wav
    python run_qwen3_asr_test.py path/to/audio.wav -o .tmp/qwen3_test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from dd_clip_miner_llm.asr_backends.funasr_backend import FunASRBackend

config = {
    "backend": "qwen3_asr",
    "funasr": {
        "model": "Qwen/Qwen3-ASR-1.7B",
        "hub": "hf",
        "device": "auto",
        "dtype": "bf16",
        "batch_size": None,
        "timestamp_chunk_seconds": 300,
        "max_workers": 2,
        "forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
        "forced_aligner_kwargs": {},
        "generate_kwargs": {"return_time_stamps": True},
        "lyrics": {
            "enabled": False,
            "max_line_chars": 24,
            "sentence_punctuation": "。！？.!?\n",
            "max_sentence_duration_ms": 5000
        }
    }
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-ASR local smoke test")
    parser.add_argument("audio", type=Path, help="Path to input WAV")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path(".tmp/qwen3_test"),
        help="Directory for transcript JSON output (default: .tmp/qwen3_test)",
    )
    args = parser.parse_args()

    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[test] Loading Qwen3-ASR model...")
    backend = FunASRBackend(config)

    print(f"[test] Transcribing: {audio_path}")
    segments = backend.transcribe(audio_path)

    print(f"\n{'=' * 60}")
    print(f"Total segments: {len(segments)}")
    print(f"{'=' * 60}\n")

    for i, seg in enumerate(segments[:30]):
        print(f"{i + 1:3d}. [{seg.start:8.2f}s - {seg.end:8.2f}s] {seg.text}")

    if len(segments) > 30:
        print(f"\n... ({len(segments) - 30} more segments)")
        print("\nLast 5 segments:")
        for i, seg in enumerate(segments[-5:]):
            idx = len(segments) - 5 + i + 1
            print(f"{idx:3d}. [{seg.start:8.2f}s - {seg.end:8.2f}s] {seg.text}")

    output_data = [
        {"start": seg.start, "end": seg.end, "text": seg.text}
        for seg in segments
    ]
    output_file = output_dir / "transcript_qwen3.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n[test] Output saved to: {output_file}")
    print("[test] DONE")


if __name__ == "__main__":
    main()
