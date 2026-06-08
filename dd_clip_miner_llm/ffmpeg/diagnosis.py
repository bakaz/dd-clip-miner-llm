from __future__ import annotations

import re
from os import devnull
from pathlib import Path

import subprocess

from ..concat.models import ProblemProfile
from .bitstream import text_indicates_bitstream_corruption, looks_like_video_bitstream_error
from .compat import pkg_attr

def analyze_ffmpeg_failure(details: str | list[str] | None) -> dict[str, object]:
    """Legacy dict version kept for compatibility. See classify_ffmpeg_output for the new structured ProblemProfile."""
    if not details:
        return {"bitstream_corruption": False, "timestamp_discontinuity": False, "duration_truncated": False, "audio_decode_fail": False, "hw_unavailable": False, "demux_error": False, "summary": "no details"}
    if isinstance(details, list):
        text = "\n".join(details)
    else:
        text = str(details)
    t = text.lower()

    bitstream = text_indicates_bitstream_corruption(text)
    ts_issue = bool(re.search(r"non.?monotonic|invalid dts|negative ts|timestamp|pts.*(invalid|discontinu)", t))
    dur_trunc = (
        "concat output duration is too short" in t
        or "concat output duration is too long" in t
        or "concat output video stream duration is too short" in t
        or "concat output video stream duration is too long" in t
    )
    audio_fail = (
        bool(re.search(r"audio.*(not decodable|fail|error|invalid|corrupt)", t))
        or bool(re.search(r"aac.*(decode|error|invalid|corrupt)", t))
        or "audio is not decodable" in t
    )
    hw_fail = bool(re.search(r"(unknown encoder|encoder not found|device.*not found|nvenc|qsv|amf).*(fail|error|unavailable|not)", t))
    demux = bool(re.search(r"(error during demuxing|error opening input|moov atom not found|partial file|invalid data found when processing input|demux)", t))

    problems = []
    if bitstream:
        problems.append("bitstream_corruption")
    if ts_issue:
        problems.append("timestamp_discontinuity")
    if dur_trunc:
        problems.append("duration_truncated")
    if audio_fail:
        problems.append("audio_decode_fail")
    if hw_fail:
        problems.append("hw_unavailable")
    if demux:
        problems.append("demux_error")

    summary = ", ".join(problems) if problems else "unknown/other"
    return {
        "bitstream_corruption": bitstream,
        "timestamp_discontinuity": ts_issue,
        "duration_truncated": dur_trunc,
        "audio_decode_fail": audio_fail,
        "hw_unavailable": hw_fail,
        "demux_error": demux,
        "summary": summary,
        "raw_snippet": text[:800] if len(text) > 800 else text,
    }


def classify_ffmpeg_output(details: str | list[str] | None) -> ProblemProfile:
    """New structured classifier. Returns ProblemProfile (the heart of output-driven decisions).
    Uses the same patterns as before but produces the dataclass used by the new pipeline.
    """
    if not details:
        return ProblemProfile(summary="no details")
    if isinstance(details, list):
        text = "\n".join(details)
    else:
        text = str(details)
    t = text.lower()

    bitstream = text_indicates_bitstream_corruption(text)
    ts_issue = bool(re.search(r"non.?monotonic|invalid dts|negative ts|timestamp|pts.*(invalid|discontinu)", t))
    dur_trunc = (
        "concat output duration is too short" in t
        or "concat output duration is too long" in t
        or "concat output video stream duration is too short" in t
        or "concat output video stream duration is too long" in t
    )
    audio_fail = (
        bool(re.search(r"audio.*(not decodable|fail|error|invalid|corrupt)", t))
        or bool(re.search(r"aac.*(decode|error|invalid|corrupt)", t))
        or "audio is not decodable" in t
    )
    hw_fail = bool(re.search(r"(unknown encoder|encoder not found|device.*not found|nvenc|qsv|amf).*(fail|error|unavailable|not)", t))
    demux = bool(re.search(r"(error during demuxing|error opening input|moov atom not found|partial file|invalid data found when processing input|demux)", t))

    # Try to extract bad segment indexes if present in logs (e.g. from repair messages or health)
    corrupt_indexes: list[int] = []
    idx_match = re.findall(r"segment\(s\)\s*([0-9,\s]+)", text, re.IGNORECASE)
    for m in idx_match:
        for num in re.findall(r"\d+", m):
            # Log messages print human-friendly 1-based segment numbers.
            corrupt_indexes.append(max(0, int(num) - 1))

    problems = []
    if bitstream:
        problems.append("bitstream_corruption")
    if ts_issue:
        problems.append("timestamp_discontinuity")
    if dur_trunc:
        problems.append("duration_truncated")
    if audio_fail:
        problems.append("audio_decode_fail")
    if hw_fail:
        problems.append("hw_unavailable")
    if demux:
        problems.append("demux_error")

    summary = ", ".join(problems) if problems else "unknown/other"
    return ProblemProfile(
        bitstream_corrupt_indexes=sorted(set(corrupt_indexes)),
        bitstream_corruption=bitstream,
        demux_errors=demux,
        timestamp_discontinuity=ts_issue,
        duration_truncated=dur_trunc,
        audio_decode_fail=audio_fail,
        hw_unavailable=hw_fail,
        summary=summary,
        raw_snippets=[text[:2000]],
    )


def find_bad_h264_segments(
    input_videos: list[str | Path],
    ffmpeg_bin: str,
    tail_seconds: float | None = None,
) -> list[int]:
    """Probe each input (or its tail) by attempting a video-only copy through the matching bitstream filter.
    Any corruption will surface as specific errors/warnings in stderr (even if rc may vary).
    Uses the centralized text_indicates_bitstream_corruption for consistent detection with fallback logic.
    """
    bad_indexes: list[int] = []
    for index, video in enumerate(input_videos):
        codec = pkg_attr("_get_video_codec")(video)
        if codec == "h264":
            bitstream_filter = "h264_mp4toannexb"
        elif codec in {"hevc", "h265"}:
            bitstream_filter = "hevc_mp4toannexb"
        else:
            continue

        input_args: list[str] = []
        if tail_seconds is not None and tail_seconds > 0:
            input_args.extend(["-sseof", f"-{tail_seconds:.3f}"])
        completed = subprocess.run(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-v",
                "warning",
                *input_args,
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                "-bsf:v",
                bitstream_filter,
                "-f",
                "null",
                devnull,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stderr = completed.stderr or ""
        if completed.returncode != 0 or text_indicates_bitstream_corruption(stderr):
            bad_indexes.append(index)
    return bad_indexes
