from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import shutil
from uuid import uuid4

from .. import ffmpeg as ffmpeg_mod
from .models import (
    AttemptRecord,
    ConcatAttempt,
    ConcatContext,
    HealthInfo,
    ProblemProfile,
    TargetProfile,
    VideoMeta,
)
from .planner import (
    build_target_profile,
    can_direct_concat_copy,
    expected_duration,
    file_matches_profile,
)
from .probe import probe_many, probe_one

def _build_health_profile(
    inputs: list[Path],
    metas: list[VideoMeta],
    ffmpeg_bin: str,
) -> dict[int, HealthInfo]:
    """Upfront health probe for all inputs. This is the key improvement:
    detect bitstream corruption *before* expensive copy attempts.

    Optimization from community best practices (ffmpeg TS remux, per-file sanitize,
    discardcorrupt for live recordings):
    - Only scan h264.
    - For small files (<120s, common for error-burst "fix" segments in splits),
      do FULL scan (not just tail) to catch corruption accurately.
    - Tail 60s for large files (typical end-of-recording corruption in live splits).
    """
    health: dict[int, HealthInfo] = {}
    h264_indexes = [
        index
        for index, meta in enumerate(metas)
        if meta.probe_ok and meta.video_codec in {"h264", "hevc", "h265"}
    ]
    h264_inputs = [inputs[index] for index in h264_indexes]
    bad_set: set[int] = set()
    if h264_inputs:
        # Per-file tail or full based on size (small files = likely the corrupt "fix" points).
        # Full scans stay serial because they can read whole videos; tail scans are cheap enough
        # to run with small bounded concurrency on network storage.
        full_scan_indexes: list[int] = []
        tail_scan_indexes: list[int] = []
        for idx in h264_indexes:
            meta = metas[idx]
            dur = meta.duration or 0
            if dur > 0 and dur < 120:
                full_scan_indexes.append(idx)
            else:
                tail_scan_indexes.append(idx)

        for inp_idx in full_scan_indexes:
            inp = inputs[inp_idx]
            single_bad = ffmpeg_mod._find_bad_h264_segments(
                [inp], ffmpeg_bin, tail_seconds=None
            )
            if single_bad:
                bad_set.add(inp_idx)

        def _tail_scan_one(inp_idx: int) -> tuple[int, bool]:
            inp = inputs[inp_idx]
            single_bad = ffmpeg_mod._find_bad_h264_segments(
                [inp], ffmpeg_bin, tail_seconds=60.0
            )
            return inp_idx, bool(single_bad)

        if tail_scan_indexes:
            max_workers = min(2, len(tail_scan_indexes))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_tail_scan_one, idx) for idx in tail_scan_indexes]
                for future in as_completed(futures):
                    inp_idx, is_bad = future.result()
                    if is_bad:
                        bad_set.add(inp_idx)

    for i, (inp, meta) in enumerate(zip(inputs, metas)):
        is_corrupt = i in bad_set
        is_small = (meta.duration or 0) < 120
        corrupt_detail = ""
        if is_corrupt:
            corrupt_detail = ("corrupt (full scan - small file)" if is_small else "corrupt tail detected by bitstream filter scan")

        health[i] = HealthInfo(
            path=inp,
            probe_ok=meta.probe_ok,
            duration=meta.duration,
            has_video=meta.has_video,
            has_audio=meta.has_audio,
            is_bitstream_corrupt=is_corrupt,
            corrupt_details=corrupt_detail,
            error=meta.error if not meta.probe_ok else None,
        )
    return health


def _get_annexb_bsf(codec: str | None) -> str:
    """Return the appropriate mp4toannexb bitstream filter for the video codec."""
    c = (codec or "").lower()
    if c in ("hevc", "h265"):
        return "hevc_mp4toannexb"
    return "h264_mp4toannexb"


def _safe_transmux_to_ts(src: Path, dst: Path, ffmpeg_bin: str, video_bsf: str = "h264_mp4toannexb") -> None:
    """Construct a TS copy from an (original or sanitized) MP4 source.

    Uses the standard safe flags + annexb bsf for live-split corruption recovery.
    This is the central place for "构造 ts copy" logic.
    """
    cmd = [
        ffmpeg_bin, "-y",
        "-fflags", "+genpts+igndts+discardcorrupt",
        "-err_detect", "ignore_err",
        "-i", str(src),
        "-map", "0:v:0?",
        "-map", "0:a:0?",
        "-c", "copy",
        "-bsf:v", video_bsf,
        "-f", "mpegts",
        str(dst),
    ]
    ffmpeg_mod.run_command(cmd, bitstream_fatal=True)


def _has_video_bitstream_corruption(context: ConcatContext) -> bool:
    if context.health and any(info.is_bitstream_corrupt for info in context.health.values()):
        return True
    return bool(context.profile and context.profile.is_bitstream_problem())

