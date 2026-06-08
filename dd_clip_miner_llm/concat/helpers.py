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

def _attempt(
    attempts: list[ConcatAttempt],
    name: str,
    action,
) -> bool:
    try:
        action()
    except ffmpeg_mod.FFmpegError as exc:
        detail = str(exc)
        analysis = ffmpeg_mod.analyze_ffmpeg_failure(detail)
        diag = analysis.get("summary", "")
        if diag and diag != "unknown/other":
            # Include diagnosis in the stored detail so that later branch decisions (_has_*) and final error
            # message clearly show what ffmpeg output indicated (core of the fallback detection).
            detail = f"{detail}\n[diagnosed: {diag}]"
        attempts.append(ConcatAttempt(name, False, detail))
        # Print short reason on every attempt failure so live batch-run output shows the
        # actual ffmpeg error text (NAL, bitstream etc.) for easier debugging and optimization.
        short = ffmpeg_mod._short_error(exc)
        print(f"[concat]   -> {name} failed: {short}")
        return False
    attempts.append(ConcatAttempt(name, True, "ok"))
    return True


def _concat_copy_with_list(
    output: Path,
    concat_file: Path,
    ffmpeg_bin: str,
    expected_duration_value: float | None,
) -> None:
    # bitstream_fatal: capture real corruption messages from ffmpeg output (e.g. Invalid NAL, missing picture)
    # even if ffmpeg rc==0, so that fallback decision has accurate diagnosis from the output.
    ffmpeg_mod.run_command(
        [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-map",
            "0:v:0?",
            "-map",
            "0:a:0?",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ],
        bitstream_fatal=True,
    )
    _validate_output(output, ffmpeg_bin, expected_duration_value)


def _remux_concat_copy(
    inputs: list[Path],
    output: Path,
    ffmpeg_bin: str,
    expected_duration_value: float | None,
) -> None:
    ffmpeg_mod._concat_remuxed_copy(
        inputs,
        output,
        ffmpeg_bin,
        expected_duration_value,
    )
    ffmpeg_mod._validate_audio_decodable(output, ffmpeg_bin)


def _remux_concat_audio_reencode(
    inputs: list[Path],
    output: Path,
    ffmpeg_bin: str,
    audio_bitrate_kbps: int,
    expected_duration_value: float | None,
) -> None:
    temp_dir, remuxed, remuxed_list = _remux_inputs(inputs, output.parent, ffmpeg_bin)
    try:
        ffmpeg_mod._concat_audio_reencoded_copy(
            output,
            remuxed_list,
            ffmpeg_bin,
            audio_bitrate_kbps,
            expected_duration_value,
        )
    finally:
        ffmpeg_mod._safe_unlink(remuxed_list)
        ffmpeg_mod._safe_rmtree(temp_dir)


def _remux_inputs(
    inputs: list[Path],
    parent: Path,
    ffmpeg_bin: str,
) -> tuple[Path, list[Path], Path]:
    temp_dir = parent / f"concat_remux_{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    remuxed: list[Path] = []
    try:
        for index, source in enumerate(inputs):
            target = temp_dir / f"{index:05d}.mp4"
            ffmpeg_mod.run_command([
                ffmpeg_bin,
                "-y",
                "-fflags",
                "+genpts",
                "-err_detect",
                "ignore_err",
                "-i",
                str(source),
                "-map",
                "0",
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                str(target),
            ])
            remuxed.append(target)
        remuxed_list = temp_dir / "concat_list.txt"
        _write_concat_list(remuxed_list, remuxed)
        return temp_dir, remuxed, remuxed_list
    except Exception:
        ffmpeg_mod._safe_rmtree(temp_dir)
        raise


def _attempt_tail_repair(
    attempts: list[ConcatAttempt],
    inputs: list[Path],
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    audio_bitrate_kbps: int,
    expected_duration_value: float | None,
) -> bool:
    try:
        bad_indexes = ffmpeg_mod._find_bad_h264_segments(
            inputs,
            ffmpeg_bin,
            tail_seconds=60.0,
        )
    except ffmpeg_mod.FFmpegError as exc:
        attempts.append(ConcatAttempt("H.264 tail scan localized reencode", False, str(exc)))
        return False
    if not bad_indexes:
        attempts.append(ConcatAttempt("H.264 tail scan localized reencode", False, "skipped: no corrupt tail segments detected"))
        return False
    print(
        "[concat] Detected possible corrupt H.264 tail segment(s) "
        f"{', '.join(str(i + 1) for i in bad_indexes)}; repairing only those segment(s)."
    )
    return _attempt(
        attempts,
        "H.264 tail scan localized reencode",
        lambda: _targeted_repair(
            inputs,
            output,
            ffmpeg_bin,
            video_codec,
            audio_bitrate_kbps,
            expected_duration_value,
            bad_indexes=bad_indexes,
        ),
    )


def _targeted_repair(
    inputs: list[Path],
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    audio_bitrate_kbps: int,
    expected_duration_value: float | None,
    bad_indexes: list[int] | None,
) -> None:
    ffmpeg_mod._concat_reencoded_bad_segments_copy(
        inputs,
        output,
        ffmpeg_bin,
        video_codec,
        audio_bitrate_kbps,
        expected_duration_value,
        bad_indexes=bad_indexes,
    )
    _validate_output(output, ffmpeg_bin, expected_duration_value)


def _selective_normalize_concat(
    inputs: list[Path],
    metas: list[VideoMeta],
    target: TargetProfile,
    target_size: tuple[int, int] | None,
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    audio_bitrate_kbps: int,
    expected_duration_value: float | None,
    force_indexes: set[int] | None = None,
    cpu_only_indexes: set[int] | None = None,
) -> None:
    temp_dir = output.parent / f"concat_selective_{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    selected: list[Path] = []
    list_file = temp_dir / "concat_list.txt"
    force_indexes = force_indexes or set()
    cpu_only_indexes = cpu_only_indexes or set()
    try:
        for index, source in enumerate(inputs):
            meta = metas[index] if index < len(metas) else None
            if (
                index not in force_indexes
                and meta is not None
                and file_matches_profile(meta, target, audio_bitrate_kbps)
            ):
                selected.append(source)
                continue
            target_file = temp_dir / f"{index:05d}.mp4"
            _normalize_to_profile(
                source,
                target_file,
                meta,
                target_size,
                ffmpeg_bin,
                video_codec,
                audio_bitrate_kbps,
                prefer_cpu=index in cpu_only_indexes,
            )
            selected.append(target_file)
        _write_concat_list(list_file, selected)
        _concat_copy_with_list(output, list_file, ffmpeg_bin, expected_duration_value)
    finally:
        ffmpeg_mod._safe_unlink(list_file)
        ffmpeg_mod._safe_rmtree(temp_dir)


def _normalize_to_profile(
    source: Path,
    output: Path,
    meta: VideoMeta | None,
    target_size: tuple[int, int] | None,
    ffmpeg_bin: str,
    video_codec: str,
    audio_bitrate_kbps: int,
    prefer_cpu: bool = False,
) -> None:
    scale_args = ffmpeg_mod._concat_scale_args(target_size)
    has_audio = bool(meta.has_audio) if meta is not None and meta.probe_ok else True
    errors: list[str] = []
    encode_candidates = (
        ffmpeg_mod._targeted_repair_encode_candidates(ffmpeg_bin, video_codec)
        if prefer_cpu
        else ffmpeg_mod._concat_reencode_arg_candidates(ffmpeg_bin, video_codec)
    )
    for encode_args in encode_candidates:
        input_args = [
            ffmpeg_bin,
            "-y",
            "-fflags",
            "+genpts",
            "-err_detect",
            "ignore_err",
            "-i",
            str(source),
        ]
        map_args = ["-map", "0:v:0?"]
        if has_audio:
            map_args += ["-map", "0:a:0?"]
            audio_input_args: list[str] = []
            shortest_args: list[str] = []
        else:
            audio_input_args = [
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
            ]
            map_args += ["-map", "1:a:0"]
            shortest_args = ["-shortest"]
        try:
            ffmpeg_mod.run_command([
                *input_args,
                *audio_input_args,
                *map_args,
                *scale_args,
                *encode_args,
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                f"{audio_bitrate_kbps}k",
                "-ar",
                "48000",
                "-ac",
                "2",
                *shortest_args,
                "-movflags",
                "+faststart",
                str(output),
            ])
            return
        except ffmpeg_mod.FFmpegError as exc:
            errors.append(str(exc))
    raise ffmpeg_mod.FFmpegError("Selective normalize failed:\n" + "\n".join(errors))


def _attempt_full_reencode(
    attempts: list[ConcatAttempt],
    output: Path,
    concat_file: Path,
    ffmpeg_bin: str,
    video_codec: str,
    target_size: tuple[int, int] | None,
    audio_bitrate_kbps: int,
    expected_duration_value: float | None,
) -> bool:
    scale_args = ffmpeg_mod._concat_scale_args(target_size)
    for encode_args in ffmpeg_mod._concat_reencode_arg_candidates(ffmpeg_bin, video_codec):
        if _attempt(
            attempts,
            "concat demuxer full reencode",
            lambda encode_args=encode_args: _concat_demuxer_full_reencode(
                output,
                concat_file,
                ffmpeg_bin,
                scale_args,
                encode_args,
                audio_bitrate_kbps,
                expected_duration_value,
            ),
        ):
            return True
    return False


def _concat_demuxer_full_reencode(
    output: Path,
    concat_file: Path,
    ffmpeg_bin: str,
    scale_args: list[str],
    encode_args: list[str],
    audio_bitrate_kbps: int,
    expected_duration_value: float | None,
) -> None:
    ffmpeg_mod.run_command([
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-map",
        "0:v:0?",
        "-map",
        "0:a:0?",
        *scale_args,
        *encode_args,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        f"{audio_bitrate_kbps}k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(output),
    ])
    _validate_output(output, ffmpeg_bin, expected_duration_value)


def _attempt_concat_filter(
    attempts: list[ConcatAttempt],
    inputs: list[Path],
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    target_size: tuple[int, int] | None,
    audio_bitrate_kbps: int,
    expected_duration_value: float | None,
) -> bool:
    for command in ffmpeg_mod._concat_filter_commands(
        inputs,
        output,
        ffmpeg_bin,
        video_codec,
        target_size,
        audio_bitrate_kbps,
    ):
        if _attempt(
            attempts,
            "concat filter full fallback",
            lambda command=command: _run_filter_command(
                command,
                output,
                ffmpeg_bin,
                expected_duration_value,
            ),
        ):
            return True
    return False


def _run_filter_command(
    command: list[str],
    output: Path,
    ffmpeg_bin: str,
    expected_duration_value: float | None,
) -> None:
    ffmpeg_mod.run_command(command, timeout=7200)
    _validate_output(output, ffmpeg_bin, expected_duration_value)


def _validate_output(
    output: Path,
    ffmpeg_bin: str,
    expected_duration_value: float | None,
) -> None:
    ffmpeg_mod._validate_concat_duration(output, expected_duration_value)
    ffmpeg_mod._validate_audio_decodable(output, ffmpeg_bin)


def _write_concat_list(
    path: Path,
    videos: list[Path],
    metas: list[VideoMeta] | None = None,
    include_duration: bool = False,
) -> None:
    durations: list[float] | None = None
    if include_duration and metas is not None and len(metas) == len(videos):
        durations = []
        for meta in metas:
            if meta.duration is None:
                durations = None
                break
            durations.append(max(0.001, float(meta.duration)))

    with path.open("w", encoding="utf-8") as handle:
        if durations is not None:
            handle.write("ffconcat version 1.0\n")
        for index, video in enumerate(videos):
            escaped = str(video).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
            if durations is not None:
                handle.write(f"duration {durations[index]:.6f}\n")


def _safe_attempt_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "attempt"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index:02d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _candidate_output_path(context: ConcatContext, name: str, index: int) -> Path:
    candidate_dir = context.output.parent / "_concat_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    return candidate_dir / f"{_safe_attempt_name(name)}_{index:02d}{context.output.suffix or '.mp4'}"


def _commit_candidate_output(candidate: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate_parent = candidate.parent
    candidate.replace(output)
    _cleanup_empty_dir(candidate_parent)


def _cleanup_empty_dir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def _target_size(target: TargetProfile) -> tuple[int, int] | None:
    if target.width and target.height:
        return target.width, target.height
    return None


def _has_video_bitstream_failure(attempts: list[ConcatAttempt]) -> bool:
    failed_details = [attempt.detail for attempt in attempts if not attempt.ok]
    analysis = ffmpeg_mod.analyze_ffmpeg_failure(failed_details)
    if analysis.get("bitstream_corruption"):
        return True
    # duration_truncated is often a *symptom* of earlier bitstream/demux problems during copy
    if analysis.get("duration_truncated"):
        return True
    return False


def _duration_failure_boundary_indexes(context: ConcatContext) -> list[int]:
    actual = _latest_failed_video_duration(context)
    if actual is None:
        return []
    return _boundary_indexes_for_duration(context.metas, actual)


def _latest_failed_video_duration(context: ConcatContext) -> float | None:
    for attempt in reversed(context.attempts):
        detail = attempt.detail or ""
        value = _extract_failed_video_duration(detail)
        if value is not None:
            return value
    return None


def _extract_failed_video_duration(detail: str) -> float | None:
    patterns = [
        r"Concat output video stream duration is too short:\s*([0-9.]+)s",
        r"Concat output duration is too short:\s*([0-9.]+)s",
    ]
    for pattern in patterns:
        match = re.search(pattern, detail, re.IGNORECASE)
        if not match:
            continue
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _boundary_indexes_for_duration(metas: list[VideoMeta], actual_duration: float) -> list[int]:
    if actual_duration <= 0:
        return []
    cumulative = 0.0
    total = sum(float(meta.duration or 0.0) for meta in metas)
    tolerance = max(2.0, total * 0.0005)
    for index, meta in enumerate(metas):
        if meta.duration is None:
            return []
        cumulative += float(meta.duration)
        if abs(cumulative - actual_duration) <= tolerance:
            candidates = {index}
            if index + 1 < len(metas):
                candidates.add(index + 1)
            if index - 1 >= 0:
                candidates.add(index - 1)
            return sorted(candidates)
    return []


def _corrupt_duration_ratio(context: ConcatContext, corrupt_indexes: list[int]) -> float | None:
    if not corrupt_indexes:
        return 0.0
    durations: list[float] = []
    for meta in context.metas:
        if meta.duration is None:
            return None
        durations.append(float(meta.duration))
    total = sum(durations)
    if total <= 0:
        return None
    bad = sum(durations[index] for index in set(corrupt_indexes) if 0 <= index < len(durations))
    return bad / total


def _corrupt_duration_seconds(context: ConcatContext, corrupt_indexes: list[int]) -> float | None:
    if not corrupt_indexes:
        return 0.0
    durations: list[float] = []
    for meta in context.metas:
        if meta.duration is None:
            return None
        durations.append(float(meta.duration))
    return sum(durations[index] for index in set(corrupt_indexes) if 0 <= index < len(durations))


def _format_failures(attempts: list[ConcatAttempt]) -> str:
    lines = ["All concat attempts failed:"]
    for attempt in attempts:
        if attempt.ok:
            continue
        detail = attempt.detail.strip()
        if len(detail) > 700:
            detail = detail[:700] + "..."
        lines.append(f"- {attempt.name}: {detail}")
    return "\n".join(lines)


def _is_output_duration_failure(detail: str) -> bool:
    text = detail.lower()
    return (
        "concat output duration is too short" in text
        or "concat output duration is too long" in text
        or "concat output video stream duration is too short" in text
        or "concat output video stream duration is too long" in text
    )
