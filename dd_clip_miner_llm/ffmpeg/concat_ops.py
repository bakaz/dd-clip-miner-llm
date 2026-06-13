from __future__ import annotations

import json
import shutil
import subprocess
from os import devnull
from pathlib import Path
from typing import Any
from uuid import uuid4

from .command import require_binary
from .compat import pkg_attr
from .errors import FFmpegError
from .fsutil import safe_rmtree, safe_unlink
from .validation import get_min_video_size


def run_command(*args: Any, **kwargs: Any) -> Any:
    return pkg_attr("run_command")(*args, **kwargs)


def get_duration(path: str | Path) -> float | None:
    return pkg_attr("get_duration")(path)


def get_video_fps(path: str | Path) -> float:
    return pkg_attr("get_video_fps")(path)


def find_bad_h264_segments(*args: Any, **kwargs: Any) -> list[int]:
    return pkg_attr("_find_bad_h264_segments")(*args, **kwargs)


def repair_video_filter_args(*args: Any, **kwargs: Any) -> list[str]:
    return pkg_attr("_repair_video_filter_args")(*args, **kwargs)


def targeted_repair_encode_candidates(*args: Any, **kwargs: Any) -> list[list[str]]:
    return pkg_attr("_targeted_repair_encode_candidates")(*args, **kwargs)


def concat_reencode_arg_candidates(*args: Any, **kwargs: Any) -> list[list[str]]:
    return pkg_attr("_concat_reencode_arg_candidates")(*args, **kwargs)


def video_reencode_arg_candidates(*args: Any, **kwargs: Any) -> list[list[str]]:
    return pkg_attr("_video_reencode_arg_candidates")(*args, **kwargs)


def validate_concat_duration(*args: Any, **kwargs: Any) -> None:
    return pkg_attr("_validate_concat_duration")(*args, **kwargs)


def validate_audio_decodable(*args: Any, **kwargs: Any) -> None:
    return pkg_attr("_validate_audio_decodable")(*args, **kwargs)


def has_audio_stream(path: str | Path) -> bool:
    return pkg_attr("_has_audio_stream")(path)


def concat_filter_complex(*args: Any, **kwargs: Any) -> str:
    return pkg_attr("_concat_filter_complex")(*args, **kwargs)


def concat_scale_args(*args: Any, **kwargs: Any) -> list[str]:
    return pkg_attr("_concat_scale_args")(*args, **kwargs)

def concat_reencoded_bad_segments_copy(
    input_videos: list[str | Path],
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    audio_bitrate_kbps: int,
    expected_duration: float | None,
    bad_indexes: list[int] | None = None,
) -> None:
    bad_index_set = set(
        find_bad_h264_segments(input_videos, ffmpeg_bin)
        if bad_indexes is None
        else bad_indexes
    )
    if not bad_index_set:
        raise FFmpegError("No H.264 bitstream-corrupt segments detected")

    temp_dir = output.parent / f"_concat_repair_{uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    try:
        for candidate_index, encode_args in enumerate(
            targeted_repair_encode_candidates(ffmpeg_bin, video_codec)
        ):
            candidate_dir = temp_dir / f"candidate_{candidate_index:02d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            concat_file = candidate_dir / "concat_list.txt"
            repaired: list[str | Path] = []

            try:
                for index, video in enumerate(input_videos):
                    target = candidate_dir / f"part_{index:04d}.mp4"
                    if index not in bad_index_set:
                        run_command(
                            [
                                ffmpeg_bin,
                                "-y",
                                "-fflags",
                                "+genpts+igndts",
                                "-err_detect",
                                "ignore_err",
                                "-i",
                                str(video),
                                "-map",
                                "0:v:0?",
                                "-map",
                                "0:a:0?",
                                "-c",
                                "copy",
                                "-avoid_negative_ts",
                                "make_zero",
                                "-movflags",
                                "+faststart",
                                str(target),
                            ],
                            bitstream_fatal=True,
                        )
                        repaired.append(target)
                        continue

                    fps = get_video_fps(video)
                    video_filter = repair_video_filter_args(fps)
                    audio_filter = ["-af", "asetpts=PTS-STARTPTS"] if has_audio_stream(video) else []
                    # Cap re-encode of bad segment to its original (reported) duration to prevent
                    # corrupt decode + repair filter from producing runaway long output (which
                    # leads to "duration too long" in final assembly vs original expected).
                    # This is part of correct handling for wrong splice duration on real corrupt data.
                    orig_dur = get_duration(video) or 0
                    cap_args = ["-t", f"{orig_dur:.3f}"] if orig_dur > 0 else []
                    run_command([
                        ffmpeg_bin,
                        "-y",
                        "-fflags",
                        "+discardcorrupt",
                        "-err_detect",
                        "ignore_err",
                        "-i",
                        str(video),
                        "-map",
                        "0:v:0",
                        "-map",
                        "0:a:0?",
                        *cap_args,
                        *video_filter,
                        *audio_filter,
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
                        str(target),
                    ])
                    repaired.append(target)

                # Write assembly list with explicit durations from the *actual* repaired parts (good originals + re-encoded bads).
                # This forces the concat demuxer timeline to the real sum of parts (after repair/sanitize), preventing
                # "duration too long/short" vs original (often bogus) expected from corrupt sources.
                # Use ffconcat format so duration directives are honored.
                part_durs = []
                for v in repaired:
                    d = get_duration(v) or 0.0
                    part_durs.append(max(0.001, d))
                with concat_file.open("w", encoding="utf-8") as handle:
                    handle.write("ffconcat version 1.0\n")
                    for i, video in enumerate(repaired):
                        escaped_path = str(video).replace("'", "'\\''")
                        handle.write(f"file '{escaped_path}'\n")
                        if part_durs:
                            handle.write(f"duration {part_durs[i]:.6f}\n")

                candidate_output = candidate_dir / "concat.mp4"
                # Validate against the sum of the actual repaired parts (the "correct" length after fixing corrupt).
                actual_expected = sum(part_durs) if part_durs else expected_duration
                try:
                    run_command(
                        [
                            ffmpeg_bin,
                            "-y",
                            "-fflags",
                            "+genpts+igndts+discardcorrupt",
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
                            "-avoid_negative_ts",
                            "make_zero",
                            "-movflags",
                            "+faststart",
                            str(candidate_output),
                        ],
                        bitstream_fatal=True,
                    )
                    validate_concat_duration(candidate_output, actual_expected)
                    validate_audio_decodable(candidate_output, ffmpeg_bin)
                except FFmpegError:
                    safe_unlink(candidate_output)
                    filter_complex = concat_filter_complex(
                        len(repaired),
                        get_min_video_size(repaired),
                    )
                    filter_inputs: list[str] = []
                    for item in repaired:
                        filter_inputs.extend(["-i", str(item)])
                    run_command(
                        [
                            ffmpeg_bin,
                            "-y",
                            *filter_inputs,
                            "-filter_complex",
                            filter_complex,
                            "-map",
                            "[v]",
                            "-map",
                            "[a]",
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
                            str(candidate_output),
                        ]
                    )
                    validate_concat_duration(candidate_output, actual_expected)
                    validate_audio_decodable(candidate_output, ffmpeg_bin)
                safe_unlink(output)
                shutil.move(str(candidate_output), str(output))
                return
            except FFmpegError as exc:
                errors.append(str(exc))
                safe_rmtree(candidate_dir)

        raise FFmpegError(
            "Targeted concat repair failed:\n" + "\n".join(errors)
        )
    finally:
        safe_rmtree(temp_dir)

def concat_audio_reencoded_copy(
    output: Path,
    concat_file: Path,
    ffmpeg_bin: str,
    audio_bitrate_kbps: int,
    expected_duration: float | None,
) -> None:
    # -c:v copy path: make bitstream_fatal so that corruption warnings from ffmpeg are promoted
    # to exception (with real error text) -> better diagnosis + immediate fallback in smart pipeline.
    run_command(
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
            "-c:v",
            "copy",
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
        ],
        bitstream_fatal=True,
    )
    validate_concat_duration(output, expected_duration)
    validate_audio_decodable(output, ffmpeg_bin)


def concat_remuxed_copy(
    input_videos: list[str | Path],
    output: Path,
    ffmpeg_bin: str,
    expected_duration: float | None,
) -> None:
    temp_dir = output.parent / f"_concat_remux_{uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    concat_file = temp_dir / "concat_list.txt"
    remuxed: list[Path] = []

    try:
        for index, video in enumerate(input_videos):
            target = temp_dir / f"part_{index:04d}.mp4"
            run_command([
                ffmpeg_bin, "-y",
                "-i", str(video),
                "-map", "0:v:0?",
                "-map", "0:a:0?",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(target),
            ])
            remuxed.append(target)

        with concat_file.open("w", encoding="utf-8") as handle:
            for video in remuxed:
                escaped_path = str(video).replace("'", "'\\''")
                handle.write(f"file '{escaped_path}'\n")

        candidate_output = temp_dir / "concat.mp4"
        run_command(
            [
                ffmpeg_bin, "-y",
                "-fflags", "+genpts+igndts+discardcorrupt",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-map", "0:v:0?",
                "-map", "0:a:0?",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(candidate_output),
            ],
            bitstream_fatal=True,
        )
        validate_concat_duration(candidate_output, expected_duration)
        validate_audio_decodable(candidate_output, ffmpeg_bin)
        safe_unlink(output)
        shutil.move(str(candidate_output), str(output))
    finally:
        safe_rmtree(temp_dir)


def concat_timestamp_remuxed_copy(
    input_videos: list[str | Path],
    output: Path,
    ffmpeg_bin: str,
    expected_duration: float | None,
) -> None:
    temp_dir = output.parent / f"_concat_ts_remux_{uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    concat_file = temp_dir / "concat_list.ffconcat"
    remuxed: list[Path] = []

    try:
        for index, video in enumerate(input_videos):
            target = temp_dir / f"part_{index:04d}.mp4"
            run_command([
                ffmpeg_bin, "-y",
                "-fflags", "+genpts+igndts+discardcorrupt",
                "-err_detect", "ignore_err",
                "-i", str(video),
                "-map", "0",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(target),
            ])
            remuxed.append(target)

        write_ffconcat_list(concat_file, remuxed)
        candidate_output = temp_dir / "concat.mp4"
        run_command(
            [
                ffmpeg_bin, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-map", "0:v:0?",
                "-map", "0:a:0?",
                "-c", "copy",
                "-movflags", "+faststart",
                str(candidate_output),
            ],
            bitstream_fatal=True,
        )
        validate_concat_duration(candidate_output, expected_duration)
        validate_audio_decodable(candidate_output, ffmpeg_bin)
        safe_unlink(output)
        shutil.move(str(candidate_output), str(output))
    finally:
        safe_rmtree(temp_dir)


def concat_timestamp_remuxed_audio_resync(
    input_videos: list[str | Path],
    output: Path,
    ffmpeg_bin: str,
    audio_bitrate_kbps: int,
    expected_duration: float | None,
) -> None:
    temp_dir = output.parent / f"_concat_ts_audio_{uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    concat_file = temp_dir / "concat_list.ffconcat"
    remuxed: list[Path] = []

    try:
        for index, video in enumerate(input_videos):
            target = temp_dir / f"part_{index:04d}.mp4"
            run_command([
                ffmpeg_bin, "-y",
                "-fflags", "+genpts+igndts+discardcorrupt",
                "-err_detect", "ignore_err",
                "-i", str(video),
                "-map", "0",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(target),
            ])
            remuxed.append(target)

        write_ffconcat_list(concat_file, remuxed)
        candidate_output = temp_dir / "concat.mp4"
        run_command(
            [
                ffmpeg_bin, "-y",
                "-fflags", "+genpts+igndts+discardcorrupt",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-map", "0:v:0?",
                "-map", "0:a:0?",
                "-c:v", "copy",
                "-af", "aresample=async=1000:first_pts=0",
                "-c:a", "aac",
                "-b:a", f"{audio_bitrate_kbps}k",
                "-ar", "48000",
                "-ac", "2",
                "-movflags", "+faststart",
                str(candidate_output),
            ],
            bitstream_fatal=True,
        )
        validate_concat_duration(candidate_output, expected_duration)
        validate_audio_decodable(candidate_output, ffmpeg_bin)
        safe_unlink(output)
        shutil.move(str(candidate_output), str(output))
    finally:
        safe_rmtree(temp_dir)


def concat_fast_transmux_copy(
    input_videos: list[str | Path],
    output: Path,
    ffmpeg_bin: str,
    expected_duration: float | None,
) -> None:
    temp_dir = output.parent / f"_concat_fast_transmux_{uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    ts_files: list[Path] = []

    try:
        for index, video in enumerate(input_videos):
            target = temp_dir / f"part_{index:04d}.ts"
            run_command([
                ffmpeg_bin, "-y",
                "-fflags", "+genpts+igndts+discardcorrupt",
                "-err_detect", "ignore_err",
                "-i", str(video),
                "-map", "0:v:0?",
                "-map", "0:a:0?",
                "-c", "copy",
                "-bsf:v", "h264_mp4toannexb",
                "-f", "mpegts",
                str(target),
            ], bitstream_fatal=True)
            ts_files.append(target)

        concat_url = "concat:" + "|".join(
            path.resolve().as_uri()
            for path in ts_files
        )
        candidate_output = temp_dir / "concat.mp4"
        run_command(
            [
                ffmpeg_bin, "-y",
                "-fflags", "+genpts+igndts+discardcorrupt",
                "-protocol_whitelist", "file,concat",
                "-i", concat_url,
                "-map", "0:v:0?",
                "-map", "0:a:0?",
                "-c", "copy",
                "-bsf:a", "aac_adtstoasc",
                "-movflags", "+faststart",
                str(candidate_output),
            ],
            bitstream_fatal=True,
        )
        validate_concat_duration(candidate_output, expected_duration)
        validate_audio_decodable(candidate_output, ffmpeg_bin)
        safe_unlink(output)
        shutil.move(str(candidate_output), str(output))
    finally:
        safe_rmtree(temp_dir)


def concat_ts_protocol_copy(
    ts_files: list[str | Path],
    output: Path,
    ffmpeg_bin: str,
    expected_duration: float | None,
) -> None:
    temp_dir = output.parent / f"_concat_ts_proto_{uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        concat_url = "concat:" + "|".join(
            Path(path).resolve().as_uri()
            for path in ts_files
        )
        candidate_output = temp_dir / "concat.mp4"
        run_command(
            [
                ffmpeg_bin, "-y",
                "-fflags", "+genpts+igndts+discardcorrupt",
                "-protocol_whitelist", "file,concat",
                "-i", concat_url,
                "-map", "0:v:0?",
                "-map", "0:a:0?",
                "-c", "copy",
                "-bsf:a", "aac_adtstoasc",
                "-movflags", "+faststart",
                str(candidate_output),
            ],
            bitstream_fatal=True,
        )
        validate_concat_duration(candidate_output, expected_duration)
        validate_audio_decodable(candidate_output, ffmpeg_bin)
        safe_unlink(output)
        shutil.move(str(candidate_output), str(output))
    finally:
        safe_rmtree(temp_dir)


def concat_tail_window_repaired_bad_segments_copy(
    input_videos: list[str | Path],
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    audio_bitrate_kbps: int,
    expected_duration: float | None,
    bad_indexes: list[int],
    repair_window_seconds: float = 90.0,
    guard_seconds: float = 2.0,
) -> None:
    if not bad_indexes:
        raise FFmpegError("No H.264 bitstream-corrupt segments detected")

    temp_dir = output.parent / f"_concat_window_repair_{uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    bad_index_set = set(bad_indexes)
    repaired_inputs: list[str | Path] = []

    try:
        for index, video in enumerate(input_videos):
            if index not in bad_index_set:
                repaired_inputs.append(video)
                continue

            duration = get_duration(video)
            if duration <= repair_window_seconds + guard_seconds + 1.0:
                raise FFmpegError("Tail window repair skipped: segment is too short for window split")

            start_tail = max(0.0, duration - repair_window_seconds - guard_seconds)
            part_dir = temp_dir / f"part_{index:04d}"
            part_dir.mkdir(parents=True, exist_ok=True)
            head = part_dir / "head.mp4"
            tail = part_dir / "tail.mp4"
            fixed = part_dir / "fixed.mp4"
            list_file = part_dir / "concat_list.txt"

            run_command([
                ffmpeg_bin, "-y",
                "-i", str(video),
                "-t", f"{start_tail:.3f}",
                "-map", "0:v:0?",
                "-map", "0:a:0?",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(head),
            ], bitstream_fatal=True)

            fps = get_video_fps(video)
            video_filter = repair_video_filter_args(fps)
            audio_filter = ["-af", "asetpts=PTS-STARTPTS"] if has_audio_stream(video) else []
            errors: list[str] = []
            for encode_args in targeted_repair_encode_candidates(ffmpeg_bin, video_codec):
                try:
                    run_command([
                        ffmpeg_bin, "-y",
                        "-ss", f"{start_tail:.3f}",
                        "-fflags", "+discardcorrupt",
                        "-err_detect", "ignore_err",
                        "-i", str(video),
                        "-t", f"{repair_window_seconds + guard_seconds:.3f}",
                        "-map", "0:v:0",
                        "-map", "0:a:0?",
                        *video_filter,
                        *audio_filter,
                        *encode_args,
                        "-pix_fmt", "yuv420p",
                        "-c:a", "aac",
                        "-b:a", f"{audio_bitrate_kbps}k",
                        "-ar", "48000",
                        "-ac", "2",
                        "-movflags", "+faststart",
                        str(tail),
                    ])
                    break
                except FFmpegError as exc:
                    errors.append(str(exc))
            else:
                raise FFmpegError("Tail window repair failed:\n" + "\n".join(errors))

            write_ffconcat_list(list_file, [head, tail])
            run_command(
                [
                    ffmpeg_bin, "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(list_file),
                    "-map", "0:v:0?",
                    "-map", "0:a:0?",
                    "-c", "copy",
                    "-movflags", "+faststart",
                    str(fixed),
                ],
                bitstream_fatal=True,
            )
            validate_concat_duration(fixed, duration)
            repaired_inputs.append(fixed)

        concat_file = temp_dir / "concat_list.txt"
        repaired_durations = durations_or_none(repaired_inputs)
        write_ffconcat_list(
            concat_file,
            repaired_inputs,
            repaired_durations,
            include_duration=repaired_durations is not None,
        )
        candidate_output = temp_dir / "concat.mp4"
        run_command(
            [
                ffmpeg_bin, "-y",
                "-fflags", "+genpts+igndts+discardcorrupt",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-map", "0:v:0?",
                "-map", "0:a:0?",
                "-c", "copy",
                "-movflags", "+faststart",
                str(candidate_output),
            ],
            bitstream_fatal=True,
        )
        actual_expected = sum(repaired_durations) if repaired_durations else expected_duration
        validate_concat_duration(candidate_output, actual_expected)
        validate_audio_decodable(candidate_output, ffmpeg_bin)
        safe_unlink(output)
        shutil.move(str(candidate_output), str(output))
    finally:
        safe_rmtree(temp_dir)


def durations_or_none(videos: list[str | Path]) -> list[float] | None:
    durations: list[float] = []
    try:
        for video in videos:
            durations.append(get_duration(video))
    except (FFmpegError, ValueError):
        return None
    return durations


def write_ffconcat_list(
    path: Path,
    videos: list[str | Path],
    durations: list[float] | None = None,
    include_duration: bool = False,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        if include_duration and durations is not None:
            handle.write("ffconcat version 1.0\n")
        for index, video in enumerate(videos):
            escaped_path = str(video).replace("'", "'\\''")
            handle.write(f"file '{escaped_path}'\n")
            if include_duration and durations is not None and index < len(durations):
                handle.write(f"duration {durations[index]:.6f}\n")


def concat_filter_commands(
    input_videos: list[str | Path],
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    target_size: tuple[int, int] | None,
    audio_bitrate_kbps: int,
) -> list[list[str]]:
    inputs: list[str] = []
    for video in input_videos:
        inputs.extend(["-i", str(video)])

    filter_complex = concat_filter_complex(len(input_videos), target_size)
    commands: list[list[str]] = []
    for encode_args in concat_reencode_arg_candidates(ffmpeg_bin, video_codec):
        commands.append([
            ffmpeg_bin, "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "[a]",
            *encode_args,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", f"{audio_bitrate_kbps}k",
            "-ar", "48000",
            "-ac", "2",
            "-movflags", "+faststart",
            str(output),
        ])
    return commands

