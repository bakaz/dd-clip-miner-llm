from __future__ import annotations

import shutil
from pathlib import Path

from .command import run_command
from .encode import concat_reencode_arg_candidates, concat_scale_args
from .errors import FFmpegError
from .probe import get_video_resolution
from .validation import sum_media_durations
from .validation import validate_audio_decodable, validate_concat_duration

def handle_single_input(
    input_video: str | Path,
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    audio_bitrate_kbps: int,
    single_file_policy: str,
) -> Path:
    policy = (single_file_policy or "copy").lower()
    expected_duration = sum_media_durations([input_video])
    if policy == "copy":
        shutil.copy2(str(input_video), str(output))
        return output
    if policy == "remux":
        remux_single_input(input_video, output, ffmpeg_bin, expected_duration)
        return output
    if policy == "normalize":
        normalize_single_input(
            input_video,
            output,
            ffmpeg_bin,
            video_codec,
            audio_bitrate_kbps,
            expected_duration,
        )
        return output
    raise ValueError(
        "single_file_policy must be one of: copy, remux, normalize"
    )


def remux_single_input(
    input_video: str | Path,
    output: Path,
    ffmpeg_bin: str,
    expected_duration: float | None,
) -> None:
    run_command([
        ffmpeg_bin,
        "-y",
        "-fflags",
        "+genpts",
        "-err_detect",
        "ignore_err",
        "-i",
        str(input_video),
        "-map",
        "0",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        str(output),
    ])
    validate_concat_duration(output, expected_duration)
    validate_audio_decodable(output, ffmpeg_bin)


def normalize_single_input(
    input_video: str | Path,
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    audio_bitrate_kbps: int,
    expected_duration: float | None,
) -> None:
    target_size = None
    try:
        target_size = get_video_resolution(input_video)
    except (FFmpegError, ValueError):
        pass
    scale_args = concat_scale_args(target_size)
    errors: list[str] = []
    for encode_args in concat_reencode_arg_candidates(ffmpeg_bin, video_codec):
        try:
            run_command([
                ffmpeg_bin,
                "-y",
                "-fflags",
                "+genpts",
                "-err_detect",
                "ignore_err",
                "-i",
                str(input_video),
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
            validate_concat_duration(output, expected_duration)
            validate_audio_decodable(output, ffmpeg_bin)
            return
        except FFmpegError as exc:
            errors.append(str(exc))
    raise FFmpegError("Single-file normalize failed:\n" + "\n".join(errors))
