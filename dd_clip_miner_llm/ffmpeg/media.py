from __future__ import annotations

from pathlib import Path

from .command import run_command, require_binary, run_command_with_fallback
from .encode import video_encode_arg_candidates


def extract_audio(
    input_video: str | Path,
    output_wav: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    ffmpeg_bin = require_binary("ffmpeg")
    output = Path(output_wav)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        ffmpeg_bin, "-y",
        "-i", str(input_video),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        str(output),
    ])
    return output


def cut_audio(
    input_media: str | Path,
    output_audio: str | Path,
    start: float,
    end: float,
    copy_codec: bool = False,
    bitrate_kbps: int | None = None,
) -> Path:
    ffmpeg_bin = require_binary("ffmpeg")
    output = Path(output_audio)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_bin, "-y", "-i", str(input_media)]
    if copy_codec:
        cmd.extend(["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-c:a", "copy", "-avoid_negative_ts", "make_zero"])
    else:
        cmd.extend(["-ss", f"{start:.3f}", "-to", f"{end:.3f}"])
        cmd.extend(_audio_encode_args(output, bitrate_kbps=bitrate_kbps))
    cmd.append(str(output))
    run_command(cmd)
    return output


def _audio_encode_args(output_audio: Path, bitrate_kbps: int | None = None) -> list[str]:
    ext = output_audio.suffix.lower().lstrip(".")
    bitrate = max(1, int(bitrate_kbps or 320))
    if ext == "wav":
        return ["-vn", "-acodec", "pcm_s16le"]
    if ext == "mp3":
        return ["-vn", "-acodec", "libmp3lame", "-b:a", f"{bitrate}k"]
    if ext in {"m4a", "aac"}:
        return ["-vn", "-acodec", "aac", "-b:a", f"{bitrate}k"]
    if ext == "flac":
        return ["-vn", "-acodec", "flac"]
    if ext == "opus":
        return ["-vn", "-acodec", "libopus", "-b:a", f"{bitrate}k"]
    return ["-vn"]


def cut_video(
    input_video: str | Path,
    output_video: str | Path,
    start: float,
    end: float,
    video_codec: str = "copy",
) -> Path:
    ffmpeg_bin = require_binary("ffmpeg")
    output = Path(output_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.001, end - start)
    base = [
        ffmpeg_bin, "-y",
        "-i", str(input_video),
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-map", "0:v:0?",
        "-map", "0:a:0?",
    ]
    commands = [base + args + ["-c:a", "copy", "-avoid_negative_ts", "make_zero", str(output)]
                for args in video_encode_arg_candidates(ffmpeg_bin, video_codec)]
    run_command_with_fallback(commands)
    return output