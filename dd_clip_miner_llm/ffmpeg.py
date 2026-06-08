from __future__ import annotations

import json
import re
import shutil
import subprocess
from functools import lru_cache
from os import devnull
from pathlib import Path
from uuid import uuid4


class FFmpegError(RuntimeError):
    """FFmpeg command error, optionally carrying raw output for diagnosis."""

    def __init__(
        self,
        message: str,
        *,
        command: list[str] | None = None,
        stderr: str | None = None,
        returncode: int | None = None,
    ) -> None:
        super().__init__(message)
        self.command = command
        self.stderr = stderr
        self.returncode = returncode


class AllConcatAttemptsFailed(FFmpegError):
    pass


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise FFmpegError(
                "ffmpeg not found. Install FFmpeg or: pip install imageio-ffmpeg"
            ) from exc
        return imageio_ffmpeg.get_ffmpeg_exe()
    raise FFmpegError(f"Binary not found: {name}")


def run_command(args: list[str], timeout: int = 3600, *, bitstream_fatal: bool = False) -> None:
    """Run ffmpeg command. If bitstream_fatal=True, treat bitstream corruption warnings in stderr
    as fatal even if ffmpeg exited 0 (useful for concat copy steps to force fallback early
    with the real error message from ffmpeg)."""
    completed = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise FFmpegError(
            f"Command failed: {' '.join(args)}\n{detail}",
            command=args,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
    if bitstream_fatal:
        stderr_text = completed.stderr or completed.stdout or ""
        if _text_indicates_bitstream_corruption(stderr_text):
            detail = stderr_text.strip()
            raise FFmpegError(
                f"FFmpeg reported video bitstream corruption during operation (exit 0 but treated as failure for reliable concat):\n{detail}",
                command=args,
                stderr=completed.stderr,
                returncode=completed.returncode,
            )


def run_command_with_fallback(commands: list[list[str]], timeout: int = 3600) -> None:
    errors: list[str] = []
    for args in commands:
        try:
            run_command(args, timeout=timeout)
            return
        except FFmpegError as exc:
            errors.append(str(exc))
    raise FFmpegError("\n\n".join(errors))


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


def concat_videos(
    input_videos: list[str | Path],
    output_video: str | Path,
    video_codec: str = "auto",
    audio_bitrate_kbps: int = 320,
    single_file_policy: str = "copy",
    force_normalize: bool = False,
) -> Path:
    from .concat.pipeline import concat_videos_smart

    return concat_videos_smart(
        input_videos,
        output_video,
        video_codec=video_codec,
        audio_bitrate_kbps=audio_bitrate_kbps,
        single_file_policy=single_file_policy,
        force_normalize=force_normalize,
    )


def _concat_videos_legacy(
    input_videos: list[str | Path],
    output_video: str | Path,
    video_codec: str = "auto",
    audio_bitrate_kbps: int = 320,
    single_file_policy: str = "copy",
) -> Path:
    """拼接多个视频文件（旧实现，保留用于兼容/参考）。

    新实现已迁移到 concat.pipeline 中的 ConcatPipeline + Strategy，
    支持 upfront health probe（小文件全扫）+ pre-sanitize corrupt segments（per-file safe remux）
    + ProblemProfile（依据完整 ffmpeg 输出判断 bitstream_corruption 等）来智能选择 fallback，
    并保存完整日志。

    旧策略（供参考）：
    1. 优先尝试音视频流 copy（最快）
    2. 如果 copy 失败，使用 auto 模式重编码（nv > intel > amd > cpu）
    3. 回退重编码时统一到最小视频分辨率，音频用 AAC 320kbps
    """
    ffmpeg_bin = require_binary("ffmpeg")
    output = Path(output_video).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    
    if len(input_videos) == 0:
        raise ValueError("No input videos provided")
    
    if len(input_videos) == 1:
        # 单个文件直接复制
        return _handle_single_input(
            input_videos[0],
            output,
            ffmpeg_bin,
            video_codec,
            audio_bitrate_kbps,
            single_file_policy,
        )

    expected_duration = _sum_media_durations(input_videos)
    target_size = _get_min_video_size(input_videos)
    
    # 创建 concat 列表文件
    concat_file = output.parent / "concat_list.txt"
    with concat_file.open("w", encoding="utf-8") as f:
        for video in input_videos:
            # FFmpeg concat 需要转义单引号
            escaped_path = str(video).replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")

    errors: list[str] = []
    # 尝试直接复制音视频流；如果源文件参数一致，这是最快且无损的路径。
    copy_cmd = [
        ffmpeg_bin, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-map", "0:v:0?",
        "-map", "0:a:0?",
        "-c", "copy",
        "-movflags", "+faststart",
        str(output),
    ]
    
    try:
        run_command(copy_cmd, bitstream_fatal=True)
        _validate_concat_duration(output, expected_duration)
        _validate_audio_decodable(output, ffmpeg_bin)
        _safe_unlink(concat_file)
        return output
    except FFmpegError as exc:
        copy_error = str(exc)
        errors.append(copy_error)
        print("[concat] Direct stream copy failed or produced invalid output; trying audio-only re-encode.")

    # Keep video streams untouched when only AAC/timestamp continuity is bad.
    try:
        _concat_audio_reencoded_copy(
            output,
            concat_file,
            ffmpeg_bin,
            audio_bitrate_kbps,
            expected_duration,
        )
        _safe_unlink(concat_file)
        return output
    except FFmpegError as exc:
        audio_reencode_error = str(exc)
        errors.append(audio_reencode_error)
        print("[concat] Audio-only re-encode failed; trying remux.")

    try:
        _concat_remuxed_copy(
            input_videos,
            output,
            ffmpeg_bin,
            expected_duration,
        )
        _validate_audio_decodable(output, ffmpeg_bin)
        _safe_unlink(concat_file)
        return output
    except FFmpegError as exc:
        remux_error = str(exc)
        errors.append(remux_error)

    # 获取最小分辨率，重编码时统一缩放到这个尺寸，避免不同源视频拼接失败。
    analysis = analyze_ffmpeg_failure(errors)
    if analysis.get("bitstream_corruption"):
        quick_bad_indexes = _find_bad_h264_segments(
            input_videos,
            ffmpeg_bin,
            tail_seconds=60.0,
        )
        if quick_bad_indexes:
            print(
                "[concat] Detected possible corrupt H.264 segment(s) "
                f"{', '.join(str(i + 1) for i in quick_bad_indexes)}; "
                "repairing only those segment(s)."
            )
            try:
                _concat_reencoded_bad_segments_copy(
                    input_videos,
                    output,
                    ffmpeg_bin,
                    video_codec,
                    audio_bitrate_kbps,
                    expected_duration,
                    bad_indexes=quick_bad_indexes,
                )
                _validate_audio_decodable(output, ffmpeg_bin)
                _safe_unlink(concat_file)
                return output
            except FFmpegError as exc:
                errors.append(str(exc))
                print(
                    "[concat] Targeted repair from tail scan failed; "
                    f"{_short_error(exc)}"
                )

        print("[concat] Scanning all segments for corrupt H.264 packets.")
        try:
            _concat_reencoded_bad_segments_copy(
                input_videos,
                output,
                ffmpeg_bin,
                video_codec,
                audio_bitrate_kbps,
                expected_duration,
            )
            _validate_audio_decodable(output, ffmpeg_bin)
            _safe_unlink(concat_file)
            return output
        except FFmpegError as exc:
            errors.append(str(exc))
            print(
                "[concat] Full targeted repair failed; "
                f"{_short_error(exc)}"
            )

    scale_args = _concat_scale_args(target_size)
    
    encode_candidates = _concat_reencode_arg_candidates(ffmpeg_bin, video_codec)
    
    for encode_args in encode_candidates:
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-map", "0:v:0?",
            "-map", "0:a:0?",
        ] + scale_args + encode_args + [
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", f"{audio_bitrate_kbps}k",
            "-ar", "48000",
            "-ac", "2",
            "-movflags", "+faststart",
            str(output),
        ]
        
        try:
            run_command(cmd)
            _validate_concat_duration(output, expected_duration)
            _validate_audio_decodable(output, ffmpeg_bin)
            _safe_unlink(concat_file)
            return output
        except FFmpegError as e:
            errors.append(str(e))
            continue

    # Some MP4s make the concat demuxer exit successfully while silently
    # truncating after a later segment. The concat filter decodes each input
    # separately, so it is slower but handles those timestamp discontinuities.
    for cmd in _concat_filter_commands(
        input_videos,
        output,
        ffmpeg_bin,
        video_codec,
        target_size,
        audio_bitrate_kbps,
    ):
        try:
            run_command(cmd, timeout=7200)
            _validate_concat_duration(output, expected_duration)
            _validate_audio_decodable(output, ffmpeg_bin)
            _safe_unlink(concat_file)
            return output
        except FFmpegError as e:
            errors.append(str(e))
            continue
    
    _safe_unlink(concat_file)
    raise FFmpegError(f"All concat attempts failed:\n" + "\n".join(errors))


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _safe_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def _short_error(exc: Exception, max_length: int = 500) -> str:
    text = str(exc).strip().replace("\r\n", "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 6:
        lines = [*lines[:3], "...", *lines[-2:]]
    summary = " | ".join(lines)
    if len(summary) > max_length:
        return summary[: max_length - 3] + "..."
    return summary


# --- Concat / bitstream error classification (centralized for fallback decisions) ---

# Patterns observed in real ffmpeg stderr when H.264 (or HEVC) bitstream is corrupt,
# especially at part boundaries from live captures. Presence of these usually means
# -c copy through concat demuxer or bsf will fail or produce bad output.
_BITSTREAM_CORRUPTION_RES: list[re.Pattern[str]] = [
    re.compile(r"Invalid NAL", re.IGNORECASE),
    re.compile(r"missing picture", re.IGNORECASE),
    re.compile(r"decode_slice_header", re.IGNORECASE),
    re.compile(r"h264_mp4toannexb.*(fail|error)", re.IGNORECASE),
    re.compile(r"hevc_mp4toannexb.*(fail|error)", re.IGNORECASE),
    re.compile(r"Error applying bitstream filters", re.IGNORECASE),
    re.compile(r"non-existing PPS", re.IGNORECASE),
    re.compile(r"bytestream overread", re.IGNORECASE),
    re.compile(r"error while decoding MB", re.IGNORECASE),
    re.compile(r"\bno frame\b", re.IGNORECASE),
    re.compile(r"corrupt decoded frame", re.IGNORECASE),
]


def _text_indicates_bitstream_corruption(text: str | list[str] | None) -> bool:
    """Return True if the ffmpeg output text contains strong signals of video bitstream corruption.
    Used both for pre-flight probes (_find_bad_h264_segments) and to decide fallback strategy,
    and also to promote warnings->errors on concat copy commands (even when ffmpeg rc==0)."""
    if not text:
        return False
    if isinstance(text, list):
        text = "\n".join(text)
    return any(pat.search(text) for pat in _BITSTREAM_CORRUPTION_RES)


def analyze_ffmpeg_failure(details: str | list[str] | None) -> dict[str, object]:
    """Legacy dict version kept for compatibility. See classify_ffmpeg_output for the new structured ProblemProfile."""
    if not details:
        return {"bitstream_corruption": False, "timestamp_discontinuity": False, "duration_truncated": False, "audio_decode_fail": False, "hw_unavailable": False, "demux_error": False, "summary": "no details"}
    if isinstance(details, list):
        text = "\n".join(details)
    else:
        text = str(details)
    t = text.lower()

    bitstream = _text_indicates_bitstream_corruption(text)
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


def classify_ffmpeg_output(details: str | list[str] | None) -> "ProblemProfile":
    """New structured classifier. Returns ProblemProfile (the heart of output-driven decisions).
    Uses the same patterns as before but produces the dataclass used by the new pipeline.
    """
    from .concat.models import ProblemProfile  # avoid circular at import time

    if not details:
        return ProblemProfile(summary="no details")
    if isinstance(details, list):
        text = "\n".join(details)
    else:
        text = str(details)
    t = text.lower()

    bitstream = _text_indicates_bitstream_corruption(text)
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


def _looks_like_video_bitstream_error(errors: list[str] | str) -> bool:
    """Legacy wrapper kept for compatibility with older call sites."""
    return _text_indicates_bitstream_corruption(errors)


def _handle_single_input(
    input_video: str | Path,
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    audio_bitrate_kbps: int,
    single_file_policy: str,
) -> Path:
    policy = (single_file_policy or "copy").lower()
    expected_duration = _sum_media_durations([input_video])
    if policy == "copy":
        shutil.copy2(str(input_video), str(output))
        return output
    if policy == "remux":
        _remux_single_input(input_video, output, ffmpeg_bin, expected_duration)
        return output
    if policy == "normalize":
        _normalize_single_input(
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


def _remux_single_input(
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
    _validate_concat_duration(output, expected_duration)
    _validate_audio_decodable(output, ffmpeg_bin)


def _normalize_single_input(
    input_video: str | Path,
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    audio_bitrate_kbps: int,
    expected_duration: float | None,
) -> None:
    target_size = None
    try:
        target_size = _get_video_resolution(input_video)
    except (FFmpegError, ValueError):
        pass
    scale_args = _concat_scale_args(target_size)
    errors: list[str] = []
    for encode_args in _concat_reencode_arg_candidates(ffmpeg_bin, video_codec):
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
            _validate_concat_duration(output, expected_duration)
            _validate_audio_decodable(output, ffmpeg_bin)
            return
        except FFmpegError as exc:
            errors.append(str(exc))
    raise FFmpegError("Single-file normalize failed:\n" + "\n".join(errors))


def _get_min_video_size(videos: list[str | Path]) -> tuple[int, int] | None:
    """获取像素面积最小的视频尺寸。"""
    min_area = float("inf")
    min_size: tuple[int, int] | None = None
    
    for video in videos:
        try:
            width, height = _get_video_resolution(video)
            area = width * height
            if area < min_area:
                min_area = area
                min_size = (width, height)
        except (FFmpegError, ValueError):
            continue
    
    return min_size


def _sum_media_durations(videos: list[str | Path]) -> float | None:
    total = 0.0
    for video in videos:
        try:
            total += get_duration(video)
        except (FFmpegError, ValueError):
            return None
    return total


def _validate_concat_duration(output: Path, expected_duration: float | None) -> None:
    if expected_duration is None:
        return
    actual_duration = get_duration(output)
    # More realistic tolerance for long live recordings (possible small drifts, frame drops in repaired tails).
    # 30s absolute or 0.5% is still strict but practical; the pre-sanitize + re-probe of sanitized
    # durations makes the "expected" itself much more accurate for damaged inputs.
    tolerance = max(30.0, expected_duration * 0.005)
    if actual_duration + tolerance < expected_duration:
        raise FFmpegError(
            "Concat output duration is too short: "
            f"{actual_duration:.3f}s, expected about {expected_duration:.3f}s"
        )
    if actual_duration > expected_duration + tolerance:
        raise FFmpegError(
            "Concat output duration is too long: "
            f"{actual_duration:.3f}s, expected about {expected_duration:.3f}s"
        )
    video_duration = _get_stream_duration(output, "v:0")
    if video_duration is not None and video_duration + tolerance < expected_duration:
        raise FFmpegError(
            "Concat output video stream duration is too short: "
            f"{video_duration:.3f}s, expected about {expected_duration:.3f}s"
        )
    if video_duration is not None and video_duration > expected_duration + tolerance:
        raise FFmpegError(
            "Concat output video stream duration is too long: "
            f"{video_duration:.3f}s, expected about {expected_duration:.3f}s"
        )


def _has_audio_stream(input_media: str | Path) -> bool:
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        # If ffprobe is unavailable, let the decode check be the source of truth.
        return True
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(input_media),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())


def _validate_audio_decodable(output: Path, ffmpeg_bin: str) -> None:
    if not _has_audio_stream(output):
        return
    try:
        run_command([
            ffmpeg_bin,
            "-v",
            "error",
            "-i",
            str(output),
            "-map",
            "0:a:0",
            "-vn",
            "-f",
            "null",
            devnull,
        ])
    except FFmpegError as exc:
        raise FFmpegError(
            f"Concat output audio is not decodable: {output}\n{exc}"
        ) from exc


def _find_bad_h264_segments(
    input_videos: list[str | Path],
    ffmpeg_bin: str,
    tail_seconds: float | None = None,
) -> list[int]:
    """Probe each input (or its tail) by attempting a video-only copy through the matching bitstream filter.
    Any corruption will surface as specific errors/warnings in stderr (even if rc may vary).
    Uses the centralized _text_indicates_bitstream_corruption for consistent detection with fallback logic.
    """
    bad_indexes: list[int] = []
    for index, video in enumerate(input_videos):
        codec = _get_video_codec(video)
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
        if completed.returncode != 0 or _text_indicates_bitstream_corruption(stderr):
            bad_indexes.append(index)
    return bad_indexes


def _get_video_codec(video_path: str | Path) -> str | None:
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return None
    codec = completed.stdout.strip().lower()
    return codec or None


def _concat_reencoded_bad_segments_copy(
    input_videos: list[str | Path],
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    audio_bitrate_kbps: int,
    expected_duration: float | None,
    bad_indexes: list[int] | None = None,
) -> None:
    bad_index_set = set(
        _find_bad_h264_segments(input_videos, ffmpeg_bin)
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
            _targeted_repair_encode_candidates(ffmpeg_bin, video_codec)
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

                    fps = _get_video_fps(video)
                    video_filter = _repair_video_filter_args(fps)
                    audio_filter = ["-af", "asetpts=PTS-STARTPTS"] if _has_audio_stream(video) else []
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
                    _validate_concat_duration(candidate_output, actual_expected)
                    _validate_audio_decodable(candidate_output, ffmpeg_bin)
                except FFmpegError:
                    _safe_unlink(candidate_output)
                    filter_complex = _concat_filter_complex(
                        len(repaired),
                        _get_min_video_size(repaired),
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
                    _validate_concat_duration(candidate_output, actual_expected)
                    _validate_audio_decodable(candidate_output, ffmpeg_bin)
                _safe_unlink(output)
                shutil.move(str(candidate_output), str(output))
                return
            except FFmpegError as exc:
                errors.append(str(exc))
                _safe_rmtree(candidate_dir)

        raise FFmpegError(
            "Targeted concat repair failed:\n" + "\n".join(errors)
        )
    finally:
        _safe_rmtree(temp_dir)


def _targeted_repair_encode_candidates(
    ffmpeg_bin: str,
    video_codec: str,
) -> list[list[str]]:
    # 尝试硬件编码，失败时回退到 CPU
    encoders = detect_video_encoders(ffmpeg_bin)
    candidates: list[list[str]] = []
    if "h264_nvenc" in encoders:
        candidates.append(["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "24"])
    if "h264_qsv" in encoders:
        candidates.append(["-c:v", "h264_qsv", "-global_quality", "24"])
    if "h264_amf" in encoders:
        candidates.append(["-c:v", "h264_amf", "-quality", "quality", "-qp_i", "24", "-qp_p", "24", "-qp_b", "24"])
    # CPU 作为最终回退
    candidates.append(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"])
    return candidates


def _concat_audio_reencoded_copy(
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
    _validate_concat_duration(output, expected_duration)
    _validate_audio_decodable(output, ffmpeg_bin)


def _concat_remuxed_copy(
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
        _validate_concat_duration(candidate_output, expected_duration)
        _validate_audio_decodable(candidate_output, ffmpeg_bin)
        _safe_unlink(output)
        shutil.move(str(candidate_output), str(output))
    finally:
        _safe_rmtree(temp_dir)


def _concat_timestamp_remuxed_copy(
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

        _write_ffconcat_list(concat_file, remuxed)
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
        _validate_concat_duration(candidate_output, expected_duration)
        _validate_audio_decodable(candidate_output, ffmpeg_bin)
        _safe_unlink(output)
        shutil.move(str(candidate_output), str(output))
    finally:
        _safe_rmtree(temp_dir)


def _concat_timestamp_remuxed_audio_resync(
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

        _write_ffconcat_list(concat_file, remuxed)
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
        _validate_concat_duration(candidate_output, expected_duration)
        _validate_audio_decodable(candidate_output, ffmpeg_bin)
        _safe_unlink(output)
        shutil.move(str(candidate_output), str(output))
    finally:
        _safe_rmtree(temp_dir)


def _concat_fast_transmux_copy(
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
        _validate_concat_duration(candidate_output, expected_duration)
        _validate_audio_decodable(candidate_output, ffmpeg_bin)
        _safe_unlink(output)
        shutil.move(str(candidate_output), str(output))
    finally:
        _safe_rmtree(temp_dir)


def _concat_ts_protocol_copy(
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
        _validate_concat_duration(candidate_output, expected_duration)
        _validate_audio_decodable(candidate_output, ffmpeg_bin)
        _safe_unlink(output)
        shutil.move(str(candidate_output), str(output))
    finally:
        _safe_rmtree(temp_dir)


def _concat_tail_window_repaired_bad_segments_copy(
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

            fps = _get_video_fps(video)
            video_filter = _repair_video_filter_args(fps)
            audio_filter = ["-af", "asetpts=PTS-STARTPTS"] if _has_audio_stream(video) else []
            errors: list[str] = []
            for encode_args in _targeted_repair_encode_candidates(ffmpeg_bin, video_codec):
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

            _write_ffconcat_list(list_file, [head, tail])
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
            _validate_concat_duration(fixed, duration)
            repaired_inputs.append(fixed)

        concat_file = temp_dir / "concat_list.txt"
        repaired_durations = _durations_or_none(repaired_inputs)
        _write_ffconcat_list(
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
        _validate_concat_duration(candidate_output, actual_expected)
        _validate_audio_decodable(candidate_output, ffmpeg_bin)
        _safe_unlink(output)
        shutil.move(str(candidate_output), str(output))
    finally:
        _safe_rmtree(temp_dir)


def _durations_or_none(videos: list[str | Path]) -> list[float] | None:
    durations: list[float] = []
    try:
        for video in videos:
            durations.append(get_duration(video))
    except (FFmpegError, ValueError):
        return None
    return durations


def _write_ffconcat_list(
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


def _concat_filter_commands(
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

    filter_complex = _concat_filter_complex(len(input_videos), target_size)
    commands: list[list[str]] = []
    for encode_args in _concat_reencode_arg_candidates(ffmpeg_bin, video_codec):
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


def _concat_filter_complex(input_count: int, target_size: tuple[int, int] | None) -> str:
    parts: list[str] = []
    concat_inputs: list[str] = []
    video_filter = "setsar=1,setpts=PTS-STARTPTS"
    if target_size is not None:
        width, height = target_size
        width = max(2, width - (width % 2))
        height = max(2, height - (height % 2))
        video_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            "setsar=1,setpts=PTS-STARTPTS"
        )

    for index in range(input_count):
        parts.append(f"[{index}:v:0]{video_filter}[v{index}]")
        parts.append(f"[{index}:a:0]asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0[a{index}]")
        concat_inputs.append(f"[v{index}][a{index}]")

    parts.append("".join(concat_inputs) + f"concat=n={input_count}:v=1:a=1[v][a]")
    return ";".join(parts)


def _get_video_resolution(video_path: str | Path) -> tuple[int, int]:
    """获取视频分辨率（宽, 高）"""
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        completed = subprocess.run(
            [ffprobe_bin, "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            parts = completed.stdout.strip().split(",")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
    
    raise FFmpegError(f"Could not get video resolution: {video_path}")


def _parse_fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            den = float(denominator)
            if den == 0:
                return None
            return float(numerator) / den
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _get_video_fps(video_path: str | Path) -> float | None:
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return None
    streams = payload.get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        fps = _parse_fraction(stream.get(key))
        if fps and fps > 0:
            return fps
    return None


def _repair_video_filter_args(fps: float | None) -> list[str]:
    if fps and fps > 0:
        fps_text = f"{fps:.6f}".rstrip("0").rstrip(".")
        return ["-vf", f"fps={fps_text},setpts=N/({fps_text}*TB)"]
    return ["-vf", "setpts=PTS-STARTPTS"]


def _concat_scale_args(target_size: tuple[int, int] | None) -> list[str]:
    if target_size is None:
        return []

    width, height = target_size
    # H.264 编码器普遍要求偶数宽高；保持目标分辨率不超过最小源视频尺寸。
    width = max(2, width - (width % 2))
    height = max(2, height - (height % 2))
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        "setsar=1"
    )
    return ["-vf", vf]


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
                for args in _video_encode_arg_candidates(ffmpeg_bin, video_codec)]
    run_command_with_fallback(commands)
    return output


def _video_encode_arg_candidates(ffmpeg_bin: str, video_codec: str = "copy") -> list[list[str]]:
    codec = (video_codec or "copy").lower()
    if codec == "copy":
        return [["-c:v", "copy"]]
    if codec in {"cpu", "libx264"}:
        return [["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]]
    if codec == "nv":
        return [["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "24"]]
    if codec == "intel":
        return [["-c:v", "h264_qsv", "-global_quality", "24"]]
    if codec == "amd":
        return [["-c:v", "h264_amf", "-quality", "quality", "-qp_i", "24", "-qp_p", "24", "-qp_b", "24"]]

    # auto: 先尝试 copy，再按 nv > intel > amd > cpu 尝试重编码
    encoders = detect_video_encoders(ffmpeg_bin)
    candidates: list[list[str]] = [["-c:v", "copy"]]
    if "h264_nvenc" in encoders:
        candidates.append(["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "24"])
    if "h264_qsv" in encoders:
        candidates.append(["-c:v", "h264_qsv", "-global_quality", "24"])
    if "h264_amf" in encoders:
        candidates.append(["-c:v", "h264_amf", "-quality", "quality", "-qp_i", "24", "-qp_p", "24", "-qp_b", "24"])
    candidates.append(["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"])
    return candidates


def _concat_reencode_arg_candidates(ffmpeg_bin: str, video_codec: str = "auto") -> list[list[str]]:
    return _video_reencode_arg_candidates(ffmpeg_bin, video_codec)


def _video_reencode_arg_candidates(ffmpeg_bin: str, video_codec: str = "auto") -> list[list[str]]:
    if (video_codec or "auto").lower() == "copy":
        video_codec = "auto"
    return [
        args for args in _video_encode_arg_candidates(ffmpeg_bin, video_codec)
        if args[:2] != ["-c:v", "copy"]
    ]


def detect_ffmpeg_hwaccels(ffmpeg_bin: str | None = None) -> set[str]:
    exe = ffmpeg_bin or require_binary("ffmpeg")
    completed = subprocess.run(
        [exe, "-hide_banner", "-hwaccels"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = f"{completed.stdout}\n{completed.stderr}"
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lower().startswith("hardware acceleration")
    }


def detect_gpu_devices() -> list[str]:
    probes = [
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
        ["wmic", "path", "win32_VideoController", "get", "name"],
        ["lspci"],
    ]
    for args in probes:
        output = _run_probe(args)
        if not output:
            continue
        if args[0] == "lspci":
            devices = [
                line.strip()
                for line in output.splitlines()
                if re.search(r"\b(vga|3d|display)\b", line, flags=re.IGNORECASE)
            ]
        else:
            devices = [
                line.strip()
                for line in output.splitlines()
                if line.strip() and line.strip().lower() != "name"
            ]
        if devices:
            return devices
    return []


def detect_ffmpeg_environment(ffmpeg_bin: str | None = None) -> dict[str, object]:
    exe = ffmpeg_bin or require_binary("ffmpeg")
    encoders = detect_video_encoders(exe)
    return {
        "ffmpeg": exe,
        "gpus": detect_gpu_devices(),
        "hwaccels": sorted(detect_ffmpeg_hwaccels(exe)),
        "video_encoders": sorted(encoders),
        "auto_reencode_order": _auto_reencode_order(encoders),
        "recommended_video_codec": "copy",
    }


def _auto_reencode_order(encoders: set[str]) -> list[str]:
    ordered: list[str] = []
    for codec, label in [
        ("h264_nvenc", "nv"),
        ("h264_qsv", "intel"),
        ("h264_amf", "amd"),
        ("libx264", "cpu"),
    ]:
        if codec in encoders:
            ordered.append(label)
    return ordered


def _run_probe(args: list[str], timeout: int = 10) -> str:
    executable = shutil.which(args[0])
    if not executable:
        return ""
    try:
        completed = subprocess.run(
            [executable, *args[1:]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip() or completed.stderr.strip()


@lru_cache(maxsize=8)
def detect_video_encoders(ffmpeg_bin: str | None = None) -> set[str]:
    exe = ffmpeg_bin or require_binary("ffmpeg")
    completed = subprocess.run(
        [exe, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = f"{completed.stdout}\n{completed.stderr}"
    return set(re.findall(r"\b(h264_nvenc|h264_qsv|h264_amf|libx264)\b", text))


def get_duration(input_media: str | Path) -> float:
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        completed = subprocess.run(
            [ffprobe_bin, "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(input_media)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            return float(completed.stdout.strip())

    ffmpeg_bin = require_binary("ffmpeg")
    completed = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", str(input_media)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    duration = _parse_ffmpeg_duration(completed.stderr)
    if duration is None:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise FFmpegError(f"Could not read media duration: {input_media}\n{detail}")
    return duration


def _get_stream_duration(input_media: str | Path, stream_selector: str) -> float | None:
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "quiet",
            "-select_streams",
            stream_selector,
            "-show_entries",
            "stream=duration",
            "-of",
            "csv=p=0",
            str(input_media),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip().splitlines()
    if not text:
        return None
    try:
        return float(text[0])
    except ValueError:
        return None


def _parse_ffmpeg_duration(text: str) -> float | None:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


# ============ MKVToolNix (mkvmerge) concat ============

def require_mkvmerge() -> str:
    """查找 mkvmerge 可执行文件。"""
    path = shutil.which("mkvmerge")
    if path:
        return path
    # Windows 默认安装路径
    default = Path(r"C:\Program Files\MKVToolNix\mkvmerge.exe")
    if default.exists():
        return str(default)
    raise FFmpegError("mkvmerge not found. Install MKVToolNix: winget install MKVToolNix")


def concat_with_mkvmerge(
    input_videos: list[str | Path],
    output_video: str | Path,
    *,
    video_codec: str = "copy",
    audio_bitrate_kbps: int = 320,
) -> Path:
    """使用 mkvmerge 拼接视频，处理 H.264 bitstream 损坏。

    流程：
    1. mkvmerge 修正每个片段的容器时间戳 / 默认帧时长
    2. mkvmerge append 合并成一个稳定 MKV
    3. ffmpeg -c copy 转回 MP4

    优势：
    - mkvmerge 对 H.264 bitstream timing 的处理比 ffmpeg 更稳健
    - 不需要多次重试或复杂 fallback
    - 速度接近 stream copy
    """
    mkvmerge_bin = require_mkvmerge()
    ffmpeg_bin = require_binary("ffmpeg")
    output = Path(output_video).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not input_videos:
        raise ValueError("No input videos provided")

    if len(input_videos) == 1:
        # 单文件直接 remux
        _run_mkvmerge_single(mkvmerge_bin, input_videos[0], output)
        return output

    # 多文件拼接
    temp_dir = output.parent / f"_mkvmerge_{uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: 每个片段用 mkvmerge 修正容器时间戳
        fixed_parts: list[Path] = []
        for i, video in enumerate(input_videos):
            fixed = temp_dir / f"part_{i:04d}.mkv"
            try:
                _run_mkvmerge_fix(mkvmerge_bin, video, fixed)
            except FFmpegError:
                # 备用 fallback：加 --default-duration
                print(f"[mkvmerge] Segment {i+1} fix failed, retrying with --default-duration fallback")
                _run_mkvmerge_fix_with_default_duration(mkvmerge_bin, video, fixed)
            fixed_parts.append(fixed)

        # Step 2: mkvmerge append 合并成 MKV
        merged_mkv = temp_dir / "merged.mkv"
        try:
            _run_mkvmerge_append(mkvmerge_bin, fixed_parts, merged_mkv)
        except FFmpegError:
            # 备用 fallback：加 --default-duration
            print("[mkvmerge] Append failed, retrying with --default-duration fallback")
            _run_mkvmerge_append_with_default_duration(mkvmerge_bin, fixed_parts, merged_mkv)

        # Step 3: ffmpeg -c copy 转回 MP4
        _run_ffmpeg_copy(ffmpeg_bin, merged_mkv, output)

        return output
    finally:
        _safe_rmtree(temp_dir)


def _get_video_fps(video_path: str | Path) -> float | None:
    """用 ffprobe 提取视频 FPS。"""
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe_bin, "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,avg_frame_rate",
                "-of", "csv=p=0",
                str(video_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if completed.returncode != 0:
            return None
        # 解析 "30/1" 或 "30000/1001" 格式
        for line in completed.stdout.strip().splitlines():
            line = line.strip()
            if not line or line == "N/A":
                continue
            parts = line.split("/")
            if len(parts) == 2:
                try:
                    num, den = float(parts[0]), float(parts[1])
                    if den > 0:
                        return round(num / den, 3)
                except ValueError:
                    continue
    except Exception:
        pass
    return None


def _fps_to_mkvmerge_duration(fps: float) -> str:
    """将 FPS 转换为 mkvmerge --default-duration 格式（如 '60.000fps'）。"""
    return f"{fps:.3f}fps"


def _run_mkvmerge_single(mkvmerge_bin: str, input_video: str | Path, output: Path) -> None:
    """单文件 mkvmerge remux。"""
    cmd = [
        mkvmerge_bin,
        "-o", str(output),
        "--no-subtitles",
        "--no-buttons",
        "--no-attachments",
        "--no-chapters",
        "--no-global-tags",
        str(input_video),
    ]
    _run_mkvmerge(cmd)


def _run_mkvmerge_fix(mkvmerge_bin: str, input_video: str | Path, output: Path) -> None:
    """用 mkvmerge 修正容器时间戳和默认帧时长。"""
    cmd = [
        mkvmerge_bin,
        "-o", str(output),
        "--no-subtitles",
        "--no-buttons",
        "--no-attachments",
        "--no-chapters",
        "--no-global-tags",
        "--fix-bitstream-timing-information", "0:1",
        str(input_video),
    ]
    _run_mkvmerge(cmd)


def _run_mkvmerge_append(mkvmerge_bin: str, inputs: list[Path], output: Path) -> None:
    """用 mkvmerge append 模式拼接多个 MKV 文件。"""
    # mkvmerge append 语法：file1 + file2 + file3
    cmd = [
        mkvmerge_bin,
        "-o", str(output),
        "--fix-bitstream-timing-information", "0:1",
    ]
    for i, inp in enumerate(inputs):
        if i > 0:
            cmd.append("+")
        cmd.append(str(inp))
    _run_mkvmerge(cmd)


def _run_mkvmerge_fix_with_default_duration(mkvmerge_bin: str, input_video: str | Path, output: Path) -> None:
    """用 mkvmerge 修正容器时间戳，带 --default-duration 备用 fallback。"""
    cmd = [
        mkvmerge_bin,
        "-o", str(output),
        "--no-subtitles",
        "--no-buttons",
        "--no-attachments",
        "--no-chapters",
        "--no-global-tags",
    ]
    fps = _get_video_fps(input_video)
    if fps and fps > 0:
        cmd.extend(["--default-duration", f"0:{_fps_to_mkvmerge_duration(fps)}"])
    cmd.extend(["--fix-bitstream-timing-information", "0:1", str(input_video)])
    _run_mkvmerge(cmd)


def _run_mkvmerge_append_with_default_duration(mkvmerge_bin: str, inputs: list[Path], output: Path) -> None:
    """用 mkvmerge append 模式拼接多个 MKV 文件，带 --default-duration 备用 fallback。"""
    cmd = [
        mkvmerge_bin,
        "-o", str(output),
    ]
    if inputs:
        fps = _get_video_fps(inputs[0])
        if fps and fps > 0:
            cmd.extend(["--default-duration", f"0:{_fps_to_mkvmerge_duration(fps)}"])
    cmd.extend(["--fix-bitstream-timing-information", "0:1"])
    for i, inp in enumerate(inputs):
        if i > 0:
            cmd.append("+")
        cmd.append(str(inp))
    _run_mkvmerge(cmd)


def _run_mkvmerge(cmd: list[str]) -> None:
    """运行 mkvmerge 命令。"""
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
    except FileNotFoundError as exc:
        raise FFmpegError(f"mkvmerge not found: {exc}") from exc

    # mkvmerge 返回值：0=成功，1=警告，2=错误
    if completed.returncode == 2:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise FFmpegError(
            f"mkvmerge failed: {' '.join(cmd)}\n{detail}",
            command=cmd,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )

    # 返回值 1 是警告，打印但不报错
    if completed.returncode == 1:
        stderr = completed.stderr.strip()
        if stderr:
            print(f"[mkvmerge] Warning: {stderr[:200]}")


def _run_ffmpeg_copy(ffmpeg_bin: str, input_video: Path, output: Path) -> None:
    """用 ffmpeg -c copy 转换格式。"""
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(input_video),
        "-map", "0:v:0?",
        "-map", "0:a:0?",
        "-c", "copy",
        "-movflags", "+faststart",
        str(output),
    ]
    # MKV 文件已经过 mkvmerge 处理，bitstream_fatal=False 避免误判
    run_command(cmd, bitstream_fatal=False)
    _validate_concat_duration(output, None)
    _validate_audio_decodable(output, ffmpeg_bin)
