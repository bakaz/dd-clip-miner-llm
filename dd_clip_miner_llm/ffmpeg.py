from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache
from os import devnull
from pathlib import Path
from uuid import uuid4


class FFmpegError(RuntimeError):
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


def run_command(args: list[str], timeout: int = 3600) -> None:
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
        raise FFmpegError(f"Command failed: {' '.join(args)}\n{detail}")


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
) -> Path:
    """拼接多个视频文件
    
    策略：
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
        shutil.copy2(str(input_videos[0]), str(output))
        return output

    expected_duration = _sum_media_durations(input_videos)
    
    # 创建 concat 列表文件
    concat_file = output.parent / "concat_list.txt"
    with concat_file.open("w", encoding="utf-8") as f:
        for video in input_videos:
            # FFmpeg concat 需要转义单引号
            escaped_path = str(video).replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")

    errors: list[str] = []
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
            _safe_unlink(concat_file)
            return output
        except FFmpegError as exc:
            errors.append(str(exc))
            print(
                "[concat] Targeted repair from quick probe failed; "
                f"{_short_error(exc)}"
            )
            print("[concat] Scanning all segments for additional corrupt H.264 packets.")
            try:
                _concat_reencoded_bad_segments_copy(
                    input_videos,
                    output,
                    ffmpeg_bin,
                    video_codec,
                    audio_bitrate_kbps,
                    expected_duration,
                )
                _safe_unlink(concat_file)
                return output
            except FFmpegError as full_scan_exc:
                errors.append(str(full_scan_exc))
                print(
                    "[concat] Full targeted repair failed; "
                    f"{_short_error(full_scan_exc)}"
                )
    
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
    
    if quick_bad_indexes:
        copy_error = "Skipped direct stream copy because corrupt H.264 segment(s) were detected."
        errors.append(copy_error)
    else:
        try:
            run_command(copy_cmd)
            _validate_concat_duration(output, expected_duration)
            _safe_unlink(concat_file)
            return output
        except FFmpegError as exc:
            copy_error = str(exc)
            errors.append(copy_error)
            print("[concat] Direct stream copy failed or produced a short output; trying targeted repair.")

    try:
        _concat_reencoded_bad_segments_copy(
            input_videos,
            output,
            ffmpeg_bin,
            video_codec,
            audio_bitrate_kbps,
            expected_duration,
        )
        _safe_unlink(concat_file)
        return output
    except FFmpegError as exc:
        targeted_reencode_error = str(exc)
        errors.append(targeted_reencode_error)

    try:
        _concat_remuxed_copy(
            input_videos,
            output,
            ffmpeg_bin,
            expected_duration,
        )
        _safe_unlink(concat_file)
        return output
    except FFmpegError as exc:
        remux_error = str(exc)
        errors.append(remux_error)

    # 获取最小分辨率，重编码时统一缩放到这个尺寸，避免不同源视频拼接失败。
    target_size = _get_min_video_size(input_videos)
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
            "-movflags", "+faststart",
            str(output),
        ]
        
        try:
            run_command(cmd)
            _validate_concat_duration(output, expected_duration)
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
    tolerance = max(2.0, expected_duration * 0.0005)
    if actual_duration + tolerance < expected_duration:
        raise FFmpegError(
            "Concat output duration is too short: "
            f"{actual_duration:.3f}s, expected about {expected_duration:.3f}s"
        )


def _find_bad_h264_segments(
    input_videos: list[str | Path],
    ffmpeg_bin: str,
    tail_seconds: float | None = None,
) -> list[int]:
    bad_indexes: list[int] = []
    suspicious = re.compile(
        r"Invalid NAL|missing picture|bitstream filters|Invalid data",
        re.IGNORECASE,
    )
    for index, video in enumerate(input_videos):
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
                "h264_mp4toannexb",
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
        if completed.returncode != 0 or suspicious.search(stderr):
            bad_indexes.append(index)
    return bad_indexes


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
                    if index not in bad_index_set:
                        repaired.append(video)
                        continue

                    target = candidate_dir / f"part_{index:04d}.mp4"
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
                        *encode_args,
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-b:a",
                        f"{audio_bitrate_kbps}k",
                        "-movflags",
                        "+faststart",
                        str(target),
                    ])
                    repaired.append(target)

                with concat_file.open("w", encoding="utf-8") as handle:
                    for video in repaired:
                        escaped_path = str(video).replace("'", "'\\''")
                        handle.write(f"file '{escaped_path}'\n")

                run_command([
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
                ])
                _validate_concat_duration(output, expected_duration)
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
    candidates = _video_reencode_arg_candidates(ffmpeg_bin, video_codec)
    cpu = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    if cpu not in candidates:
        candidates.append(cpu)
    return candidates


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

        run_command([
            ffmpeg_bin, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-map", "0:v:0?",
            "-map", "0:a:0?",
            "-c", "copy",
            "-movflags", "+faststart",
            str(output),
        ])
        _validate_concat_duration(output, expected_duration)
    finally:
        _safe_rmtree(temp_dir)


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
        return [["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]]
    if codec == "nv":
        return [["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19"]]
    if codec == "intel":
        return [["-c:v", "h264_qsv", "-global_quality", "20"]]
    if codec == "amd":
        return [["-c:v", "h264_amf", "-quality", "quality", "-qp_i", "20", "-qp_p", "20", "-qp_b", "20"]]

    # auto: 先尝试 copy，再按 nv > intel > amd > cpu 尝试重编码
    encoders = detect_video_encoders(ffmpeg_bin)
    candidates: list[list[str]] = [["-c:v", "copy"]]
    if "h264_nvenc" in encoders:
        candidates.append(["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19"])
    if "h264_qsv" in encoders:
        candidates.append(["-c:v", "h264_qsv", "-global_quality", "20"])
    if "h264_amf" in encoders:
        candidates.append(["-c:v", "h264_amf", "-quality", "quality", "-qp_i", "20", "-qp_p", "20", "-qp_b", "20"])
    candidates.append(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"])
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


def _parse_ffmpeg_duration(text: str) -> float | None:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds
