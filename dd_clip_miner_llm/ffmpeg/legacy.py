from __future__ import annotations

from pathlib import Path

from .command import run_command
from .concat_ops import (
    concat_audio_reencoded_copy,
    concat_filter_commands,
    concat_reencoded_bad_segments_copy,
    concat_reencode_arg_candidates,
    concat_remuxed_copy,
    concat_scale_args,
)
from .diagnosis import analyze_ffmpeg_failure, find_bad_h264_segments
from .encode import concat_reencode_arg_candidates
from .errors import FFmpegError
from .fsutil import safe_unlink, short_error
from .validation import get_min_video_size, sum_media_durations
from .single_input import handle_single_input
from .validation import validate_audio_decodable, validate_concat_duration

def concat_videos_legacy(
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
        return handle_single_input(
            input_videos[0],
            output,
            ffmpeg_bin,
            video_codec,
            audio_bitrate_kbps,
            single_file_policy,
        )

    expected_duration = sum_media_durations(input_videos)
    target_size = get_min_video_size(input_videos)
    
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
        validate_concat_duration(output, expected_duration)
        validate_audio_decodable(output, ffmpeg_bin)
        safe_unlink(concat_file)
        return output
    except FFmpegError as exc:
        copy_error = str(exc)
        errors.append(copy_error)
        print("[concat] Direct stream copy failed or produced invalid output; trying audio-only re-encode.")

    # Keep video streams untouched when only AAC/timestamp continuity is bad.
    try:
        concat_audio_reencoded_copy(
            output,
            concat_file,
            ffmpeg_bin,
            audio_bitrate_kbps,
            expected_duration,
        )
        safe_unlink(concat_file)
        return output
    except FFmpegError as exc:
        audio_reencode_error = str(exc)
        errors.append(audio_reencode_error)
        print("[concat] Audio-only re-encode failed; trying remux.")

    try:
        concat_remuxed_copy(
            input_videos,
            output,
            ffmpeg_bin,
            expected_duration,
        )
        validate_audio_decodable(output, ffmpeg_bin)
        safe_unlink(concat_file)
        return output
    except FFmpegError as exc:
        remux_error = str(exc)
        errors.append(remux_error)

    # 获取最小分辨率，重编码时统一缩放到这个尺寸，避免不同源视频拼接失败。
    analysis = analyze_ffmpeg_failure(errors)
    if analysis.get("bitstream_corruption"):
        quick_bad_indexes = find_bad_h264_segments(
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
                concat_reencoded_bad_segments_copy(
                    input_videos,
                    output,
                    ffmpeg_bin,
                    video_codec,
                    audio_bitrate_kbps,
                    expected_duration,
                    bad_indexes=quick_bad_indexes,
                )
                validate_audio_decodable(output, ffmpeg_bin)
                safe_unlink(concat_file)
                return output
            except FFmpegError as exc:
                errors.append(str(exc))
                print(
                    "[concat] Targeted repair from tail scan failed; "
                    f"{short_error(exc)}"
                )

        print("[concat] Scanning all segments for corrupt H.264 packets.")
        try:
            concat_reencoded_bad_segments_copy(
                input_videos,
                output,
                ffmpeg_bin,
                video_codec,
                audio_bitrate_kbps,
                expected_duration,
            )
            validate_audio_decodable(output, ffmpeg_bin)
            safe_unlink(concat_file)
            return output
        except FFmpegError as exc:
            errors.append(str(exc))
            print(
                "[concat] Full targeted repair failed; "
                f"{short_error(exc)}"
            )

    scale_args = concat_scale_args(target_size)
    
    encode_candidates = concat_reencode_arg_candidates(ffmpeg_bin, video_codec)
    
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
            validate_concat_duration(output, expected_duration)
            validate_audio_decodable(output, ffmpeg_bin)
            safe_unlink(concat_file)
            return output
        except FFmpegError as e:
            errors.append(str(e))
            continue

    # Some MP4s make the concat demuxer exit successfully while silently
    # truncating after a later segment. The concat filter decodes each input
    # separately, so it is slower but handles those timestamp discontinuities.
    for cmd in concat_filter_commands(
        input_videos,
        output,
        ffmpeg_bin,
        video_codec,
        target_size,
        audio_bitrate_kbps,
    ):
        try:
            run_command(cmd, timeout=7200)
            validate_concat_duration(output, expected_duration)
            validate_audio_decodable(output, ffmpeg_bin)
            safe_unlink(concat_file)
            return output
        except FFmpegError as e:
            errors.append(str(e))
            continue
    
    safe_unlink(concat_file)
    raise FFmpegError(f"All concat attempts failed:\n" + "\n".join(errors))
