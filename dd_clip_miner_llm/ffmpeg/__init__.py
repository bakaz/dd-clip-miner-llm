"""FFmpeg helpers — split into focused submodules, re-exported here for compatibility."""
from __future__ import annotations

import subprocess

from .bitstream import (
    looks_like_video_bitstream_error,
    text_indicates_bitstream_corruption,
)
from .command import require_binary, run_command, run_command_with_fallback
from .concat_ops import (
    concat_audio_reencoded_copy,
    concat_fast_transmux_copy,
    concat_filter_commands,
    concat_reencoded_bad_segments_copy,
    concat_remuxed_copy,
    concat_tail_window_repaired_bad_segments_copy,
    concat_timestamp_remuxed_audio_resync,
    concat_timestamp_remuxed_copy,
    concat_ts_protocol_copy,
    durations_or_none,
    write_ffconcat_list,
)
from .diagnosis import (
    analyze_ffmpeg_failure,
    classify_ffmpeg_output,
    find_bad_h264_segments,
)
from .encode import (
    concat_filter_complex,
    concat_reencode_arg_candidates,
    concat_scale_args,
    repair_video_filter_args,
    targeted_repair_encode_candidates,
    video_encode_arg_candidates,
    video_reencode_arg_candidates,
)
from .errors import AllConcatAttemptsFailed, FFmpegError
from .fsutil import safe_rmtree, safe_unlink, short_error
from .legacy import concat_videos_legacy
from .media import cut_audio, cut_video, extract_audio
from .mkvmerge import concat_with_mkvmerge, require_mkvmerge, run_ffmpeg_copy
from .probe import (
    auto_reencode_order,
    detect_ffmpeg_environment,
    detect_ffmpeg_hwaccels,
    detect_gpu_devices,
    detect_video_encoders,
    get_duration,
    get_stream_duration,
    get_video_codec,
    get_video_fps,
    get_video_resolution,
)
from .single_input import handle_single_input, normalize_single_input, remux_single_input
from .validation import (
    get_min_video_size,
    has_audio_stream,
    sum_media_durations,
    validate_audio_decodable,
    validate_concat_duration,
)

# Backward-compatible private aliases (tests & concat modules monkeypatch these names)
_text_indicates_bitstream_corruption = text_indicates_bitstream_corruption
_looks_like_video_bitstream_error = looks_like_video_bitstream_error
_find_bad_h264_segments = find_bad_h264_segments
_get_video_codec = get_video_codec
_get_video_fps = get_video_fps
_get_stream_duration = get_stream_duration
_get_video_resolution = get_video_resolution
_get_min_video_size = get_min_video_size
_sum_media_durations = sum_media_durations
_validate_concat_duration = validate_concat_duration
_validate_audio_decodable = validate_audio_decodable
_has_audio_stream = has_audio_stream
_safe_unlink = safe_unlink
_safe_rmtree = safe_rmtree
_short_error = short_error
_handle_single_input = handle_single_input
_remux_single_input = remux_single_input
_normalize_single_input = normalize_single_input
_concat_videos_legacy = concat_videos_legacy
_concat_reencoded_bad_segments_copy = concat_reencoded_bad_segments_copy
_concat_tail_window_repaired_bad_segments_copy = concat_tail_window_repaired_bad_segments_copy
_concat_timestamp_remuxed_audio_resync = concat_timestamp_remuxed_audio_resync
_concat_audio_reencoded_copy = concat_audio_reencoded_copy
_concat_remuxed_copy = concat_remuxed_copy
_concat_timestamp_remuxed_copy = concat_timestamp_remuxed_copy
_concat_fast_transmux_copy = concat_fast_transmux_copy
_concat_ts_protocol_copy = concat_ts_protocol_copy
_write_ffconcat_list = write_ffconcat_list
_concat_filter_commands = concat_filter_commands
_concat_filter_complex = concat_filter_complex
_concat_scale_args = concat_scale_args
_concat_reencode_arg_candidates = concat_reencode_arg_candidates
_video_encode_arg_candidates = video_encode_arg_candidates
_video_reencode_arg_candidates = video_reencode_arg_candidates
_targeted_repair_encode_candidates = targeted_repair_encode_candidates
_repair_video_filter_args = repair_video_filter_args
_durations_or_none = durations_or_none
_run_ffmpeg_copy = run_ffmpeg_copy


def concat_videos(
    input_videos: list[str | Path],
    output_video: str | Path,
    video_codec: str = "auto",
    audio_bitrate_kbps: int = 320,
    single_file_policy: str = "copy",
    force_normalize: bool = False,
) -> Path:
    from pathlib import Path

    from ..concat.pipeline import concat_videos_smart

    return concat_videos_smart(
        input_videos,
        output_video,
        video_codec=video_codec,
        audio_bitrate_kbps=audio_bitrate_kbps,
        single_file_policy=single_file_policy,
        force_normalize=force_normalize,
    )


__all__ = [
    "AllConcatAttemptsFailed",
    "FFmpegError",
    "analyze_ffmpeg_failure",
    "classify_ffmpeg_output",
    "concat_videos",
    "concat_with_mkvmerge",
    "cut_audio",
    "cut_video",
    "detect_ffmpeg_environment",
    "detect_ffmpeg_hwaccels",
    "detect_gpu_devices",
    "detect_video_encoders",
    "extract_audio",
    "get_duration",
    "require_binary",
    "require_mkvmerge",
    "run_command",
    "run_command_with_fallback",
    "subprocess",
]
