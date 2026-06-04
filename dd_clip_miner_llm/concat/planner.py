from __future__ import annotations

from .models import TargetProfile, VideoMeta

# 这些辅助函数仍被新 ConcatPipeline / Strategy 使用（health probe 后计算 target profile 等）。


def expected_duration(metas: list[VideoMeta]) -> float | None:
    total = 0.0
    for meta in metas:
        if meta.duration is None:
            return None
        total += meta.duration
    return total


def build_target_profile(
    metas: list[VideoMeta],
    audio_bitrate_kbps: int,
) -> TargetProfile:
    video_metas = [m for m in metas if m.has_video and m.width and m.height]
    if video_metas:
        target = min(video_metas, key=lambda m: int(m.width or 0) * int(m.height or 0))
        width = target.width
        height = target.height
    else:
        width = None
        height = None
    fps_values = [round(m.fps, 3) for m in metas if m.fps]
    fps = min(fps_values) if fps_values else None
    return TargetProfile(
        width=width,
        height=height,
        fps=fps,
        audio_bitrate_kbps=audio_bitrate_kbps,
    )


def can_direct_concat_copy(metas: list[VideoMeta]) -> bool:
    if not metas or any(not m.probe_ok for m in metas):
        return False
    if any(not m.has_video for m in metas):
        return False
    return (
        _same(m.video_codec for m in metas)
        and _same(m.width for m in metas)
        and _same(m.height for m in metas)
        and _same(round(m.fps, 3) if m.fps else None for m in metas)
        and _same(m.pix_fmt for m in metas)
        and _same(m.sar for m in metas)
        and _same(m.has_audio for m in metas)
        and _same(m.audio_codec for m in metas)
        and _same(m.audio_sample_rate for m in metas)
        and _same(m.audio_channels for m in metas)
    )


def file_matches_profile(meta: VideoMeta, target: TargetProfile) -> bool:
    if not meta.probe_ok or not meta.has_video:
        return False
    video_ok = (
        meta.video_codec == "h264"
        and meta.width == target.width
        and meta.height == target.height
        and meta.pix_fmt == "yuv420p"
        and (meta.sar in (None, "1:1"))
    )
    audio_ok = (
        meta.has_audio
        and meta.audio_codec == target.audio_codec
        and meta.audio_sample_rate == target.audio_sample_rate
        and meta.audio_channels == target.audio_channels
    )
    return video_ok and audio_ok


def _same(values: object) -> bool:
    items = list(values)
    return len(set(items)) <= 1
