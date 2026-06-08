from __future__ import annotations

import re

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


def text_indicates_bitstream_corruption(text: str | list[str] | None) -> bool:
    """Return True if the ffmpeg output text contains strong signals of video bitstream corruption.
    Used both for pre-flight probes (_find_bad_h264_segments) and to decide fallback strategy,
    and also to promote warnings->errors on concat copy commands (even when ffmpeg rc==0)."""
    if not text:
        return False
    if isinstance(text, list):
        text = "\n".join(text)
    return any(pat.search(text) for pat in _BITSTREAM_CORRUPTION_RES)
def looks_like_video_bitstream_error(errors: list[str] | str) -> bool:
    """Legacy wrapper kept for compatibility with older call sites."""
    return text_indicates_bitstream_corruption(errors)
