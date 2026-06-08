from __future__ import annotations

from .command import require_binary
from .compat import pkg_attr


def targeted_repair_encode_candidates(
    ffmpeg_bin: str,
    video_codec: str,
) -> list[list[str]]:
    encoders = pkg_attr("detect_video_encoders")(ffmpeg_bin)
    candidates: list[list[str]] = []
    if "h264_nvenc" in encoders:
        candidates.append(["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "24"])
    if "h264_qsv" in encoders:
        candidates.append(["-c:v", "h264_qsv", "-global_quality", "24"])
    if "h264_amf" in encoders:
        candidates.append(["-c:v", "h264_amf", "-quality", "quality", "-qp_i", "24", "-qp_p", "24", "-qp_b", "24"])
    candidates.append(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"])
    return candidates


def concat_filter_complex(input_count: int, target_size: tuple[int, int] | None) -> str:
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


def repair_video_filter_args(fps: float | None) -> list[str]:
    if fps and fps > 0:
        fps_text = f"{fps:.6f}".rstrip("0").rstrip(".")
        return ["-vf", f"fps={fps_text},setpts=N/({fps_text}*TB)"]
    return ["-vf", "setpts=PTS-STARTPTS"]


def concat_scale_args(target_size: tuple[int, int] | None) -> list[str]:
    if target_size is None:
        return []

    width, height = target_size
    width = max(2, width - (width % 2))
    height = max(2, height - (height % 2))
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        "setsar=1"
    )
    return ["-vf", vf]


def video_encode_arg_candidates(ffmpeg_bin: str, video_codec: str = "copy") -> list[list[str]]:
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

    encoders = pkg_attr("detect_video_encoders")(ffmpeg_bin)
    candidates: list[list[str]] = [["-c:v", "copy"]]
    if "h264_nvenc" in encoders:
        candidates.append(["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "24"])
    if "h264_qsv" in encoders:
        candidates.append(["-c:v", "h264_qsv", "-global_quality", "24"])
    if "h264_amf" in encoders:
        candidates.append(["-c:v", "h264_amf", "-quality", "quality", "-qp_i", "24", "-qp_p", "24", "-qp_b", "24"])
    candidates.append(["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"])
    return candidates


def concat_reencode_arg_candidates(ffmpeg_bin: str, video_codec: str = "auto") -> list[list[str]]:
    return video_reencode_arg_candidates(ffmpeg_bin, video_codec)


def video_reencode_arg_candidates(ffmpeg_bin: str, video_codec: str = "auto") -> list[list[str]]:
    if (video_codec or "auto").lower() == "copy":
        video_codec = "auto"
    return [
        args for args in video_encode_arg_candidates(ffmpeg_bin, video_codec)
        if args[:2] != ["-c:v", "copy"]
    ]