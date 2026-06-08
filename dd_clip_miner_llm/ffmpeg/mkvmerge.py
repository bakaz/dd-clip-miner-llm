from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from .command import run_command, require_binary
from .errors import FFmpegError
from .fsutil import safe_rmtree
from .probe import get_video_fps
from .validation import validate_audio_decodable, validate_concat_duration

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
    expected_duration: float | None = None,
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
        run_mkvmerge_single(mkvmerge_bin, input_videos[0], output)
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
                run_mkvmerge_fix(mkvmerge_bin, video, fixed)
            except FFmpegError:
                # 备用 fallback：加 --default-duration
                print(f"[mkvmerge] Segment {i+1} fix failed, retrying with --default-duration fallback")
                run_mkvmerge_fix_with_default_duration(mkvmerge_bin, video, fixed)
            fixed_parts.append(fixed)

        # Step 2: mkvmerge append 合并成 MKV
        merged_mkv = temp_dir / "merged.mkv"
        try:
            run_mkvmerge_append(mkvmerge_bin, fixed_parts, merged_mkv)
        except FFmpegError:
            # 备用 fallback：加 --default-duration
            print("[mkvmerge] Append failed, retrying with --default-duration fallback")
            run_mkvmerge_append_with_default_duration(mkvmerge_bin, fixed_parts, merged_mkv)

        # Step 3: ffmpeg -c copy 转回 MP4
        run_ffmpeg_copy(ffmpeg_bin, merged_mkv, output, expected_duration)

        return output
    finally:
        safe_rmtree(temp_dir)


def fps_to_mkvmerge_duration(fps: float) -> str:
    """将 FPS 转换为 mkvmerge --default-duration 格式（如 '60.000fps'）。"""
    return f"{fps:.3f}fps"


def run_mkvmerge_single(mkvmerge_bin: str, input_video: str | Path, output: Path) -> None:
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
    run_mkvmerge(cmd)


def run_mkvmerge_fix(mkvmerge_bin: str, input_video: str | Path, output: Path) -> None:
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
    run_mkvmerge(cmd)


def run_mkvmerge_append(mkvmerge_bin: str, inputs: list[Path], output: Path) -> None:
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
    run_mkvmerge(cmd)


def run_mkvmerge_fix_with_default_duration(mkvmerge_bin: str, input_video: str | Path, output: Path) -> None:
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
    fps = get_video_fps(input_video)
    if fps and fps > 0:
        cmd.extend(["--default-duration", f"0:{fps_to_mkvmerge_duration(fps)}"])
    cmd.extend(["--fix-bitstream-timing-information", "0:1", str(input_video)])
    run_mkvmerge(cmd)


def run_mkvmerge_append_with_default_duration(mkvmerge_bin: str, inputs: list[Path], output: Path) -> None:
    """用 mkvmerge append 模式拼接多个 MKV 文件，带 --default-duration 备用 fallback。"""
    cmd = [
        mkvmerge_bin,
        "-o", str(output),
    ]
    if inputs:
        fps = get_video_fps(inputs[0])
        if fps and fps > 0:
            cmd.extend(["--default-duration", f"0:{fps_to_mkvmerge_duration(fps)}"])
    cmd.extend(["--fix-bitstream-timing-information", "0:1"])
    for i, inp in enumerate(inputs):
        if i > 0:
            cmd.append("+")
        cmd.append(str(inp))
    run_mkvmerge(cmd)


def run_mkvmerge(cmd: list[str]) -> None:
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


def run_ffmpeg_copy(
    ffmpeg_bin: str,
    input_video: Path,
    output: Path,
    expected_duration: float | None = None,
) -> None:
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
    validate_concat_duration(output, expected_duration)
    validate_audio_decodable(output, ffmpeg_bin)
