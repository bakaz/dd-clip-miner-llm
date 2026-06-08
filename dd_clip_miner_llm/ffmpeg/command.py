from __future__ import annotations

import shutil
import subprocess

from .bitstream import text_indicates_bitstream_corruption
from .errors import FFmpegError


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
        if text_indicates_bitstream_corruption(stderr_text):
            detail = stderr_text.strip()
            raise FFmpegError(
                "FFmpeg reported video bitstream corruption during operation (exit 0 but treated as failure for reliable concat):\n"
                f"{detail}",
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
