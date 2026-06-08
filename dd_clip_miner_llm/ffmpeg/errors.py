from __future__ import annotations

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
