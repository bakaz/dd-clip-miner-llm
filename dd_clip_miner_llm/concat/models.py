from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    duration: float | None
    has_video: bool
    has_audio: bool
    video_codec: str | None
    width: int | None
    height: int | None
    fps: float | None
    pix_fmt: str | None
    sar: str | None
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None
    audio_layout: str | None
    audio_bit_rate: int | None = None
    probe_ok: bool = True
    error: str | None = None


@dataclass(frozen=True)
class TargetProfile:
    width: int | None
    height: int | None
    fps: float | None
    audio_codec: str = "aac"
    audio_bitrate_kbps: int = 320
    audio_sample_rate: int = 48000
    audio_channels: int = 2


@dataclass
class ConcatAttempt:
    name: str
    ok: bool
    detail: str = ""


# === New structured types for the refactored output-driven concat pipeline ===

@dataclass(frozen=True)
class HealthInfo:
    """Per-input health from upfront probe (probe + bitstream scan)."""
    path: Path
    probe_ok: bool
    duration: float | None
    has_video: bool
    has_audio: bool
    is_bitstream_corrupt: bool = False
    corrupt_details: str = ""
    error: str | None = None


@dataclass
class ProblemProfile:
    """Structured diagnosis derived from ffmpeg output (and health probes).
    This is the core of '依据ffmpeg的输出，判断是什么问题'.
    """
    bitstream_corrupt_indexes: list[int] = None  # 0-based indexes of bad parts
    bitstream_corruption: bool = False
    demux_errors: bool = False
    timestamp_discontinuity: bool = False
    duration_truncated: bool = False
    audio_decode_fail: bool = False
    hw_unavailable: bool = False
    profile_mismatch: bool = False
    summary: str = "unknown"
    raw_snippets: list[str] = None

    def __post_init__(self):
        if self.bitstream_corrupt_indexes is None:
            self.bitstream_corrupt_indexes = []
        if self.raw_snippets is None:
            self.raw_snippets = []

    def is_bitstream_problem(self) -> bool:
        return self.bitstream_corruption or bool(self.bitstream_corrupt_indexes)

    def merge(self, other: "ProblemProfile") -> "ProblemProfile":
        """Merge another profile (e.g. from a new attempt failure)."""
        merged = ProblemProfile(
            bitstream_corrupt_indexes=sorted(set(self.bitstream_corrupt_indexes) | set(other.bitstream_corrupt_indexes)),
            bitstream_corruption=self.bitstream_corruption or other.bitstream_corruption,
            demux_errors=self.demux_errors or other.demux_errors,
            timestamp_discontinuity=self.timestamp_discontinuity or other.timestamp_discontinuity,
            duration_truncated=self.duration_truncated or other.duration_truncated,
            audio_decode_fail=self.audio_decode_fail or other.audio_decode_fail,
            hw_unavailable=self.hw_unavailable or other.hw_unavailable,
            profile_mismatch=self.profile_mismatch or other.profile_mismatch,
            summary=self.summary if self.summary != "unknown" else other.summary,
            raw_snippets=(self.raw_snippets or []) + (other.raw_snippets or []),
        )
        return merged


@dataclass
class AttemptRecord:
    """Rich record of one strategy attempt, including full ffmpeg output for diagnosis."""
    name: str
    ok: bool
    detail: str = ""
    profile: ProblemProfile | None = None
    log_path: Path | None = None  # saved full raw log for post-mortem


@dataclass
class ConcatContext:
    """Shared state passed through the pipeline and strategies."""
    inputs: list[Path]
    metas: list[VideoMeta]
    output: Path
    ffmpeg_bin: str
    video_codec: str
    audio_bitrate_kbps: int
    single_file_policy: str
    force_normalize: bool
    expected_duration: float | None
    target: "TargetProfile"  # from planner
    target_size: tuple[int, int] | None
    health: dict[int, HealthInfo] | None = None
    profile: ProblemProfile | None = None
    attempts: list[AttemptRecord] = None
    concat_file: Path | None = None
    sanitized_inputs: dict[int, Path] | None = None  # pre-sanitized versions of corrupt segments (using safe remux)
    ts_inputs: dict[int, Path] | None = None  # reusable TS working copies created by successful transmux steps
    original_inputs: list[Path] | None = None  # kept for reference if needed

    def __post_init__(self):
        if self.attempts is None:
            self.attempts = []
        if self.sanitized_inputs is None:
            self.sanitized_inputs = {}
        if self.ts_inputs is None:
            self.ts_inputs = {}
        if self.original_inputs is None:
            self.original_inputs = list(self.inputs) if self.inputs else []
