from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import re
from uuid import uuid4

from .. import ffmpeg as ffmpeg_mod
from .models import (
    AttemptRecord,
    ConcatAttempt,
    ConcatContext,
    HealthInfo,
    ProblemProfile,
    TargetProfile,
    VideoMeta,
)
from .planner import (
    build_target_profile,
    can_direct_concat_copy,
    expected_duration,
    file_matches_profile,
)
from .probe import probe_many, probe_one


def _build_health_profile(
    inputs: list[Path],
    metas: list[VideoMeta],
    ffmpeg_bin: str,
) -> dict[int, HealthInfo]:
    """Upfront health probe for all inputs. This is the key improvement:
    detect bitstream corruption *before* expensive copy attempts.
    """
    health: dict[int, HealthInfo] = {}
    h264_indexes = [
        index
        for index, meta in enumerate(metas)
        if meta.probe_ok and meta.video_codec == "h264"
    ]
    h264_inputs = [inputs[index] for index in h264_indexes]
    bad_set: set[int] = set()
    if h264_inputs:
        bad_relative_indexes = ffmpeg_mod._find_bad_h264_segments(
            h264_inputs,
            ffmpeg_bin,
            tail_seconds=60.0,
        )
        bad_set = {
            h264_indexes[index]
            for index in bad_relative_indexes
            if 0 <= index < len(h264_indexes)
        }

    for i, (inp, meta) in enumerate(zip(inputs, metas)):
        is_corrupt = i in bad_set
        corrupt_detail = ""
        if is_corrupt:
            # Re-run probe on tail for detail (or use previous)
            corrupt_detail = "corrupt tail detected by h264_mp4toannexb scan"

        health[i] = HealthInfo(
            path=inp,
            probe_ok=meta.probe_ok,
            duration=meta.duration,
            has_video=meta.has_video,
            has_audio=meta.has_audio,
            is_bitstream_corrupt=is_corrupt,
            corrupt_details=corrupt_detail,
            error=meta.error if not meta.probe_ok else None,
        )
    return health


class Strategy(ABC):
    """Base for a concat recovery strategy. Each knows when it is applicable
    based on current ProblemProfile + Context, and can execute (capturing full log).
    """
    name: str = "base"

    @abstractmethod
    def is_applicable(self, context: ConcatContext) -> bool:
        ...

    @abstractmethod
    def execute(self, context: ConcatContext) -> bool:
        """Run the strategy. Return True on success (after validate).
        On failure, should record AttemptRecord with full log into context.
        """
        ...

    def _save_log(self, context: ConcatContext, log_text: str) -> Path | None:
        try:
            log_dir = context.output.parent / "concat_attempts"
            log_dir.mkdir(parents=True, exist_ok=True)
            safe_name = _safe_attempt_name(self.name)
            log_path = _unique_path(log_dir / f"{safe_name}.log")
            log_path.write_text(log_text, encoding="utf-8", errors="replace")
            return log_path
        except Exception:
            return None

    def _save_named_log(self, context: ConcatContext, name: str, log_text: str) -> Path | None:
        try:
            log_dir = context.output.parent / "concat_attempts"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = _unique_path(log_dir / f"{_safe_attempt_name(name)}.log")
            log_path.write_text(log_text, encoding="utf-8", errors="replace")
            return log_path
        except Exception:
            return None

    def _record_failure(self, context: ConcatContext, exc: Exception) -> None:
        full_log = str(exc)
        profile = ffmpeg_mod.classify_ffmpeg_output(full_log)
        log_path = self._save_log(context, full_log)
        context.attempts.append(AttemptRecord(self.name, False, full_log, profile, log_path))
        if context.profile is None:
            context.profile = profile
        else:
            context.profile = context.profile.merge(profile)


class DirectCopyStrategy(Strategy):
    name = "direct concat copy"

    def is_applicable(self, context: ConcatContext) -> bool:
        if context.force_normalize:
            return False
        if any(not meta.probe_ok for meta in context.metas):
            return True
        return can_direct_concat_copy(context.metas)

    def execute(self, context: ConcatContext) -> bool:
        concat_file = context.concat_file or (context.output.parent / "concat_list.txt")
        _write_concat_list(concat_file, context.inputs)
        context.concat_file = concat_file

        try:
            ffmpeg_mod.run_command(
                [
                    context.ffmpeg_bin, "-y", "-f", "concat", "-safe", "0",
                    "-i", str(concat_file),
                    "-map", "0:v:0?", "-map", "0:a:0?",
                    "-c", "copy", "-movflags", "+faststart",
                    str(context.output),
                ],
                bitstream_fatal=True,
            )
            _validate_output(context.output, context.ffmpeg_bin, context.expected_duration)
            context.attempts.append(AttemptRecord(self.name, True, "ok"))
            return True
        except ffmpeg_mod.FFmpegError as exc:
            full_log = str(exc)
            profile = ffmpeg_mod.classify_ffmpeg_output(full_log)
            log_path = self._save_log(context, full_log)
            rec = AttemptRecord(self.name, False, full_log, profile, log_path)
            context.attempts.append(rec)
            if context.profile is None:
                context.profile = profile
            else:
                context.profile = context.profile.merge(profile)
            return False


class AudioReencodeStrategy(Strategy):
    name = "audio-only reencode"

    def is_applicable(self, context: ConcatContext) -> bool:
        if context.profile and context.profile.is_bitstream_problem():
            return False
        return True

    def execute(self, context: ConcatContext) -> bool:
        concat_file = context.concat_file or (context.output.parent / "concat_list.txt")
        _write_concat_list(concat_file, context.inputs)
        context.concat_file = concat_file

        try:
            ffmpeg_mod._concat_audio_reencoded_copy(
                context.output,
                concat_file,
                context.ffmpeg_bin,
                context.audio_bitrate_kbps,
                context.expected_duration,
            )
            context.attempts.append(AttemptRecord(self.name, True, "ok"))
            return True
        except ffmpeg_mod.FFmpegError as exc:
            full_log = str(exc)
            profile = ffmpeg_mod.classify_ffmpeg_output(full_log)
            log_path = self._save_log(context, full_log)
            rec = AttemptRecord(self.name, False, full_log, profile, log_path)
            context.attempts.append(rec)
            if context.profile is None:
                context.profile = profile
            else:
                context.profile = context.profile.merge(profile)
            return False


class TimestampIndexRemuxCopyStrategy(Strategy):
    name = "timestamp/index remux copy"

    def is_applicable(self, context: ConcatContext) -> bool:
        if context.force_normalize:
            return False
        profile = context.profile
        if profile and profile.is_bitstream_problem():
            return False
        if profile and (
            profile.timestamp_discontinuity
            or profile.duration_truncated
            or profile.summary in {"unknown", "unknown/other", "no details"}
        ):
            return True
        return True

    def execute(self, context: ConcatContext) -> bool:
        try:
            ffmpeg_mod._concat_timestamp_remuxed_copy(
                context.inputs,
                context.output,
                context.ffmpeg_bin,
                context.expected_duration,
            )
            context.attempts.append(AttemptRecord(self.name, True, "ok"))
            return True
        except ffmpeg_mod.FFmpegError as exc:
            self._record_failure(context, exc)
            return False


class TimestampIndexRemuxAudioResyncStrategy(Strategy):
    name = "timestamp/index remux + audio resync"

    def is_applicable(self, context: ConcatContext) -> bool:
        profile = context.profile
        if profile and profile.is_bitstream_problem():
            return False
        if profile and (
            profile.timestamp_discontinuity
            or profile.duration_truncated
            or profile.audio_decode_fail
        ):
            return True
        return True

    def execute(self, context: ConcatContext) -> bool:
        try:
            ffmpeg_mod._concat_timestamp_remuxed_audio_resync(
                context.inputs,
                context.output,
                context.ffmpeg_bin,
                context.audio_bitrate_kbps,
                context.expected_duration,
            )
            context.attempts.append(AttemptRecord(self.name, True, "ok"))
            return True
        except ffmpeg_mod.FFmpegError as exc:
            self._record_failure(context, exc)
            return False


class FastTransmuxStrategy(Strategy):
    name = "fast transmux (mp4-ts-mp4)"

    def is_applicable(self, context: ConcatContext) -> bool:
        if context.force_normalize:
            return False
        profile = context.profile
        if profile and profile.is_bitstream_problem():
            return False
        if any(meta.video_codec not in {None, "h264"} for meta in context.metas):
            return False
        return True

    def execute(self, context: ConcatContext) -> bool:
        try:
            ffmpeg_mod._concat_fast_transmux_copy(
                context.inputs,
                context.output,
                context.ffmpeg_bin,
                context.expected_duration,
            )
            context.attempts.append(AttemptRecord(self.name, True, "ok"))
            return True
        except ffmpeg_mod.FFmpegError as exc:
            self._record_failure(context, exc)
            return False


class RemuxThenCopyStrategy(Strategy):
    name = "remux then concat copy"

    def is_applicable(self, context: ConcatContext) -> bool:
        return True

    def execute(self, context: ConcatContext) -> bool:
        try:
            ffmpeg_mod._concat_remuxed_copy(
                context.inputs,
                context.output,
                context.ffmpeg_bin,
                context.expected_duration,
            )
            ffmpeg_mod._validate_audio_decodable(context.output, context.ffmpeg_bin)
            context.attempts.append(AttemptRecord(self.name, True, "ok"))
            return True
        except ffmpeg_mod.FFmpegError as exc:
            full_log = str(exc)
            profile = ffmpeg_mod.classify_ffmpeg_output(full_log)
            log_path = self._save_log(context, full_log)
            rec = AttemptRecord(self.name, False, full_log, profile, log_path)
            context.attempts.append(rec)
            if context.profile is None:
                context.profile = profile
            else:
                context.profile = context.profile.merge(profile)
            return False


class TargetedRepairStrategy(Strategy):
    """The smart one: uses health/profile to repair only bad segments."""
    name = "targeted H.264 repair"

    def is_applicable(self, context: ConcatContext) -> bool:
        if context.profile and context.profile.is_bitstream_problem():
            return True
        if context.health:
            return any(h.is_bitstream_corrupt for h in context.health.values())
        return False

    def execute(self, context: ConcatContext) -> bool:
        # Determine bad indexes from profile or health
        bad = []
        if context.profile and context.profile.bitstream_corrupt_indexes:
            bad = context.profile.bitstream_corrupt_indexes
        elif context.health:
            bad = [i for i, h in context.health.items() if h.is_bitstream_corrupt]

        if not bad:
            # Fallback to tail scan
            try:
                bad = ffmpeg_mod._find_bad_h264_segments(
                    context.inputs, context.ffmpeg_bin, tail_seconds=60.0
                )
            except Exception:
                bad = []
        if not bad and context.profile and context.profile.is_bitstream_problem():
            try:
                print("[concat] Tail scan found no corrupt segment; scanning full inputs for H.264 corruption.")
                bad = ffmpeg_mod._find_bad_h264_segments(
                    context.inputs, context.ffmpeg_bin, tail_seconds=None
                )
            except Exception:
                bad = []

        if not bad:
            # Record skip
            context.attempts.append(AttemptRecord(self.name, False, "skipped: no corrupt segments detected"))
            return False

        print(
            f"[concat] Detected possible corrupt H.264 segment(s) "
            f"{', '.join(str(i + 1) for i in bad)}; repairing only those segment(s)."
        )

        try:
            print("[concat] Trying fixed tail-window H.264 repair before full bad-segment reencode.")
            ffmpeg_mod._concat_tail_window_repaired_bad_segments_copy(
                context.inputs,
                context.output,
                context.ffmpeg_bin,
                context.video_codec,
                context.audio_bitrate_kbps,
                context.expected_duration,
                bad_indexes=bad,
                repair_window_seconds=90.0,
                guard_seconds=2.0,
            )
            _validate_output(context.output, context.ffmpeg_bin, context.expected_duration)
            context.attempts.append(AttemptRecord(f"{self.name} (tail window)", True, "ok"))
            return True
        except ffmpeg_mod.FFmpegError as exc:
            context.attempts.append(
                AttemptRecord(
                    f"{self.name} (tail window)",
                    False,
                    str(exc),
                    ffmpeg_mod.classify_ffmpeg_output(str(exc)),
                    self._save_log(context, str(exc)),
                )
            )

        try:
            ffmpeg_mod._concat_reencoded_bad_segments_copy(
                context.inputs,
                context.output,
                context.ffmpeg_bin,
                context.video_codec,
                context.audio_bitrate_kbps,
                context.expected_duration,
                bad_indexes=bad,
            )
            _validate_output(context.output, context.ffmpeg_bin, context.expected_duration)
            context.attempts.append(AttemptRecord(self.name, True, "ok"))
            return True
        except ffmpeg_mod.FFmpegError as exc:
            full_log = str(exc)
            profile = ffmpeg_mod.classify_ffmpeg_output(full_log)
            log_path = self._save_log(context, full_log)
            rec = AttemptRecord(self.name, False, full_log, profile, log_path)
            context.attempts.append(rec)
            if context.profile is None:
                context.profile = profile
            else:
                context.profile = context.profile.merge(profile)
            return False


# More strategies can be added (Selective, FullReencode, Filter) similarly.
# For full refactor we wrap the existing functions for the remaining fallbacks.


class SelectiveNormalizeStrategy(Strategy):
    name = "selective normalize"

    def is_applicable(self, context: ConcatContext) -> bool:
        return True

    def execute(self, context: ConcatContext) -> bool:
        # Delegates to the existing _selective_normalize_concat helper
        # (full port to pure Strategy can be done later if needed).
        metas = probe_many(context.inputs)
        target = context.target
        target_size = context.target_size
        force_indexes = {
            index
            for index, health in (context.health or {}).items()
            if health.is_bitstream_corrupt
        }

        try:
            # Call the existing helper (kept for compatibility during refactor)
            _selective_normalize_concat(
                context.inputs, metas, target, target_size,
                context.output, context.ffmpeg_bin, context.video_codec,
                context.audio_bitrate_kbps, context.expected_duration,
                force_indexes=force_indexes,
            )
            context.attempts.append(AttemptRecord(self.name, True, "ok"))
            return True
        except ffmpeg_mod.FFmpegError as exc:
            full_log = str(exc)
            profile = ffmpeg_mod.classify_ffmpeg_output(full_log)
            log_path = self._save_log(context, full_log)
            rec = AttemptRecord(self.name, False, full_log, profile, log_path)
            context.attempts.append(rec)
            if context.profile is None:
                context.profile = profile
            else:
                context.profile = context.profile.merge(profile)
            return False


class BoundaryNormalizeStrategy(Strategy):
    name = "structural duration boundary normalize"

    def is_applicable(self, context: ConcatContext) -> bool:
        if context.profile and context.profile.is_bitstream_problem():
            return False
        return bool(_duration_failure_boundary_indexes(context))

    def execute(self, context: ConcatContext) -> bool:
        indexes = _duration_failure_boundary_indexes(context)
        if not indexes:
            context.attempts.append(AttemptRecord(self.name, False, "skipped: no duration boundary detected"))
            return False
        print(
            "[concat] Structural duration failure appears near segment(s) "
            f"{', '.join(str(i + 1) for i in indexes)}; normalizing only boundary segment(s)."
        )
        try:
            _selective_normalize_concat(
                context.inputs,
                context.metas,
                context.target,
                context.target_size,
                context.output,
                context.ffmpeg_bin,
                context.video_codec,
                context.audio_bitrate_kbps,
                context.expected_duration,
                force_indexes=set(indexes),
            )
            context.attempts.append(AttemptRecord(self.name, True, "ok"))
            return True
        except ffmpeg_mod.FFmpegError as exc:
            self._record_failure(context, exc)
            return False


class FullReencodeStrategy(Strategy):
    name = "full reencode (demuxer + filter fallback)"

    def is_applicable(self, context: ConcatContext) -> bool:
        return True

    def execute(self, context: ConcatContext) -> bool:
        concat_file = context.concat_file or (context.output.parent / "concat_list.txt")
        _write_concat_list(concat_file, context.inputs)
        context.concat_file = concat_file

        scale_args = ffmpeg_mod._concat_scale_args(context.target_size)
        for candidate_index, encode_args in enumerate(
            ffmpeg_mod._concat_reencode_arg_candidates(context.ffmpeg_bin, context.video_codec)
        ):
            candidate = _candidate_output_path(context, f"{self.name}_demuxer", candidate_index)
            try:
                # Use existing demuxer reencode helper
                _concat_demuxer_full_reencode(
                    candidate, concat_file, context.ffmpeg_bin,
                    scale_args, encode_args, context.audio_bitrate_kbps,
                    context.expected_duration,
                )
                _commit_candidate_output(candidate, context.output)
                context.attempts.append(AttemptRecord(f"{self.name} (demuxer)", True, "ok"))
                return True
            except ffmpeg_mod.FFmpegError as exc:
                full_log = str(exc)
                profile = ffmpeg_mod.classify_ffmpeg_output(full_log)
                log_path = self._save_named_log(
                    context,
                    f"{self.name} demuxer candidate {candidate_index:02d}",
                    full_log,
                )
                context.attempts.append(AttemptRecord(f"{self.name} (demuxer)", False, full_log, profile, log_path))
                ffmpeg_mod._safe_unlink(candidate)
                _cleanup_empty_dir(candidate.parent)
                if context.profile is None:
                    context.profile = profile
                else:
                    context.profile = context.profile.merge(profile)
                if _is_output_duration_failure(full_log):
                    print(
                        "[concat] Demuxer full reencode produced invalid stream duration; "
                        "switching to concat filter fallback."
                    )
                    break

        # Last resort: concat filter (decodes everything, handles worst cases)
        for candidate_index, command in enumerate(ffmpeg_mod._concat_filter_commands(
            context.inputs, context.output, context.ffmpeg_bin,
            context.video_codec, context.target_size, context.audio_bitrate_kbps,
        )):
            candidate = _candidate_output_path(context, f"{self.name}_filter", candidate_index)
            command = [str(candidate) if arg == str(context.output) else arg for arg in command]
            try:
                ffmpeg_mod.run_command(command, timeout=7200)
                _validate_output(candidate, context.ffmpeg_bin, context.expected_duration)
                _commit_candidate_output(candidate, context.output)
                context.attempts.append(AttemptRecord(f"{self.name} (filter)", True, "ok"))
                return True
            except ffmpeg_mod.FFmpegError as exc:
                full_log = str(exc)
                profile = ffmpeg_mod.classify_ffmpeg_output(full_log)
                log_path = self._save_named_log(
                    context,
                    f"{self.name} filter candidate {candidate_index:02d}",
                    full_log,
                )
                context.attempts.append(AttemptRecord(f"{self.name} (filter)", False, full_log, profile, log_path))
                ffmpeg_mod._safe_unlink(candidate)
                _cleanup_empty_dir(candidate.parent)
                if context.profile is None:
                    context.profile = profile
                else:
                    context.profile = context.profile.merge(profile)
                continue

        return False


class ConcatPipeline:
    """Refactored output-driven concat pipeline (核心按 ffmpeg 输出判断问题来选 fallback)。

    流程：
    1. Upfront health probe（ffprobe + tail bitstream scan）得到 HealthInfo。
    2. 初始 ProblemProfile（来自 classify_ffmpeg_output）。
    3. 按 Strategy 列表尝试（每个 Strategy.is_applicable 基于当前 profile）。
    4. 失败时用完整 ffmpeg 输出重新 classify + merge profile，保存日志到 concat_attempts/。
    5. TargetedRepair 等策略会优先使用诊断出的坏段索引，只修坏的。

    保留了所有原有恢复路径，同时极大提升了可观测性和决策准确性。
    """
    def __init__(self):
        self.strategies: list[Strategy] = [
            DirectCopyStrategy(),
            TimestampIndexRemuxCopyStrategy(),
            AudioReencodeStrategy(),
            TimestampIndexRemuxAudioResyncStrategy(),
            FastTransmuxStrategy(),
            RemuxThenCopyStrategy(),
            TargetedRepairStrategy(),
            SelectiveNormalizeStrategy(),
            BoundaryNormalizeStrategy(),
            FullReencodeStrategy(),
        ]

    def run(self, context: ConcatContext) -> Path:
        # 1. Upfront health
        context.health = _build_health_profile(context.inputs, context.metas, context.ffmpeg_bin)

        # 2. Initial profile from health
        initial_profile = ProblemProfile()
        corrupt = [i for i, h in context.health.items() if h.is_bitstream_corrupt]
        if corrupt:
            initial_profile.bitstream_corrupt_indexes = corrupt
            initial_profile.bitstream_corruption = True
            initial_profile.summary = "bitstream_corruption (from upfront probe)"
        context.profile = initial_profile

        # 3. Run strategies
        concat_file = context.output.parent / "concat_list.txt"
        _write_concat_list(concat_file, context.inputs)
        context.concat_file = concat_file

        strategies = self.strategies
        if corrupt:
            print(
                "[concat] Upfront H.264 tail probe found corrupt segment(s) "
                f"{', '.join(str(i + 1) for i in corrupt)}; skipping copy-based concat attempts."
            )
            corrupt_ratio = _corrupt_duration_ratio(context, corrupt)
            count_ratio = len(corrupt) / max(1, len(context.inputs))
            should_skip_mixed = (
                corrupt_ratio > 0.5
                if corrupt_ratio is not None
                else count_ratio > 0.5
            )
            if should_skip_mixed:
                ratio_text = (
                    f"{corrupt_ratio:.2%} of input duration"
                    if corrupt_ratio is not None
                    else f"{count_ratio:.2%} of input files"
                )
                print(
                    f"[concat] Corrupt H.264 segment(s) cover {ratio_text}; skipping mixed copy repair paths "
                    "and going straight to full reencode."
                )
                strategies = [FullReencodeStrategy()]
            else:
                strategies = [
                    TargetedRepairStrategy(),
                    SelectiveNormalizeStrategy(),
                    BoundaryNormalizeStrategy(),
                    FullReencodeStrategy(),
                ]

        for strat in strategies:
            if not strat.is_applicable(context):
                continue
            print(f"[concat] Trying strategy: {strat.name}")
            success = strat.execute(context)
            if success:
                ffmpeg_mod._safe_unlink(concat_file)
                return context.output

        # All failed
        details = []
        for a in context.attempts:
            if not a.ok:
                d = a.detail[:700] + "..." if len(a.detail) > 700 else a.detail
                details.append(f"- {a.name}: {d}")
        raise ffmpeg_mod.AllConcatAttemptsFailed(
            "All concat attempts failed (new pipeline):\n" + "\n".join(details)
        )


def concat_videos_smart(
    input_videos: list[str | Path],
    output_video: str | Path,
    video_codec: str = "auto",
    audio_bitrate_kbps: int = 320,
    single_file_policy: str = "copy",
    force_normalize: bool = False,
) -> Path:
    ffmpeg_bin = ffmpeg_mod.require_binary("ffmpeg")
    inputs = [Path(video).resolve() for video in input_videos]
    output = Path(output_video).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not inputs:
        raise ValueError("No input videos provided")

    if len(inputs) == 1:
        return ffmpeg_mod._handle_single_input(
            inputs[0],
            output,
            ffmpeg_bin,
            video_codec,
            audio_bitrate_kbps,
            single_file_policy,
        )

    # Thin wrapper: builds ConcatContext and delegates to the refactored
    # ConcatPipeline (new output-driven architecture with Strategy + ProblemProfile).
    # Public API (concat_videos_smart) remains fully compatible.
    metas = probe_many(inputs)
    expected = expected_duration(metas)
    if expected is None:
        expected = ffmpeg_mod._sum_media_durations(inputs)
    target = build_target_profile(metas, audio_bitrate_kbps)
    target_size = _target_size(target) or ffmpeg_mod._get_min_video_size(inputs)

    context = ConcatContext(
        inputs=inputs,
        metas=metas,
        output=output,
        ffmpeg_bin=ffmpeg_bin,
        video_codec=video_codec,
        audio_bitrate_kbps=audio_bitrate_kbps,
        single_file_policy=single_file_policy,
        force_normalize=force_normalize,
        expected_duration=expected,
        target=target,
        target_size=target_size,
    )

    pipeline = ConcatPipeline()
    return pipeline.run(context)

    # (old internal ladder removed; _attempt / _has / _format_failures kept only for backward compat with existing tests)


def _attempt(
    attempts: list[ConcatAttempt],
    name: str,
    action,
) -> bool:
    try:
        action()
    except ffmpeg_mod.FFmpegError as exc:
        detail = str(exc)
        analysis = ffmpeg_mod.analyze_ffmpeg_failure(detail)
        diag = analysis.get("summary", "")
        if diag and diag != "unknown/other":
            # Include diagnosis in the stored detail so that later branch decisions (_has_*) and final error
            # message clearly show what ffmpeg output indicated (core of the fallback detection).
            detail = f"{detail}\n[diagnosed: {diag}]"
        attempts.append(ConcatAttempt(name, False, detail))
        # Print short reason on every attempt failure so live batch-run output shows the
        # actual ffmpeg error text (NAL, bitstream etc.) for easier debugging and optimization.
        short = ffmpeg_mod._short_error(exc)
        print(f"[concat]   -> {name} failed: {short}")
        return False
    attempts.append(ConcatAttempt(name, True, "ok"))
    return True


def _concat_copy_with_list(
    output: Path,
    concat_file: Path,
    ffmpeg_bin: str,
    expected_duration_value: float | None,
) -> None:
    # bitstream_fatal: capture real corruption messages from ffmpeg output (e.g. Invalid NAL, missing picture)
    # even if ffmpeg rc==0, so that fallback decision has accurate diagnosis from the output.
    ffmpeg_mod.run_command(
        [
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
        ],
        bitstream_fatal=True,
    )
    _validate_output(output, ffmpeg_bin, expected_duration_value)


def _remux_concat_copy(
    inputs: list[Path],
    output: Path,
    ffmpeg_bin: str,
    expected_duration_value: float | None,
) -> None:
    ffmpeg_mod._concat_remuxed_copy(
        inputs,
        output,
        ffmpeg_bin,
        expected_duration_value,
    )
    ffmpeg_mod._validate_audio_decodable(output, ffmpeg_bin)


def _remux_concat_audio_reencode(
    inputs: list[Path],
    output: Path,
    ffmpeg_bin: str,
    audio_bitrate_kbps: int,
    expected_duration_value: float | None,
) -> None:
    temp_dir, remuxed, remuxed_list = _remux_inputs(inputs, output.parent, ffmpeg_bin)
    try:
        ffmpeg_mod._concat_audio_reencoded_copy(
            output,
            remuxed_list,
            ffmpeg_bin,
            audio_bitrate_kbps,
            expected_duration_value,
        )
    finally:
        ffmpeg_mod._safe_unlink(remuxed_list)
        ffmpeg_mod._safe_rmtree(temp_dir)


def _remux_inputs(
    inputs: list[Path],
    parent: Path,
    ffmpeg_bin: str,
) -> tuple[Path, list[Path], Path]:
    temp_dir = parent / f"concat_remux_{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    remuxed: list[Path] = []
    try:
        for index, source in enumerate(inputs):
            target = temp_dir / f"{index:05d}.mp4"
            ffmpeg_mod.run_command([
                ffmpeg_bin,
                "-y",
                "-fflags",
                "+genpts",
                "-err_detect",
                "ignore_err",
                "-i",
                str(source),
                "-map",
                "0",
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                str(target),
            ])
            remuxed.append(target)
        remuxed_list = temp_dir / "concat_list.txt"
        _write_concat_list(remuxed_list, remuxed)
        return temp_dir, remuxed, remuxed_list
    except Exception:
        ffmpeg_mod._safe_rmtree(temp_dir)
        raise


def _attempt_tail_repair(
    attempts: list[ConcatAttempt],
    inputs: list[Path],
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    audio_bitrate_kbps: int,
    expected_duration_value: float | None,
) -> bool:
    try:
        bad_indexes = ffmpeg_mod._find_bad_h264_segments(
            inputs,
            ffmpeg_bin,
            tail_seconds=60.0,
        )
    except ffmpeg_mod.FFmpegError as exc:
        attempts.append(ConcatAttempt("H.264 tail scan localized reencode", False, str(exc)))
        return False
    if not bad_indexes:
        attempts.append(ConcatAttempt("H.264 tail scan localized reencode", False, "skipped: no corrupt tail segments detected"))
        return False
    print(
        "[concat] Detected possible corrupt H.264 tail segment(s) "
        f"{', '.join(str(i + 1) for i in bad_indexes)}; repairing only those segment(s)."
    )
    return _attempt(
        attempts,
        "H.264 tail scan localized reencode",
        lambda: _targeted_repair(
            inputs,
            output,
            ffmpeg_bin,
            video_codec,
            audio_bitrate_kbps,
            expected_duration_value,
            bad_indexes=bad_indexes,
        ),
    )


def _targeted_repair(
    inputs: list[Path],
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    audio_bitrate_kbps: int,
    expected_duration_value: float | None,
    bad_indexes: list[int] | None,
) -> None:
    ffmpeg_mod._concat_reencoded_bad_segments_copy(
        inputs,
        output,
        ffmpeg_bin,
        video_codec,
        audio_bitrate_kbps,
        expected_duration_value,
        bad_indexes=bad_indexes,
    )
    _validate_output(output, ffmpeg_bin, expected_duration_value)


def _selective_normalize_concat(
    inputs: list[Path],
    metas: list[VideoMeta],
    target: TargetProfile,
    target_size: tuple[int, int] | None,
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    audio_bitrate_kbps: int,
    expected_duration_value: float | None,
    force_indexes: set[int] | None = None,
) -> None:
    temp_dir = output.parent / f"concat_selective_{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    selected: list[Path] = []
    list_file = temp_dir / "concat_list.txt"
    force_indexes = force_indexes or set()
    try:
        for index, source in enumerate(inputs):
            meta = metas[index] if index < len(metas) else None
            if index not in force_indexes and meta is not None and file_matches_profile(meta, target):
                selected.append(source)
                continue
            target_file = temp_dir / f"{index:05d}.mp4"
            _normalize_to_profile(
                source,
                target_file,
                meta,
                target_size,
                ffmpeg_bin,
                video_codec,
                audio_bitrate_kbps,
            )
            selected.append(target_file)
        _write_concat_list(list_file, selected)
        _concat_copy_with_list(output, list_file, ffmpeg_bin, expected_duration_value)
    finally:
        ffmpeg_mod._safe_unlink(list_file)
        ffmpeg_mod._safe_rmtree(temp_dir)


def _normalize_to_profile(
    source: Path,
    output: Path,
    meta: VideoMeta | None,
    target_size: tuple[int, int] | None,
    ffmpeg_bin: str,
    video_codec: str,
    audio_bitrate_kbps: int,
) -> None:
    scale_args = ffmpeg_mod._concat_scale_args(target_size)
    has_audio = bool(meta.has_audio) if meta is not None and meta.probe_ok else True
    errors: list[str] = []
    for encode_args in ffmpeg_mod._concat_reencode_arg_candidates(ffmpeg_bin, video_codec):
        input_args = [
            ffmpeg_bin,
            "-y",
            "-fflags",
            "+genpts",
            "-err_detect",
            "ignore_err",
            "-i",
            str(source),
        ]
        map_args = ["-map", "0:v:0?"]
        if has_audio:
            map_args += ["-map", "0:a:0?"]
            audio_input_args: list[str] = []
            shortest_args: list[str] = []
        else:
            audio_input_args = [
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
            ]
            map_args += ["-map", "1:a:0"]
            shortest_args = ["-shortest"]
        try:
            ffmpeg_mod.run_command([
                *input_args,
                *audio_input_args,
                *map_args,
                *scale_args,
                *encode_args,
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                f"{audio_bitrate_kbps}k",
                "-ar",
                "48000",
                "-ac",
                "2",
                *shortest_args,
                "-movflags",
                "+faststart",
                str(output),
            ])
            return
        except ffmpeg_mod.FFmpegError as exc:
            errors.append(str(exc))
    raise ffmpeg_mod.FFmpegError("Selective normalize failed:\n" + "\n".join(errors))


def _attempt_full_reencode(
    attempts: list[ConcatAttempt],
    output: Path,
    concat_file: Path,
    ffmpeg_bin: str,
    video_codec: str,
    target_size: tuple[int, int] | None,
    audio_bitrate_kbps: int,
    expected_duration_value: float | None,
) -> bool:
    scale_args = ffmpeg_mod._concat_scale_args(target_size)
    for encode_args in ffmpeg_mod._concat_reencode_arg_candidates(ffmpeg_bin, video_codec):
        if _attempt(
            attempts,
            "concat demuxer full reencode",
            lambda encode_args=encode_args: _concat_demuxer_full_reencode(
                output,
                concat_file,
                ffmpeg_bin,
                scale_args,
                encode_args,
                audio_bitrate_kbps,
                expected_duration_value,
            ),
        ):
            return True
    return False


def _concat_demuxer_full_reencode(
    output: Path,
    concat_file: Path,
    ffmpeg_bin: str,
    scale_args: list[str],
    encode_args: list[str],
    audio_bitrate_kbps: int,
    expected_duration_value: float | None,
) -> None:
    ffmpeg_mod.run_command([
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
        *scale_args,
        *encode_args,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        f"{audio_bitrate_kbps}k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(output),
    ])
    _validate_output(output, ffmpeg_bin, expected_duration_value)


def _attempt_concat_filter(
    attempts: list[ConcatAttempt],
    inputs: list[Path],
    output: Path,
    ffmpeg_bin: str,
    video_codec: str,
    target_size: tuple[int, int] | None,
    audio_bitrate_kbps: int,
    expected_duration_value: float | None,
) -> bool:
    for command in ffmpeg_mod._concat_filter_commands(
        inputs,
        output,
        ffmpeg_bin,
        video_codec,
        target_size,
        audio_bitrate_kbps,
    ):
        if _attempt(
            attempts,
            "concat filter full fallback",
            lambda command=command: _run_filter_command(
                command,
                output,
                ffmpeg_bin,
                expected_duration_value,
            ),
        ):
            return True
    return False


def _run_filter_command(
    command: list[str],
    output: Path,
    ffmpeg_bin: str,
    expected_duration_value: float | None,
) -> None:
    ffmpeg_mod.run_command(command, timeout=7200)
    _validate_output(output, ffmpeg_bin, expected_duration_value)


def _validate_output(
    output: Path,
    ffmpeg_bin: str,
    expected_duration_value: float | None,
) -> None:
    ffmpeg_mod._validate_concat_duration(output, expected_duration_value)
    ffmpeg_mod._validate_audio_decodable(output, ffmpeg_bin)


def _write_concat_list(path: Path, videos: list[Path]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for video in videos:
            escaped = str(video).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")


def _safe_attempt_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "attempt"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index:02d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _candidate_output_path(context: ConcatContext, name: str, index: int) -> Path:
    candidate_dir = context.output.parent / "_concat_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    return candidate_dir / f"{_safe_attempt_name(name)}_{index:02d}{context.output.suffix or '.mp4'}"


def _commit_candidate_output(candidate: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate_parent = candidate.parent
    candidate.replace(output)
    _cleanup_empty_dir(candidate_parent)


def _cleanup_empty_dir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def _target_size(target: TargetProfile) -> tuple[int, int] | None:
    if target.width and target.height:
        return target.width, target.height
    return None


def _has_video_bitstream_failure(attempts: list[ConcatAttempt]) -> bool:
    failed_details = [attempt.detail for attempt in attempts if not attempt.ok]
    analysis = ffmpeg_mod.analyze_ffmpeg_failure(failed_details)
    if analysis.get("bitstream_corruption"):
        return True
    # duration_truncated is often a *symptom* of earlier bitstream/demux problems during copy
    if analysis.get("duration_truncated"):
        return True
    return False


def _duration_failure_boundary_indexes(context: ConcatContext) -> list[int]:
    actual = _latest_failed_video_duration(context)
    if actual is None:
        return []
    return _boundary_indexes_for_duration(context.metas, actual)


def _latest_failed_video_duration(context: ConcatContext) -> float | None:
    for attempt in reversed(context.attempts):
        detail = attempt.detail or ""
        value = _extract_failed_video_duration(detail)
        if value is not None:
            return value
    return None


def _extract_failed_video_duration(detail: str) -> float | None:
    patterns = [
        r"Concat output video stream duration is too short:\s*([0-9.]+)s",
        r"Concat output duration is too short:\s*([0-9.]+)s",
    ]
    for pattern in patterns:
        match = re.search(pattern, detail, re.IGNORECASE)
        if not match:
            continue
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _boundary_indexes_for_duration(metas: list[VideoMeta], actual_duration: float) -> list[int]:
    if actual_duration <= 0:
        return []
    cumulative = 0.0
    total = sum(float(meta.duration or 0.0) for meta in metas)
    tolerance = max(2.0, total * 0.0005)
    for index, meta in enumerate(metas):
        if meta.duration is None:
            return []
        cumulative += float(meta.duration)
        if abs(cumulative - actual_duration) <= tolerance:
            candidates = {index}
            if index + 1 < len(metas):
                candidates.add(index + 1)
            if index - 1 >= 0:
                candidates.add(index - 1)
            return sorted(candidates)
    return []


def _corrupt_duration_ratio(context: ConcatContext, corrupt_indexes: list[int]) -> float | None:
    if not corrupt_indexes:
        return 0.0
    durations: list[float] = []
    for meta in context.metas:
        if meta.duration is None:
            return None
        durations.append(float(meta.duration))
    total = sum(durations)
    if total <= 0:
        return None
    bad = sum(durations[index] for index in set(corrupt_indexes) if 0 <= index < len(durations))
    return bad / total


def _format_failures(attempts: list[ConcatAttempt]) -> str:
    lines = ["All concat attempts failed:"]
    for attempt in attempts:
        if attempt.ok:
            continue
        detail = attempt.detail.strip()
        if len(detail) > 700:
            detail = detail[:700] + "..."
        lines.append(f"- {attempt.name}: {detail}")
    return "\n".join(lines)


def _is_output_duration_failure(detail: str) -> bool:
    text = detail.lower()
    return (
        "concat output duration is too short" in text
        or "concat output duration is too long" in text
        or "concat output video stream duration is too short" in text
        or "concat output video stream duration is too long" in text
    )
