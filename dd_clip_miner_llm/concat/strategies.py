from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import shutil
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

from .health import _has_video_bitstream_corruption
from .helpers import (
    _candidate_output_path,
    _cleanup_empty_dir,
    _commit_candidate_output,
    _concat_demuxer_full_reencode,
    _corrupt_duration_ratio,
    _is_output_duration_failure,
    _safe_attempt_name,
    _selective_normalize_concat,
    _unique_path,
    _validate_output,
    _write_concat_list,
)

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
            return False
        return can_direct_concat_copy(context.metas)

    def execute(self, context: ConcatContext) -> bool:
        concat_file = context.concat_file or (context.output.parent / "concat_list.txt")
        _write_concat_list(concat_file, context.inputs, context.metas)
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


class DiscardCorruptCopyStrategy(Strategy):
    """使用 +discardcorrupt 在 demux 层处理损坏，单次 ffmpeg 调用。

    这是处理损坏文件的主要快速路径：在 demux 层直接丢弃损坏的包，
    避免进入解码器，单次 ffmpeg 调用即可完成 concat。
    """
    name = "discard corrupt copy"

    def is_applicable(self, context: ConcatContext) -> bool:
        # 总是适用，除非强制 normalize
        return not context.force_normalize

    def execute(self, context: ConcatContext) -> bool:
        concat_file = context.concat_file or (context.output.parent / "concat_list.txt")
        _write_concat_list(concat_file, context.inputs, context.metas)
        context.concat_file = concat_file

        try:
            candidate = _candidate_output_path(context, self.name, 0)
            ffmpeg_mod.run_command(
                [
                    context.ffmpeg_bin, "-y",
                    "-fflags", "+genpts+igndts+discardcorrupt",
                    "-f", "concat", "-safe", "0",
                    "-i", str(concat_file),
                    "-map", "0:v:0?", "-map", "0:a:0?",
                    "-c", "copy",
                    "-avoid_negative_ts", "make_zero",
                    "-movflags", "+faststart",
                    str(candidate),
                ],
                bitstream_fatal=False,  # 不把 corruption 当作 fatal
            )
            _validate_output(candidate, context.ffmpeg_bin, context.expected_duration)
            _commit_candidate_output(candidate, context.output)
            context.attempts.append(AttemptRecord(self.name, True, "ok"))
            return True
        except ffmpeg_mod.FFmpegError as exc:
            if "candidate" in locals():
                ffmpeg_mod._safe_unlink(candidate)
            self._record_failure(context, exc)
            return False


class TargetedRepairStrategy(Strategy):
    """The smart one: uses health/profile to repair only bad segments."""
    name = "targeted H.264 repair"

    def is_applicable(self, context: ConcatContext) -> bool:
        if not _has_video_bitstream_corruption(context):
            return False
        corrupt = []
        if context.profile and context.profile.bitstream_corrupt_indexes:
            corrupt = list(context.profile.bitstream_corrupt_indexes)
        elif context.health:
            corrupt = [i for i, h in context.health.items() if h.is_bitstream_corrupt]
        if not corrupt:
            return True
        duration_ratio = _corrupt_duration_ratio(context, corrupt)
        if duration_ratio is not None:
            return duration_ratio <= 0.5
        count_ratio = len(corrupt) / max(1, len(context.inputs))
        return count_ratio <= 0.5

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

        window_bad: list[int] = []
        skipped_window: list[int] = []
        for index in bad:
            try:
                duration = ffmpeg_mod.get_duration(context.inputs[index])
            except Exception:
                duration = 0.0
            if duration > 93.0:
                window_bad.append(index)
            else:
                skipped_window.append(index)

        if window_bad:
            try:
                if skipped_window:
                    print(
                        "[concat] Tail-window repair will skip short corrupt segment(s) "
                        f"{', '.join(str(i + 1) for i in skipped_window)}; those remain for full bad-segment repair if needed."
                    )
                print("[concat] Trying fixed tail-window H.264 repair before full bad-segment reencode.")
                candidate = _candidate_output_path(context, f"{self.name}_tail_window", 0)
                ffmpeg_mod._concat_tail_window_repaired_bad_segments_copy(
                    context.inputs,
                    candidate,
                    context.ffmpeg_bin,
                    context.video_codec,
                    context.audio_bitrate_kbps,
                    context.expected_duration,
                    bad_indexes=window_bad,
                    repair_window_seconds=90.0,
                    guard_seconds=2.0,
                )
                _validate_output(candidate, context.ffmpeg_bin, context.expected_duration)
                _commit_candidate_output(candidate, context.output)
                context.attempts.append(AttemptRecord(f"{self.name} (tail window)", True, "ok"))
                return True
            except ffmpeg_mod.FFmpegError as exc:
                if "candidate" in locals():
                    ffmpeg_mod._safe_unlink(candidate)
                context.attempts.append(
                    AttemptRecord(
                        f"{self.name} (tail window)",
                        False,
                        str(exc),
                        ffmpeg_mod.classify_ffmpeg_output(str(exc)),
                        self._save_log(context, str(exc)),
                    )
                )
        else:
            context.attempts.append(
                AttemptRecord(
                    f"{self.name} (tail window)",
                    False,
                    "skipped: corrupt segments are too short for tail-window split",
                )
            )

        try:
            candidate = _candidate_output_path(context, self.name, 0)
            ffmpeg_mod._concat_reencoded_bad_segments_copy(
                context.inputs,
                candidate,
                context.ffmpeg_bin,
                context.video_codec,
                context.audio_bitrate_kbps,
                context.expected_duration,
                bad_indexes=bad,
            )
            _validate_output(candidate, context.ffmpeg_bin, context.expected_duration)
            _commit_candidate_output(candidate, context.output)
            context.attempts.append(AttemptRecord(self.name, True, "ok"))
            return True
        except ffmpeg_mod.FFmpegError as exc:
            if "candidate" in locals():
                ffmpeg_mod._safe_unlink(candidate)
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


class MkvMergeStrategy(Strategy):
    """使用 mkvmerge 拼接，处理 H.264 bitstream 损坏。

    流程：mkvmerge 修正容器时间戳 → mkvmerge append 合并 → ffmpeg -c copy 转 MP4。
    优势：mkvmerge 对 H.264 bitstream timing 处理比 ffmpeg 更稳健。
    """
    name = "mkvmerge concat"

    def is_applicable(self, context: ConcatContext) -> bool:
        # 总是适用，除非强制 normalize
        return not context.force_normalize

    def execute(self, context: ConcatContext) -> bool:
        try:
            candidate = _candidate_output_path(context, self.name, 0)
            ffmpeg_mod.concat_with_mkvmerge(
                context.inputs,
                candidate,
                video_codec=context.video_codec,
                audio_bitrate_kbps=context.audio_bitrate_kbps,
                expected_duration=context.expected_duration,
            )
            _commit_candidate_output(candidate, context.output)
            context.attempts.append(AttemptRecord(self.name, True, "ok"))
            return True
        except (ffmpeg_mod.FFmpegError, FileNotFoundError) as exc:
            if "candidate" in locals():
                ffmpeg_mod._safe_unlink(candidate)
            self._record_failure(context, exc)
            return False


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
                cpu_only_indexes=force_indexes,
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


class FullReencodeStrategy(Strategy):
    name = "full reencode (demuxer + filter fallback)"

    def is_applicable(self, context: ConcatContext) -> bool:
        return True

    def execute(self, context: ConcatContext) -> bool:
        concat_file = context.concat_file or (context.output.parent / "concat_list.txt")
        _write_concat_list(concat_file, context.inputs, context.metas)
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

