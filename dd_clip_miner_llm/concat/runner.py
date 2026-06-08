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

from .health import _build_health_profile, _get_annexb_bsf, _safe_transmux_to_ts
from .helpers import (
    _corrupt_duration_ratio,
    _target_size,
    _validate_output,
    _write_concat_list,
    _candidate_output_path,
    _commit_candidate_output,
)
from .strategies import (
    DirectCopyStrategy,
    DiscardCorruptCopyStrategy,
    FullReencodeStrategy,
    MkvMergeStrategy,
    SelectiveNormalizeStrategy,
    TargetedRepairStrategy,
)

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
            DirectCopyStrategy(),           # 1. 最快，参数一致时
            MkvMergeStrategy(),             # 2. mkvmerge 拼接（处理 H.264 bitstream 损坏）
            DiscardCorruptCopyStrategy(),   # 3. 单次 ffmpeg + discardcorrupt
            TargetedRepairStrategy(),       # 4. 只修坏段
            SelectiveNormalizeStrategy(),   # 5. 只重编码不匹配的段
            FullReencodeStrategy(),         # 6. 最后兜底
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

        # 3. 写初始 concat list
        concat_file = context.output.parent / "concat_list.txt"
        _write_concat_list(concat_file, context.inputs, context.metas)
        context.concat_file = concat_file

        # 4. 先尝试 mkvmerge（能处理 H.264 bitstream 损坏，无需 pre-sanitize）
        context.mkvmerge_attempted = False
        if corrupt:
            print(f"[concat] Detected {len(corrupt)} corrupt segment(s), trying mkvmerge first...")
            mkvmerge_strat = MkvMergeStrategy()
            if mkvmerge_strat.is_applicable(context):
                print("[concat] Trying strategy: mkvmerge concat (on original files)")
                context.mkvmerge_attempted = True
                if mkvmerge_strat.execute(context):
                    self._cleanup(context)
                    return context.output
                print("[concat] mkvmerge failed, falling back to pre-sanitize + other strategies")

        # 5. Pre-sanitize corrupt segments（mkvmerge 失败后才执行）
        if not context.original_inputs:
            context.original_inputs = list(context.inputs)
        if not context.sanitized_inputs:
            context.sanitized_inputs = {}
        if corrupt:
            sanitize_dir = context.output.parent / f"_pre_sanitize_{uuid4().hex[:8]}"
            sanitize_dir.mkdir(parents=True, exist_ok=True)
            for i in corrupt:
                src = context.original_inputs[i]
                dst = sanitize_dir / f"sanitized_{i:04d}.mp4"
                ts_temp = sanitize_dir / f"sanitized_{i:04d}.ts"
                sanitized = False
                try:
                    # Preferred: Per-file sanitize via TS intermediate (mp4 -> ts with appropriate annexb bsf + flags -> mp4)
                    # This matches the common effective scheme from searches for H.264/HEVC live split corruption.
                    # Use the centralized _safe_transmux_to_ts for the TS construction part.
                    meta = context.metas[i] if i < len(context.metas) else None
                    bsf = _get_annexb_bsf(getattr(meta, "video_codec", None))
                    _safe_transmux_to_ts(src, ts_temp, context.ffmpeg_bin, bsf)
                    ffmpeg_mod.run_command([
                        context.ffmpeg_bin, "-y",
                        "-i", str(ts_temp),
                        "-map", "0:v:0?",
                        "-map", "0:a:0?",
                        "-c", "copy",
                        "-bsf:a", "aac_adtstoasc",
                        "-movflags", "+faststart",
                        str(dst),
                    ], bitstream_fatal=True)
                    context.sanitized_inputs[i] = dst
                    context.ts_inputs[i] = ts_temp
                    print(f"[concat] Pre-sanitized corrupt segment {i+1} via TS intermediate ({bsf} bsf + discardcorrupt etc.)")
                    sanitized = True
                except Exception as exc:
                    ffmpeg_mod._safe_unlink(ts_temp)
                    print(f"[concat] TS sanitize for segment {i+1} failed (bitstream too severe for bsf), falling back to plain flagged mp4 remux: {exc}")
                if not sanitized:
                    try:
                        # Fallback: plain per-file remux with flags (no bsf), still applies discardcorrupt etc.
                        ffmpeg_mod.run_command([
                            context.ffmpeg_bin, "-y",
                            "-fflags", "+genpts+discardcorrupt+igndts",
                            "-err_detect", "ignore_err",
                            "-i", str(src),
                            "-map", "0",
                            "-c", "copy",
                            "-avoid_negative_ts", "make_zero",
                            "-movflags", "+faststart",
                            str(dst),
                        ])
                        context.sanitized_inputs[i] = dst
                        print(f"[concat] Pre-sanitized corrupt segment {i+1} (plain mp4 remux with discardcorrupt etc.)")
                    except Exception as exc2:
                        print(f"[concat] Pre-sanitize for segment {i+1} failed (will use original): {exc2}")

        # Use sanitized where available for the concat processing (transparent to most strategies)
        effective_inputs = [
            context.sanitized_inputs.get(i, context.original_inputs[i])
            for i in range(len(context.original_inputs))
        ]
        context.inputs = effective_inputs

        # Always re-probe the effective inputs (the ones that will actually be fed to the concat strategies,
        # after any pre-sanitize/plain fallback or TS promotion).
        # This ensures expected_duration and the metas used for _write_concat_list (with explicit duration lines)
        # are based on the *actual* probed durations of the (possibly sanitized) files.
        # Critical to avoid "duration too long/short" in the final splice when original ffprobe on corrupt _fix files
        # (or even after plain remux) reports inaccurate/bogus durations.
        print("[concat] Re-probing effective inputs for accurate durations.")
        try:
            new_metas = probe_many(context.inputs)
            context.metas = new_metas
            new_exp = expected_duration(new_metas)
            if new_exp is not None and new_exp > 0:
                old_exp = context.expected_duration
                context.expected_duration = new_exp
                print(f"[concat]   expected_duration updated: {old_exp} -> {new_exp} (based on effective inputs)")
        except Exception as reerr:
            print(f"[concat]   Re-probe of effective inputs failed, keeping previous expected: {reerr}")

        # TS workspace for full TS format or partial promotion.
        # Per user rule: if error parts > 50% duration (force_full_ts), force convert *all* segments to .ts
        # so we can use the robust TS concat path (ReuseTsWorkspaceStrategy).
        # For normal cases, only promote remaining if we already have a good number of TS from pre-sanitize.
        corrupt_ratio = _corrupt_duration_ratio(context, corrupt) or 0.0
        force_full_ts = corrupt_ratio > 0.5
        do_full_ts_force = getattr(context, 'force_full_ts_workspace', False) or force_full_ts
        promote_remaining = bool(context.ts_inputs) and (
            do_full_ts_force or len(context.ts_inputs) >= max(1, len(corrupt) // 2)
        )
        if promote_remaining:
            if context.ts_inputs:
                ts_dir = next(iter(context.ts_inputs.values())).parent
            else:
                ts_dir = context.output.parent / f"_ts_workspace_{uuid4().hex[:8]}"
                ts_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(context.inputs):
                if i in context.ts_inputs:
                    continue
                ts_target = ts_dir / f"working_{i:04d}.ts"
                try:
                    # Use centralized helper for constructing the TS copy.
                    # Determine correct bsf from the (original or refreshed) meta for this segment.
                    meta = context.metas[i] if i < len(context.metas) else None
                    bsf = _get_annexb_bsf(getattr(meta, "video_codec", None))
                    _safe_transmux_to_ts(src, ts_target, context.ffmpeg_bin, bsf)
                    context.ts_inputs[i] = ts_target
                    if do_full_ts_force:
                        print(f"[concat] Forced full TS for segment {i+1} (error duration > 50%, bsf={bsf}).")
                except Exception as ts_work_err:
                    ffmpeg_mod._safe_unlink(ts_target)
                    msg = f"TS working copy for segment {i+1} failed"
                    if do_full_ts_force:
                        msg += " (full TS mode, will try to continue with what we have)"
                    else:
                        msg += "; TS workspace fallback disabled unless all segments are available"
                    print(f"[concat] {msg}: {ts_work_err}")
                    if not do_full_ts_force:
                        break  # only break early in non-force mode
            # After the loop, if force_full_ts and we have at least one, we may still try Reuse if all succeeded.

        # Additional health/profile refresh if there were sanitized inputs (to update the corrupt list for routing).
        # Note: the expected_duration re-probe above already happened for all effective inputs.
        if context.sanitized_inputs:
            try:
                print("[concat] Re-running health probe on sanitized inputs for routing.")
                context.health = _build_health_profile(context.inputs, context.metas, context.ffmpeg_bin)
                corrupt = [i for i, h in context.health.items() if h.is_bitstream_corrupt]
                refreshed_profile = ProblemProfile()
                if corrupt:
                    refreshed_profile.bitstream_corrupt_indexes = corrupt
                    refreshed_profile.bitstream_corruption = True
                    refreshed_profile.summary = "bitstream_corruption (after pre-sanitize)"
                else:
                    refreshed_profile.summary = "pre_sanitize_clean"
                    refreshed_profile.bitstream_corrupt_indexes = []
                    refreshed_profile.bitstream_corruption = False
                    print("[concat]   Sanitized inputs passed health probe; copy/remux strategies remain available.")
                context.profile = refreshed_profile
            except Exception as health_err:
                print(f"[concat]   Health probe after pre-sanitize failed, keeping previous routing profile: {health_err}")

        # 3. Run strategies
        concat_file = context.output.parent / "concat_list.txt"
        _write_concat_list(concat_file, context.inputs, context.metas)
        context.concat_file = concat_file

        strategies = self.strategies
        force_flag = getattr(context, 'force_full_ts_workspace', False) or force_full_ts
        if corrupt or force_flag:
            corrupt_ratio = _corrupt_duration_ratio(context, corrupt)
            count_ratio = len(corrupt) / max(1, len(context.inputs))
            ratio_text = (
                f"{corrupt_ratio:.2%} of input duration"
                if corrupt_ratio is not None
                else f"{count_ratio:.2%} of input files"
            )
            print(
                "[concat] Upfront H.264 tail probe found corrupt segment(s) "
                f"{', '.join(str(i + 1) for i in corrupt)} ({ratio_text})."
            )
            if force_flag:
                print(
                    f"[concat] Forcing full TS format because error parts duration > 50%. "
                    f"Current corrupt after sanitize: {len(corrupt)} segments."
                )
            else:
                print(
                    "[concat] Will try discard-corrupt copy, mkvmerge on sanitized inputs, "
                    "then targeted repair if needed."
                )

        for strat in strategies:
            # 跳过已在原始文件上尝试过的 mkvmerge；sanitize 后可再试
            if isinstance(strat, MkvMergeStrategy) and getattr(context, "mkvmerge_attempted", False):
                continue
            if not strat.is_applicable(context):
                continue
            print(f"[concat] Trying strategy: {strat.name}")
            success = strat.execute(context)
            if success:
                self._cleanup(context)
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

    def _cleanup(self, context: ConcatContext) -> None:
        """清理临时目录。"""
        if context.concat_file:
            ffmpeg_mod._safe_unlink(context.concat_file)
        try:
            import shutil
            parent = context.output.parent
            for pat in ("_pre_sanitize_*", "_concat_repair_*", "_ts_workspace_*", "_concat_*_tmp*", "_mkvmerge_*", "_concat_candidates"):
                for d in parent.glob(pat):
                    if d.is_dir():
                        shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


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
