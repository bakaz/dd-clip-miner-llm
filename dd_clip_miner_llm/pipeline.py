"""核心流水线

编排完整的内容识别流程：
1. 音频提取
2. ASR 转写
3. LLM 识别（通过识别器架构）
4. 片段导出
5. 报告生成
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .asr import Transcriber
from .config import get_padding_config
from .ffmpeg import cut_audio, cut_video, extract_audio, get_duration
from .merger import build_content_results
from .models import ContentMatch, ContentResult, TranscriptSegment
from .asr_backends import resolve_asr_model_name
from .paths import safe_path_part, stage_input_for_ffmpeg
from .recognizers import get_recognizer, list_recognizers
from .clip_naming import ClipNamingProfile, resolve_clip_naming_profile, resolve_export_stem
from .report import write_match_context_reports, write_reports


def _safe_filename(value: str, fallback: str = "untitled") -> str:
    return safe_path_part(value, fallback=fallback)


def _check_previous_run(out: Path, input_path: Path) -> dict[str, Any] | None:
    progress_path = out / "progress.json"
    if not progress_path.exists():
        return None
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        prev_input = progress.get("input_video", "")
        if Path(prev_input).resolve() == input_path.resolve():
            return progress
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_progress(out: Path, input_path: Path, step: str, data: dict[str, Any] | None = None) -> None:
    progress_path = out / "progress.json"
    try:
        progress = {}
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress["input_video"] = str(input_path)
        progress["last_completed_step"] = step
        if data:
            progress[step] = data
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def _load_previous_segments(asr_dir: Path) -> list[TranscriptSegment] | None:
    transcript_path = asr_dir / "transcript.json"
    if not transcript_path.exists():
        return None
    try:
        return [
            TranscriptSegment(start=s["start"], end=s["end"], text=s["text"])
            for s in json.loads(transcript_path.read_text(encoding="utf-8"))
        ]
    except (json.JSONDecodeError, OSError):
        return None


def _load_previous_matches(llm_dir: Path, content_type: str) -> list[ContentMatch] | None:
    matches_path = llm_dir / "matches.json"
    if not matches_path.exists():
        return None
    try:
        return [
            ContentMatch(
                content_type=m.get("content_type", content_type),
                title=m["title"],
                segment_indices=m.get("segment_indices", []),
                confidence=m.get("confidence", 0.5),
                tags=m.get("tags", []),
                description=m.get("description", ""),
                artist=m.get("artist", ""),
                lyrics_snippet=m.get("lyrics_snippet", ""),
            )
            for m in json.loads(matches_path.read_text(encoding="utf-8"))
        ]
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def _load_previous_summary(llm_dir: Path) -> dict[str, Any] | None:
    summary_path = llm_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or not summary:
            return None
        if summary.get("error"):
            return None
        if not isinstance(summary.get("level_1"), list) and not isinstance(summary.get("overall"), dict):
            return None
        return summary
    except (json.JSONDecodeError, OSError):
        return None


def _is_summary_only(recognizer: Any, config: dict[str, Any]) -> bool:
    type_config = config.get(recognizer.name, {})
    default_config = getattr(recognizer, "default_config", {})
    return bool(type_config.get("summary_only", default_config.get("summary_only", False)))


def _write_structured_summary(
    summary: dict[str, Any],
    recognizer: Any,
    llm_dir: Path,
    reports_dir: Path,
    content_type: str,
    config: dict[str, Any],
) -> None:
    for target_dir in (llm_dir, reports_dir / content_type):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        formatter = getattr(recognizer, "format_summary_markdown", None)
        if callable(formatter):
            markdown = formatter(summary, config)
        else:
            markdown = "```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n"
        (target_dir / "summary.md").write_text(markdown, encoding="utf-8")


class _OffsetRecognizer:
    def __init__(self, recognizer: Any, offset: int) -> None:
        self._recognizer = recognizer
        self._offset = offset

    @property
    def name(self) -> str:
        return self._recognizer.name

    @property
    def default_config(self) -> dict[str, Any]:
        return getattr(self._recognizer, "default_config", {})

    def build_prompt(
        self,
        segments: list[TranscriptSegment],
        batch_start: int,
        config: dict[str, Any],
    ) -> str:
        return self._recognizer.build_prompt(segments, self._offset + batch_start, config)

    def parse_response(self, items: list[dict[str, Any]], config: dict[str, Any]) -> list[ContentMatch]:
        return self._recognizer.parse_response(items, config)

    def get_tools(self, config: dict[str, Any]) -> Any:
        return self._recognizer.get_tools(config)


def _uncovered_segment_ranges(
    segment_count: int,
    matches: list[ContentMatch],
    min_gap_segments: int = 1,
) -> list[tuple[int, int]]:
    covered: set[int] = set()
    for match in matches:
        covered.update(i for i in match.segment_indices if 0 <= i < segment_count)

    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(segment_count):
        if index in covered:
            if start is not None and index - start >= min_gap_segments:
                ranges.append((start, index - 1))
            start = None
            continue
        if start is None:
            start = index

    if start is not None and segment_count - start >= min_gap_segments:
        ranges.append((start, segment_count - 1))
    return ranges


def _split_segment_ranges(
    ranges: list[tuple[int, int]],
    batch_size: int,
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    batch_size = max(1, batch_size)
    for start, end in ranges:
        current = start
        while current <= end:
            chunk_end = min(end, current + batch_size - 1)
            result.append((current, chunk_end))
            current = chunk_end + 1
    return result


def _segment_range_duration_seconds(
    segments: list[TranscriptSegment],
    start: int,
    end: int,
) -> float:
    if not segments or start < 0 or end < start or end >= len(segments):
        return 0.0
    return max(0.0, float(segments[end].end) - float(segments[start].start))


def _filter_short_segment_ranges(
    segments: list[TranscriptSegment],
    ranges: list[tuple[int, int]],
    min_duration_seconds: float,
) -> tuple[list[tuple[int, int]], int]:
    if min_duration_seconds <= 0:
        return ranges, 0

    kept: list[tuple[int, int]] = []
    skipped = 0
    for start, end in ranges:
        if _segment_range_duration_seconds(segments, start, end) < min_duration_seconds:
            skipped += 1
            continue
        kept.append((start, end))
    return kept, skipped


def _expand_segment_range(
    start: int,
    end: int,
    segment_count: int,
    context_segments: int,
) -> tuple[int, int]:
    context_segments = max(0, context_segments)
    return (
        max(0, start - context_segments),
        min(segment_count - 1, end + context_segments),
    )


def _filter_matches_to_segment_range(
    matches: list[ContentMatch],
    start: int,
    end: int,
) -> list[ContentMatch]:
    filtered: list[ContentMatch] = []
    for match in matches:
        segment_indices = sorted({i for i in match.segment_indices if start <= i <= end})
        if not segment_indices:
            continue
        filtered.append(
            ContentMatch(
                content_type=match.content_type,
                title=match.title,
                segment_indices=segment_indices,
                confidence=match.confidence,
                tags=match.tags,
                description=match.description,
                artist=match.artist,
                lyrics_snippet=match.lyrics_snippet,
            )
        )
    return filtered


def _clone_match_with_indices(
    match: ContentMatch,
    segment_indices: list[int],
) -> ContentMatch:
    return ContentMatch(
        content_type=match.content_type,
        title=match.title,
        segment_indices=segment_indices,
        confidence=match.confidence,
        tags=match.tags,
        description=match.description,
        artist=match.artist,
        lyrics_snippet=match.lyrics_snippet,
    )


def _split_indices_by_time_gap_for_recheck(
    segments: list[TranscriptSegment],
    indices: list[int],
    merge_gap_seconds: float,
) -> list[list[int]]:
    if not indices:
        return []

    groups: list[list[int]] = [[indices[0]]]
    for index in indices[1:]:
        previous = groups[-1][-1]
        gap = float(segments[index].start) - float(segments[previous].end)
        if gap > merge_gap_seconds:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def _match_groups_over_max_song_seconds(
    segments: list[TranscriptSegment],
    matches: list[ContentMatch],
    max_song_seconds: float,
    merge_gap_seconds: float,
) -> bool:
    if max_song_seconds <= 0:
        return False
    for match in matches:
        valid_indices = sorted({i for i in match.segment_indices if 0 <= i < len(segments)})
        for group in _split_indices_by_time_gap_for_recheck(
            segments,
            valid_indices,
            merge_gap_seconds,
        ):
            if _segment_range_duration_seconds(segments, min(group), max(group)) > max_song_seconds:
                return True
    return False


def _recheck_overlong_song_matches(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    llm_dir: Path,
) -> list[ContentMatch]:
    if not segments or not matches:
        return matches

    recheck_config = config.get("song", {}).get("missed_recheck", {})
    if recheck_config.get("enabled", True) is False:
        return matches

    padding_config = get_padding_config(config, "song")
    max_song_seconds = float(padding_config.get("max_song_seconds", 360.0))
    merge_gap_seconds = float(padding_config.get("merge_gap_seconds", 20.0))
    if max_song_seconds <= 0:
        return matches

    context_segments = int(recheck_config.get("context_segments", 10) or 0)

    from .llm import identify_content

    recheck_root = llm_dir / "overlong_recheck"
    replacement_matches: list[ContentMatch] = []
    rechecked_count = 0
    replaced_count = 0
    kept_count = 0
    all_rechecked_matches: list[ContentMatch] = []

    for match in matches:
        valid_indices = sorted({i for i in match.segment_indices if 0 <= i < len(segments)})
        groups = _split_indices_by_time_gap_for_recheck(
            segments,
            valid_indices,
            merge_gap_seconds,
        )
        if not groups:
            continue

        if not any(
            _segment_range_duration_seconds(segments, min(group), max(group)) > max_song_seconds
            for group in groups
        ):
            replacement_matches.append(match)
            continue

        recheck_root.mkdir(parents=True, exist_ok=True)
        for group in groups:
            group_start = min(group)
            group_end = max(group)
            group_match = _clone_match_with_indices(match, group)
            if _segment_range_duration_seconds(segments, group_start, group_end) <= max_song_seconds:
                replacement_matches.append(group_match)
                continue

            rechecked_count += 1
            context_start, context_end = _expand_segment_range(
                group_start,
                group_end,
                len(segments),
                context_segments,
            )
            chunk = segments[context_start:context_end + 1]
            debug_dir = recheck_root / f"{group_start:06d}_{group_end:06d}"
            offset_recognizer = _OffsetRecognizer(recognizer, context_start)
            rechecked_matches = _filter_matches_to_segment_range(
                identify_content(chunk, config, offset_recognizer, debug_dir=debug_dir),
                group_start,
                group_end,
            )
            all_rechecked_matches.extend(rechecked_matches)

            if rechecked_matches and not _match_groups_over_max_song_seconds(
                segments,
                rechecked_matches,
                max_song_seconds,
                merge_gap_seconds,
            ):
                replacement_matches.extend(rechecked_matches)
                replaced_count += 1
            else:
                replacement_matches.append(group_match)
                kept_count += 1

    if not rechecked_count:
        return matches

    if all_rechecked_matches:
        (recheck_root / "matches.json").write_text(
            json.dumps([m.to_dict() for m in all_rechecked_matches], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        "  Song overlong recheck: "
        f"checked {rechecked_count} range(s), replaced {replaced_count}, kept original {kept_count}"
    )
    return replacement_matches


def _recheck_uncovered_song_segments(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    llm_dir: Path,
) -> list[ContentMatch]:
    recheck_config = config.get("song", {}).get("missed_recheck", {})
    if recheck_config.get("enabled", True) is False:
        return matches

    min_gap_segments = int(recheck_config.get("min_gap_segments", 1) or 1)
    context_segments = int(recheck_config.get("context_segments", 10) or 0)
    padding_config = get_padding_config(config, "song")
    min_song_seconds = float(padding_config.get("min_song_seconds", 75.0))
    batch_size_value = recheck_config.get("batch_size", config.get("llm", {}).get("batch_size") or 500)
    batch_size = int(batch_size_value or 500)
    ranges = _split_segment_ranges(
        _uncovered_segment_ranges(len(segments), matches, min_gap_segments=min_gap_segments),
        batch_size,
    )
    ranges, skipped_short = _filter_short_segment_ranges(segments, ranges, min_song_seconds)
    if skipped_short:
        print(
            f"  Song missed recheck: skipped {skipped_short} short ASR range(s) "
            f"below min_song_seconds={min_song_seconds:g}"
        )
    if not ranges:
        return matches

    from .llm import identify_content

    print(f"  Song missed recheck: {len(ranges)} uncovered ASR range(s)")
    extra_matches: list[ContentMatch] = []
    recheck_root = llm_dir / "missed_recheck"
    recheck_root.mkdir(parents=True, exist_ok=True)

    for start, end in ranges:
        context_start, context_end = _expand_segment_range(
            start,
            end,
            len(segments),
            context_segments,
        )
        chunk = segments[context_start:context_end + 1]
        debug_dir = recheck_root / f"{start:06d}_{end:06d}"
        offset_recognizer = _OffsetRecognizer(recognizer, context_start)
        rechecked_matches = identify_content(chunk, config, offset_recognizer, debug_dir=debug_dir)
        extra_matches.extend(
            _filter_matches_to_segment_range(rechecked_matches, start, end)
        )

    if extra_matches:
        (recheck_root / "matches.json").write_text(
            json.dumps([m.to_dict() for m in extra_matches], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Song missed recheck: found {len(extra_matches)} additional match(es)")
        return [*matches, *extra_matches]

    print("  Song missed recheck: no additional matches")
    return matches


def _export_results(
    results: list[ContentResult],
    input_path: Path,
    clips_dir: Path,
    config: dict[str, Any],
    content_type: str,
    naming_profile: ClipNamingProfile | None = None,
) -> None:
    """导出音视频片段"""
    audio_ext = str(config["output"].get("audio_extension", "mp3")).lstrip(".")
    audio_bitrate_kbps = int(config["output"].get("audio_bitrate_kbps") or 320)
    video_ext = str(config["output"].get("video_extension", "mp4")).lstrip(".")
    video_codec = str(config["output"].get("video_codec", "copy"))
    
    audio_dir_out = clips_dir / "audio" / content_type
    video_dir_out = clips_dir / "video" / content_type

    for result in results:
        stem = resolve_export_stem(
            result,
            config,
            content_type,
            naming_profile,
            legacy_safe_filename=_safe_filename,
        )

        if config["output"].get("audio_segments", True):
            try:
                target = audio_dir_out / f"{stem}.{audio_ext}"
                copy_audio = audio_ext.lower() in {"aac", "m4a"}
                cut_audio(input_path, target, result.start, result.end, copy_codec=copy_audio, bitrate_kbps=audio_bitrate_kbps)
                result.audio_path = target
            except Exception as exc:
                result.errors.append(f"audio export failed: {exc}")

        if config["output"].get("video_clips", True):
            try:
                target = video_dir_out / f"{stem}.{video_ext}"
                cut_video(input_path, target, result.start, result.end, video_codec=video_codec)
                result.video_path = target
            except Exception as exc:
                result.errors.append(f"video export failed: {exc}")


def _get_content_types(config: dict[str, Any]) -> list[str]:
    """获取要处理的内容类型列表"""
    content_types = config.get("content_types", {})
    
    # 新格式：字典 {"song": true, "dialogue": false, ...}
    if isinstance(content_types, dict):
        return [ct for ct, enabled in content_types.items() if enabled]
    
    # 旧格式兼容：列表 ["song", "dialogue", ...]
    if isinstance(content_types, list) and content_types:
        return content_types
    
    # 向后兼容：检查各个类型的 enabled 状态
    available = list_recognizers()
    result = []
    for ct in available:
        type_config = config.get(ct, {})
        if type_config.get("enabled", True):
            result.append(ct)
    return result if result else ["song"]


def run_pipeline(
    input_video: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, list[ContentResult]]:
    """
    运行完整流水线，返回按类型分组的结果。
    
    Returns:
        {"song": [...], "dialogue": [...], "highlight": [...], "funny": [...]}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_path = stage_input_for_ffmpeg(input_video, out / "00_input").resolve()

    naming_profile = resolve_clip_naming_profile(
        input_video,
        config,
        config_path=Path(config_path).parent if config_path else None,
        extra_texts=[out.name],
    )
    if naming_profile is not None:
        profile_path = out / "clip_naming.json"
        profile_path.write_text(
            json.dumps(naming_profile.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[naming] 【{naming_profile.streamer}】*-{naming_profile.date} "
            f"({naming_profile.source}, score={naming_profile.score:.2f})"
        )

    audio_dir = out / "01_audio"
    asr_dir = out / "02_asr"
    clips_dir = out / "03_clips"
    reports_dir = out / "04_reports"
    for d in [audio_dir, asr_dir, clips_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 获取要识别的内容类型
    content_types = _get_content_types(config)
    
    # 检查是否有可复用的上次运行结果
    prev_progress = _check_previous_run(out, input_path)
    reuse_audio = False
    reuse_asr = False

    if prev_progress:
        last_step = prev_progress.get("last_completed_step", "")
        print(f"[info] 检测到上次运行结果（完成到 {last_step}），检查可复用的部分...")
        reuse_audio = last_step in ("audio", "asr", "llm", "done") and (audio_dir / "source.wav").exists()
        reuse_asr = last_step in ("asr", "llm", "done") and (asr_dir / "transcript.json").exists()
        print(f"  音频提取: {'复用' if reuse_audio else '需要重新运行'}")
        print(f"  ASR 转写: {'复用' if reuse_asr else '需要重新运行'}")

    # Step 1: 音频提取
    source_wav = audio_dir / "source.wav"
    if reuse_audio:
        print("[1/3] 音频提取: 复用已有结果")
    else:
        print("[1/3] Extracting audio...")
        extract_audio(
            input_path, source_wav,
            sample_rate=int(config["audio"]["sample_rate"]),
            channels=int(config["audio"]["channels"]),
        )
    _save_progress(out, input_path, "audio")

    total_duration = get_duration(input_path)

    # Step 2: ASR 转写
    if reuse_asr:
        print("[2/3] ASR 转写: 复用已有结果")
        segments = _load_previous_segments(asr_dir)
        if segments is None:
            print("  [warn] 无法加载之前的 ASR 结果，重新运行...")
            reuse_asr = False

    if not reuse_asr:
        print("[2/3] Running Whisper ASR...")
        transcriber = Transcriber(config)
        segments = transcriber.transcribe(source_wav)
        transcript_path = asr_dir / "transcript.json"
        transcript_path.write_text(
            json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    _save_progress(out, input_path, "asr")
    print(f"  Transcribed {len(segments)} segments")

    # Step 3: LLM 识别（通过识别器架构）
    print("[3/3] Identifying content with LLM...")
    
    all_results: dict[str, list[ContentResult]] = {}

    for content_type in content_types:
        # 获取识别器
        recognizer = get_recognizer(content_type)
        if recognizer is None:
            print(f"  [warn] 未找到识别器: {content_type}")
            continue
        
        # 检查是否启用
        type_config = config.get(content_type, {})
        if not type_config.get("enabled", True):
            print(f"  {content_type}: 已禁用，跳过")
            continue

        print(f"\n  === {content_type} 识别 ===")
        llm_dir = asr_dir / "llm" / content_type
        llm_dir.mkdir(parents=True, exist_ok=True)

        if _is_summary_only(recognizer, config):
            reuse_summary = False
            summary = None
            if prev_progress:
                summary = _load_previous_summary(llm_dir)
                reuse_summary = summary is not None

            if reuse_summary:
                print("  LLM 总结: 复用已有结果")
            else:
                from .llm import identify_structured_content
                summary = identify_structured_content(segments, config, recognizer, debug_dir=llm_dir)

            _write_structured_summary(summary or {}, recognizer, llm_dir, reports_dir, content_type, config)
            print(f"  Wrote {content_type} summary")
            all_results[content_type] = []
            continue

        # 检查是否复用 LLM 结果
        reuse_llm = False
        if prev_progress:
            reuse_llm = llm_dir.exists() and (llm_dir / "matches.json").exists()

        if reuse_llm:
            print(f"  LLM 识别: 复用已有结果")
            matches = _load_previous_matches(llm_dir, content_type)
            if matches is None:
                reuse_llm = False

        if not reuse_llm:
            # 使用识别器进行识别
            from .llm import identify_content
            matches = identify_content(segments, config, recognizer, debug_dir=llm_dir)
            if content_type == "song":
                matches = _recheck_overlong_song_matches(
                    segments, config, recognizer, matches, llm_dir
                )
                matches = _recheck_uncovered_song_segments(
                    segments, config, recognizer, matches, llm_dir
                )

            (llm_dir / "matches.json").write_text(
                json.dumps([m.to_dict() for m in matches], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            write_match_context_reports(
                matches, segments, llm_dir,
                context_segments=int(config["output"].get("match_context_segments", 10)),
                content_type=content_type,
            )

        print(f"  Found {len(matches)} {content_type} matches")

        # 构建结果
        results = build_content_results(segments, matches, total_duration, config, content_type)

        # 导出片段
        _export_results(results, input_path, clips_dir, config, content_type, naming_profile)

        # 写入报告
        type_reports_dir = reports_dir / content_type
        type_reports_dir.mkdir(parents=True, exist_ok=True)
        write_reports(results, type_reports_dir, content_type)

        all_results[content_type] = results

    _save_progress(out, input_path, "export")

    # 输出识别结果摘要
    _print_summary(all_results)

    # 写入 manifest
    manifest = {
        "input_video": str(input_path),
        "total_duration": total_duration,
        "segment_count": len(segments),
        "content_types": {ct: len(results) for ct, results in all_results.items()},
        "config": {
            "asr_model": resolve_asr_model_name(config["asr"]),
            "llm_model": config["llm"]["model"],
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_progress(out, input_path, "done")

    return all_results


# 兼容旧项目的函数别名
def run_pipeline_songs(
    input_video: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[ContentResult]:
    """运行流水线，仅返回歌曲结果（兼容旧项目）"""
    config_copy = config.copy()
    config_copy["content_types"] = ["song"]
    results = run_pipeline(input_video, output_dir, config_copy)
    return results.get("song", [])


def _print_summary(all_results: dict[str, list[ContentResult]]) -> None:
    """输出识别结果摘要"""
    print(f"\n{'='*60}")
    print(f"识别结果摘要:")
    print(f"{'='*60}")
    
    for content_type, results in all_results.items():
        print(f"\n  {content_type}: {len(results)} 个片段")
        for r in results[:5]:  # 最多显示5个
            tc_start = f"{int(r.start//3600):02d}:{int((r.start%3600)//60):02d}:{int(r.start%60):02d}"
            tc_end = f"{int(r.end//3600):02d}:{int((r.end%3600)//60):02d}:{int(r.end%60):02d}"
            print(f"    [{r.index}] {r.title} ({tc_start}-{tc_end}, {r.duration:.1f}s)")
        if len(results) > 5:
            print(f"    ... 还有 {len(results) - 5} 个")
    
    print(f"\n{'='*60}")
