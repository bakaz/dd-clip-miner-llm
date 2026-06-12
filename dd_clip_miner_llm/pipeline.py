"""核心流水线

编排完整的内容识别流程：
1. 音频提取
2. ASR 转写
3. LLM 识别（通过识别器架构）
4. 片段导出
5. 报告生成
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
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


def _fingerprint_payload(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _config_fingerprint(config: dict[str, Any]) -> str:
    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: sanitize(item)
                for key, item in value.items()
                if key not in {"api_key", "api_key_env"} and not key.startswith("_")
            }
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        return value

    return _fingerprint_payload(sanitize(config))


def _transcript_fingerprint(segments: list[TranscriptSegment]) -> str:
    return _fingerprint_payload([segment.to_dict() for segment in segments])


def _profile_state_matches(
    state_path: Path,
    *,
    input_path: Path,
    config_fingerprint: str,
    transcript_fingerprint: str,
) -> bool:
    if not state_path.exists():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        state.get("input_video") == str(input_path)
        and state.get("config_fingerprint") == config_fingerprint
        and state.get("transcript_fingerprint") == transcript_fingerprint
        and state.get("status") == "complete"
    )


def _write_profile_state(
    state_path: Path,
    *,
    input_path: Path,
    config: dict[str, Any],
    config_fingerprint: str,
    transcript_fingerprint: str,
    status: str,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "profile": config.get("_profile_name"),
                "input_video": str(input_path),
                "config_fingerprint": config_fingerprint,
                "transcript_fingerprint": transcript_fingerprint,
                "model": config.get("llm", {}).get("model"),
                "cache_friendly_prompt_layout": bool(
                    config.get("llm", {}).get("cache_friendly_prompt_layout", False)
                ),
                "compact_segment_ranges": bool(
                    config.get("llm", {}).get("compact_segment_ranges", False)
                ),
                "status": status,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _profile_usage_totals(profile_dir: Path) -> dict[str, int]:
    totals = {
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "completion_tokens": 0,
    }
    manifests = list(profile_dir.rglob("valid_debug_files.json"))
    if manifests:
        paths: set[Path] = set()
        for manifest_path in manifests:
            try:
                values = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(values, list):
                continue
            for value in values:
                path = manifest_path.parent / str(value)
                if path.is_file():
                    paths.add(path)
    else:
        paths = set(profile_dir.rglob("llm_batch_*.json"))

    for path in sorted(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for usage in payload.get("usage", []):
            if not isinstance(usage, dict):
                continue
            for key in totals:
                totals[key] += int(usage.get(key) or 0)
    return totals


def _write_valid_debug_manifest(llm_dir: Path) -> None:
    paths: set[Path] = set(llm_dir.glob("llm_batch_*.json"))

    review_root = llm_dir / "review"
    if review_root.exists():
        for phase_dir in review_root.iterdir():
            if not phase_dir.is_dir():
                continue
            summary_path = phase_dir / "summary.json"
            if not summary_path.exists():
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                cluster_count = int(summary.get("cluster_count") or 0)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            cluster_records = summary.get("clusters")
            if isinstance(cluster_records, list):
                active_indices = [
                    int(item.get("cluster"))
                    for item in cluster_records
                    if (
                        isinstance(item, dict)
                        and item.get("cluster") is not None
                        and item.get("resolution")
                        != "unchanged_pre_audit_cluster"
                    )
                ]
            else:
                active_indices = list(range(1, cluster_count + 1))
            for index in active_indices:
                paths.update(
                    (phase_dir / f"cluster_{index:03d}").glob("llm_batch_*.json")
                )

    overlong_root = llm_dir / "overlong_recheck"
    if overlong_root.exists():
        paths.update(overlong_root.glob("*/llm_batch_*.json"))

    missed_root = llm_dir / "missed_recheck"
    missed_audit_path = missed_root / "audit.json"
    if missed_audit_path.exists():
        try:
            audit = json.loads(missed_audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            audit = {}
        for value in audit.get("active_debug_files", []):
            path = missed_root / str(value)
            if path.is_file():
                paths.add(path)

    relative_paths = sorted(
        str(path.relative_to(llm_dir)).replace("\\", "/")
        for path in paths
        if path.is_file()
    )
    (llm_dir / "valid_debug_files.json").write_text(
        json.dumps(relative_paths, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _match_overlap_count(matches: list[dict[str, Any]]) -> int:
    count = 0
    for left_index, left in enumerate(matches):
        left_indices = set(left.get("segment_indices", []))
        left_title = str(left.get("title", "")).strip().casefold()
        for right in matches[left_index + 1:]:
            right_title = str(right.get("title", "")).strip().casefold()
            if left_title == right_title:
                continue
            if left_indices & set(right.get("segment_indices", [])):
                count += 1
    return count


def _write_profile_comparison(profiles_root: Path) -> None:
    profiles: dict[str, dict[str, Any]] = {}
    if not profiles_root.exists():
        return
    for profile_dir in profiles_root.iterdir():
        if not profile_dir.is_dir():
            continue
        state_path = profile_dir / "profile.json"
        matches_path = profile_dir / "song" / "matches.json"
        if not state_path.exists() or not matches_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            matches = json.loads(matches_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("status") != "complete" or not isinstance(matches, list):
            continue
        usage = _profile_usage_totals(profile_dir)
        input_tokens = (
            usage["prompt_cache_hit_tokens"] + usage["prompt_cache_miss_tokens"]
        )
        profile_name = str(state.get("profile") or profile_dir.name)
        profiles[profile_name] = {
            "model": state.get("model"),
            "match_count": len(matches),
            "known_title_count": sum(
                1
                for match in matches
                if not str(match.get("title", "")).strip().startswith("未知歌曲")
            ),
            "unknown_title_count": sum(
                1
                for match in matches
                if str(match.get("title", "")).strip().startswith("未知歌曲")
            ),
            "different_title_overlap_pairs": _match_overlap_count(matches),
            **usage,
            "cache_hit_ratio": (
                usage["prompt_cache_hit_tokens"] / input_tokens
                if input_tokens
                else None
            ),
        }
    if len(profiles) < 2:
        return

    payload = {"profiles": profiles}
    (profiles_root / "profile_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# LLM profile comparison",
        "",
        "| Profile | Matches | Known | Unknown | Conflicts | Cache hit | Cache miss | Completion | Hit ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in sorted(profiles.items()):
        ratio = metrics["cache_hit_ratio"]
        ratio_text = f"{ratio:.1%}" if isinstance(ratio, float) else "n/a"
        lines.append(
            f"| {name} | {metrics['match_count']} | {metrics['known_title_count']} | "
            f"{metrics['unknown_title_count']} | {metrics['different_title_overlap_pairs']} | "
            f"{metrics['prompt_cache_hit_tokens']} | {metrics['prompt_cache_miss_tokens']} | "
            f"{metrics['completion_tokens']} | {ratio_text} |"
        )
    (profiles_root / "profile_comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


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
    naming_profile: Any = None,
) -> None:
    # 构建文件名：【streamername】summary-YYMMDD 或 summary
    if naming_profile and naming_profile.streamer and naming_profile.date:
        stem = f"【{naming_profile.streamer}】summary-{naming_profile.date}"
    else:
        stem = "summary"

    for target_dir in (llm_dir, reports_dir / content_type):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / f"{stem}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        formatter = getattr(recognizer, "format_summary_markdown", None)
        if callable(formatter):
            markdown = formatter(summary, config)
        else:
            markdown = "```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n"
        (target_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")


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

    def transcript_index_start(self, batch_start: int) -> int:
        return self._offset + batch_start

    def build_prompt(
        self,
        segments: list[TranscriptSegment],
        batch_start: int,
        config: dict[str, Any],
    ) -> str:
        return self._recognizer.build_prompt(segments, self._offset + batch_start, config)

    def build_system_prompt(self, config: dict[str, Any]) -> str | None:
        return self._recognizer.build_system_prompt(config)

    def parse_response(self, items: list[dict[str, Any]], config: dict[str, Any]) -> list[ContentMatch]:
        return self._recognizer.parse_response(items, config)

    def get_tools(self, config: dict[str, Any]) -> Any:
        return self._recognizer.get_tools(config)


class _SongReviewRecognizer:
    name = "song"

    def __init__(self, base_recognizer: Any, candidates: list[ContentMatch]) -> None:
        self._base_recognizer = base_recognizer
        self._candidates = candidates

    @property
    def default_config(self) -> dict[str, Any]:
        return getattr(self._base_recognizer, "default_config", {})

    def build_prompt(
        self,
        segments: list[TranscriptSegment],
        batch_start: int,
        config: dict[str, Any],
    ) -> str:
        lines = [
            f"[{batch_start + index}] ({segment.start:.1f}s-{segment.end:.1f}s) {segment.text}"
            for index, segment in enumerate(segments)
        ]
        candidates = []
        for match in self._candidates:
            candidates.append({
                "title": match.title,
                "artist": match.artist,
                "segment_ranges": _indices_to_ranges(match.segment_indices),
                "confidence": match.confidence,
            })
        return f"""你是歌曲识别结果复核器。请只裁决候选结果之间的冲突，不要总结整场内容。

候选结果：
{json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))}

要求：
1. 根据 ASR 歌词判断正确歌名和精确边界。
2. 删除重复、错误标题和非演唱片段。
3. 不同歌曲的 segment_ranges 不得重叠。
4. 只返回 JSON 数组，每项包含 content_type、title、artist、segment_ranges、confidence、tags、description。
5. 可以使用 search_lyrics 一次确认歌词；无法确认时保留“未知歌曲：代表歌词”。

完整 ASR 转写片段：
{chr(10).join(lines)}"""

    def build_system_prompt(self, config: dict[str, Any]) -> str | None:
        return None

    def parse_response(
        self,
        items: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[ContentMatch]:
        return self._base_recognizer.parse_response(items, config)

    def get_tools(self, config: dict[str, Any]) -> Any:
        return self._base_recognizer.get_tools(config)


class _SongCoverageAuditRecognizer:
    name = "song"

    def __init__(
        self,
        base_recognizer: Any,
        target_ranges: list[tuple[int, int]],
        candidates: list[ContentMatch],
    ) -> None:
        self._base_recognizer = base_recognizer
        self._target_ranges = target_ranges
        self._candidates = candidates

    @property
    def default_config(self) -> dict[str, Any]:
        return getattr(self._base_recognizer, "default_config", {})

    def build_prompt(
        self,
        segments: list[TranscriptSegment],
        batch_start: int,
        config: dict[str, Any],
    ) -> str:
        lines = [
            f"[{batch_start + index}] ({segment.start:.1f}s-{segment.end:.1f}s) {segment.text}"
            for index, segment in enumerate(segments)
        ]
        target_ranges = [[start, end] for start, end in self._target_ranges]
        return f"""你是歌曲漏检覆盖审计器。完整 ASR 已经做过一次歌曲识别。

以下范围由已有歌曲结果的覆盖区间反向计算得到。只检查这些目标区间：
{json.dumps(target_ranges, ensure_ascii=False, separators=(",", ":"))}

要求：
1. 未覆盖区间通常包含聊天、报幕、静音和歌曲之间的空隙。不要为了覆盖目标区间而输出结果。
2. 必须按时间顺序检查每个目标区间。连续多句 ASR 呈现歌词、押韵、重复或明显演唱结构时，必须返回；不要只寻找能确认歌名的歌曲。
3. 目标区间可能包含完整漏检歌曲，也可能包含已有歌曲未覆盖的演唱续段。两者都要返回精确的演唱子区间。
4. 不要把整个目标区间直接当作歌曲。聊天、感谢、报幕、点歌、歌曲讨论和零散哼唱必须排除。
5. 不要返回目标区间之外的 segment。本地会再次裁剪并与已有结果去重。
6. 搜索工具只用于确认歌名。搜索失败或无法确认时仍必须返回“未知歌曲：代表歌词”。
7. 使用紧凑 segment_ranges，区间起止均包含。confidence 必须明确填写 0 到 1。
8. 只返回 JSON 数组，每项包含 content_type、title、artist、segment_ranges、confidence、tags、description。
9. 先完整扫描全部目标区间并形成候选列表，再决定是否使用唯一一次歌词搜索。不能发现第一首候选后就停止扫描。
10. 搜索只用于给候选命名，不能让最终结果只包含被搜索的对象。没有搜索的候选也必须保留。
11. 不要输出逐区间分析、解释或 Markdown。不需要搜索时立即返回 JSON。
12. 如果没有足够证据证明存在漏检歌曲，返回合法空数组 []。

完整 ASR 转写片段：
{chr(10).join(lines)}"""

    def build_system_prompt(self, config: dict[str, Any]) -> str | None:
        return self._base_recognizer.build_system_prompt(config)

    def parse_response(
        self,
        items: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[ContentMatch]:
        return self._base_recognizer.parse_response(items, config)

    def get_tools(self, config: dict[str, Any]) -> Any:
        return self._base_recognizer.get_tools(config)


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


def _group_segment_ranges(
    ranges: list[tuple[int, int]],
    max_span_segments: int,
) -> list[list[tuple[int, int]]]:
    """把多个目标区间合并到有限跨度的请求窗口中。"""
    if not ranges:
        return []

    max_span_segments = max(1, max_span_segments)
    groups: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    group_start = 0

    for start, end in ranges:
        if not current:
            current = [(start, end)]
            group_start = start
            continue
        if end - group_start + 1 <= max_span_segments:
            current.append((start, end))
            continue
        groups.append(current)
        current = [(start, end)]
        group_start = start

    if current:
        groups.append(current)
    return groups


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


def _filter_matches_to_segment_ranges(
    matches: list[ContentMatch],
    ranges: list[tuple[int, int]],
) -> list[ContentMatch]:
    allowed: set[int] = set()
    for start, end in ranges:
        allowed.update(range(start, end + 1))

    filtered: list[ContentMatch] = []
    for match in matches:
        segment_indices = sorted(set(match.segment_indices) & allowed)
        if segment_indices:
            filtered.append(_clone_match_with_indices(match, segment_indices))
    return filtered


def _merge_adjacent_same_title_matches(
    segments: list[TranscriptSegment],
    matches: list[ContentMatch],
    max_gap_seconds: float,
) -> tuple[list[ContentMatch], list[dict[str, Any]]]:
    if max_gap_seconds < 0:
        return matches, []

    merged: list[ContentMatch] = []
    events: list[dict[str, Any]] = []
    for match in sorted(matches, key=lambda item: min(item.segment_indices)):
        candidate_index = None
        candidate_gap = None
        title_key = match.title.strip().casefold()
        for index in range(len(merged) - 1, -1, -1):
            previous = merged[index]
            if previous.title.strip().casefold() != title_key:
                continue
            previous_end = max(previous.segment_indices)
            match_start = min(match.segment_indices)
            if match_start <= previous_end:
                gap_seconds = 0.0
            else:
                gap_seconds = max(
                    0.0,
                    segments[match_start].start - segments[previous_end].end,
                )
            if gap_seconds <= max_gap_seconds:
                candidate_index = index
                candidate_gap = gap_seconds
            break

        if candidate_index is None:
            merged.append(match)
            continue

        previous = merged[candidate_index]
        combined = _clone_match_with_indices(
            previous if previous.confidence >= match.confidence else match,
            sorted(set(previous.segment_indices) | set(match.segment_indices)),
        )
        merged[candidate_index] = combined
        events.append({
            "type": "adjacent_same_title_merge",
            "title": combined.title,
            "gap_seconds": candidate_gap,
            "ranges": _indices_to_ranges(combined.segment_indices),
        })

    merged.sort(key=lambda item: min(item.segment_indices))
    return merged, events


def _is_invalid_audit_title(title: str) -> bool:
    normalized = title.strip().casefold()
    return normalized in {
        "",
        "unknown",
        "未知",
        "未知歌曲",
        "未知歌曲：代表歌词",
        "未知歌曲:代表歌词",
        "对话",
        "对话片段",
        "聊天",
        "聊天片段",
        "非歌曲",
        "无歌曲",
    }


def _sanitize_full_transcript_review_results(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    matches: list[ContentMatch],
    candidate_matches: list[ContentMatch],
    target_ranges: list[tuple[int, int]],
) -> tuple[list[ContentMatch], list[dict[str, Any]]]:
    candidate_coverage = {
        index
        for candidate in candidate_matches
        for index in candidate.segment_indices
    }
    events: list[dict[str, Any]] = []
    sanitized: list[ContentMatch] = []
    for match in matches:
        if (
            match.content_type.strip().casefold()
            not in {"song", "music"}
            or _is_invalid_audit_title(match.title)
        ):
            events.append({
                "type": "invalid_audit_result",
                "title": match.title,
                "content_type": match.content_type,
                "ranges": _indices_to_ranges(match.segment_indices),
            })
            continue
        if candidate_coverage.intersection(match.segment_indices):
            sanitized.append(match)
            continue
        cropped = _filter_matches_to_segment_ranges([match], target_ranges)
        if not cropped:
            events.append({
                "type": "review_result_outside_audit_targets",
                "title": match.title,
                "ranges": _indices_to_ranges(match.segment_indices),
            })
            continue
        if cropped[0].segment_indices != match.segment_indices:
            events.append({
                "type": "review_result_cropped_to_audit_targets",
                "title": match.title,
                "before": _indices_to_ranges(match.segment_indices),
                "after": _indices_to_ranges(cropped[0].segment_indices),
            })
        sanitized.extend(cropped)

    sanitized, merge_events = _merge_adjacent_same_title_matches(
        segments,
        sanitized,
        float(
            get_padding_config(config, "song").get(
                "merge_gap_seconds",
                20.0,
            )
        ),
    )
    events.extend(merge_events)
    sanitized, normalization_events, _ = _normalize_song_matches(
        segments,
        config,
        sanitized,
    )
    events.extend(normalization_events)
    return sanitized, events


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


def _content_match_from_dict(
    item: dict[str, Any],
    default_content_type: str = "song",
) -> ContentMatch:
    return ContentMatch(
        content_type=str(item.get("content_type", default_content_type)),
        title=str(item.get("title", "")),
        segment_indices=[
            int(index)
            for index in item.get("segment_indices", [])
            if not isinstance(index, bool)
        ],
        confidence=float(item.get("confidence", 0.5)),
        tags=list(item.get("tags", [])),
        description=str(item.get("description", "")),
        artist=str(item.get("artist", "")),
        lyrics_snippet=str(item.get("lyrics_snippet", "")),
    )


def _load_cached_identify_matches(
    debug_dir: Path,
    recognizer: Any,
    config: dict[str, Any],
) -> list[ContentMatch] | None:
    paths = sorted(debug_dir.glob("llm_batch_*.json"))
    if not paths:
        return None
    matches: list[ContentMatch] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("error") or payload.get("parse_valid") is not True:
            return None
        items = payload.get("parsed_items")
        if not isinstance(items, list):
            return None
        matches.extend(recognizer.parse_response(items, config))
    return matches


def _llm_debug_structural_failures(debug_dir: Path) -> list[str]:
    paths = sorted(debug_dir.glob("llm_batch_*.json"))
    if not paths:
        return ["missing_debug"]

    failures: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append("invalid_debug_json")
            continue
        if payload.get("error"):
            failures.append("api_error")
        if payload.get("parse_valid") is not True:
            failures.append("invalid_result_json")
        if payload.get("json_fix_rounds"):
            failures.append("json_repair")
        if any(
            item.get("finish_reason") == "length"
            for item in payload.get("tool_rounds", [])
            if isinstance(item, dict)
        ):
            failures.append("output_truncated")
    return sorted(set(failures))


def _indices_to_ranges(indices: list[int]) -> list[list[int]]:
    values = sorted(set(indices))
    if not values:
        return []
    ranges: list[list[int]] = []
    start = previous = values[0]
    for index in values[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append([start, previous])
        start = previous = index
    ranges.append([start, previous])
    return ranges


def _match_key(match: ContentMatch) -> tuple[str, tuple[int, ...]]:
    return (match.title.strip().casefold(), tuple(sorted(set(match.segment_indices))))


def _matches_overlap(left: ContentMatch, right: ContentMatch) -> bool:
    return bool(set(left.segment_indices) & set(right.segment_indices))


def _normalize_song_matches(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    matches: list[ContentMatch],
) -> tuple[list[ContentMatch], list[dict[str, Any]], set[tuple[str, tuple[int, ...]]]]:
    padding_config = get_padding_config(config, "song")
    merge_gap_seconds = float(padding_config.get("merge_gap_seconds", 20.0))
    max_song_seconds = float(padding_config.get("max_song_seconds", 360.0))
    normalized: list[ContentMatch] = []
    events: list[dict[str, Any]] = []
    suspicious: set[tuple[str, tuple[int, ...]]] = set()
    seen: set[tuple[str, tuple[int, ...]]] = set()

    for match in matches:
        valid_indices = sorted({i for i in match.segment_indices if 0 <= i < len(segments)})
        if not valid_indices:
            events.append({"type": "invalid_range", "title": match.title})
            continue
        groups = _split_indices_by_time_gap_for_recheck(
            segments,
            valid_indices,
            merge_gap_seconds,
        )
        if len(groups) > 1:
            events.append({
                "type": "disjoint_ranges",
                "title": match.title,
                "ranges": [_indices_to_ranges(group) for group in groups],
            })
        for group in groups:
            candidate = _clone_match_with_indices(match, group)
            key = _match_key(candidate)
            if key in seen:
                events.append({
                    "type": "exact_duplicate",
                    "title": candidate.title,
                    "ranges": _indices_to_ranges(candidate.segment_indices),
                })
                continue
            seen.add(key)
            normalized.append(candidate)
            if len(groups) > 1:
                suspicious.add(key)
            duration = _segment_range_duration_seconds(
                segments,
                min(group),
                max(group),
            )
            if max_song_seconds > 0 and duration > max_song_seconds:
                suspicious.add(key)
                events.append({
                    "type": "overlong",
                    "title": candidate.title,
                    "duration_seconds": duration,
                })

    normalized.sort(key=lambda match: min(match.segment_indices))
    return normalized, events, suspicious


def _build_song_review_clusters(
    matches: list[ContentMatch],
    suspicious: set[tuple[str, tuple[int, ...]]],
    max_span_segments: int = 500,
    nearby_title_conflict_gap_segments: int = -1,
) -> list[list[ContentMatch]]:
    adjacency: list[set[int]] = [set() for _ in matches]
    included: set[int] = set()
    for index, match in enumerate(matches):
        if _match_key(match) in suspicious:
            included.add(index)
    for left_index, left in enumerate(matches):
        for right_index in range(left_index + 1, len(matches)):
            right = matches[right_index]
            same_title = left.title.strip().casefold() == right.title.strip().casefold()
            same_suspicious_source = (
                same_title
                and _match_key(left) in suspicious
                and _match_key(right) in suspicious
                and (
                    max(max(left.segment_indices), max(right.segment_indices))
                    - min(min(left.segment_indices), min(right.segment_indices))
                    + 1
                    <= max(1, max_span_segments)
                )
            )
            left_start, left_end = min(left.segment_indices), max(left.segment_indices)
            right_start, right_end = min(right.segment_indices), max(right.segment_indices)
            if left_end < right_start:
                gap_segments = right_start - left_end - 1
            elif right_end < left_start:
                gap_segments = left_start - right_end - 1
            else:
                gap_segments = -1
            nearby_title_conflict = (
                not same_title
                and nearby_title_conflict_gap_segments >= 0
                and 0 <= gap_segments <= nearby_title_conflict_gap_segments
                and max(left_end, right_end) - min(left_start, right_start) + 1
                <= max(1, max_span_segments)
            )
            if (
                same_suspicious_source
                or (not same_title and _matches_overlap(left, right))
                or nearby_title_conflict
            ):
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
                included.update({left_index, right_index})

    clusters: list[list[ContentMatch]] = []
    visited: set[int] = set()
    for start in sorted(included):
        if start in visited:
            continue
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)
        clusters.append([matches[index] for index in sorted(component)])
    return clusters


def _llm_debug_has_structural_issue(debug_root: Path) -> bool:
    for path in debug_root.glob("llm_batch_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("json_fix_rounds"):
            return True
        for tool_round in payload.get("tool_rounds", []):
            if tool_round.get("finish_reason") == "length":
                return True
    return False


def _review_debug_succeeded(debug_root: Path) -> bool:
    paths = list(debug_root.glob("llm_batch_*.json"))
    if not paths:
        return False
    try:
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    except (OSError, json.JSONDecodeError):
        return False
    return all(not payload.get("error") and payload.get("parse_valid") is True for payload in payloads)


def _load_searched_titles(debug_root: Path) -> set[str]:
    titles: set[str] = set()
    for path in debug_root.rglob("llm_batch_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for call in payload.get("tool_calls_log", []):
            arguments = call.get("arguments", {})
            title = str(arguments.get("title", "")).strip()
            if title:
                titles.add(title.casefold())
    return titles


def _local_best_song_cluster(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    cluster: list[ContentMatch],
    searched_titles: set[str],
) -> tuple[list[ContentMatch], list[dict[str, Any]]]:
    padding_config = get_padding_config(config, "song")
    max_song_seconds = float(padding_config.get("max_song_seconds", 360.0))
    ranked: list[tuple[tuple[Any, ...], int, ContentMatch]] = []
    for index, match in enumerate(cluster):
        duration = _segment_range_duration_seconds(
            segments,
            min(match.segment_indices),
            max(match.segment_indices),
        )
        title = match.title.strip()
        known_title = not title.startswith("未知歌曲")
        has_search_evidence = any(
            title.casefold() in searched or searched in title.casefold()
            for searched in searched_titles
        )
        duration_valid = max_song_seconds <= 0 or duration <= max_song_seconds
        score = (
            int(has_search_evidence),
            int(known_title),
            float(match.confidence),
            int(duration_valid),
            -abs(duration - min(max_song_seconds, duration)) if max_song_seconds > 0 else 0.0,
            -index,
        )
        ranked.append((score, index, match))

    selected: list[ContentMatch] = []
    decisions: list[dict[str, Any]] = []
    for score, original_index, match in sorted(ranked, key=lambda item: item[0], reverse=True):
        if any(_matches_overlap(match, kept) for kept in selected):
            decisions.append({
                "title": match.title,
                "action": "discard",
                "reason": "lower local evidence score in overlapping cluster",
                "score": list(score),
            })
            continue
        selected.append(match)
        decisions.append({
            "title": match.title,
            "action": "keep",
            "reason": "highest non-overlapping local evidence score",
            "score": list(score),
            "original_index": original_index,
        })
    selected.sort(key=lambda match: min(match.segment_indices))
    return selected, decisions


def _review_loses_known_title(
    cluster: list[ContentMatch],
    reviewed_matches: list[ContentMatch],
) -> bool:
    return (
        any(not match.title.strip().startswith("未知歌曲") for match in cluster)
        and not any(
            not match.title.strip().startswith("未知歌曲")
            for match in reviewed_matches
        )
    )


def _review_song_matches(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    llm_dir: Path,
    *,
    phase: str,
) -> list[ContentMatch]:
    review_config = config.get("song", {}).get("review", {})
    if review_config.get("enabled", False) is False:
        return matches

    normalized, normalization_events, suspicious = _normalize_song_matches(
        segments,
        config,
        matches,
    )
    if phase == "before_missed_recheck" and _llm_debug_has_structural_issue(llm_dir):
        suspicious.update(_match_key(match) for match in normalized)
        normalization_events.append({"type": "main_output_truncated_or_repaired"})
    max_cluster_span = max(
        1,
        int(review_config.get("max_window_segments", 500) or 500)
        - 2 * int(review_config.get("context_segments", 10) or 0),
    )
    clusters = _build_song_review_clusters(
        normalized,
        suspicious,
        max_span_segments=max_cluster_span,
        nearby_title_conflict_gap_segments=int(
            review_config.get("nearby_title_conflict_gap_segments", 2) or 0
        ),
    )
    review_root = llm_dir / "review" / phase
    review_root.mkdir(parents=True, exist_ok=True)
    full_audit: dict[str, Any] | None = None
    full_audit_candidate_keys: set[tuple[str, tuple[int, ...]]] | None = None
    if (
        phase == "after_missed_recheck"
        and str(
            config.get("song", {})
            .get("missed_recheck", {})
            .get("strategy", "windowed")
        ).strip().lower() == "full_transcript"
    ):
        full_audit_path = llm_dir / "missed_recheck" / "audit.json"
        try:
            loaded_audit = json.loads(full_audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded_audit = None
        if isinstance(loaded_audit, dict):
            full_audit = loaded_audit
            full_audit_candidates = [
                _content_match_from_dict(item)
                for item in loaded_audit.get("candidates", [])
                if isinstance(item, dict)
            ]
            full_audit_candidate_keys = {
                _match_key(match) for match in full_audit_candidates
            }
    searched_titles = _load_searched_titles(llm_dir)
    context_segments = int(review_config.get("context_segments", 10) or 0)
    max_window_segments = int(review_config.get("max_window_segments", 500) or 500)
    resolved: list[ContentMatch] = []
    cluster_member_ids = {id(match) for cluster in clusters for match in cluster}
    resolved.extend(match for match in normalized if id(match) not in cluster_member_ids)
    audit_clusters: list[dict[str, Any]] = []

    for cluster_index, cluster in enumerate(clusters, 1):
        target_start = min(min(match.segment_indices) for match in cluster)
        target_end = max(max(match.segment_indices) for match in cluster)
        context_start, context_end = _expand_segment_range(
            target_start,
            target_end,
            len(segments),
            context_segments,
        )
        audit: dict[str, Any] = {
            "cluster": cluster_index,
            "target_range": [target_start, target_end],
            "context_range": [context_start, context_end],
            "before": [match.to_dict() for match in cluster],
            "resolution": None,
            "after": [],
        }
        if (
            full_audit_candidate_keys is not None
            and all(_match_key(match) in full_audit_candidate_keys for match in cluster)
        ):
            resolved.extend(cluster)
            audit["resolution"] = "unchanged_pre_audit_cluster"
            audit["after"] = [match.to_dict() for match in cluster]
            audit_clusters.append(audit)
            continue
        reviewed_matches: list[ContentMatch] = []
        review_succeeded = False
        audit_path = review_root / f"cluster_{cluster_index:03d}.json"
        current_before = [match.to_dict() for match in cluster]
        if audit_path.exists():
            try:
                cached_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached_audit = None
            if (
                isinstance(cached_audit, dict)
                and cached_audit.get("before") == current_before
                and cached_audit.get("resolution") == "llm"
                and isinstance(cached_audit.get("after"), list)
            ):
                replacement = [
                    _content_match_from_dict(item)
                    for item in cached_audit["after"]
                    if isinstance(item, dict) and item.get("title")
                ]
                if not _review_loses_known_title(cluster, replacement):
                    resolved.extend(replacement)
                    audit_clusters.append(cached_audit)
                    continue
        if context_end - context_start + 1 <= max_window_segments:
            local_config = deepcopy(config)
            local_config["llm"]["max_tokens"] = int(
                review_config.get("max_completion_tokens", 4096) or 4096
            )
            local_config["llm"]["max_completion_tokens"] = int(
                review_config.get("max_completion_tokens", 4096) or 4096
            )
            local_config["llm"]["final_tool_max_tokens"] = int(
                review_config.get("max_completion_tokens", 4096) or 4096
            )
            local_config["llm"]["max_tool_rounds"] = int(
                review_config.get("max_tool_rounds", 1) or 0
            )
            local_config["llm"]["compact_segment_ranges"] = True
            review_recognizer = _SongReviewRecognizer(recognizer, cluster)
            offset_recognizer = _OffsetRecognizer(review_recognizer, context_start)
            cluster_debug_dir = review_root / f"cluster_{cluster_index:03d}"
            from .llm import identify_content
            reviewed_matches = identify_content(
                segments[context_start:context_end + 1],
                local_config,
                offset_recognizer,
                debug_dir=cluster_debug_dir,
            )
            reviewed_matches = _filter_matches_to_segment_range(
                reviewed_matches,
                context_start,
                context_end,
            )
            reviewed_matches, review_events, _ = _normalize_song_matches(
                segments,
                config,
                reviewed_matches,
            )
            has_conflict = any(
                left.title.strip().casefold() != right.title.strip().casefold()
                and _matches_overlap(left, right)
                for left_index, left in enumerate(reviewed_matches)
                for right in reviewed_matches[left_index + 1:]
            )
            loses_known_title = _review_loses_known_title(
                cluster,
                reviewed_matches,
            )
            review_succeeded = (
                _review_debug_succeeded(cluster_debug_dir)
                and not has_conflict
                and not loses_known_title
            )
            audit["review_normalization_events"] = review_events
            if loses_known_title:
                audit["review_rejected_reason"] = "known_title_downgraded_to_unknown"
        else:
            audit["skip_reason"] = "review window exceeds max_window_segments"

        if review_succeeded:
            replacement = reviewed_matches
            audit["resolution"] = "llm"
        else:
            replacement, decisions = _local_best_song_cluster(
                segments,
                config,
                cluster,
                searched_titles,
            )
            audit["resolution"] = "local_best"
            audit["local_decisions"] = decisions
        resolved.extend(replacement)
        audit["after"] = [match.to_dict() for match in replacement]
        audit_clusters.append(audit)
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    final_matches, final_events, _ = _normalize_song_matches(segments, config, resolved)
    residual_clusters = _build_song_review_clusters(final_matches, set())
    if residual_clusters:
        residual_ids = {id(match) for cluster in residual_clusters for match in cluster}
        conflict_free = [match for match in final_matches if id(match) not in residual_ids]
        for cluster in residual_clusters:
            replacement, decisions = _local_best_song_cluster(
                segments,
                config,
                cluster,
                searched_titles,
            )
            conflict_free.extend(replacement)
            final_events.append({
                "type": "residual_conflict_local_best",
                "decisions": decisions,
            })
        final_matches = sorted(
            conflict_free,
            key=lambda match: min(match.segment_indices),
        )
    sanitation_events: list[dict[str, Any]] = []
    if full_audit is not None:
        full_audit_candidates = [
            _content_match_from_dict(item)
            for item in full_audit.get("candidates", [])
            if isinstance(item, dict)
        ]
        target_ranges = [
            (int(item[0]), int(item[1]))
            for item in full_audit.get("target_ranges", [])
            if isinstance(item, list) and len(item) == 2
        ]
        final_matches, sanitation_events = _sanitize_full_transcript_review_results(
            segments,
            config,
            final_matches,
            full_audit_candidates,
            target_ranges,
        )
    summary = {
        "phase": phase,
        "input_count": len(matches),
        "normalized_count": len(normalized),
        "cluster_count": len(clusters),
        "output_count": len(final_matches),
        "normalization_events": normalization_events,
        "final_normalization_events": final_events,
        "full_transcript_sanitation_events": sanitation_events,
        "clusters": audit_clusters,
    }
    (review_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return final_matches


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

    for mi, match in enumerate(matches, 1):
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
        for gi, group in enumerate(groups, 1):
            group_start = min(group)
            group_end = max(group)
            group_match = _clone_match_with_indices(match, group)
            if _segment_range_duration_seconds(segments, group_start, group_end) <= max_song_seconds:
                replacement_matches.append(group_match)
                continue

            rechecked_count += 1
            print(f"  Song overlong recheck: match {mi}/{len(matches)}, group {gi}/{len(groups)} (segments {group_start}-{group_end})...")
            context_start, context_end = _expand_segment_range(
                group_start,
                group_end,
                len(segments),
                context_segments,
            )
            chunk = segments[context_start:context_end + 1]
            debug_dir = recheck_root / f"{group_start:06d}_{group_end:06d}"
            offset_recognizer = _OffsetRecognizer(recognizer, context_start)
            cached_matches = _load_cached_identify_matches(
                debug_dir,
                offset_recognizer,
                config,
            )
            raw_rechecked_matches = (
                cached_matches
                if cached_matches is not None
                else identify_content(chunk, config, offset_recognizer, debug_dir=debug_dir)
            )
            rechecked_matches = _filter_matches_to_segment_range(
                raw_rechecked_matches,
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


def _run_windowed_missed_recheck(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    recheck_root: Path,
) -> tuple[list[ContentMatch], list[str]]:
    recheck_config = config.get("song", {}).get("missed_recheck", {})
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
        return [], []

    from .llm import identify_content

    range_groups = _group_segment_ranges(ranges, batch_size)
    print(
        f"  Song missed recheck: {len(ranges)} uncovered ASR range(s), "
        f"combined into {len(range_groups)} request window(s)"
    )
    extra_matches: list[ContentMatch] = []
    recheck_root.mkdir(parents=True, exist_ok=True)
    active_debug_files: list[str] = []

    for ri, target_ranges in enumerate(range_groups, 1):
        start = target_ranges[0][0]
        end = target_ranges[-1][1]
        context_start, context_end = _expand_segment_range(
            start,
            end,
            len(segments),
            context_segments,
        )
        chunk = segments[context_start:context_end + 1]
        debug_dir = recheck_root / f"{start:06d}_{end:06d}"
        offset_recognizer = _OffsetRecognizer(recognizer, context_start)
        print(
            f"  Song missed recheck: processing window {ri}/{len(range_groups)} "
            f"(segments {start}-{end}, {len(target_ranges)} target range(s))..."
        )
        cached_matches = _load_cached_identify_matches(
            debug_dir,
            offset_recognizer,
            config,
        )
        if cached_matches is not None:
            rechecked_matches = cached_matches
            print(
                f"  Song missed recheck: reused cached result for window {ri}/{len(range_groups)}"
            )
        else:
            rechecked_matches = identify_content(
                chunk,
                config,
                offset_recognizer,
                debug_dir=debug_dir,
            )
        active_debug_files.extend(
            str(path.relative_to(recheck_root)).replace("\\", "/")
            for path in sorted(debug_dir.glob("llm_batch_*.json"))
        )
        for target_start, target_end in target_ranges:
            extra_matches.extend(
                _filter_matches_to_segment_range(
                    rechecked_matches,
                    target_start,
                    target_end,
                )
            )
        print(
            f"  Song missed recheck: window {ri}/{len(range_groups)} done, "
            f"found {len(rechecked_matches)} match(es)"
        )

    if extra_matches:
        (recheck_root / "matches.json").write_text(
            json.dumps([m.to_dict() for m in extra_matches], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Song missed recheck: found {len(extra_matches)} additional match(es)")
        return extra_matches, active_debug_files

    print("  Song missed recheck: no additional matches")
    return [], active_debug_files


def _missed_recheck_fingerprint(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    matches: list[ContentMatch],
    target_ranges: list[tuple[int, int]],
) -> dict[str, str]:
    candidates = sorted(
        (
            {
                "title": match.title,
                "artist": match.artist,
                "segment_ranges": _indices_to_ranges(match.segment_indices),
                "confidence": match.confidence,
            }
            for match in matches
        ),
        key=lambda item: (
            item["segment_ranges"][0][0] if item["segment_ranges"] else -1,
            item["title"],
        ),
    )
    return {
        "protocol": "full_transcript_coverage_audit_v5",
        "transcript": _transcript_fingerprint(segments),
        "candidates": _fingerprint_payload(candidates),
        "target_ranges": _fingerprint_payload(target_ranges),
        "config": _config_fingerprint(config),
    }


def _load_missed_recheck_audit(
    audit_path: Path,
    fingerprints: dict[str, str],
) -> dict[str, Any] | None:
    if not audit_path.exists():
        return None
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if audit.get("fingerprints") != fingerprints:
        return None
    return audit


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

    strategy = str(recheck_config.get("strategy", "windowed")).strip().lower()
    if strategy not in {"windowed", "full_transcript"}:
        raise ValueError(
            "song.missed_recheck.strategy must be 'windowed' or 'full_transcript'"
        )

    recheck_root = llm_dir / "missed_recheck"
    recheck_root.mkdir(parents=True, exist_ok=True)
    if strategy == "windowed":
        extra_matches, active_files = _run_windowed_missed_recheck(
            segments,
            config,
            recognizer,
            matches,
            recheck_root,
        )
        (recheck_root / "audit.json").write_text(
            json.dumps(
                {
                    "strategy": "windowed",
                    "status": "success",
                    "fallback_used": False,
                    "active_debug_files": active_files,
                    "additional_match_count": len(extra_matches),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return [*matches, *extra_matches]

    min_gap_segments = int(recheck_config.get("min_gap_segments", 1) or 1)
    min_song_seconds = float(
        get_padding_config(config, "song").get("min_song_seconds", 75.0)
    )
    target_ranges, skipped_short = _filter_short_segment_ranges(
        segments,
        _uncovered_segment_ranges(
            len(segments),
            matches,
            min_gap_segments=min_gap_segments,
        ),
        min_song_seconds,
    )
    if skipped_short:
        print(
            f"  Song missed recheck: skipped {skipped_short} short ASR range(s) "
            f"below min_song_seconds={min_song_seconds:g}"
        )
    if not target_ranges:
        return matches

    fingerprints = _missed_recheck_fingerprint(
        segments,
        config,
        matches,
        target_ranges,
    )
    audit_path = recheck_root / "audit.json"
    cached_audit = _load_missed_recheck_audit(audit_path, fingerprints)
    fallback_strategy = str(
        recheck_config.get(
            "fallback_strategy",
            "windowed_on_structural_failure",
        )
    ).strip().lower()

    if cached_audit and cached_audit.get("status") == "fallback_success":
        extra_matches, active_files = _run_windowed_missed_recheck(
            segments,
            config,
            recognizer,
            matches,
            recheck_root / "fallback_windowed",
        )
        cached_audit["active_debug_files"] = [
            f"fallback_windowed/{path}" for path in active_files
        ]
        audit_path.write_text(
            json.dumps(cached_audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return [*matches, *extra_matches]

    audit_recognizer = _SongCoverageAuditRecognizer(
        recognizer,
        target_ranges,
        matches,
    )
    audit_debug_dir = recheck_root / "full_transcript"
    raw_matches: list[ContentMatch] | None = None
    cache_reused = False
    if cached_audit and cached_audit.get("status") == "success":
        if not _llm_debug_structural_failures(audit_debug_dir):
            raw_matches = _load_cached_identify_matches(
                audit_debug_dir,
                audit_recognizer,
                config,
            )
            cache_reused = raw_matches is not None

    if raw_matches is None:
        from .llm import identify_content

        local_config = deepcopy(config)
        max_completion_tokens = int(
            recheck_config.get("max_completion_tokens", 4096) or 4096
        )
        local_config["llm"]["max_tokens"] = max_completion_tokens
        local_config["llm"]["max_completion_tokens"] = max_completion_tokens
        local_config["llm"]["final_tool_max_tokens"] = max_completion_tokens
        local_config["llm"]["max_tool_rounds"] = int(
            recheck_config.get("max_tool_rounds", 1) or 0
        )
        local_config["llm"]["force_final_tool_round"] = True
        local_config["llm"]["final_tool_instruction"] = (
            "歌词搜索只用于确认某个候选的名称，不能替代完整覆盖审计。"
            "现在重新检查原任务列出的每一个目标区间，返回所有有明确连续演唱证据的候选；"
            "每个 segment_ranges 必须与目标区间相交，不能只返回刚刚搜索的对象，"
            "也不能返回已有覆盖区间。不要再调用工具。只返回 JSON 数组，不要解释或 Markdown。"
        )
        local_config["llm"]["retry_empty_with_reasoning"] = False
        local_config["llm"]["json_fix_rounds"] = 0
        raw_matches = identify_content(
            segments,
            local_config,
            audit_recognizer,
            debug_dir=audit_debug_dir,
        )

    structural_failures = _llm_debug_structural_failures(audit_debug_dir)
    active_debug_files = [
        f"full_transcript/{path.name}"
        for path in sorted(audit_debug_dir.glob("llm_batch_*.json"))
    ]
    audit: dict[str, Any] = {
        "strategy": "full_transcript",
        "status": "success" if not structural_failures else "structural_failure",
        "fallback_strategy": fallback_strategy,
        "fallback_used": False,
        "fingerprints": fingerprints,
        "target_ranges": [[start, end] for start, end in target_ranges],
        "candidate_count": len(matches),
        "candidates": [match.to_dict() for match in matches],
        "cache_reused": cache_reused,
        "structural_failures": structural_failures,
        "active_debug_files": active_debug_files,
    }
    review_trigger_matches: list[ContentMatch] = []

    if (
        structural_failures
        and fallback_strategy == "windowed_on_structural_failure"
    ):
        print(
            "  Song missed recheck: full transcript audit failed structurally "
            f"({', '.join(structural_failures)}); falling back to windowed"
        )
        extra_matches, fallback_files = _run_windowed_missed_recheck(
            segments,
            config,
            recognizer,
            matches,
            recheck_root / "fallback_windowed",
        )
        audit["status"] = "fallback_success"
        audit["fallback_used"] = True
        audit["active_debug_files"] = [
            f"fallback_windowed/{path}" for path in fallback_files
        ]
    elif structural_failures:
        extra_matches = []
    else:
        cropped_matches = _filter_matches_to_segment_ranges(
            raw_matches,
            target_ranges,
        )
        review_trigger_matches = [
            match
            for match in cropped_matches
            if _is_invalid_audit_title(match.title)
        ]
        rejected_titles = [
            {
                "title": match.title,
                "ranges": _indices_to_ranges(match.segment_indices),
            }
            for match in review_trigger_matches
        ]
        if rejected_titles:
            audit["rejected_invalid_titles"] = rejected_titles
        extra_matches = [
            match
            for match in cropped_matches
            if not _is_invalid_audit_title(match.title)
        ]
        if config.get("song", {}).get("review", {}).get("enabled", False):
            audit["review_trigger_count"] = len(review_trigger_matches)
        else:
            review_trigger_matches = []

    audit["additional_match_count"] = len(extra_matches)
    combined_matches = [
        *matches,
        *extra_matches,
        *review_trigger_matches,
    ]
    if not structural_failures:
        combined_matches, merge_events = _merge_adjacent_same_title_matches(
            segments,
            combined_matches,
            float(
                get_padding_config(config, "song").get(
                    "merge_gap_seconds",
                    20.0,
                )
            ),
        )
        audit["same_title_merge_events"] = merge_events
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if extra_matches:
        (recheck_root / "matches.json").write_text(
            json.dumps(
                [match.to_dict() for match in extra_matches],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  Song missed recheck: found {len(extra_matches)} additional match(es)")
    else:
        print("  Song missed recheck: no additional matches")
    return combined_matches


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
    profile_enabled = bool(config.get("_profile_enabled", False))
    profile_name = safe_path_part(str(config.get("_profile_name") or "default"))
    llm_base_dir = asr_dir / "llm"
    clips_dir = out / "03_clips"
    reports_dir = out / "04_reports"
    if profile_enabled:
        llm_base_dir = llm_base_dir / profile_name
        clips_dir = clips_dir / profile_name
        reports_dir = reports_dir / profile_name
    for d in [audio_dir, asr_dir, llm_base_dir, clips_dir, reports_dir]:
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

    config_fingerprint = _config_fingerprint(config)
    transcript_fingerprint = _transcript_fingerprint(segments)
    profile_state_path = llm_base_dir / "profile.json"
    profile_reusable = (
        profile_enabled
        and _profile_state_matches(
            profile_state_path,
            input_path=input_path,
            config_fingerprint=config_fingerprint,
            transcript_fingerprint=transcript_fingerprint,
        )
    )
    if profile_enabled and not profile_reusable:
        _write_profile_state(
            profile_state_path,
            input_path=input_path,
            config=config,
            config_fingerprint=config_fingerprint,
            transcript_fingerprint=transcript_fingerprint,
            status="running",
        )

    # Step 3: LLM 识别（通过识别器架构）
    print("[3/3] Identifying content with LLM...")
    
    all_results: dict[str, list[ContentResult]] = {}

    for ct_idx, content_type in enumerate(content_types, 1):
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

        print(f"\n  === {content_type} 识别 ({ct_idx}/{len(content_types)}) ===")
        llm_dir = llm_base_dir / content_type
        llm_dir.mkdir(parents=True, exist_ok=True)

        if _is_summary_only(recognizer, config):
            reuse_summary = False
            summary = None
            if prev_progress and (not profile_enabled or profile_reusable):
                summary = _load_previous_summary(llm_dir)
                reuse_summary = summary is not None

            if reuse_summary:
                print("  LLM 总结: 复用已有结果")
            else:
                from .llm import identify_structured_content
                summary = identify_structured_content(segments, config, recognizer, debug_dir=llm_dir)

            _write_structured_summary(summary or {}, recognizer, llm_dir, reports_dir, content_type, config, naming_profile)
            _write_valid_debug_manifest(llm_dir)
            print(f"  Wrote {content_type} summary")
            all_results[content_type] = []
            continue

        # 检查是否复用 LLM 结果
        reuse_llm = False
        if prev_progress and (not profile_enabled or profile_reusable):
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
                (llm_dir / "initial_matches.json").write_text(
                    json.dumps([m.to_dict() for m in matches], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                matches = _review_song_matches(
                    segments,
                    config,
                    recognizer,
                    matches,
                    llm_dir,
                    phase="before_missed_recheck",
                )
                matches = _recheck_overlong_song_matches(
                    segments, config, recognizer, matches, llm_dir
                )
                matches = _recheck_uncovered_song_segments(
                    segments, config, recognizer, matches, llm_dir
                )
                matches = _review_song_matches(
                    segments,
                    config,
                    recognizer,
                    matches,
                    llm_dir,
                    phase="after_missed_recheck",
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

        _write_valid_debug_manifest(llm_dir)
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
        "profile": config.get("_profile_name"),
        "total_duration": total_duration,
        "segment_count": len(segments),
        "content_types": {ct: len(results) for ct, results in all_results.items()},
        "config": {
            "asr_model": resolve_asr_model_name(config["asr"]),
            "llm_model": config["llm"]["model"],
        },
    }
    manifest_path = out / (
        f"manifest.{profile_name}.json" if profile_enabled else "manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if profile_enabled:
        _write_profile_state(
            profile_state_path,
            input_path=input_path,
            config=config,
            config_fingerprint=config_fingerprint,
            transcript_fingerprint=transcript_fingerprint,
            status="complete",
        )
        _write_profile_comparison(asr_dir / "llm")
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
