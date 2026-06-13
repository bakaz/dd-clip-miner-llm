from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..config import get_llm_config, get_padding_config, get_song_recheck_config, get_song_review_config
from ..models import ContentMatch, TranscriptSegment
from ..song_adaptive import (
    ensure_song_adaptive_strategies,
    load_adaptive_strategies_cache,
    resolve_review_transcript_scope,
)
from .normalize import (
    _clone_match_with_indices,
    _content_match_from_dict,
    _expand_segment_range,
    _filter_matches_to_segment_range,
    _filter_matches_to_segment_ranges,
    _filter_short_segment_ranges,
    _indices_to_ranges,
    _is_invalid_audit_title,
    _match_key,
    _matches_overlap,
    _merge_adjacent_same_title_matches,
    _normalize_song_matches,
    _segment_range_duration_seconds,
    _split_indices_by_time_gap_for_recheck,
    _uncovered_segment_ranges,
)
from .risk import (
    load_supported_search_titles,
    score_song_match_risks,
)
from ..config import is_risk_routed_v3


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
5. 可以使用 search_lyrics 一次确认歌词。查询必须使用 ASR 中最有辨识度的歌词原句，不要只用候选歌名反向搜索；无法确认时保留“未知歌曲：代表歌词”。

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


class _SongFullReviewRecognizer:
    name = "song"

    def __init__(
        self,
        base_recognizer: Any,
        candidates: list[ContentMatch],
        *,
        target_start: int,
        target_end: int,
        allowed_start: int,
        allowed_end: int,
    ) -> None:
        self._base_recognizer = base_recognizer
        self._candidates = candidates
        self._target_start = target_start
        self._target_end = target_end
        self._allowed_start = allowed_start
        self._allowed_end = allowed_end

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

目标 segment ranges：
[[{self._target_start},{self._target_end}]]

允许输出的 segment ranges 必须落在：
[[{self._allowed_start},{self._allowed_end}]]

要求：
1. 根据 ASR 歌词判断正确歌名和精确边界。
2. 删除重复、错误标题和非演唱片段。
3. 不同歌曲的 segment_ranges 不得重叠。
4. 只返回 JSON 数组，每项包含 content_type、title、artist、segment_ranges、confidence、tags、description。
5. 可以使用 search_lyrics 一次确认歌词。查询必须使用 ASR 中最有辨识度的歌词原句，不要只用候选歌名反向搜索；无法确认时保留“未知歌曲：代表歌词”。

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
        output_mode = str(
            config.get("song", {}).get("missed_recheck", {}).get("output_mode", "matches")
        ).strip().lower()
        anchor_instruction = ""
        naming_instruction = """6. 搜索工具只用于确认歌名。搜索失败或无法确认时仍必须返回“未知歌曲：代表歌词”。
9. 先完整扫描全部目标区间并形成候选列表，再决定是否使用唯一一次歌词搜索。不能发现第一首候选后就停止扫描。
10. 搜索只用于给候选命名，不能让最终结果只包含被搜索的对象。没有搜索的候选也必须保留。"""
        if output_mode == "anchors":
            scan_window_segments = max(1, int(
                config.get("song", {}).get("pipeline", {}).get(
                    "scan_window_segments", 300
                ) or 300
            ))
            scan_windows = []
            scan_number = 0
            for target_start, target_end in self._target_ranges:
                for window_start in range(target_start, target_end + 1, scan_window_segments):
                    scan_windows.append({
                        "scan_id": f"audit_{scan_number:03d}",
                        "segment_range": [window_start, min(window_start + scan_window_segments - 1, target_end)],
                    })
                    scan_number += 1
            anchor_instruction = f"""
13. 只输出用于证明存在演唱的短锚点，每个锚点最多覆盖 12 个 ASR segment。连续两句歌词或约 10 秒连续演唱即可保留。
14. 分段阶段不识别歌名。候选只包含 content_type、scan_id、title、segment_ranges、confidence、anchor_text；title 使用“未知歌曲：anchor_text”。
15. 必须依次扫描以下窗口，并在每个窗口后输出 scan_checkpoint；即使没有歌曲也必须输出 checkpoint：
{json.dumps(scan_windows, ensure_ascii=False, separators=(",", ":"))}
16. checkpoint 格式固定为 {{"content_type":"scan_checkpoint","scan_id":"audit_000","segment_ranges":[]}}。
"""
            naming_instruction = """6. 本阶段只发现演唱分段，不搜索或猜测歌名；歌名留给后续阶段。
9. 必须完整扫描全部 scan window，不能发现第一首候选后停止。
10. 所有满足连续两句歌词或约 10 秒连续演唱的候选都必须保留。"""
        return f"""你是歌曲漏检覆盖审计器。完整 ASR 已经做过一次歌曲识别。

以下范围由已有歌曲结果的覆盖区间反向计算得到。只检查这些目标区间：
{json.dumps(target_ranges, ensure_ascii=False, separators=(",", ":"))}

要求：
1. 未覆盖区间通常包含聊天、报幕、静音和歌曲之间的空隙。不要为了覆盖目标区间而输出结果。
2. 必须按时间顺序检查每个目标区间。连续多句 ASR 呈现歌词、押韵、重复或明显演唱结构时，必须返回；不要只寻找能确认歌名的歌曲。
3. 目标区间可能包含完整漏检歌曲，也可能包含已有歌曲未覆盖的演唱续段。两者都要返回精确的演唱子区间。
4. 不要把整个目标区间直接当作歌曲。聊天、感谢、报幕、点歌、歌曲讨论和零散哼唱必须排除。
5. 不要返回目标区间之外的 segment。本地会再次裁剪并与已有结果去重。
{naming_instruction}
7. 使用紧凑 segment_ranges，区间起止均包含。confidence 必须明确填写 0 到 1。
8. 只返回 JSON 数组。anchors 模式使用紧凑候选和 checkpoint；其他模式保持完整歌曲字段。
11. 不要输出逐区间分析、解释或 Markdown。不需要搜索时立即返回 JSON。
12. 如果没有足够证据证明存在漏检歌曲，返回合法空数组 []。
{anchor_instruction}

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


def _build_song_review_clusters(
    matches: list[ContentMatch],
    suspicious: set[tuple[str, tuple[int, ...]]],
    max_span_segments: int = 500,
    nearby_title_conflict_gap_segments: int = -1,
) -> list[list[ContentMatch]]:
    matches = [m for m in matches if m.segment_indices]
    if not matches:
        return []

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


def _batch_risk_review_clusters(
    clusters: list[list[ContentMatch]],
    *,
    max_span_segments: int,
    max_candidates: int,
) -> list[list[ContentMatch]]:
    """Pack nearby risk candidates into bounded local review requests."""
    if not clusters:
        return []
    ordered = sorted(
        clusters,
        key=lambda cluster: min(min(match.segment_indices) for match in cluster),
    )
    batched: list[list[ContentMatch]] = []
    current: list[ContentMatch] = []
    current_start = 0
    for cluster in ordered:
        cluster_items = [match for match in cluster if match.segment_indices]
        if not cluster_items:
            continue
        cluster_start = min(min(match.segment_indices) for match in cluster_items)
        cluster_end = max(max(match.segment_indices) for match in cluster_items)
        can_append = (
            current
            and len(current) + len(cluster_items) <= max(1, max_candidates)
            and cluster_end - current_start + 1 <= max(1, max_span_segments)
        )
        if can_append:
            current.extend(cluster_items)
            continue
        if current:
            batched.append(current)
        current = list(cluster_items)
        current_start = cluster_start
    if current:
        batched.append(current)
    return batched


def _local_best_song_cluster(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    cluster: list[ContentMatch],
    searched_titles: set[str],
) -> tuple[list[ContentMatch], list[dict[str, Any]]]:
    cluster = [m for m in cluster if m.segment_indices]
    if not cluster:
        return [], []

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


def _load_previously_reviewed_keys(llm_dir: Path) -> set[tuple[str, tuple[int, ...]]]:
    reviewed: set[tuple[str, tuple[int, ...]]] = set()
    root = llm_dir / "review" / "before_missed_recheck"
    for path in root.glob("cluster_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in payload.get("after", []):
            if isinstance(item, dict):
                reviewed.add(_match_key(_content_match_from_dict(item)))
    return reviewed


def _review_introduces_unsupported_title(
    cluster: list[ContentMatch],
    reviewed_matches: list[ContentMatch],
    supported_titles: set[str],
) -> bool:
    original_titles = {
        match.title.strip().casefold()
        for match in cluster
        if not match.title.strip().startswith("未知歌曲")
    }
    for match in reviewed_matches:
        title = match.title.strip().casefold()
        if not title or match.title.strip().startswith("未知歌曲"):
            continue
        if title not in original_titles and title not in supported_titles:
            return True
    return False


def _resolve_review_context(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    llm_dir: Path,
    *,
    phase: str,
) -> tuple[
    list[ContentMatch],
    list[dict[str, Any]],
    list[list[ContentMatch]],
    dict[str, Any],
    set[str],
    set[tuple[str, tuple[int, ...]]] | None,
    Path,
    str,
    str,
    str,
]:
    """Resolve normalized matches, clusters, adaptive scope, and audit context."""
    from .recheck import (
        _llm_debug_has_structural_issue,
        _load_searched_titles,
    )

    review_config = get_song_review_config(config)
    normalized, normalization_events, suspicious = _normalize_song_matches(
        segments, config, matches,
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
        normalized, suspicious,
        max_span_segments=max_cluster_span,
        nearby_title_conflict_gap_segments=int(
            review_config.get("nearby_title_conflict_gap_segments", 2) or 0
        ),
    )
    review_root = llm_dir / "review" / phase
    review_root.mkdir(parents=True, exist_ok=True)

    full_audit_candidate_keys: set[tuple[str, tuple[int, ...]]] | None = None
    missed_strategy = str(
        get_song_recheck_config(config).get("strategy", "windowed")
    ).strip().lower()
    missed_audit_path = llm_dir / "missed_recheck" / "audit.json"
    missed_strategy_resolved = missed_strategy
    if missed_audit_path.exists():
        try:
            missed_audit = json.loads(missed_audit_path.read_text(encoding="utf-8"))
            missed_strategy_resolved = str(
                missed_audit.get("strategy_resolved")
                or missed_audit.get("strategy")
                or missed_strategy
            ).strip().lower()
        except (OSError, json.JSONDecodeError):
            missed_strategy_resolved = missed_strategy
    if phase == "after_missed_recheck" and missed_strategy_resolved == "full_transcript":
        full_audit_path = llm_dir / "missed_recheck" / "audit.json"
        try:
            loaded_audit = json.loads(full_audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded_audit = None
        if isinstance(loaded_audit, dict):
            full_audit_candidates = [
                _content_match_from_dict(item)
                for item in loaded_audit.get("candidates", [])
                if isinstance(item, dict)
            ]
            full_audit_candidate_keys = {
                _match_key(match) for match in full_audit_candidates
            }

    searched_titles = _load_searched_titles(llm_dir)
    transcript_scope_requested = str(
        review_config.get("transcript_scope", "local")
    ).strip().lower()
    scope_cost_details: dict[str, Any] = {}
    if phase == "before_missed_recheck":
        min_gap_segments = int(
            get_song_recheck_config(config).get("min_gap_segments", 1) or 1
        )
        min_song_seconds = float(
            get_padding_config(config, "song").get("min_song_seconds", 75.0)
        )
        preview_ranges, _ = _filter_short_segment_ranges(
            segments,
            _uncovered_segment_ranges(len(segments), normalized, min_gap_segments=min_gap_segments),
            min_song_seconds,
        )
        adaptive_resolution = ensure_song_adaptive_strategies(
            llm_dir, config, clusters=clusters, segments=segments,
            matches=normalized, target_ranges=preview_ranges, recognizer=recognizer,
        )
    else:
        adaptive_resolution = load_adaptive_strategies_cache(llm_dir) or {}

    if adaptive_resolution.get("review_scope_resolved"):
        transcript_scope = str(adaptive_resolution["review_scope_resolved"])
        transcript_scope_reason = str(adaptive_resolution.get("reason", "cached_joint"))
        scope_cost_details = dict(adaptive_resolution.get("review_details") or {})
        if scope_cost_details:
            scope_cost_details["joint_resolution_mode"] = adaptive_resolution.get("resolution_mode")
            scope_cost_details["joint_combinations"] = adaptive_resolution.get("combinations", [])
            scope_cost_details["joint_chosen_total_usd"] = adaptive_resolution.get("chosen_total_usd")
    else:
        transcript_scope, transcript_scope_reason, scope_cost_details = (
            resolve_review_transcript_scope(
                config, clusters=clusters, segments=segments, recognizer=recognizer,
            )
        )

    if phase == "before_missed_recheck" and adaptive_resolution.get("resolution_mode") == "joint_cost_estimate":
        print(
            "  Song adaptive (joint): "
            f"review={adaptive_resolution.get('review_scope_resolved')} + "
            f"missed={adaptive_resolution.get('missed_strategy_resolved')} "
            f"({adaptive_resolution.get('reason')}, "
            f"pipeline=${adaptive_resolution.get('chosen_total_usd')}, "
            f"main=${adaptive_resolution.get('main_cost_usd')}, "
            f"review_before=${adaptive_resolution.get('review_before_cost_usd')}, "
            f"overlong=${adaptive_resolution.get('overlong_cost_usd')}, "
            f"missed=${adaptive_resolution.get('missed_cost_usd')}, "
            f"review_after=${adaptive_resolution.get('review_after_cost_usd')})"
        )
    elif transcript_scope != transcript_scope_requested:
        print(
            f"  Song review ({phase}): adaptive transcript_scope "
            f"{transcript_scope_requested} -> {transcript_scope} "
            f"({transcript_scope_reason}, clusters={len(clusters)})"
        )

    return (
        normalized, normalization_events, clusters, scope_cost_details,
        searched_titles, full_audit_candidate_keys, review_root,
        transcript_scope, transcript_scope_requested, transcript_scope_reason,
    )


def _review_single_cluster(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    cluster: list[ContentMatch],
    cluster_index: int,
    *,
    phase: str,
    review_root: Path,
    context_segments: int,
    max_window_segments: int,
    transcript_scope: str,
    review_config: dict[str, Any],
    full_audit_candidate_keys: set[tuple[str, tuple[int, ...]]] | None,
    searched_titles: set[str],
) -> tuple[list[ContentMatch], dict[str, Any]]:
    """Review a single cluster and return replacement matches + audit entry."""
    from .recheck import _review_debug_succeeded

    target_start = min(min(match.segment_indices) for match in cluster)
    target_end = max(max(match.segment_indices) for match in cluster)
    context_start, context_end = _expand_segment_range(
        target_start, target_end, len(segments), context_segments,
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
        audit["resolution"] = "unchanged_pre_audit_cluster"
        audit["after"] = [match.to_dict() for match in cluster]
        return list(cluster), audit

    reviewed_matches: list[ContentMatch] = []
    review_succeeded = False
    cluster_span = target_end - target_start + 1
    debug_phase = "review_before" if phase == "before_missed_recheck" else "review_after"

    if cluster_span <= max_window_segments:
        local_config = deepcopy(config)
        local_config["llm"]["max_tokens"] = int(review_config.get("max_completion_tokens", 4096) or 4096)
        local_config["llm"]["max_completion_tokens"] = int(review_config.get("max_completion_tokens", 4096) or 4096)
        local_config["llm"]["final_tool_max_tokens"] = int(review_config.get("max_completion_tokens", 4096) or 4096)
        local_config["llm"]["max_tool_rounds"] = int(review_config.get("max_tool_rounds", 1) or 0)
        local_config["llm"]["compact_segment_ranges"] = True
        cluster_debug_dir = review_root / f"cluster_{cluster_index:03d}"
        from ..llm import identify_content
        if transcript_scope == "full":
            review_recognizer = _SongFullReviewRecognizer(
                recognizer, cluster,
                target_start=target_start, target_end=target_end,
                allowed_start=context_start, allowed_end=context_end,
            )
            reviewed_matches = identify_content(
                segments, local_config, review_recognizer,
                debug_dir=cluster_debug_dir, debug_phase=debug_phase,
            )
        else:
            review_recognizer = _SongReviewRecognizer(recognizer, cluster)
            offset_recognizer = _OffsetRecognizer(review_recognizer, context_start)
            reviewed_matches = identify_content(
                segments[context_start:context_end + 1], local_config, offset_recognizer,
                debug_dir=cluster_debug_dir, debug_phase=debug_phase,
            )
        reviewed_matches = _filter_matches_to_segment_range(reviewed_matches, context_start, context_end)
        # Fused C: if this LLM review "merged" what was a disjoint same-title cluster (e.g. 囚鸟),
        # pass the key to force_merge_same_title so the normalize call here skips re-split.
        # This respects the LLM's decision to treat as one continuous performance.
        force = set()
        if (
            not is_risk_routed_v3(config)
            and reviewed_matches
            and len(reviewed_matches) < len(cluster)
        ):
            for m in reviewed_matches:
                force.add(_match_key(m))
        reviewed_matches, review_events, _ = _normalize_song_matches(segments, config, reviewed_matches, force_merge_same_title=force)
        has_conflict = any(
            left.title.strip().casefold() != right.title.strip().casefold()
            and _matches_overlap(left, right)
            for left_index, left in enumerate(reviewed_matches)
            for right in reviewed_matches[left_index + 1:]
        )
        loses_known_title = _review_loses_known_title(cluster, reviewed_matches)
        introduces_unsupported_title = (
            False
            and bool(
                config.get("song", {}).get("naming", {}).get(
                    "preserve_unknown_on_weak_evidence", True
                )
            )
            and _review_introduces_unsupported_title(
                cluster, reviewed_matches, searched_titles,
            )
        )
        review_succeeded = (
            _review_debug_succeeded(cluster_debug_dir)
            and not has_conflict
            and not loses_known_title
            and not introduces_unsupported_title
        )
        audit["review_normalization_events"] = review_events
        if loses_known_title:
            audit["review_rejected_reason"] = "known_title_downgraded_to_unknown"
        elif introduces_unsupported_title:
            audit["review_rejected_reason"] = "unsupported_title_replacement"
    else:
        audit["skip_reason"] = "review cluster exceeds max_window_segments"

    if review_succeeded:
        replacement = reviewed_matches
        audit["resolution"] = "llm"
    else:
        replacement, decisions = _local_best_song_cluster(segments, config, cluster, searched_titles)
        audit["resolution"] = "local_best"
        audit["local_decisions"] = decisions
    audit["after"] = [match.to_dict() for match in replacement]
    audit_path = review_root / f"cluster_{cluster_index:03d}.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return replacement, audit


def _review_song_matches(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    llm_dir: Path,
    *,
    phase: str,
) -> list[ContentMatch]:
    """Review and refine song matches via LLM or local heuristics.

    Builds clusters of conflicting/suspicious matches, then either sends them
    to LLM for re-evaluation or falls back to local best-selection.

    Args:
        segments: Full ASR transcript segments
        config: Complete configuration dict
        recognizer: Song recognizer instance
        matches: Initial song matches to review
        llm_dir: Directory for LLM debug output
        phase: Either "before_missed_recheck" or "after_missed_recheck"

    Returns:
        Reviewed and refined song matches
    """
    review_config = get_song_review_config(config)
    if review_config.get("enabled", False) is False:
        return matches

    (
        normalized, normalization_events, clusters, scope_cost_details,
        searched_titles, full_audit_candidate_keys, review_root,
        transcript_scope, transcript_scope_requested, transcript_scope_reason,
    ) = _resolve_review_context(
        segments, config, recognizer, matches, llm_dir, phase=phase,
    )

    context_segments = int(review_config.get("context_segments", 10) or 0)
    max_window_segments = int(review_config.get("max_window_segments", 500) or 500)

    resolved: list[ContentMatch] = []
    cluster_member_ids = {id(match) for cluster in clusters for match in cluster}
    resolved.extend(match for match in normalized if id(match) not in cluster_member_ids)
    audit_clusters: list[dict[str, Any]] = []

    for cluster_index, cluster in enumerate(clusters, 1):
        replacement, audit = _review_single_cluster(
            segments, config, recognizer, cluster, cluster_index,
            phase=phase, review_root=review_root,
            context_segments=context_segments, max_window_segments=max_window_segments,
            transcript_scope=transcript_scope, review_config=review_config,
            full_audit_candidate_keys=full_audit_candidate_keys,
            searched_titles=searched_titles,
        )
        resolved.extend(replacement)
        audit_clusters.append(audit)

    final_matches, final_events, _ = _normalize_song_matches(segments, config, resolved)
    residual_clusters = _build_song_review_clusters(final_matches, set())
    if residual_clusters:
        residual_ids = {id(match) for cluster in residual_clusters for match in cluster}
        conflict_free = [match for match in final_matches if id(match) not in residual_ids]
        for cluster in residual_clusters:
            replacement, decisions = _local_best_song_cluster(segments, config, cluster, searched_titles)
            conflict_free.extend(replacement)
            final_events.append({"type": "residual_conflict_local_best", "decisions": decisions})
        final_matches = sorted(conflict_free, key=lambda match: min(match.segment_indices))

    # Fused B (from plan): unconditional merge_adjacent_same_title at end of review main path
    # (before sanitation). This ensures LLM review decisions to merge same-title across gaps
    # (e.g. 囚鸟 case) are respected and applied, instead of being lost to prior normalize splits.
    # Uses the (possibly tuned via A) merge_gap_seconds from config.
    # Only skip for "conservative" policy.
    review_config = get_song_review_config(config)
    merge_policy = review_config.get("merge_policy", "conservative")
    if merge_policy != "conservative":
        merge_gap = float(
            get_padding_config(config, "song").get("merge_gap_seconds", 40.0)
        )
        final_matches, merge_events = _merge_adjacent_same_title_matches(
            segments, final_matches, merge_gap
        )
        final_events.extend(merge_events)

    sanitation_events: list[dict[str, Any]] = []
    if full_audit_candidate_keys is not None:
        full_audit_path = llm_dir / "missed_recheck" / "audit.json"
        try:
            loaded_audit = json.loads(full_audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded_audit = None
        if isinstance(loaded_audit, dict):
            full_audit_candidates = [
                _content_match_from_dict(item)
                for item in loaded_audit.get("candidates", [])
                if isinstance(item, dict)
            ]
            target_ranges = [
                (int(item[0]), int(item[1]))
                for item in loaded_audit.get("target_ranges", [])
                if isinstance(item, list) and len(item) == 2
            ]
            final_matches, sanitation_events = _sanitize_full_transcript_review_results(
                segments, config, final_matches, full_audit_candidates, target_ranges,
            )

    summary = {
        "phase": phase,
        "input_count": len(matches),
        "normalized_count": len(normalized),
        "cluster_count": len(clusters),
        "output_count": len(final_matches),
        "transcript_scope_requested": transcript_scope_requested,
        "transcript_scope_resolved": transcript_scope,
        "transcript_scope_reason": transcript_scope_reason,
        **scope_cost_details,
        "normalization_events": normalization_events,
        "final_normalization_events": final_events,
        "full_transcript_sanitation_events": sanitation_events,
        "clusters": audit_clusters,
    }
    (review_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return final_matches


