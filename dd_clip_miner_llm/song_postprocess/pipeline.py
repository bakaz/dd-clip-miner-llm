from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..models import ContentMatch, TranscriptSegment
from .normalize import _normalize_song_matches
from .risk import (
    repair_song_boundaries,
    score_song_match_risks,
    write_risk_audit,
)
from ..config import song_pipeline_strategy


@dataclass(slots=True)
class SongPipelineContext:
    segments: list[TranscriptSegment]
    config: dict[str, Any]
    recognizer: Any
    llm_dir: Path
    matches: list[ContentMatch]
    stage_history: list[dict[str, Any]] = field(default_factory=list)


class SongPipelineStage(Protocol):
    name: str

    def run(self, context: SongPipelineContext) -> None: ...


class BoundaryRiskStage:
    def __init__(
        self,
        name: str,
        source: str,
        *,
        preserve_candidate_ranges: bool = False,
    ) -> None:
        self.name = name
        self.source = source
        self.preserve_candidate_ranges = preserve_candidate_ranges

    def run(self, context: SongPipelineContext) -> None:
        if self.preserve_candidate_ranges:
            repaired = context.matches
            boundary_events = []
        else:
            repaired, boundary_events = repair_song_boundaries(
                context.segments, context.config, context.matches,
            )
        normalized, normalization_events, _ = _normalize_song_matches(
            context.segments,
            context.config,
            repaired,
            preserve_time_gaps=self.preserve_candidate_ranges,
        )
        records, _ = score_song_match_risks(
            context.segments,
            context.config,
            normalized,
            source=self.source,
        )
        context.matches = normalized
        write_risk_audit(
            context.llm_dir / "risk" / f"{self.name}.json",
            strategy=song_pipeline_strategy(context.config),
            source=self.source,
            records=records,
            boundary_events=[*boundary_events, *normalization_events],
        )
        context.stage_history.append({
            "stage": self.name,
            "input_count": len(repaired),
            "output_count": len(normalized),
            "review_count": sum(record.action != "accept" for record in records),
        })


class ReviewStage:
    def __init__(self, phase: str) -> None:
        self.phase = phase
        self.name = f"review_{phase}"

    def run(self, context: SongPipelineContext) -> None:
        from .review import _review_song_matches

        before = len(context.matches)
        context.matches = _review_song_matches(
            context.segments,
            context.config,
            context.recognizer,
            context.matches,
            context.llm_dir,
            phase=self.phase,
        )
        context.stage_history.append({
            "stage": self.name,
            "input_count": before,
            "output_count": len(context.matches),
        })


class MissedAuditStage:
    name = "missed_recheck"

    def run(self, context: SongPipelineContext) -> None:
        from .recheck import _recheck_uncovered_song_segments

        before = len(context.matches)
        context.matches = _recheck_uncovered_song_segments(
            context.segments,
            context.config,
            context.recognizer,
            context.matches,
            context.llm_dir,
        )
        context.stage_history.append({
            "stage": self.name,
            "input_count": before,
            "output_count": len(context.matches),
        })


class TemporalAdjudicationStage:
    name = "temporal_adjudication"

    def run(self, context: SongPipelineContext) -> None:
        from .temporal import run_temporal_adjudication

        before = len(context.matches)
        context.matches, audit = run_temporal_adjudication(
            context.segments,
            context.config,
            context.recognizer,
            context.matches,
            context.llm_dir,
        )
        context.stage_history.append({
            "stage": self.name,
            "input_count": before,
            "output_count": len(context.matches),
            "status": audit.get("status"),
        })


class FinalAdjudicationStage:
    name = "final_adjudication"

    def run(self, context: SongPipelineContext) -> None:
        from .review import _build_song_review_clusters, _local_best_song_cluster
        from .risk import load_supported_search_titles

        normalized, events, _ = _normalize_song_matches(
            context.segments, context.config, context.matches,
        )
        clusters = _build_song_review_clusters(normalized, set())
        member_ids = {id(match) for cluster in clusters for match in cluster}
        resolved = [match for match in normalized if id(match) not in member_ids]
        searched_titles = load_supported_search_titles(context.llm_dir, normalized)
        decisions: list[dict[str, Any]] = []
        for cluster in clusters:
            replacement, cluster_decisions = _local_best_song_cluster(
                context.segments, context.config, cluster, searched_titles,
            )
            resolved.extend(replacement)
            decisions.extend(cluster_decisions)
        context.matches = sorted(resolved, key=lambda match: min(match.segment_indices))
        context.stage_history.append({
            "stage": self.name,
            "input_count": len(normalized),
            "output_count": len(context.matches),
            "normalization_events": events,
            "decisions": decisions,
        })


def _is_unknown_title(title: str) -> bool:
    """判断是否为未知歌曲标题。"""
    normalized = title.strip().casefold()
    return normalized.startswith(("未知歌曲", "unknown"))


def _extract_search_query(match: ContentMatch, segments: list[TranscriptSegment] | None = None) -> str:
    """从 match 中提取辨识度高的歌词作为搜索查询。

    优先使用 lyrics_snippet 和 description。如果都为空，从 transcript segments 中提取。
    """
    import re as _re

    if match.lyrics_snippet and len(match.lyrics_snippet) >= 5:
        return match.lyrics_snippet
    if match.description and len(match.description) >= 5:
        clean = _re.sub(r"\[严重程度:\d+/5\]\s*", "", match.description)
        clean = _re.sub(r"\[场景:[ABC]\]\s*", "", clean)
        return clean.strip()[:100]

    # 从 transcript segments 提取歌词
    if segments and match.segment_indices:
        texts = []
        for idx in sorted(match.segment_indices):
            if 0 <= idx < len(segments):
                text = segments[idx].text.strip()
                if text and len(text) > 2:
                    texts.append(text)
        if texts:
            # 取中间部分的歌词（更可能是副歌/代表性歌词）
            mid = len(texts) // 2
            start = max(0, mid - 2)
            end = min(len(texts), mid + 3)
            return " ".join(texts[start:end])[:100]

    return ""


def _parse_search_result(result: dict[str, Any]) -> dict[str, str] | None:
    """从搜索结果中提取歌名，清理 web 标题后缀。

    不猜测横线两侧的歌手/歌名顺序：无结构化来源时将清理后的完整标题作为歌名，
    歌手留空。仅在搜索结果提供明确 artist/title 字段时设置歌手。
    """
    import re as _re

    results = result.get("results", [])
    if not results:
        return None

    raw_title = results[0].get("title", "").strip()
    if not raw_title:
        return None

    # 清理后缀：先处理括号内的后缀
    cleaned = _re.sub(
        r"\s*\(.*?(lyrics|歌词|MV|Official|Audio|HD|歌詞|高清|现场|Live).*?\)\s*$",
        "", raw_title, flags=_re.IGNORECASE,
    )
    # 清理后缀：分隔符后的后缀（-、–、—、|），支持多个关键词
    cleaned = _re.sub(
        r"\s*[-–—|]\s*(lyrics|歌词|MV|Official|Audio|HD|歌詞|高清|现场|Live)(\s+(lyrics|歌词|MV|Official|Audio|HD|歌詞|高清|现场|Live))*\s*$",
        "", cleaned, flags=_re.IGNORECASE,
    )
    # 清理后缀：空格后直接跟的后缀（如 "昨日青空 lyrics"），支持多个关键词
    cleaned = _re.sub(
        r"\s+(lyrics|歌词|MV|Official|Audio|HD|歌詞|高清|现场|Live)(\s+(lyrics|歌词|MV|Official|Audio|HD|歌詞|高清|现场|Live))*\s*$",
        "", cleaned, flags=_re.IGNORECASE,
    )
    cleaned = _re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or len(cleaned) < 2:
        return None

    # 检查是否有结构化 artist/title 字段
    structured_artist = results[0].get("artist", "").strip()
    structured_title = results[0].get("title", "").strip()
    if structured_artist and structured_title:
        return {"title": structured_title, "artist": structured_artist}

    # 无结构化来源：清理后的完整标题作为歌名，歌手留空
    return {"title": cleaned}


def _parse_search_result_per_item(result: dict[str, Any]) -> list[dict[str, Any]]:
    """逐条解析搜索结果，返回 [{title, artist?, snippet, lyrics_hints}] 列表。

    每条结果独立绑定自己的标题和摘要，不跨结果混用。
    """
    import re as _re

    def _clean_title(raw: str) -> str:
        cleaned = _re.sub(
            r"\s*\(.*?(lyrics|歌词|MV|Official|Audio|HD|歌詞|高清|现场|Live).*?\)\s*$",
            "", raw, flags=_re.IGNORECASE,
        )
        cleaned = _re.sub(
            r"\s*[-–—|]\s*(lyrics|歌词|MV|Official|Audio|HD|歌詞|高清|现场|Live)(\s+(lyrics|歌词|MV|Official|Audio|HD|歌詞|高清|现场|Live))*\s*$",
            "", cleaned, flags=_re.IGNORECASE,
        )
        cleaned = _re.sub(
            r"\s+(lyrics|歌词|MV|Official|Audio|HD|歌詞|高清|现场|Live)(\s+(lyrics|歌词|MV|Official|Audio|HD|歌詞|高清|现场|Live))*\s*$",
            "", cleaned, flags=_re.IGNORECASE,
        )
        # 清理 "With Pinyin By xxx" 等后缀
        cleaned = _re.sub(r"\s+With\s+\w+\s+By\s+.*$", "", cleaned, flags=_re.IGNORECASE)
        # 清理 "歌詞" 后面的所有内容
        cleaned = _re.sub(r"\s+歌詞.*$", "", cleaned, flags=_re.IGNORECASE)
        # 清理 "The + 英文翻译" 后缀（如果前面有中文歌名）
        if _re.search(r"[\u4e00-\u9fff]", cleaned):
            cleaned = _re.sub(r"\s+The\s+\w+(\s+\w+)*$", "", cleaned, flags=_re.IGNORECASE)
        # 清理中文前的拼音/罗马字前缀（如 "Qiu Niao 囚鸟" → "囚鸟"）
        if _re.search(r"[\u4e00-\u9fff]", cleaned):
            cleaned = _re.sub(r"^[A-Za-z][A-Za-z\s]+(?=[\u4e00-\u9fff])", "", cleaned).strip()
        # 清理 web 页面后缀
        web_suffixes = [
            r"\s*[-–—|]\s*TikTok\s*$",
            r"\s*[-–—|]\s*Shazam\s*$",
            r"\s*[-–—|]\s*Spotify\s*$",
            r"\s*[-–—|]\s*YouTube\s*$",
            r"\s*[-–—|]\s*KKBOX\s*$",
            r"\s*[-–—|]\s*Genius\s*$",
            r"\s*[-–—|]\s*AZLyrics\.com\s*$",
            r"\s*[-–—|]\s*JustSomeLyrics\s*$",
            r"\s*[-–—|]\s*hymnal\.net\s*$",
            r"\s+Song Lyrics, Music Videos & Concerts\s*$",
            r"\s+Music Videos & Concerts\s*$",
            r"\s+arranged by\s+.*$",
            r"\s*\|\s*Genius\s*$",
            r"\s*\|\s*AZLyrics\.com\s*$",
            r"\s*\(book\)\s*$",
        ]
        for pattern in web_suffixes:
            cleaned = _re.sub(pattern, "", cleaned, flags=_re.IGNORECASE)
        return _re.sub(r"\s+", " ", cleaned).strip()

    items: list[dict[str, Any]] = []
    for r in result.get("results", [])[:3]:
        raw_title = r.get("title", "").strip()
        if not raw_title:
            continue
        cleaned = _clean_title(raw_title)
        if not cleaned or len(cleaned) < 2:
            continue

        structured_artist = r.get("artist", "").strip()
        entry: dict[str, Any] = {"title": cleaned, "snippet": r.get("snippet", "").strip()}
        if structured_artist:
            entry["artist"] = structured_artist
        items.append(entry)

    # 将 lyrics_hints 绑定到第一条结果
    lyrics_hints = result.get("lyrics_hints", [])
    if items and lyrics_hints:
        items[0]["lyrics_hints"] = [h.strip() for h in lyrics_hints if isinstance(h, str) and h.strip()]

    return items


def _verify_search_evidence(
    query_text: str, item: dict[str, Any],
) -> tuple[bool, float, str]:
    """验证单条搜索结果是否有歌词证据支持。

    只使用该条自身的 snippet 和 lyrics_hints，不跨结果混用。

    Returns:
        (accepted, score, reason)
    """
    from .normalize import _v2_text_similarity

    title = item.get("title", "").strip()
    if not title or len(title) < 2:
        return False, 0.0, "empty_title"

    # 收集该条自身的证据
    evidence_texts: list[str] = []
    snippet = item.get("snippet", "").strip()
    if snippet:
        evidence_texts.append(snippet)
    for hint in item.get("lyrics_hints", []):
        if hint:
            evidence_texts.append(hint)

    if not evidence_texts:
        return False, 0.0, "no_evidence_text"

    score = _v2_text_similarity([query_text], evidence_texts)
    if score >= 0.15:
        return True, score, "lyrics_evidence_found"

    return False, score, f"insufficient_evidence(score={score:.3f})"


class SearchVerificationStage:
    """对未知歌曲进行歌词搜索验证，只修改 title/artist，不修改 segment_indices。"""
    name = "search_verification"

    def run(self, context: SongPipelineContext) -> None:
        from ..config import get_song_search_config
        from ..search_tools import search_lyrics

        search_cfg = get_song_search_config(context.config)
        if not search_cfg.get("enabled", False):
            return

        max_searches = int(search_cfg.get("max_searches", 25))
        search_unknown_only = bool(search_cfg.get("search_unknown_only", True))
        search_count = 0
        changes: list[dict[str, Any]] = []

        for match in context.matches:
            if search_count >= max_searches:
                break
            if search_unknown_only and not _is_unknown_title(match.title):
                continue
            query_text = _extract_search_query(match, context.segments)
            if not query_text or len(query_text) < 5:
                continue

            result = search_lyrics(query_text[:100], "")
            search_count += 1

            # 逐条解析和验证
            items = _parse_search_result_per_item(result)
            best_item = None
            best_score = 0.0
            best_reason = "no_items"

            for item in items:
                accepted, score, reason = _verify_search_evidence(query_text, item)
                if accepted and score > best_score:
                    best_item = item
                    best_score = score
                    best_reason = reason
                    break  # 取第一条通过的

            audit_entry: dict[str, Any] = {
                "query": result.get("query"),
                "old_title": match.title,
                "candidates": [
                    {"title": it.get("title"), "snippet": it.get("snippet", "")[:80]}
                    for it in items
                ],
                "best_title": best_item.get("title") if best_item else None,
                "score": round(best_score, 4),
                "accepted": best_item is not None,
                "reason": best_reason,
            }

            if best_item:
                match.title = best_item["title"]
                if best_item.get("artist"):
                    match.artist = best_item["artist"]
                match.tags.append("search_verified")
                audit_entry["new_title"] = match.title
                audit_entry["new_artist"] = match.artist
                changes.append(audit_entry)
            else:
                audit_entry["rejected"] = True
                changes.append(audit_entry)

        if changes:
            (context.llm_dir / "search_audit.json").write_text(
                __import__("json").dumps(changes, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        context.stage_history.append({
            "stage": self.name,
            "search_count": search_count,
            "accepted": sum(1 for c in changes if c.get("accepted")),
            "rejected": sum(1 for c in changes if c.get("rejected")),
        })


class AnchorMissedRecheckStage:
    """对 uncovered 区间进行 anchor-based 补查。默认关闭。"""
    name = "anchor_missed_recheck"

    def run(self, context: SongPipelineContext) -> None:
        from ..config import get_song_recheck_config
        from .normalize import _uncovered_segment_ranges

        recheck_cfg = get_song_recheck_config(context.config)
        if not recheck_cfg.get("enabled", False):
            return

        min_seconds = float(recheck_cfg.get("min_uncovered_seconds", 10.0))
        check_all = bool(recheck_cfg.get("check_all_uncovered", True))
        min_gap_segments = int(recheck_cfg.get("min_gap_segments", 1))

        uncovered = _uncovered_segment_ranges(
            len(context.segments), context.matches, min_gap_segments=min_gap_segments,
        )

        qualifying: list[tuple[int, int]] = []
        for start, end in uncovered:
            duration = float(context.segments[end].end) - float(context.segments[start].start)
            if duration >= min_seconds:
                qualifying.append((start, end))

        if not qualifying:
            context.stage_history.append({"stage": self.name, "status": "no_uncovered"})
            return

        if not check_all:
            qualifying = qualifying[:1]

        try:
            extra_matches = self._run_anchor_recheck(
                context.segments, context.config, context.recognizer,
                context.matches, qualifying, context.llm_dir, recheck_cfg,
            )
        except Exception as exc:
            import traceback
            context.stage_history.append({
                "stage": self.name,
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc()[-500:],
            })
            return

        if extra_matches:
            context.matches.extend(extra_matches)
            context.matches.sort(key=lambda m: min(m.segment_indices))

        context.stage_history.append({
            "stage": self.name,
            "uncovered_count": len(qualifying),
            "extra_matches": len(extra_matches) if extra_matches else 0,
        })

    def _run_anchor_recheck(
        self,
        segments: list[TranscriptSegment],
        config: dict[str, Any],
        recognizer: Any,
        existing_matches: list[ContentMatch],
        target_ranges: list[tuple[int, int]],
        llm_dir: Path,
        recheck_cfg: dict[str, Any],
    ) -> list[ContentMatch]:
        from ..llm import identify_content
        from .review import _SongCoverageAuditRecognizer
        from .risk import expand_song_anchors

        audit_recognizer = _SongCoverageAuditRecognizer(recognizer, target_ranges, existing_matches)
        debug_dir = llm_dir / "anchor_recheck"

        local_config = __import__("copy").deepcopy(config)
        max_tokens = int(recheck_cfg.get("max_completion_tokens", 32768) or 32768)
        local_config["llm"]["max_tokens"] = max_tokens
        local_config["llm"]["max_completion_tokens"] = max_tokens
        # 强制单批次、无工具、无 JSON 修复、无 reasoning follow-up
        local_config["llm"]["batch_size"] = None
        local_config["llm"]["max_tool_rounds"] = 0
        local_config["llm"]["use_tools"] = False
        local_config["llm"]["retry_empty_with_reasoning"] = False
        local_config["llm"]["json_fix_rounds"] = 0

        raw_matches = identify_content(
            segments, local_config, audit_recognizer,
            debug_dir=debug_dir, debug_phase="anchor_recheck",
        )

        if not raw_matches:
            return []

        from .normalize import _filter_matches_to_segment_ranges

        cropped = _filter_matches_to_segment_ranges(raw_matches, target_ranges)
        cropped = [
            match for match in cropped
            if _anchor_has_minimum_evidence(segments, match)
        ]

        if not cropped:
            return []

        output_mode = str(recheck_cfg.get("output_mode", "anchors")).strip().lower()
        if output_mode == "anchors":
            max_anchor_segments = int(recheck_cfg.get("max_anchor_segments", 12) or 12)
            anchor_max_expansion = float(recheck_cfg.get("anchor_max_expansion_seconds", 420) or 420)
            cropped = self._constrain_anchors(cropped, max_anchor_segments)
            try:
                expanded, expansion_events = expand_song_anchors(
                    segments, config, cropped, target_ranges, existing_matches,
                )
                cropped = expanded
            except Exception as exc:
                import logging
                logging.getLogger(__name__).debug("Anchor expansion failed: %s", exc)

        for match in cropped:
            if "anchor_expanded" not in match.tags:
                match.tags.append("anchor_expanded")

        return cropped

    @staticmethod
    def _constrain_anchors(
        matches: list[ContentMatch], max_segments: int,
    ) -> list[ContentMatch]:
        constrained: list[ContentMatch] = []
        for match in matches:
            if len(match.segment_indices) <= max_segments:
                constrained.append(match)
            else:
                indices = match.segment_indices[:max_segments]
                constrained.append(ContentMatch(
                    content_type=match.content_type,
                    title=match.title,
                    segment_indices=indices,
                    confidence=match.confidence,
                    tags=list(match.tags),
                    description=match.description,
                    artist=match.artist,
                    lyrics_snippet=match.lyrics_snippet,
                ))
        return constrained


def _anchor_has_minimum_evidence(
    segments: list[TranscriptSegment],
    match: ContentMatch,
    *,
    minimum_seconds: float = 10.0,
) -> bool:
    """Keep two-line anchors or a single ASR span that covers enough singing time."""
    if not match.segment_indices:
        return False
    if len(match.segment_indices) >= 2:
        return True
    start = min(match.segment_indices)
    end = max(match.segment_indices)
    if start < 0 or end >= len(segments):
        return False
    return (
        float(segments[end].end) - float(segments[start].start)
    ) >= minimum_seconds


__all__ = [
    "BoundaryRiskStage",
    "FinalAdjudicationStage",
    "SearchVerificationStage",
    "AnchorMissedRecheckStage",
    "SongPipelineContext",
    "SongPipelineStage",
]
