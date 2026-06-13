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
    resolve_missed_recheck_strategy,
)
from ..profile_state import (
    _config_fingerprint,
    _fingerprint_payload,
    _transcript_fingerprint,
)
from .normalize import (
    _clone_match_with_indices,
    _content_match_from_dict,
    _expand_segment_range,
    _filter_matches_to_segment_range,
    _filter_matches_to_segment_ranges,
    _filter_short_segment_ranges,
    _group_segment_ranges,
    _indices_to_ranges,
    _is_invalid_audit_title,
    _match_groups_over_max_song_seconds,
    _merge_adjacent_same_title_matches,
    _normalize_song_matches,
    _segment_range_duration_seconds,
    _split_segment_ranges,
    _split_indices_by_time_gap_for_recheck,
    _uncovered_segment_ranges,
)
from .risk import expand_song_anchors
from .review import (
    _OffsetRecognizer,
    _SongCoverageAuditRecognizer,
    _build_song_review_clusters,
)


def _load_cached_identify_matches(
    debug_dir: Path,
    recognizer: Any,
    config: dict[str, Any],
    segments: list[TranscriptSegment],
    *,
    debug_phase: str | None = None,
) -> list[ContentMatch] | None:
    from ..llm import (
        _try_load_cached_batch,
        build_llm_messages,
        build_providers,
        build_request_debug_metadata,
    )

    providers = [provider for provider in build_providers(config) if provider.api_key]
    if not providers:
        return None
    batch_size = get_llm_config(config).get("batch_size")
    if batch_size in (None, "", 0, "0"):
        batches = [(0, segments)]
    else:
        size = int(batch_size)
        batches = [
            (start, segments[start:start + size])
            for start in range(0, len(segments), size)
        ]
    tools = recognizer.get_tools(config)
    matches: list[ContentMatch] = []
    for batch_start, batch_segments in batches:
        cached_items: list[dict[str, Any]] | None = None
        messages = build_llm_messages(
            recognizer,
            batch_segments,
            batch_start,
            config,
        )
        for provider in providers:
            metadata = build_request_debug_metadata(
                messages,
                config=config,
                provider=provider,
                recognizer=recognizer,
                segments=batch_segments,
                batch_start=batch_start,
                tools=tools,
                debug_phase=debug_phase,
            )
            cached = _try_load_cached_batch(
                debug_dir,
                batch_start,
                expected_metadata=metadata,
            )
            if cached is not None:
                _, cached_items = cached
                break
        if cached_items is None:
            return None
        matches.extend(recognizer.parse_response(cached_items, config))
    return matches


def _active_debug_files(
    debug_dir: Path,
    *,
    relative_to: Path,
) -> list[str]:
    manifest_path = debug_dir / "active_debug_files.json"
    try:
        values = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        path = debug_dir / str(value)
        if path.is_file():
            result.append(str(path.relative_to(relative_to)).replace("\\", "/"))
    return result


def _cache_reuse_count(debug_dir: Path) -> int:
    count = 0
    for path in debug_dir.glob("llm_batch_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        reuse = payload.get("cache_reuse")
        if isinstance(reuse, dict):
            count += int(reuse.get("count") or 0)
    return count


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
        if payload.get("finish_reason") == "length":
            failures.append("output_truncated")
        if any(
            item.get("finish_reason") == "length"
            for item in payload.get("tool_rounds", [])
            if isinstance(item, dict)
        ):
            failures.append("output_truncated")
    return sorted(set(failures))


def _llm_debug_has_structural_issue(debug_root: Path) -> bool:
    for path in debug_root.glob("llm_batch_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("json_fix_rounds"):
            return True
        if payload.get("finish_reason") == "length":
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


def _recheck_overlong_song_matches(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    llm_dir: Path,
) -> list[ContentMatch]:
    if not segments or not matches:
        return matches

    recheck_config = get_song_recheck_config(config)
    if recheck_config.get("enabled", True) is False:
        return matches

    padding_config = get_padding_config(config, "song")
    max_song_seconds = float(padding_config.get("max_song_seconds", 360.0))
    merge_gap_seconds = float(padding_config.get("merge_gap_seconds", 40.0))
    if max_song_seconds <= 0:
        return matches

    context_segments = int(recheck_config.get("context_segments", 10) or 0)

    from ..llm import identify_content

    recheck_root = llm_dir / "overlong_recheck"
    replacement_matches: list[ContentMatch] = []
    rechecked_count = 0
    replaced_count = 0
    kept_count = 0
    all_rechecked_matches: list[ContentMatch] = []
    active_debug_files: list[str] = []

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
            raw_rechecked_matches = identify_content(
                chunk,
                config,
                offset_recognizer,
                debug_dir=debug_dir,
                debug_phase="overlong",
            )
            active_debug_files.extend(
                _active_debug_files(debug_dir, relative_to=recheck_root)
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
    (recheck_root / "audit.json").write_text(
        json.dumps(
            {
                "status": "success",
                "active_debug_files": sorted(set(active_debug_files)),
                "rechecked_count": rechecked_count,
                "replaced_count": replaced_count,
                "kept_count": kept_count,
            },
            ensure_ascii=False,
            indent=2,
        ),
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
    recheck_config = get_song_recheck_config(config)
    min_gap_segments = int(recheck_config.get("min_gap_segments", 1) or 1)
    context_segments = int(recheck_config.get("context_segments", 10) or 0)
    padding_config = get_padding_config(config, "song")
    min_song_seconds = float(padding_config.get("min_song_seconds", 75.0))
    batch_size_value = recheck_config.get("batch_size", get_llm_config(config).get("batch_size") or 500)
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

    from ..llm import identify_content

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
        rechecked_matches = identify_content(
            chunk,
            config,
            offset_recognizer,
            debug_dir=debug_dir,
            debug_phase="missed_recheck",
        )
        active_debug_files.extend(
            _active_debug_files(debug_dir, relative_to=recheck_root)
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

    if (
        extra_matches
        and str(recheck_config.get("output_mode", "matches")).strip().lower() == "anchors"
    ):
        extra_matches = _constrain_missed_anchor_matches(
            extra_matches,
            max_anchor_segments=int(recheck_config.get("max_anchor_segments", 12) or 12),
        )
        extra_matches, expansion_events = expand_song_anchors(
            segments, config, extra_matches, ranges, matches,
        )
        (recheck_root / "anchor_expansion.json").write_text(
            json.dumps(expansion_events, ensure_ascii=False, indent=2),
            encoding="utf-8",
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


def _finalize_windowed_missed_recheck_matches(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    matches: list[ContentMatch],
    extra_matches: list[ContentMatch],
) -> tuple[list[ContentMatch], dict[str, Any]]:
    """Normalize windowed missed-recheck output before returning."""
    combined = [*matches, *extra_matches]
    before_count = len(combined)
    normalized, normalization_events, _ = _normalize_song_matches(
        segments,
        config,
        combined,
    )
    merge_gap_seconds = float(
        get_padding_config(config, "song").get("merge_gap_seconds", 40.0)
    )
    merged, merge_events = _merge_adjacent_same_title_matches(
        segments,
        normalized,
        merge_gap_seconds,
    )
    events = [*normalization_events, *merge_events]
    return merged, {
        "before_count": before_count,
        "after_count": len(merged),
        "normalization_events": events,
    }


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
        "protocol": "full_transcript_coverage_audit_v6_anchors",
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


def _resolve_missed_strategy(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    llm_dir: Path,
) -> tuple[str, str, dict[str, Any], list[tuple[int, int]]]:
    """Resolve the missed recheck strategy (windowed/full_transcript/adaptive)."""
    recheck_config = get_song_recheck_config(config)
    strategy_requested = str(recheck_config.get("strategy", "windowed")).strip().lower()

    min_gap_segments = int(recheck_config.get("min_gap_segments", 1) or 1)
    min_song_seconds = float(get_padding_config(config, "song").get("min_song_seconds", 75.0))
    preview_ranges, _ = _filter_short_segment_ranges(
        segments,
        _uncovered_segment_ranges(len(segments), matches, min_gap_segments=min_gap_segments),
        min_song_seconds,
    )

    adaptive_resolution = load_adaptive_strategies_cache(llm_dir) or {}
    if not adaptive_resolution:
        review_config = get_song_review_config(config)
        normalized, _, suspicious = _normalize_song_matches(segments, config, matches)
        max_cluster_span = max(
            1,
            int(review_config.get("max_window_segments", 500) or 500)
            - 2 * int(review_config.get("context_segments", 10) or 0),
        )
        clusters = (
            _build_song_review_clusters(
                normalized, suspicious,
                max_span_segments=max_cluster_span,
                nearby_title_conflict_gap_segments=int(review_config.get("nearby_title_conflict_gap_segments", 2) or 0),
            )
            if review_config.get("enabled", False) else []
        )
        adaptive_resolution = ensure_song_adaptive_strategies(
            llm_dir, config, clusters=clusters, segments=segments,
            matches=normalized, target_ranges=preview_ranges, recognizer=recognizer,
        )

    if adaptive_resolution.get("missed_strategy_resolved"):
        strategy = str(adaptive_resolution["missed_strategy_resolved"])
        strategy_reason = str(adaptive_resolution.get("reason", "cached_joint"))
        strategy_cost_details = dict(adaptive_resolution.get("missed_details") or {})
        if strategy_cost_details:
            strategy_cost_details["joint_resolution_mode"] = adaptive_resolution.get("resolution_mode")
            strategy_cost_details["joint_combinations"] = adaptive_resolution.get("combinations", [])
            strategy_cost_details["joint_chosen_total_usd"] = adaptive_resolution.get("chosen_total_usd")
    else:
        strategy, strategy_reason, strategy_cost_details = resolve_missed_recheck_strategy(
            config, segments=segments, matches=matches,
            target_ranges=preview_ranges, recognizer=recognizer,
        )

    if strategy != strategy_requested and adaptive_resolution.get("resolution_mode") != "joint_cost_estimate":
        print(
            f"  Song missed recheck: adaptive strategy {strategy_requested} -> "
            f"{strategy} ({strategy_reason}, targets={len(preview_ranges)})"
        )

    return strategy, strategy_reason, strategy_cost_details, preview_ranges


def _execute_full_transcript_audit(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    recheck_root: Path,
    target_ranges: list[tuple[int, int]],
    fingerprints: dict[str, str],
    *,
    strategy_requested: str,
    strategy: str,
    strategy_reason: str,
    strategy_cost_details: dict[str, Any],
) -> list[ContentMatch]:
    """Execute full transcript audit strategy with fallback support."""
    recheck_config = get_song_recheck_config(config)
    fallback_strategy = str(recheck_config.get("fallback_strategy", "windowed_on_structural_failure")).strip().lower()
    audit_path = recheck_root / "audit.json"
    cached_audit = _load_missed_recheck_audit(audit_path, fingerprints)

    if cached_audit and cached_audit.get("status") == "fallback_success":
        extra_matches, active_files = _run_windowed_missed_recheck(
            segments, config, recognizer, matches, recheck_root / "fallback_windowed",
        )
        combined = [*matches, *extra_matches]
        combined, merge_events = _merge_adjacent_same_title_matches(
            segments, combined,
            float(get_padding_config(config, "song").get("merge_gap_seconds", 40.0)),
        )
        cached_audit["active_debug_files"] = [f"fallback_windowed/{path}" for path in active_files]
        cached_audit["same_title_merge_events"] = merge_events
        audit_path.write_text(json.dumps(cached_audit, ensure_ascii=False, indent=2), encoding="utf-8")
        return combined

    audit_recognizer = _SongCoverageAuditRecognizer(recognizer, target_ranges, matches)
    audit_debug_dir = recheck_root / "full_transcript"
    reuse_count_before = _cache_reuse_count(audit_debug_dir)

    from ..llm import identify_content
    local_config = deepcopy(config)
    max_completion_tokens = int(recheck_config.get("max_completion_tokens", 4096) or 4096)
    local_config["llm"]["max_tokens"] = max_completion_tokens
    local_config["llm"]["max_completion_tokens"] = max_completion_tokens
    local_config["llm"]["final_tool_max_tokens"] = max_completion_tokens
    local_config["llm"]["max_tool_rounds"] = int(recheck_config.get("max_tool_rounds", 1) or 0)
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
        segments, local_config, audit_recognizer,
        debug_dir=audit_debug_dir, debug_phase="missed_recheck",
    )

    cache_reused = _cache_reuse_count(audit_debug_dir) > reuse_count_before
    structural_failures = _llm_debug_structural_failures(audit_debug_dir)
    active_debug_files = _active_debug_files(audit_debug_dir, relative_to=recheck_root)

    audit: dict[str, Any] = {
        "strategy": "full_transcript",
        "strategy_requested": strategy_requested,
        "strategy_resolved": strategy,
        "strategy_reason": strategy_reason,
        **strategy_cost_details,
        "status": "success" if not structural_failures else "structural_failure",
        "fallback_strategy": fallback_strategy,
        "fallback_used": False,
        "fingerprints": fingerprints,
        "target_ranges": [[start, end] for start, end in target_ranges],
        "output_mode": str(recheck_config.get("output_mode", "matches")),
        "candidate_count": len(matches),
        "candidates": [match.to_dict() for match in matches],
        "cache_reused": cache_reused,
        "structural_failures": structural_failures,
        "active_debug_files": active_debug_files,
    }
    review_trigger_matches: list[ContentMatch] = []

    if structural_failures and fallback_strategy == "windowed_on_structural_failure":
        print(
            "  Song missed recheck: full transcript audit failed structurally "
            f"({', '.join(structural_failures)}); falling back to windowed"
        )
        extra_matches, fallback_files = _run_windowed_missed_recheck(
            segments, config, recognizer, matches, recheck_root / "fallback_windowed",
        )
        audit["status"] = "fallback_success"
        audit["fallback_used"] = True
        audit["active_debug_files"] = [f"fallback_windowed/{path}" for path in fallback_files]
    elif structural_failures:
        extra_matches = []
    else:
        cropped_matches = _filter_matches_to_segment_ranges(raw_matches, target_ranges)
        if str(recheck_config.get("output_mode", "matches")).strip().lower() == "anchors":
            cropped_matches = _constrain_missed_anchor_matches(
                cropped_matches,
                max_anchor_segments=int(recheck_config.get("max_anchor_segments", 12) or 12),
            )
            cropped_matches, expansion_events = expand_song_anchors(
                segments, config, cropped_matches, target_ranges, matches,
            )
            audit["anchor_expansion_events"] = expansion_events
        review_trigger_matches = [
            match for match in cropped_matches if _is_invalid_audit_title(match.title)
        ]
        rejected_titles = [
            {"title": match.title, "ranges": _indices_to_ranges(match.segment_indices)}
            for match in review_trigger_matches
        ]
        if rejected_titles:
            audit["rejected_invalid_titles"] = rejected_titles
        extra_matches = [
            match for match in cropped_matches if not _is_invalid_audit_title(match.title)
        ]
        if get_song_review_config(config).get("enabled", False):
            audit["review_trigger_count"] = len(review_trigger_matches)
        else:
            review_trigger_matches = []

    audit["additional_match_count"] = len(extra_matches)
    combined_matches = [*matches, *extra_matches, *review_trigger_matches]
    if not structural_failures:
        combined_matches, merge_events = _merge_adjacent_same_title_matches(
            segments, combined_matches,
            float(get_padding_config(config, "song").get("merge_gap_seconds", 40.0)),
        )
        audit["same_title_merge_events"] = merge_events
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if extra_matches:
        (recheck_root / "matches.json").write_text(
            json.dumps([match.to_dict() for match in extra_matches], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Song missed recheck: found {len(extra_matches)} additional match(es)")
    else:
        print("  Song missed recheck: no additional matches")
    return combined_matches


def _constrain_missed_anchor_matches(
    matches: list[ContentMatch],
    *,
    max_anchor_segments: int,
) -> list[ContentMatch]:
    """Prevent a coverage-audit range from becoming a final large song boundary."""
    max_anchor_segments = max(1, max_anchor_segments)
    constrained: list[ContentMatch] = []
    for match in matches:
        indices = sorted(set(match.segment_indices))
        if len(indices) <= max_anchor_segments:
            constrained.append(match)
            continue
        width = max(1, max_anchor_segments // 3)
        centers = (0, max(0, (len(indices) - width) // 2), max(0, len(indices) - width))
        anchors: list[int] = []
        for start in centers:
            anchors.extend(indices[start:start + width])
        constrained.append(_clone_match_with_indices(match, sorted(set(anchors))))
    return constrained


def _recheck_uncovered_song_segments(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: Any,
    matches: list[ContentMatch],
    llm_dir: Path,
) -> list[ContentMatch]:
    """Recheck ASR segments not covered by existing song matches.

    Uses either windowed or full_transcript strategy to find missed songs.
    Falls back to windowed on structural LLM failures.

    Args:
        segments: Full ASR transcript segments
        config: Complete configuration dict
        recognizer: Song recognizer instance
        matches: Existing song matches
        llm_dir: Directory for LLM debug output

    Returns:
        Original matches plus any newly discovered songs
    """
    recheck_config = get_song_recheck_config(config)
    if recheck_config.get("enabled", True) is False:
        return matches

    recheck_root = llm_dir / "missed_recheck"
    recheck_root.mkdir(parents=True, exist_ok=True)

    strategy, strategy_reason, strategy_cost_details, preview_ranges = _resolve_missed_strategy(
        segments, config, recognizer, matches, llm_dir,
    )
    strategy_requested = str(recheck_config.get("strategy", "windowed")).strip().lower()

    if strategy == "windowed":
        extra_matches, active_files = _run_windowed_missed_recheck(
            segments, config, recognizer, matches, recheck_root,
        )
        finalized_matches, normalization_summary = _finalize_windowed_missed_recheck_matches(
            segments, config, matches, extra_matches,
        )
        (recheck_root / "audit.json").write_text(
            json.dumps({
                "strategy": "windowed",
                "strategy_requested": strategy_requested,
                "strategy_resolved": strategy,
                "strategy_reason": strategy_reason,
                **strategy_cost_details,
                "status": "success",
                "fallback_used": False,
                "active_debug_files": active_files,
                "additional_match_count": len(extra_matches),
                **normalization_summary,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return finalized_matches

    min_gap_segments = int(recheck_config.get("min_gap_segments", 1) or 1)
    min_song_seconds = float(get_padding_config(config, "song").get("min_song_seconds", 75.0))
    target_ranges, skipped_short = _filter_short_segment_ranges(
        segments,
        _uncovered_segment_ranges(len(segments), matches, min_gap_segments=min_gap_segments),
        min_song_seconds,
    )
    if skipped_short:
        print(
            f"  Song missed recheck: skipped {skipped_short} short ASR range(s) "
            f"below min_song_seconds={min_song_seconds:g}"
        )
    if not target_ranges:
        return matches

    fingerprints = _missed_recheck_fingerprint(segments, config, matches, target_ranges)
    return _execute_full_transcript_audit(
        segments, config, recognizer, matches, recheck_root, target_ranges, fingerprints,
        strategy_requested=strategy_requested, strategy=strategy,
        strategy_reason=strategy_reason, strategy_cost_details=strategy_cost_details,
    )


