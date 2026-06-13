"""Profile fingerprinting, usage summaries, and comparison reports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)

from .config import get_llm_config
from .models import TranscriptSegment

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
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Failed to read profile state: %s", exc)
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
                "model": get_llm_config(config).get("model"),
                "cache_friendly_prompt_layout": bool(
                    get_llm_config(config).get("cache_friendly_prompt_layout", False)
                ),
                "compact_segment_ranges": bool(
                    get_llm_config(config).get("compact_segment_ranges", False)
                ),
                "song_pipeline_strategy": config.get("song", {}).get(
                    "pipeline", {}
                ).get("strategy", "legacy"),
                "song_runtime_adaptive": config.get("song", {}).get(
                    "pipeline", {}
                ).get("runtime_adaptive", "disabled"),
                "status": status,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


_USAGE_PHASES = (
    "main",
    "v3_discovery",
    "v3_recall_audit",
    "v3_adjudication",
    "temporal_adjudication",
    "review_before",
    "overlong",
    "missed_recheck",
    "review_after",
)


def _infer_debug_phase(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/")
    if normalized.startswith("v3/discovery/"):
        return "v3_discovery"
    if normalized.startswith("v3/recall_audit/"):
        return "v3_recall_audit"
    if normalized.startswith("v3/adjudication/"):
        return "v3_adjudication"
    if normalized.startswith("temporal_adjudication/"):
        return "temporal_adjudication"
    if normalized.startswith("review/before_missed_recheck/"):
        return "review_before"
    if normalized.startswith("review/after_missed_recheck/"):
        return "review_after"
    if normalized.startswith("overlong_recheck/"):
        return "overlong"
    if normalized.startswith("missed_recheck/"):
        return "missed_recheck"
    return "main"


def _summarize_usage_records(usage_records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "calls": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "completion_tokens": 0,
    }
    for usage in usage_records:
        if not isinstance(usage, dict):
            continue
        totals["calls"] += 1
        totals["prompt_cache_hit_tokens"] += int(usage.get("prompt_cache_hit_tokens") or 0)
        totals["prompt_cache_miss_tokens"] += int(usage.get("prompt_cache_miss_tokens") or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    input_tokens = (
        totals["prompt_cache_hit_tokens"] + totals["prompt_cache_miss_tokens"]
    )
    totals["prompt_tokens"] = input_tokens
    totals["cache_hit_ratio"] = (
        totals["prompt_cache_hit_tokens"] / input_tokens if input_tokens else None
    )
    return totals


def _write_usage_summary(profile_llm_dir: Path) -> dict[str, Any]:
    phase_usage: dict[str, list[dict[str, Any]]] = {
        phase: [] for phase in _USAGE_PHASES
    }
    content_usage: dict[str, list[dict[str, Any]]] = {}
    content_dirs = [
        path
        for path in profile_llm_dir.iterdir()
        if path.is_dir() and (path / "valid_debug_files.json").is_file()
    ] if profile_llm_dir.exists() else []
    if (profile_llm_dir / "valid_debug_files.json").is_file():
        content_dirs.append(profile_llm_dir)

    for content_dir in content_dirs:
        manifest_path = content_dir / "valid_debug_files.json"
        try:
            values = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            values = []
        if not isinstance(values, list):
            continue
        content_type = (
            content_dir.name
            if content_dir != profile_llm_dir
            else str(content_dir.name or "default")
        )
        records = content_usage.setdefault(content_type, [])
        for value in values:
            relative_path = str(value)
            path = content_dir / relative_path
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("Skipping item: %s", exc)
                continue
            phase = str(payload.get("phase") or _infer_debug_phase(relative_path))
            if phase not in phase_usage:
                phase = "main"
            for usage in payload.get("usage", []):
                if isinstance(usage, dict):
                    phase_usage[phase].append(usage)
                    records.append(usage)

    phases = {
        phase: _summarize_usage_records(phase_usage[phase])
        for phase in _USAGE_PHASES
        if phase_usage[phase]
    }
    all_records = [record for records in phase_usage.values() for record in records]
    summary = {
        "phases": phases,
        "content_types": {
            content_type: _summarize_usage_records(records)
            for content_type, records in sorted(content_usage.items())
            if records
        },
        "totals": _summarize_usage_records(all_records),
    }
    profile_llm_dir.mkdir(parents=True, exist_ok=True)
    (profile_llm_dir / "usage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _format_usage_summary_console(summary: dict[str, Any]) -> str:
    totals = summary.get("totals", {})
    if not isinstance(totals, dict) or not totals.get("calls"):
        return ""
    ratio = totals.get("cache_hit_ratio")
    ratio_text = f"{ratio:.1%}" if isinstance(ratio, float) else "n/a"
    return (
        "LLM usage: "
        f"{totals.get('calls', 0)} calls, "
        f"prompt hit/miss {totals.get('prompt_cache_hit_tokens', 0)}/"
        f"{totals.get('prompt_cache_miss_tokens', 0)}, "
        f"completion {totals.get('completion_tokens', 0)}, "
        f"hit ratio {ratio_text}"
    )


def _profile_usage_totals(profile_dir: Path) -> dict[str, int]:
    totals = {
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "completion_tokens": 0,
    }
    usage_summary_path = profile_dir / "usage_summary.json"
    if usage_summary_path.exists():
        try:
            summary = json.loads(usage_summary_path.read_text(encoding="utf-8"))
            summary_totals = summary.get("totals", {})
            if isinstance(summary_totals, dict):
                for key in totals:
                    totals[key] = int(summary_totals.get(key) or 0)
                return totals
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Skipping: %s", exc)

    manifests = list(profile_dir.rglob("valid_debug_files.json"))
    if manifests:
        paths: set[Path] = set()
        for manifest_path in manifests:
            try:
                values = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("Skipping item: %s", exc)
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
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Skipping item: %s", exc)
            continue
        for usage in payload.get("usage", []):
            if not isinstance(usage, dict):
                continue
            for key in totals:
                totals[key] += int(usage.get(key) or 0)
    return totals


def _read_active_debug_paths(
    debug_dir: Path,
    *,
    relative_to: Path,
) -> set[Path]:
    manifest_path = debug_dir / "active_debug_files.json"
    try:
        values = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(values, list):
        return set()
    paths: set[Path] = set()
    for value in values:
        path = debug_dir / str(value)
        if path.is_file():
            try:
                path.relative_to(relative_to)
            except ValueError as exc:
                logger.debug("Path outside tree: %s", exc)
                continue
            paths.add(path)
    return paths


def _write_valid_debug_manifest(llm_dir: Path) -> None:
    paths = _read_active_debug_paths(llm_dir, relative_to=llm_dir)

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
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.debug("Skipping item: %s", exc)
                continue
            cluster_records = summary.get("clusters")
            if isinstance(cluster_records, list):
                active_indices = [
                    int(item["cluster"])
                    for item in cluster_records
                    if (
                        isinstance(item, dict)
                        and isinstance(item.get("cluster"), (int, float))
                        and item.get("resolution")
                        != "unchanged_pre_audit_cluster"
                    )
                ]
            else:
                active_indices = list(range(1, cluster_count + 1))
            for index in active_indices:
                paths.update(
                    _read_active_debug_paths(
                        phase_dir / f"cluster_{index:03d}",
                        relative_to=llm_dir,
                    )
                )

    overlong_root = llm_dir / "overlong_recheck"
    overlong_audit_path = overlong_root / "audit.json"
    if overlong_audit_path.exists():
        try:
            audit = json.loads(overlong_audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            audit = {}
        for value in audit.get("active_debug_files", []):
            path = overlong_root / str(value)
            if path.is_file():
                paths.add(path)

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

    temporal_root = llm_dir / "temporal_adjudication"
    if temporal_root.exists():
        paths.update(
            _read_active_debug_paths(temporal_root, relative_to=llm_dir)
        )

    anchor_root = llm_dir / "anchor_recheck"
    if anchor_root.exists():
        paths.update(
            _read_active_debug_paths(anchor_root, relative_to=llm_dir)
        )

    v3_root = llm_dir / "v3"
    if v3_root.exists():
        for stage_name in ("discovery", "recall_audit", "adjudication"):
            paths.update(
                _read_active_debug_paths(
                    v3_root / stage_name,
                    relative_to=llm_dir,
                )
            )

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
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Skipping item: %s", exc)
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
