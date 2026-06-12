"""Probe adaptive strategy costs from local ASR fixtures and optional usage history."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dd_clip_miner_llm.config import load_config
from dd_clip_miner_llm.models import ContentMatch, TranscriptSegment
from dd_clip_miner_llm.song_adaptive import resolve_song_adaptive_strategies
from dd_clip_miner_llm.song_adaptive_cost import (
    estimate_main_cost,
    estimate_missed_cost,
    estimate_overlong_cost,
    estimate_review_after_cost,
    estimate_review_cost,
)
from dd_clip_miner_llm.song_postprocess import (
    _build_song_review_clusters,
    _content_match_from_dict,
    _filter_short_segment_ranges,
    _group_segment_ranges,
    _normalize_song_matches,
    _uncovered_segment_ranges,
)


def _load_segments(path: Path) -> list[TranscriptSegment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected transcript list in {path}")
    return [
        TranscriptSegment(
            start=float(item["start"]),
            end=float(item["end"]),
            text=str(item.get("text") or ""),
        )
        for item in payload
        if isinstance(item, dict)
    ]


def _load_matches(path: Path) -> list[ContentMatch]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected matches list in {path}")
    return [
        _content_match_from_dict(item)
        for item in payload
        if isinstance(item, dict)
    ]


def _stream_features(
    segments: list[TranscriptSegment],
    matches: list[ContentMatch],
    config: dict[str, Any],
) -> dict[str, Any]:
    recheck_config = config.get("song", {}).get("missed_recheck", {})
    min_gap_segments = int(recheck_config.get("min_gap_segments", 1) or 1)
    min_song_seconds = float(
        config.get("song", {}).get("padding", {}).get("min_song_seconds", 75.0)
    )
    batch_size = int(recheck_config.get("batch_size", 500) or 500)

    normalized, _, suspicious = _normalize_song_matches(segments, config, matches)
    review_config = config.get("song", {}).get("review", {})
    clusters = _build_song_review_clusters(
        normalized,
        suspicious,
        max_span_segments=max(
            1,
            int(review_config.get("max_window_segments", 500) or 500)
            - 2 * int(review_config.get("context_segments", 10) or 0),
        ),
        nearby_title_conflict_gap_segments=int(
            review_config.get("nearby_title_conflict_gap_segments", 2) or -1
        ),
    )
    preview_ranges, _ = _filter_short_segment_ranges(
        segments,
        _uncovered_segment_ranges(
            len(segments),
            normalized,
            min_gap_segments=min_gap_segments,
        ),
        min_song_seconds,
    )
    window_groups = _group_segment_ranges(preview_ranges, batch_size)
    char_total = sum(len(segment.text) for segment in segments)
    duration = float(segments[-1].end) if segments else 0.0
    return {
        "segment_count": len(segments),
        "char_total": char_total,
        "duration_seconds": duration,
        "cluster_count": len(clusters),
        "target_range_count": len(preview_ranges),
        "window_count": len(window_groups),
        "clusters": clusters,
        "preview_ranges": preview_ranges,
        "normalized_matches": normalized,
    }


def _token_cost_usd(
    *,
    hit: int,
    miss: int,
    completion: int,
    pricing: dict[str, float] | None = None,
) -> float:
    rates = pricing or {
        "input_cache_hit_per_1m": 0.0028,
        "input_cache_miss_per_1m": 0.14,
        "output_per_1m": 0.28,
    }
    return (
        hit * rates["input_cache_hit_per_1m"] / 1_000_000
        + miss * rates["input_cache_miss_per_1m"] / 1_000_000
        + completion * rates["output_per_1m"] / 1_000_000
    )


def _phase_costs(usage_summary: dict[str, Any]) -> dict[str, float]:
    phases = usage_summary.get("phases", {})
    costs: dict[str, float] = {}
    for name, phase in phases.items():
        if not isinstance(phase, dict):
            continue
        costs[name] = round(
            _token_cost_usd(
                hit=int(phase.get("prompt_cache_hit_tokens") or 0),
                miss=int(phase.get("prompt_cache_miss_tokens") or 0),
                completion=int(phase.get("completion_tokens") or 0),
            ),
            6,
        )
    return costs


def _usage_actual_cost(usage_summary: dict[str, Any]) -> dict[str, Any] | None:
    totals = usage_summary.get("totals", {})
    if not totals:
        return None
    pricing = {
        "input_cache_hit_per_1m": 0.0028,
        "input_cache_miss_per_1m": 0.14,
        "output_per_1m": 0.28,
    }
    hit = int(totals.get("prompt_cache_hit_tokens") or 0)
    miss = int(totals.get("prompt_cache_miss_tokens") or 0)
    completion = int(totals.get("completion_tokens") or 0)
    actual_usd = _token_cost_usd(hit=hit, miss=miss, completion=completion, pricing=pricing)
    return {
        "prompt_cache_hit_tokens": hit,
        "prompt_cache_miss_tokens": miss,
        "completion_tokens": completion,
        "calls": int(totals.get("calls") or 0),
        "actual_usd": round(actual_usd, 6),
        "phases": usage_summary.get("phases", {}),
        "phase_costs_usd": _phase_costs(usage_summary),
    }


def probe_run_dir(
    run_dir: Path,
    *,
    config_path: Path,
    profile: str = "kv_optimized",
) -> dict[str, Any]:
    transcript_path = run_dir / "02_asr" / "transcript.json"
    matches_path = (
        run_dir / "02_asr" / "llm" / profile / "song" / "initial_matches.json"
    )
    usage_path = run_dir / "02_asr" / "llm" / profile / "usage_summary.json"

    segments = _load_segments(transcript_path)
    matches = _load_matches(matches_path)
    config = load_config(config_path, profile=profile)
    features = _stream_features(segments, matches, config)

    main_cost = estimate_main_cost(config, segments=segments)
    overlong_cost = estimate_overlong_cost(
        config,
        segments=segments,
        matches=features["normalized_matches"],
    )
    review_local = estimate_review_cost(
        config,
        scope="local",
        clusters=features["clusters"],
        segments=segments,
    )
    review_full = estimate_review_cost(
        config,
        scope="full",
        clusters=features["clusters"],
        segments=segments,
    )
    review_after_local_windowed = estimate_review_after_cost(
        config,
        scope="local",
        missed_strategy="windowed",
        clusters=features["clusters"],
        segments=segments,
        target_ranges=features["preview_ranges"],
    )
    review_after_full_full = estimate_review_after_cost(
        config,
        scope="full",
        missed_strategy="full_transcript",
        clusters=features["clusters"],
        segments=segments,
        target_ranges=features["preview_ranges"],
    )
    missed_windowed = estimate_missed_cost(
        config,
        strategy="windowed",
        segments=segments,
        matches=features["normalized_matches"],
        target_ranges=features["preview_ranges"],
    )
    missed_full = estimate_missed_cost(
        config,
        strategy="full_transcript",
        segments=segments,
        matches=features["normalized_matches"],
        target_ranges=features["preview_ranges"],
        include_fallback_penalty=True,
    )

    joint = resolve_song_adaptive_strategies(
        config,
        clusters=features["clusters"],
        segments=segments,
        matches=features["normalized_matches"],
        target_ranges=features["preview_ranges"],
    )

    report: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "profile": profile,
        "features": {
            key: value
            for key, value in features.items()
            if key not in {"clusters", "preview_ranges", "normalized_matches"}
        },
        "estimates": {
            "main_usd": round(main_cost.total_usd, 6),
            "overlong_usd": round(overlong_cost.total_usd, 6),
            "review_local_usd": round(review_local.total_usd, 6),
            "review_full_usd": round(review_full.total_usd, 6),
            "review_after_local_windowed_usd": round(
                review_after_local_windowed.total_usd,
                6,
            ),
            "review_after_full_full_usd": round(
                review_after_full_full.total_usd,
                6,
            ),
            "missed_windowed_usd": round(missed_windowed.total_usd, 6),
            "missed_full_usd": round(missed_full.total_usd, 6),
            "main": main_cost.to_dict(),
            "overlong": overlong_cost.to_dict(),
            "review_local": review_local.to_dict(),
            "review_full": review_full.to_dict(),
            "review_after_local_windowed": review_after_local_windowed.to_dict(),
            "review_after_full_full": review_after_full_full.to_dict(),
            "missed_windowed": missed_windowed.to_dict(),
            "missed_full": missed_full.to_dict(),
        },
        "adaptive_decisions": {
            "review_scope": joint.get("review_scope_resolved"),
            "review_reason": joint.get("reason"),
            "review_details": joint.get("review_details", {}),
            "missed_strategy": joint.get("missed_strategy_resolved"),
            "missed_reason": joint.get("reason"),
            "missed_details": joint.get("missed_details", {}),
            "joint_combinations": joint.get("combinations", []),
            "chosen_total_usd": joint.get("chosen_total_usd"),
            "resolution_mode": joint.get("resolution_mode"),
        },
    }

    if usage_path.exists():
        usage_summary = json.loads(usage_path.read_text(encoding="utf-8"))
        actual = _usage_actual_cost(usage_summary)
        if actual is not None:
            report["actual_usage"] = actual
            phase_costs = actual.get("phase_costs_usd", {})
            review_actual = (
                phase_costs.get("review_before", 0.0)
                + phase_costs.get("review_after", 0.0)
            )
            missed_actual = phase_costs.get("missed_recheck", 0.0)
            main_actual = phase_costs.get("main", 0.0)
            overlong_actual = phase_costs.get("overlong", 0.0)
            chosen_combo = next(
                (
                    item
                    for item in joint.get("combinations", [])
                    if item.get("review_scope") == joint.get("review_scope_resolved")
                    and item.get("missed_strategy")
                    == joint.get("missed_strategy_resolved")
                ),
                {},
            )
            chosen_total = float(
                joint.get("chosen_total_usd") or chosen_combo.get("total_usd") or 0.0
            )
            chosen_main_est = float(
                joint.get("main_cost_usd") or chosen_combo.get("main_cost_usd") or 0.0
            )
            chosen_overlong_est = float(
                joint.get("overlong_cost_usd")
                or chosen_combo.get("overlong_cost_usd")
                or 0.0
            )
            chosen_review_est = float(chosen_combo.get("review_cost_usd") or 0.0)
            chosen_missed_est = float(chosen_combo.get("missed_cost_usd") or 0.0)
            report["calibration"] = {
                "estimated_pipeline_usd": round(chosen_total, 6),
                "estimated_main_usd": round(chosen_main_est, 6),
                "estimated_overlong_usd": round(chosen_overlong_est, 6),
                "estimated_review_usd": round(chosen_review_est, 6),
                "estimated_missed_usd": round(chosen_missed_est, 6),
                "actual_main_usd": main_actual,
                "actual_overlong_usd": overlong_actual,
                "actual_review_usd": review_actual,
                "actual_missed_usd": missed_actual,
                "actual_pipeline_usd": round(
                    main_actual + overlong_actual + review_actual + missed_actual,
                    6,
                ),
                "actual_total_usd": actual["actual_usd"],
                "main_delta_usd": round(main_actual - chosen_main_est, 6),
                "overlong_delta_usd": round(overlong_actual - chosen_overlong_est, 6),
                "review_delta_usd": round(review_actual - chosen_review_est, 6),
                "missed_delta_usd": round(missed_actual - chosen_missed_est, 6),
                "pipeline_delta_usd": round(
                    main_actual
                    + overlong_actual
                    + review_actual
                    + missed_actual
                    - chosen_total,
                    6,
                ),
            }

    return report


def _write_markdown(report_path: Path, report: dict[str, Any]) -> None:
    features = report["features"]
    estimates = report["estimates"]
    decisions = report["adaptive_decisions"]
    lines = [
        f"# Adaptive cost probe ({report.get('profile', 'kv_optimized')})",
        "",
        f"Run dir: `{report['run_dir']}`",
        "",
        "## Stream features",
        "",
        f"- segments: {features['segment_count']}",
        f"- chars: {features['char_total']}",
        f"- duration: {features['duration_seconds']:.1f}s",
        f"- clusters: {features['cluster_count']}",
        f"- target ranges: {features['target_range_count']}",
        f"- window groups: {features['window_count']}",
        "",
        "## Cost estimates (USD)",
        "",
        "| Strategy | Estimated |",
        "| --- | --- |",
        f"| main | {estimates['main_usd']} |",
        f"| overlong | {estimates['overlong_usd']} |",
        f"| review local (before) | {estimates['review_local_usd']} |",
        f"| review full (before) | {estimates['review_full_usd']} |",
        f"| review after (local + windowed) | {estimates['review_after_local_windowed_usd']} |",
        f"| review after (full + full_transcript) | {estimates['review_after_full_full_usd']} |",
        f"| missed windowed | {estimates['missed_windowed_usd']} |",
        f"| missed full (+fallback penalty) | {estimates['missed_full_usd']} |",
        "",
        "## Adaptive decisions",
        "",
        f"- review: `{decisions['review_scope']}` ({decisions['review_reason']})",
        f"- missed: `{decisions['missed_strategy']}` ({decisions['missed_reason']})",
    ]
    if "actual_usage" in report:
        actual = report["actual_usage"]
        lines.extend(
            [
                "",
                "## Actual usage",
                "",
                f"- calls: {actual['calls']}",
                f"- cache hit/miss: {actual['prompt_cache_hit_tokens']}/"
                f"{actual['prompt_cache_miss_tokens']}",
                f"- completion: {actual['completion_tokens']}",
                f"- actual USD: {actual['actual_usd']}",
            ]
        )
    if "calibration" in report:
        cal = report["calibration"]
        lines.extend(
            [
                "",
                "## Calibration",
                "",
                f"- estimated pipeline USD: {cal.get('estimated_pipeline_usd')}",
                f"- estimated main / overlong / review / missed USD: "
                f"{cal.get('estimated_main_usd')} / {cal.get('estimated_overlong_usd')} / "
                f"{cal.get('estimated_review_usd')} / {cal.get('estimated_missed_usd')}",
                f"- actual main / overlong / review / missed USD: "
                f"{cal.get('actual_main_usd')} / {cal.get('actual_overlong_usd')} / "
                f"{cal.get('actual_review_usd')} / {cal.get('actual_missed_usd')}",
                f"- pipeline delta (actual - est): {cal.get('pipeline_delta_usd')}",
            ]
        )
    report_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_calibration_summary(
    reports: list[dict[str, Any]],
    out_path: Path,
) -> None:
    lines = [
        "# Adaptive cost calibration summary",
        "",
        "Pricing: DeepSeek v4-flash (hit $0.0028 / miss $0.14 / output $0.28 per 1M)",
        "",
        "| Run | Review est | Review actual | Missed est | Missed actual | Adaptive picks |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        name = Path(report["run_dir"]).name
        decisions = report["adaptive_decisions"]
        estimates = report["estimates"]
        cal = report.get("calibration", {})
        picks = (
            f"{decisions['review_scope']} + "
            f"{decisions['missed_strategy']} "
            f"(${decisions.get('chosen_total_usd', 'n/a')})"
        )
        lines.append(
            f"| {name} | "
            f"{estimates['review_local_usd']}/{estimates['review_full_usd']} | "
            f"{cal.get('actual_review_usd', 'n/a')} | "
            f"{estimates['missed_windowed_usd']}/{estimates['missed_full_usd']} | "
            f"{cal.get('actual_missed_usd', 'n/a')} | {picks} |"
        )
    lines.extend(
        [
            "",
            "## Calibrated defaults",
            "",
            "- `token_chars_ratio: 0.62`",
            "- `estimated_completion_tokens.review: 420`",
            "- `estimated_completion_tokens.missed_full: 1100`",
            "- `estimated_completion_tokens.missed_window: 360`",
            "- KV full transcript prefix billed as cache hit after main phase",
            "- Tool rounds included via `1 + max_tool_rounds`",
            "- `full_transcript_max_segments: 3500`",
            "- Fallback penalty only when `segment_count > full_transcript_max_segments`",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="*",
        help="Run directories containing 02_asr/transcript.json",
    )
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--profile", default="kv_optimized")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / ".tmp"),
        help="Directory for probe JSON/Markdown reports",
    )
    args = parser.parse_args()

    run_dirs = [Path(path) for path in args.run_dirs]
    if not run_dirs:
        print("No run directories provided.")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)

    reports: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        report = probe_run_dir(run_dir, config_path=config_path, profile=args.profile)
        reports.append(report)
        report_path = out_dir / f"adaptive_cost_probe_{run_id}.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_markdown(report_path, report)
        print(f"Probe report: {report_path}")
        print(f"Markdown: {report_path.with_suffix('.md')}")
        cal = report.get("calibration", {})
        print(
            f"  adaptive={report['adaptive_decisions']['review_scope']}+"
            f"{report['adaptive_decisions']['missed_strategy']} "
            f"review_est={cal.get('estimated_review_usd', 'n/a')} "
            f"review_actual={cal.get('actual_review_usd', 'n/a')} "
            f"missed_est={cal.get('estimated_missed_usd', 'n/a')} "
            f"missed_actual={cal.get('actual_missed_usd', 'n/a')}"
        )

    if reports:
        summary_path = out_dir / "adaptive_cost_calibration_summary.md"
        _write_calibration_summary(reports, summary_path)
        print(f"Calibration summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())