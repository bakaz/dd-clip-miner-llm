"""Run local vs full song.review.transcript_scope A/B on a seeded ASR fixture."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dd_clip_miner_llm.config import load_config
from dd_clip_miner_llm.pipeline import run_pipeline

GATES = {
    "review_cache_miss_baseline": 46_842,
    "review_cache_miss_max": 32_789,
    "total_calls_max": 24,
    "completion_tokens_max": 13_776,
    "main_candidates_min": 23,
    "final_named_min": 22,
}

REQUIRED_TITLES = (
    "七月七日晴",
    "说了再见",
    "雨天",
    "下雨天",
)
REQUIRED_EXACT = ("天后",)
REQUIRED_TITLE_ALIASES: dict[str, frozenset[str]] = {
    "七月七日晴": frozenset({"七月七日晴", "七月七日清"}),
    "说了再见": frozenset({"说了再见", "說了再見"}),
    "雨天": frozenset({"雨天"}),
    "下雨天": frozenset({"下雨天"}),
}
_TRADITIONAL_CHAR_MAP = str.maketrans(
    {
        "說": "说",
        "見": "见",
        "聽": "听",
        "後": "后",
        "裡": "里",
        "國": "国",
        "會": "会",
        "這": "这",
        "們": "们",
        "時": "时",
        "間": "间",
        "愛": "爱",
        "戀": "恋",
        "陽": "阳",
        "陰": "阴",
        "雲": "云",
        "風": "风",
        "車": "车",
        "長": "长",
        "廣": "广",
        "華": "华",
        "樂": "乐",
        "歡": "欢",
        "離": "离",
        "難": "难",
        "夢": "梦",
        "憶": "忆",
        "憂": "忧",
        "鬱": "郁",
        "淚": "泪",
        "聲": "声",
        "響": "响",
        "體": "体",
        "關": "关",
        "開": "开",
        "門": "门",
        "問": "问",
        "題": "题",
        "頭": "头",
        "臉": "脸",
        "發": "发",
        "現": "现",
        "實": "实",
        "無": "无",
        "為": "为",
        "與": "与",
        "從": "从",
        "來": "来",
        "個": "个",
        "倆": "俩",
        "麼": "么",
        "麗": "丽",
        "點": "点",
        "線": "线",
        "經": "经",
        "過": "过",
        "還": "还",
        "進": "进",
        "遠": "远",
        "邊": "边",
        "裏": "里",
        "裡": "里",
    }
)


def _seed_run_dir(fixture: Path, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("00_input", "01_audio"):
        src = fixture / sub
        dst = out_dir / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    asr_dir = out_dir / "02_asr"
    asr_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fixture / "02_asr" / "transcript.json", asr_dir / "transcript.json")

    inputs = sorted((out_dir / "00_input").glob("*"))
    if not inputs:
        raise FileNotFoundError(f"No input video under {out_dir / '00_input'}")
    video = inputs[0]

    progress = {
        "input_video": str(video.resolve()),
        "last_completed_step": "asr",
    }
    (out_dir / "progress.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_dir, video


def _named_title(title: str) -> bool:
    return bool(title) and not title.startswith("未知歌曲")


def _canonical_title(title: str) -> str:
    stripped = title.strip()
    if not stripped:
        return stripped
    try:
        import zhconv

        return zhconv.convert(stripped, "zh-cn")
    except ImportError:
        return stripped.translate(_TRADITIONAL_CHAR_MAP)


def _title_matches_canonical(actual: str, canonical: str) -> bool:
    stripped = actual.strip()
    if not stripped:
        return False
    aliases = REQUIRED_TITLE_ALIASES.get(canonical, frozenset({canonical}))
    if stripped in aliases:
        return True
    return _canonical_title(stripped) == canonical


def _required_title_checks(titles: list[str]) -> dict[str, bool]:
    checks = {
        canonical: any(_title_matches_canonical(title, canonical) for title in titles)
        for canonical in REQUIRED_TITLES
    }
    for title in REQUIRED_EXACT:
        checks[title] = sum(
            1 for candidate in titles if _title_matches_canonical(candidate, title)
        ) == 1
    return checks


def _intervals_from_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in matches:
        indices = match.get("segment_indices") or []
        if not indices:
            continue
        rows.append(
            {
                "title": str(match.get("title") or ""),
                "start": min(indices),
                "end": max(indices),
                "indices": sorted(indices),
            }
        )
    return rows


def _overlap_issues(intervals: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for i, left in enumerate(intervals):
        for right in intervals[i + 1 :]:
            if left["title"] == right["title"]:
                if left["start"] == right["start"] and left["end"] == right["end"]:
                    issues.append(f"duplicate interval: {left['title']}")
                continue
            overlap_start = max(left["start"], right["start"])
            overlap_end = min(left["end"], right["end"])
            if overlap_start <= overlap_end:
                issues.append(
                    f"cross-title overlap: {left['title']} vs {right['title']} "
                    f"segments {overlap_start}-{overlap_end}"
                )
    return issues


def _evaluate_run(profile_llm_dir: Path) -> dict[str, Any]:
    song_dir = profile_llm_dir / "song"
    initial_path = song_dir / "initial_matches.json"
    matches_path = song_dir / "matches.json"
    usage_path = profile_llm_dir / "usage_summary.json"

    initial_matches: list[dict[str, Any]] = []
    if initial_path.exists():
        initial_matches = json.loads(initial_path.read_text(encoding="utf-8"))
    final_matches = json.loads(matches_path.read_text(encoding="utf-8"))

    final_named = [m for m in final_matches if _named_title(str(m.get("title") or ""))]
    titles = [str(m.get("title") or "") for m in final_named]

    title_checks = _required_title_checks(titles)

    usage_summary: dict[str, Any] = {}
    if usage_path.exists():
        usage_summary = json.loads(usage_path.read_text(encoding="utf-8"))

    phases = usage_summary.get("phases", {})
    review_miss = 0
    for phase in ("review_before", "review_after"):
        phase_totals = phases.get(phase, {})
        if isinstance(phase_totals, dict):
            review_miss += int(phase_totals.get("prompt_cache_miss_tokens") or 0)

    totals = usage_summary.get("totals", {})
    total_calls = int(totals.get("calls") or 0)
    completion_tokens = int(totals.get("completion_tokens") or 0)

    gate_results = {
        "main_candidates_ge_23": len(initial_matches) >= GATES["main_candidates_min"],
        "final_named_ge_22": len(final_named) >= GATES["final_named_min"],
        "required_titles": all(title_checks.values()),
        "no_overlap_or_dupes": not _overlap_issues(_intervals_from_matches(final_matches)),
        "review_cache_miss_le_32789": review_miss <= GATES["review_cache_miss_max"],
        "total_calls_le_24": total_calls <= GATES["total_calls_max"],
        "completion_le_13776": completion_tokens <= GATES["completion_tokens_max"],
    }

    return {
        "main_candidates": len(initial_matches),
        "final_named": len(final_named),
        "titles": titles,
        "title_checks": title_checks,
        "overlap_issues": _overlap_issues(_intervals_from_matches(final_matches)),
        "usage": {
            "review_cache_miss": review_miss,
            "total_calls": total_calls,
            "completion_tokens": completion_tokens,
            "phases": phases,
            "totals": totals,
        },
        "gates": gate_results,
        "all_gates_pass": all(gate_results.values()),
    }


def _default_run_id(fixture: Path) -> str:
    name = fixture.name
    if "-" in name:
        suffix = name.rsplit("-", 1)[-1].strip()
        if suffix:
            return suffix
    return name


def _run_variant(
    *,
    fixture: Path,
    out_root: Path,
    scope: str,
    config_path: Path,
    run_id: str,
) -> dict[str, Any]:
    out_dir = out_root / f"{run_id}_ab_{scope}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    _, video = _seed_run_dir(fixture, out_dir)

    config = load_config(config_path, profile="kv_optimized")
    config["content_types"] = {"song": True}
    config["daily_summary"] = {**config.get("daily_summary", {}), "enabled": False}
    config["output"]["video_clips"] = False
    config["output"]["audio_segments"] = False
    config.setdefault("song", {}).setdefault("review", {})["enabled"] = True
    config["song"]["review"]["transcript_scope"] = scope

    print(f"\n=== Running kv_optimized review transcript_scope={scope} ===")
    print(f"Output: {out_dir}")
    run_pipeline(video, out_dir, config, config_path=str(config_path))

    profile_llm_dir = out_dir / "02_asr" / "llm" / "kv_optimized"
    metrics = _evaluate_run(profile_llm_dir)
    metrics["scope"] = scope
    metrics["out_dir"] = str(out_dir)
    return metrics


def _write_report(
    report_path: Path,
    local: dict[str, Any],
    full: dict[str, Any],
    *,
    run_id: str,
) -> None:
    lines = [
        f"# Review transcript_scope A/B ({run_id})",
        "",
        f"Fixture: `{local.get('fixture', '')}`",
        "",
        "## Gates",
        "",
        "| Gate | Local | Full |",
        "| --- | --- | --- |",
    ]
    for gate in local["gates"]:
        lines.append(
            f"| {gate} | {local['gates'][gate]} | {full['gates'][gate]} |"
        )
    lines.extend(
        [
            "",
            f"**Local all pass:** {local['all_gates_pass']}",
            f"**Full all pass:** {full['all_gates_pass']}",
            "",
            "## Metrics",
            "",
            "| Metric | Local | Full |",
            "| --- | --- | --- |",
            f"| main_candidates | {local['main_candidates']} | {full['main_candidates']} |",
            f"| final_named | {local['final_named']} | {full['final_named']} |",
            f"| review_cache_miss | {local['usage']['review_cache_miss']} | {full['usage']['review_cache_miss']} |",
            f"| total_calls | {local['usage']['total_calls']} | {full['usage']['total_calls']} |",
            f"| completion_tokens | {local['usage']['completion_tokens']} | {full['usage']['completion_tokens']} |",
            "",
            "## Recommendation",
            "",
        ]
    )
    if full["all_gates_pass"] and local["all_gates_pass"]:
        if full["usage"]["review_cache_miss"] < local["usage"]["review_cache_miss"]:
            lines.append(
                "Both variants pass gates; `full` reduces review cache miss — safe to default `transcript_scope: full`."
            )
        else:
            lines.append(
                "Both pass gates but `full` does not beat `local` on review cache miss — keep default `local`."
            )
    elif full["all_gates_pass"]:
        lines.append("`full` passes all gates — safe to default `transcript_scope: full`.")
    else:
        lines.append("Do **not** switch default to `full` until all gates pass.")

    report = {
        "gates": GATES,
        "local": local,
        "full": full,
        "recommendation": lines[-1],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fixture",
        help="Run directory with 00_input, 01_audio, 02_asr/transcript.json",
    )
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--out-root", default=str(ROOT / ".tmp"))
    parser.add_argument("--run-id", default=None, help="Output dir suffix; default from fixture name")
    parser.add_argument("--report", default=None, help="Report JSON path; default .tmp/review_scope_ab_<run-id>.json")
    args = parser.parse_args()

    fixture = Path(args.fixture).resolve()
    run_id = str(args.run_id or _default_run_id(fixture))
    if not (fixture / "02_asr" / "transcript.json").exists():
        print(f"Missing ASR fixture: {fixture}")
        return 1

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    local = _run_variant(
        fixture=fixture,
        out_root=out_root,
        scope="local",
        config_path=Path(args.config),
        run_id=run_id,
    )
    local["fixture"] = str(fixture)
    local["run_id"] = run_id

    full = _run_variant(
        fixture=fixture,
        out_root=out_root,
        scope="full",
        config_path=Path(args.config),
        run_id=run_id,
    )
    full["fixture"] = str(fixture)
    full["run_id"] = run_id

    report_path = Path(args.report or (ROOT / ".tmp" / f"review_scope_ab_{run_id}.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(report_path, local, full, run_id=run_id)

    print(f"\nReport: {report_path}")
    print(f"Markdown: {report_path.with_suffix('.md')}")
    print(f"Local all gates pass: {local['all_gates_pass']}")
    print(f"Full all gates pass: {full['all_gates_pass']}")
    if not full["all_gates_pass"]:
        print("Note: run completed; acceptance gates not met (exit 0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
