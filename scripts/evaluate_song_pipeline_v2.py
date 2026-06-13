from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dd_clip_miner_llm.song_evaluation import evaluate_song_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate song profiles without API calls.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline", default="accuracy")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    report = evaluate_song_run(args.run_dir, baseline_profile=args.baseline)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if args.enforce:
        candidates = [
            metrics for name, metrics in report["profiles"].items()
            if name != args.baseline
        ]
        return 0 if candidates and all(item.get("passes_all_gates") for item in candidates) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
