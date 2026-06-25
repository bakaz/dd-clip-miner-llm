"""Resolve config.yaml -> cut_copy.conf -> source.path for task setup.

Uses !include-aware YAML loading from dd_clip_miner_llm.config so modular
configs (with !include tags and profiles) are fully supported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dd_clip_miner_llm.resolve_batch_config import resolve_batch_config


def resolve(config_yaml: str | Path) -> dict:
    payload = resolve_batch_config(config_yaml, Path("."))
    return {
        "enabled": payload.get("enabled", False),
        "cut_copy_conf": payload.get("cut_copy_conf", ""),
        "source_path": payload.get("source_path", ""),
        "error": payload.get("error", ""),
    }


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 1:
        print("usage: resolve_batch_config.py <config.yaml>", file=sys.stderr)
        return 2

    payload = resolve(args[0])
    json.dump(payload, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())