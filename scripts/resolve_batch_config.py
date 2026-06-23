"""Resolve config.yaml -> cut_copy.conf -> source.path for task setup."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def resolve(config_yaml: str | Path) -> dict:
    config_path = Path(config_yaml)
    config_dir = config_path.parent
    result = {
        "enabled": False,
        "cut_copy_conf": "",
        "source_path": "",
        "error": "",
    }

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        cc = cfg.get("cut_copy", {}) or {}
        result["enabled"] = bool(cc.get("enabled", False))

        conf_rel = cc.get("conf_path", "cut_copy.conf")
        conf_path = Path(conf_rel)
        if not conf_path.is_absolute():
            conf_path = config_dir / conf_path
        result["cut_copy_conf"] = str(conf_path)

        if conf_path.is_file():
            with conf_path.open("r", encoding="utf-8") as handle:
                cc_cfg = yaml.safe_load(handle) or {}
            result["source_path"] = str(
                (cc_cfg.get("source", {}) or {}).get("path", "") or ""
            )
        else:
            result["error"] = f"cut_copy.conf not found: {conf_path}"
    except Exception as exc:
        result["error"] = str(exc)

    return result


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