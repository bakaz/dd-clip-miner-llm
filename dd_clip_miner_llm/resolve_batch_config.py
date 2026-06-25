"""Resolve config.yaml -> cut_copy.conf -> source.path for task setup.

Uses !include-aware YAML loading from config.py so modular configs
(with !include tags and profiles) are fully supported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import _load_yaml_with_includes


def resolve_batch_config(config_path: str | Path, video: Path) -> dict[str, Any]:
    """Resolve a batch configuration from a YAML config file.

    Loads the config via the !include-aware loader, resolves the
    cut_copy.conf path, and returns a dict with the resolved
    configuration and metadata.

    Args:
        config_path: Path to the main YAML config file.
        video: Path to the video file (reserved for future use).

    Returns:
        A dict with keys: config, enabled, cut_copy_conf,
        source_path, error.
    """
    config_path = Path(config_path)
    config_dir = config_path.parent
    result: dict[str, Any] = {
        "enabled": False,
        "cut_copy_conf": "",
        "source_path": "",
        "error": "",
    }

    try:
        cfg = _load_yaml_with_includes(config_path)
        result["config"] = cfg

        cc = cfg.get("cut_copy", {}) or {}
        result["enabled"] = bool(cc.get("enabled", False))

        conf_rel = cc.get("conf_path", "cut_copy.conf")
        conf_path = Path(conf_rel)
        if not conf_path.is_absolute():
            conf_path = config_dir / conf_path
        result["cut_copy_conf"] = str(conf_path)

        if conf_path.is_file():
            cc_cfg = _load_yaml_with_includes(conf_path)
            result["source_path"] = str(
                (cc_cfg.get("source", {}) or {}).get("path", "") or ""
            )
        else:
            result["error"] = f"cut_copy.conf not found: {conf_path}"
    except Exception as exc:
        result["error"] = str(exc)

    return result
