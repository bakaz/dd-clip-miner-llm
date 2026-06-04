from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
    },
    "asr": {
        "model": "small",
        "device": "auto",
        "compute_type": "default",
        "language": None,
        "beam_size": 5,
        "vad_filter": True,
        "initial_prompt": None,
    },
    "llm": {
        "api_key": None,
        "api_key_env": None,
        "base_url": None,
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 8192,
        "max_completion_tokens": None,
        "retry_empty_with_reasoning": True,
        "reasoning_followup_rounds": 5,
        "reasoning_followup_max_tokens": 32768,
        "batch_size": None,
        "use_tools": True,
        "verify_with_search": True,
        "json_fix_rounds": 3,
        "fallbacks": [],
    },
    # 兼容旧项目 dd-song-miner-llm 的顶层 padding 配置
    "padding": {
        "before_seconds": 15.0,
        "after_seconds": 15.0,
        "after_next_asr_end_guard_seconds": 2.0,
        "min_song_seconds": 75.0,
        "merge_gap_seconds": 20.0,
    },
    # 内容识别类型（true/false 控制启用/禁用）
    "content_types": {
        "song": True,
        "dialogue": True,
        "highlight": True,
        "funny": True,
        "cringe": True,
        "daily_summary": False,
    },
    # 歌曲识别配置
    "song": {
        "enabled": True,
        "padding": {
            "before_seconds": 15.0,
            "after_seconds": 15.0,
            "after_next_asr_end_guard_seconds": 2.0,
            "min_song_seconds": 75.0,
            "merge_gap_seconds": 20.0,
        },
        "missed_recheck": {
            "enabled": True,
            "batch_size": 500,
            "min_gap_segments": 1,
        },
    },
    # 对话识别配置
    "dialogue": {
        "enabled": True,
        "min_duration": 10.0,
        "max_duration": 300.0,
        "min_confidence": 0.6,
        "merge_gap_seconds": 10.0,
        "tags": ["搞笑", "吐槽", "名场面", "金句", "互动", "高能"],
    },
    # 高能时刻配置
    "highlight": {
        "enabled": False,
        "min_duration": 5.0,
        "max_duration": 120.0,
        "min_confidence": 0.6,
        "merge_gap_seconds": 15.0,
    },
    # 搞笑片段配置
    "funny": {
        "enabled": False,
        "min_duration": 5.0,
        "max_duration": 180.0,
        "min_confidence": 0.6,
        "merge_gap_seconds": 15.0,
    },
    # 下头对话配置
    "cringe": {
        "enabled": True,
        "min_duration": 5.0,
        "max_duration": 120.0,
        "min_confidence": 0.6,
        "merge_gap_seconds": 15.0,
    },
    "daily_summary": {
        "enabled": False,
        "summary_only": True,
        "language": "zh-CN",
        "title": "当天直播内容总结",
        "max_level1_items": 6,
        "max_level2_per_level1": 5,
        "max_level3_per_level2": 4,
        "include_timeline": True,
        "include_quotes": True,
        "include_open_questions": True,
    },
    "output": {
        "video_clips": True,
        "audio_segments": True,
        "audio_extension": "mp3",
        "audio_bitrate_kbps": 320,
        "video_extension": "mp4",
        "video_codec": "copy",
        "match_context_segments": 10,
        "concat_videos": False,
        "clip_naming": {
            "enabled": False,
            "dictionary_path": "streamer_dictionary.json",
            "default_streamer": "StreamerName",
            "min_score": 0.65,
            "apply_to": ["song"],
        },
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _migrate_padding_config(config: dict[str, Any]) -> dict[str, Any]:
    """兼容旧项目 dd-song-miner-llm 的 padding 配置结构。
    
    旧项目将 padding 放在顶层，新项目将其放在 song.padding。
    为了兼容，如果顶层有 padding 配置，会同时同步到 song.padding。
    """
    if "padding" in config:
        # 将顶层 padding 同步到 song.padding
        if "song" not in config:
            config["song"] = {}
        if "padding" not in config["song"]:
            config["song"]["padding"] = {}
        # 合并顶层 padding 到 song.padding（song.padding 优先）
        config["song"]["padding"] = deep_merge(config["padding"], config["song"]["padding"])
    return config


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return _migrate_padding_config(deepcopy(DEFAULT_CONFIG))

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install PyYAML") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    config = deep_merge(DEFAULT_CONFIG, loaded)
    return _migrate_padding_config(config)


def get_padding_config(config: dict[str, Any], content_type: str = "song") -> dict[str, Any]:
    """获取 padding 配置，兼容新旧配置结构。
    
    优先使用 content_type.padding，如果不存在则使用顶层 padding。
    """
    # 首先尝试从内容类型配置中获取
    type_config = config.get(content_type, {})
    if "padding" in type_config:
        return type_config["padding"]
    
    # 回退到顶层 padding 配置（兼容旧项目）
    if "padding" in config:
        return config["padding"]
    
    # 返回默认值
    return {
        "before_seconds": 15.0,
        "after_seconds": 15.0,
        "after_next_asr_end_guard_seconds": 2.0,
        "min_song_seconds": 75.0,
        "merge_gap_seconds": 20.0,
    }
