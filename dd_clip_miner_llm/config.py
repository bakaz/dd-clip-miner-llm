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
        "backend": "funasr",
        "model": "small",
        "device": "auto",
        "compute_type": "default",
        "language": None,
        "beam_size": 5,
        "vad_filter": True,
        "initial_prompt": None,
        "funasr": {
            "model": "Qwen/Qwen3-ASR-0.6B",
            "hub": "hf",
            "trust_remote_code": True,
            "device": "auto",
            "batch_size": 1,
            "language": None,
            "vad_model": None,
            "punc_model": None,
            "spk_model": None,
            "generate_kwargs": {},
        },
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
        "cache_friendly_prompt_layout": False,
        "compact_segment_ranges": False,
        "max_tool_rounds": 2,
        "final_tool_max_tokens": None,
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
        "adaptive_silence_padding": True,
        "adaptive_silence_gap_threshold_seconds": 25.0,
        "adaptive_silence_gap_ratio": 0.95,
        "adaptive_max_before_seconds": 45.0,
        "adaptive_max_after_seconds": 45.0,
        "min_song_seconds": 75.0,
        "max_song_seconds": 360.0,
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
            "adaptive_silence_padding": True,
            "adaptive_silence_gap_threshold_seconds": 25.0,
            "adaptive_silence_gap_ratio": 0.95,
            "adaptive_max_before_seconds": 45.0,
            "adaptive_max_after_seconds": 45.0,
            "min_song_seconds": 75.0,
            "max_song_seconds": 360.0,
            "merge_gap_seconds": 20.0,
        },
        "missed_recheck": {
            "enabled": True,
            "strategy": "windowed",
            "fallback_strategy": "windowed_on_structural_failure",
            "batch_size": 500,
            "min_gap_segments": 1,
            "context_segments": 10,
            "max_completion_tokens": 4096,
            "max_tool_rounds": 1,
        },
        "review": {
            "enabled": False,
            "context_segments": 10,
            "max_window_segments": 500,
            "max_completion_tokens": 4096,
            "max_tool_rounds": 1,
            "fallback": "local_best",
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
        "single_file_policy": "copy",
        "concat_force_normalize": False,  # 新 pipeline 下仍先做 health probe + pre-sanitize + ProblemProfile 分类
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


def load_config(
    path: str | Path | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    if path is None:
        if profile:
            raise ValueError("A profile can only be selected from a YAML config with a profiles mapping.")
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

    profiles = loaded.get("profiles")
    if profiles is None:
        if profile:
            raise ValueError(f"Config does not define profiles; cannot select profile: {profile}")
        config = deep_merge(DEFAULT_CONFIG, loaded)
        return _migrate_padding_config(config)

    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Config profiles must be a non-empty mapping.")

    selected_profile = profile or loaded.get("default_profile")
    if not selected_profile:
        selected_profile = next(iter(profiles))
    selected_profile = str(selected_profile)
    if selected_profile not in profiles:
        available = ", ".join(sorted(str(name) for name in profiles))
        raise ValueError(
            f"Unknown config profile {selected_profile!r}. Available profiles: {available}"
        )
    profile_override = profiles[selected_profile]
    if not isinstance(profile_override, dict):
        raise ValueError(f"Config profile {selected_profile!r} must be a mapping.")

    common = {
        key: value
        for key, value in loaded.items()
        if key not in {"profiles", "default_profile"}
    }
    config = deep_merge(DEFAULT_CONFIG, common)
    config = deep_merge(config, profile_override)
    config["_profile_name"] = selected_profile
    config["_profile_enabled"] = True
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
        "adaptive_silence_padding": True,
        "adaptive_silence_gap_threshold_seconds": 25.0,
        "adaptive_silence_gap_ratio": 0.95,
        "adaptive_max_before_seconds": 45.0,
        "adaptive_max_after_seconds": 45.0,
        "min_song_seconds": 75.0,
        "max_song_seconds": 360.0,
        "merge_gap_seconds": 20.0,
    }
