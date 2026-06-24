from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
    },
    "asr": {
        "mode": "local",
        "local": {
            "backend": "qwen3_asr",
            "funasr": {
                "device": "auto",
                "trust_remote_code": True,
                "batch_size": None,
                "language": None,
                "vad_model": None,
                "punc_model": None,
                "spk_model": None,
                "generate_kwargs": {"return_time_stamps": True},
                "keep_chunk_audio": False,
                "lyrics": {
                    "enabled": False,
                    "max_line_chars": 24,
                    "sentence_punctuation": "。！？.!?\n",
                    "max_sentence_duration_ms": 5000,
                },
                "fallback": {
                    "enabled": True,
                    "chunk_seconds": 5,
                    "max_segment_seconds": 15,
                    "sparse_chars_per_sec": 1.0,
                    "repeat_threshold": 3,
                    "padding_seconds": 2.0,
                    "max_workers": 1,
                    "min_gap_seconds": 10.0,
                    "merge_policy": "fill_fix_asr",
                    "zero_duration_policy": "merge_adjacent",
                },
            },
            "faster_whisper": {
                "device": "auto",
                "compute_type": "default",
                "cpu_threads": 8,
                "num_workers": 1,
                "language": None,
                "beam_size": 5,
                "initial_prompt": None,
                "word_timestamps": True,
                "split_on_word_gaps": True,
                "word_gap_seconds": 2.0,
                "max_segment_seconds": 15.0,
                "batch": {
                    "model": "small",
                    "inference_mode": "batched",
                    "batch_size": 8,
                    "vad_filter": True,
                },
                "standard": {
                    "model": "small",
                    "inference_mode": "standard",
                    "batch_size": 0,
                    "vad_filter": False,
                },
                "fallback": {
                    "enabled": True,
                    "primary_mode": "batch",
                    "fallback_mode": "standard",
                    "min_gap_seconds": 10.0,
                    "padding_seconds": 2.0,
                    "max_workers": 1,
                    "max_segment_seconds": 15.0,
                    "sparse_chars_per_sec": 1.0,
                    "repeat_threshold": 3,
                    "merge_policy": "fill_fix_asr",
                },
            },
            "gpu": {
                "funasr": {
                    "model": "Qwen/Qwen3-ASR-1.7B",
                    "hub": "ms",
                    "device": "cuda:0",
                    "dtype": "bf16",
                    "timestamp_chunk_seconds": 180,
                    "max_workers": 1,
                    "forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
                    "forced_aligner_kwargs": {},
                },
                "faster_whisper": {
                    "batch": {"model": "turbo"},
                    "standard": {"model": "turbo"},
                },
            },
            "cpu": {
                "backend": "faster_whisper",
                "faster_whisper": {
                    "compute_type": "int8",
                    "batch": {"model": "small"},
                    "standard": {"model": "small"},
                },
            },
        },
    },
    "llm": {
        "active_provider": "default",
        "provider_route": None,
        "providers": {
            "default": {
                "api_key": None,
                "api_key_env": None,
                "base_url": None,
                "model": "gpt-4o",
                "temperature": 0.1,
                "max_tokens": 8192,
                "max_completion_tokens": None,
                "timeout": 300,
                "max_retries": 3,
                "thinking": None,
                "timeout_schedule": None,
                "retry_backoff_seconds": [2, 5],
                "retry_jitter_ratio": 0.25,
                "result_retries": 2,
            },
        },
        "retry_empty_with_reasoning": True,
        "reasoning_followup_rounds": 5,
        "reasoning_followup_max_tokens": 32768,
        "max_completion_tokens": 32768,
        "batch_size": None,
        "cache_friendly_prompt_layout": False,
        "compact_segment_ranges": False,
        "max_tool_rounds": 2,
        "final_tool_max_tokens": 32768,
        "continuation_on_length": True,
        "max_continuation_rounds": 8,
        "debug_store_requests": False,
        "reuse_valid_batches": True,
        "use_tools": True,
        "verify_with_search": True,
        "json_fix_rounds": 3,
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
        "merge_gap_seconds": 40.0,
    },
    # 内容识别类型（true/false 控制启用/禁用）
    "content_types": {
        "song": True,
        "dialogue": False,
        "highlight": False,
        "funny": False,
        "cringe": False,
        "daily_summary": True,
    },
    # 歌曲识别配置
    "song": {
        "enabled": True,
        "pipeline": {
            "strategy": "legacy",
            "runtime_adaptive": "disabled",
            "stages": {
                "discovery": "precision",
                "recall_audit": "uncovered_evidence",
                "adjudication": "full_transcript",
            },
            "continuation_overlap_segments": 50,
            "allow_final_discovery": True,
            "anchor_boundary_expansion": False,
            "protocol_guard": {
                "min_candidate_limit": 64,
                "max_candidates_per_hour": 40.0,
                "short_candidate_ratio_threshold": 0.70,
                "max_final_additions": 10,
            },
            "temporal_adjudication": {
                "enabled": False,
                "preserve_source_names": True,
                "max_completion_tokens": 32768,
            },
        },
        "risk": {
            "duration_weight": 1.0,
            "boundary_expansion_weight": 2.0,
            "overlap_weight": 2.0,
            "evidence_weight": 2.0,
            "review_threshold": 0.55,
            "reject_threshold": 0.90,
            "soft_min_seconds": 150.0,
            "soft_max_seconds": 360.0,
            "boundary_gap_seconds": 20.0,
            "low_confidence": 0.65,
        },
        "naming": {
            "search_query_source": "title",
            "preserve_unknown_on_weak_evidence": True,
        },
        "normalization": {
            "force_merge_same_title": False,
            "chorus_aware_split": False,
            "chorus_gap_seconds": 120.0,
            "chorus_similarity_threshold": 0.3,
            "chorus_context_segments": 3,
        },
        "search": {
            "enabled": False,
            "search_unknown_only": True,
            "max_searches": 25,
        },
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
            "merge_gap_seconds": 40.0,
        },
        "missed_recheck": {
            "enabled": True,
            "strategy": "windowed",
            "fallback_strategy": "windowed_on_structural_failure",
            "batch_size": 500,
            "min_gap_segments": 1,
            "context_segments": 10,
            "max_completion_tokens": 32768,
            "max_tool_rounds": 1,
            "output_mode": "matches",
            "min_uncovered_seconds": 10.0,
            "max_anchor_segments": 12,
            "anchor_max_expansion_seconds": 420.0,
            "adaptive": {
                "full_transcript_max_segments": 3500,
                "windowed_min_target_ranges": 19,
            },
        },
        "review": {
            "enabled": False,
            "transcript_scope": "local",
            "context_segments": 10,
            "max_window_segments": 500,
            "max_candidates_per_request": 6,
            "max_completion_tokens": 32768,
            "max_tool_rounds": 1,
            "fallback": "local_best",
            "nearby_title_conflict_gap_segments": 2,
            "merge_policy": "conservative",  # conservative | llm_guided | aggressive ; for fused ABCF, accuracy uses llm_guided to force merge same-title per review decision (e.g. 囚鸟)
            "adaptive": {
                "local_max_clusters": 3,
                "full_min_clusters": 6,
                "full_min_segments": 2000,
            },
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
        "enabled": True,
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
        "max_export_workers": 4,
        "concat_videos": True,
        "single_file_policy": "copy",
        "concat_force_normalize": False,  # 新 pipeline 下仍先做 health probe + pre-sanitize + ProblemProfile 分类
        "clip_naming": {
            "enabled": True,
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


PROFILE_ALL = "all"


def list_profile_names(loaded: dict[str, Any]) -> list[str]:
    profiles = loaded.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        return []
    default_profile = loaded.get("default_profile")
    names = [str(name) for name in profiles]
    if default_profile and str(default_profile) in names:
        ordered = [str(default_profile), *[
            name for name in names if name != str(default_profile)
        ]]
        return ordered
    return names


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
    if selected_profile == PROFILE_ALL:
        raise ValueError(
            f"{PROFILE_ALL!r} is a reserved CLI value; load one profile at a time."
        )
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
    # 只在 YAML 显式定义了 providers 时才解析 active_provider
    has_explicit_providers = "providers" in (loaded.get("llm") or {})
    config = _migrate_padding_config(config)
    if has_explicit_providers:
        llm_cfg = config.get("llm") or {}
        if not llm_cfg.get("provider_route"):
            config = _resolve_llm_provider(config)
    return config


def _resolve_llm_provider(config: dict[str, Any]) -> dict[str, Any]:
    """将 llm.active_provider 解析为顶层 llm 配置。

    新格式下 llm 包含 providers 字典和 active_provider 选择器。
    此函数将选中的 provider 配置合并到 llm 顶层，保持旧代码兼容。
    如果配置中没有 providers 字典（旧格式），直接返回。
    """
    llm = config.get("llm", {})
    providers = llm.get("providers")
    if not isinstance(providers, dict) or not providers:
        return config

    active = str(llm.get("active_provider") or "").strip()
    if not active:
        return config

    provider_cfg = providers.get(active)
    if not isinstance(provider_cfg, dict):
        return config

    # 将 provider 配置合并到 llm 顶层（provider 优先，但保留共享参数）
    resolved = {k: v for k, v in llm.items() if k not in {"providers", "active_provider"}}
    for key, value in provider_cfg.items():
        # provider 的 None 值不覆盖顶层已有的非 None 共享参数
        if value is None and resolved.get(key) is not None:
            continue
        resolved[key] = value
    config["llm"] = resolved
    config["llm"]["_active_provider"] = active
    return config


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
        "merge_gap_seconds": 40.0,
    }


def get_song_review_config(config: dict[str, Any]) -> dict[str, Any]:
    """获取 song.review 子配置。"""
    return config.get("song", {}).get("review", {})


def get_song_recheck_config(config: dict[str, Any]) -> dict[str, Any]:
    """获取 song.missed_recheck 子配置。"""
    return config.get("song", {}).get("missed_recheck", {})


def get_llm_config(config: dict[str, Any]) -> dict[str, Any]:
    """获取 llm 子配置。"""
    return config.get("llm", {})


def get_llm_setting(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """获取单个 llm 配置值，消除 config.get('llm', {}).get(key, default) 链。"""
    return config.get("llm", {}).get(key, default)


def get_output_config(config: dict[str, Any]) -> dict[str, Any]:
    """获取 output 子配置。"""
    return config.get("output", {})


def song_pipeline_strategy(config: dict[str, Any]) -> str:
    """获取 song.pipeline.strategy 配置值。"""
    return str(
        config.get("song", {}).get("pipeline", {}).get("strategy", "legacy")
    ).strip().lower()


def is_risk_routed_v2(config: dict[str, Any]) -> bool:
    """判断是否使用 risk_routed_v2 流程。"""
    return song_pipeline_strategy(config) == "risk_routed_v2"


def is_risk_routed_kv(config: dict[str, Any]) -> bool:
    """Return whether the strict three-stage KV song pipeline is enabled."""
    return song_pipeline_strategy(config) == "risk_routed_kv"


def is_kv_v3(config: dict[str, Any]) -> bool:
    """Return whether the kv_v3 optimized pipeline is enabled (now in kv_v2)."""
    return config.get("_profile_name") in {"kv_v2", "kv_v3"}


def is_risk_routed(config: dict[str, Any]) -> bool:
    """Return whether either risk-routed song pipeline is enabled."""
    return song_pipeline_strategy(config) in {"risk_routed_v2", "risk_routed_kv"}


def get_song_normalization_config(config: dict[str, Any]) -> dict[str, Any]:
    """获取 song.normalization 子配置。"""
    return config.get("song", {}).get("normalization", {})


def get_song_search_config(config: dict[str, Any]) -> dict[str, Any]:
    """获取 song.search 子配置。"""
    return config.get("song", {}).get("search", {})


def get_asr_inference_mode(settings: dict[str, Any]) -> str:
    """Resolve the primary ASR inference mode."""
    mode = str(settings.get("mode", "")).lower()
    if mode == "local":
        local_cfg = settings.get("local", {}) or {}
        backend = str(local_cfg.get("backend", "faster_whisper")).lower().replace("-", "_")
        if backend in ("faster_whisper", "whisper"):
            fw = local_cfg.get("faster_whisper", {}) or local_cfg
            fallback = fw.get("fallback", {}) if isinstance(fw, dict) else {}
            primary_mode = str(fallback.get("primary_mode") or "batch")
            mode_cfg = fw.get(primary_mode, {}) if isinstance(fw, dict) else {}
            return _resolve_inference_mode(mode_cfg)
        return "standard"
    return "standard"


def _resolve_inference_mode(fw: dict[str, Any]) -> str:
    mode = str(fw.get("inference_mode", "")).lower().strip()
    if mode in ("batched", "standard"):
        return mode
    if mode:
        raise ValueError(
            f"Invalid inference_mode '{mode}' for faster_whisper. Must be 'batched' or 'standard'."
        )
    batch_size = int(fw.get("batch_size", 0))
    return "batched" if batch_size > 0 else "standard"


def get_asr_fingerprint(config: dict[str, Any]) -> str:
    """Fingerprint of ASR-relevant settings used for transcript reuse."""
    payload = deepcopy(config.get("asr", {}) or {})
    local = payload.get("local", {}) if isinstance(payload, dict) else {}
    if isinstance(local, dict):
        fw = local.get("faster_whisper", {})
        if isinstance(fw, dict) and str(fw.get("device", "auto")).lower() == "auto":
            current_ct = str(fw.get("compute_type", "default")).lower()
            if current_ct in ("", "default"):
                try:
                    import ctranslate2
                    if ctranslate2.get_cuda_device_count() == 0:
                        fw["compute_type"] = "int8"
                except Exception:
                    fw["compute_type"] = "int8"
    return _fingerprint_payload(payload)


def _fingerprint_payload(value: Any) -> str:
    """Internal sha256 json fingerprint (duplicated from profile_state to avoid import cycle)."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
