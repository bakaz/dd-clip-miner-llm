from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_CONFIG, PROFILE_ALL, list_profile_names, load_config
from .ffmpeg import detect_ffmpeg_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dd-clip-miner-llm",
        description="基于 Whisper ASR + LLM 的直播内容挖掘工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run 命令
    run_parser = subparsers.add_parser("run", help="处理单个视频文件")
    run_parser.add_argument("video", help="输入视频文件")
    run_parser.add_argument("--out", default=None, help="输出目录")
    run_parser.add_argument("--out-root", default="runs", help="自动创建运行目录的根目录")
    run_parser.add_argument("--config", default=None, help="YAML 配置文件")
    run_parser.add_argument(
        "--profile",
        default=None,
        help="配置 profile 名称，或 all 串行运行全部 profile",
    )
    run_parser.add_argument("--content-types", default=None, help="要识别的内容类型，逗号分隔 (song,dialogue,highlight,funny)。不指定则使用配置文件")
    run_parser.add_argument("--asr-model", default=None, help="Whisper 模型")
    run_parser.add_argument("--asr-language", default=None, help="ASR 语言提示")
    run_parser.add_argument("--llm-model", default=None, help="LLM 模型名")
    run_parser.add_argument("--llm-api-key", default=None, help="LLM API key")
    run_parser.add_argument("--llm-base-url", default=None, help="LLM API base URL")
    run_parser.add_argument("--padding-before", type=float, default=None, help="歌曲开始前 padding（秒）")
    run_parser.add_argument("--padding-after", type=float, default=None, help="歌曲结束后 padding（秒）")
    run_parser.add_argument("--no-video-clips", action="store_true", help="不导出视频片段")
    run_parser.add_argument("--export-audio", default=None, help="音频导出格式")
    run_parser.add_argument("--export-video", default=None, help="视频导出格式")
    run_parser.add_argument("--video-codec", default=None, help="视频编码器")
    run_parser.add_argument("--audio-bitrate-kbps", type=int, default=None, help="音频码率")

    # batch-run 命令
    batch_parser = subparsers.add_parser("batch-run", help="批量处理目录下的视频")
    batch_parser.add_argument("input_root", help="要扫描的目录")
    batch_parser.add_argument("--result-root", required=True, help="结果输出目录")
    batch_parser.add_argument("--work-root", default="runs/batch", help="工作目录")
    batch_parser.add_argument("--config", default=None, help="YAML 配置文件")
    batch_parser.add_argument(
        "--profile",
        default=None,
        help="配置 profile 名称，或 all 串行运行全部 profile",
    )
    batch_parser.add_argument("--content-types", default=None, help="要识别的内容类型，逗号分隔")
    batch_parser.add_argument("--marker", default=".dd_clip_miner_done.json", help="完成标记文件")
    batch_parser.add_argument("--extensions", default=None, help="视频扩展名，逗号分隔")
    batch_parser.add_argument("--concat", action="store_true", help="合并目录下的多个视频后再处理")
    batch_parser.add_argument("--video-codec", default=None, help="视频编码器")
    batch_parser.add_argument("--audio-bitrate-kbps", type=int, default=None, help="音频码率")
    batch_parser.add_argument("--cut-copy-conf", default=None, help="cut_copy 配置文件路径，batch-run 完成后自动执行 cut_copy 后处理")

    # manual-cut 命令（兼容旧项目）
    manual_parser = subparsers.add_parser("manual-cut", help="从编辑后的 CSV 重新切割片段")
    manual_parser.add_argument("run_dir", help="已有的运行输出目录")
    manual_parser.add_argument("--csv", default=None, help="编辑后的 CSV 路径，默认为 RUN_DIR/04_reports/songs.csv")
    manual_parser.add_argument("--video", default=None, help="输入视频覆盖，默认从 manifest 读取")
    manual_parser.add_argument("--out", default=None, help="手动输出目录，默认为 RUN_DIR/05_manual")
    manual_parser.add_argument("--config", default=None, help="YAML 配置文件")
    manual_parser.add_argument("--content-type", default="song", help="内容类型 (song/dialogue)")
    manual_parser.add_argument("--video-codec", default=None, help="视频编码器")
    manual_parser.add_argument("--audio-bitrate-kbps", type=int, default=None, help="音频码率")

    # post-merge 命令：拖拽两个已导出片段后，从原始输入重新切为一个片段
    post_merge_parser = subparsers.add_parser("post-merge", help="从两个已导出歌曲片段反查 ASR 并重新切为一个片段")
    post_merge_parser.add_argument("file1", help="第一个已导出 MP4/MP3")
    post_merge_parser.add_argument("file2", help="第二个已导出 MP4/MP3")
    post_merge_parser.add_argument("--context", required=True, help="merge_recut_context.json 路径")

    # manual-cut-context 命令：从 context JSON 读取视频路径，手动切片到同目录
    mcc_parser = subparsers.add_parser("manual-cut-context", help="从 context JSON 手动切片到同目录")
    mcc_parser.add_argument("--context", required=True, help="manual_cut_context.json 路径")
    mcc_parser.add_argument("--start", required=True, help="开始时间 (如 10:30 或 630)")
    mcc_parser.add_argument("--end", required=True, help="结束时间 (如 15:45 或 945)")
    mcc_parser.add_argument("--filename", default=None, help="输出文件名（不含扩展名，留空自动生成）")

    # init-config 命令
    init_parser = subparsers.add_parser("init-config", help="生成默认配置文件")
    init_parser.add_argument("--out", default="config.yaml", help="输出路径")

    # ffmpeg-info 命令
    info_parser = subparsers.add_parser("ffmpeg-info", help="显示 GPU 和 FFmpeg 编码器信息")
    info_parser.add_argument("--ffmpeg", default=None, help="FFmpeg 路径")

    # cut-copy 命令
    cc_parser = subparsers.add_parser(
        "cut-copy",
        help="录播自动处理工作流：扫描 DDTV 输出 → 处理 → 复制到 SMB → 关机",
    )
    cc_parser.add_argument(
        "--conf", default="cut_copy.conf",
        help="cut_copy 配置文件路径 (默认: cut_copy.conf)",
    )
    cc_parser.add_argument(
        "--dry-run", action="store_true",
        help="只显示待处理文件，不执行处理/复制/关机",
    )
    cc_parser.add_argument(
        "--no-shutdown", action="store_true",
        help="本次运行不关机（覆盖配置文件设置）",
    )

    return parser


def _config_example_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.example.yaml"


def _generate_config_yaml() -> str:
    """Return the shipped config template (config.example.yaml)."""
    template = _config_example_path()
    if not template.is_file():
        raise FileNotFoundError(
            f"Config template not found: {template}. "
            "Copy config.example.yaml manually if running outside the project tree."
        )
    return template.read_text(encoding="utf-8")


def _apply_run_overrides(config: dict, args: argparse.Namespace) -> None:
    if args.content_types:
        # 将逗号分隔的列表转换为字典格式
        types_list = [ct.strip() for ct in args.content_types.split(",") if ct.strip()]
        config["content_types"] = {ct: True for ct in types_list}
    if args.asr_model:
        from .asr_backends import apply_asr_model_override

        apply_asr_model_override(config["asr"], args.asr_model)
    if args.asr_language:
        config["asr"]["language"] = args.asr_language
    if args.llm_model:
        config["llm"]["model"] = args.llm_model
    if args.llm_api_key:
        config["llm"]["api_key"] = args.llm_api_key
    if args.llm_base_url:
        config["llm"]["base_url"] = args.llm_base_url
    # 兼容旧项目的 padding 参数
    if args.padding_before is not None:
        # 同时更新顶层和 song.padding
        config["padding"]["before_seconds"] = args.padding_before
        if "song" in config and "padding" in config["song"]:
            config["song"]["padding"]["before_seconds"] = args.padding_before
    if args.padding_after is not None:
        config["padding"]["after_seconds"] = args.padding_after
        if "song" in config and "padding" in config["song"]:
            config["song"]["padding"]["after_seconds"] = args.padding_after
    if args.no_video_clips:
        config["output"]["video_clips"] = False
    if args.export_audio:
        config["output"]["audio_segments"] = True
        config["output"]["audio_extension"] = args.export_audio.lstrip(".")
    if args.export_video:
        config["output"]["video_clips"] = True
        config["output"]["video_extension"] = args.export_video.lstrip(".")
    _apply_output_overrides(config, args)


def _apply_output_overrides(config: dict, args: argparse.Namespace) -> None:
    if getattr(args, "video_codec", None):
        config["output"]["video_codec"] = args.video_codec
    if getattr(args, "audio_bitrate_kbps", None) is not None:
        config["output"]["audio_bitrate_kbps"] = args.audio_bitrate_kbps


def _has_api_key(config: dict) -> bool:
    import os
    llm = config.get("llm", {})

    # 新格式：provider_route 检查 providers 内的 key
    provider_route = llm.get("provider_route")
    if provider_route:
        providers = llm.get("providers", {})
        for name in provider_route:
            pcfg = providers.get(str(name), {})
            if not isinstance(pcfg, dict):
                continue
            key = pcfg.get("api_key", "")
            key_env = pcfg.get("api_key_env")
            if not key and key_env:
                key = os.environ.get(str(key_env), "")
            if key:
                return True
        return False

    # 旧格式：顶层 api_key
    api_key = llm.get("api_key")
    api_key_env = llm.get("api_key_env")
    if not api_key and api_key_env:
        api_key = os.environ.get(str(api_key_env), "")
    return bool(api_key)


def _load_raw_yaml_config(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install PyYAML") from exc
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return loaded


def _resolve_profile_names(
    config_path: str | Path | None,
    profile: str | None,
) -> list[str | None]:
    if profile != PROFILE_ALL:
        return [profile]
    if config_path is None:
        raise ValueError("--profile all requires a YAML config with profiles.")
    names = list_profile_names(_load_raw_yaml_config(config_path))
    if not names:
        raise ValueError("Config does not define profiles; cannot use --profile all.")
    return names


def _print_ffmpeg_info(ffmpeg_bin: str | None = None) -> None:
    info = detect_ffmpeg_environment(ffmpeg_bin)
    print(f"FFmpeg: {info['ffmpeg']}")

    gpus = list(info["gpus"])
    if gpus:
        print("GPU:")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("GPU: not detected")

    hwaccels = list(info["hwaccels"])
    print("FFmpeg hwaccels: " + (", ".join(hwaccels) if hwaccels else "none detected"))

    encoders = list(info["video_encoders"])
    print("FFmpeg H.264 encoders: " + (", ".join(encoders) if encoders else "none detected"))

    auto_order = list(info["auto_reencode_order"])
    print("Auto re-encode order: " + (" > ".join(auto_order) if auto_order else "none"))
    print("Recommended for fastest lossless-quality clipping: --video-codec copy")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init-config":
        content = _generate_config_yaml()
        out_path = Path(args.out)
        out_path.write_text(content, encoding="utf-8")
        print(f"Wrote config from {_config_example_path()}: {out_path}")
        return 0

    if args.command == "ffmpeg-info":
        _print_ffmpeg_info(args.ffmpeg)
        return 0

    if args.command == "cut-copy":
        from .cut_copy import run_cut_copy
        return run_cut_copy(args.conf, dry_run=args.dry_run, no_shutdown=args.no_shutdown)

    if args.command == "run":
        from .pipeline import run_pipeline

        try:
            profile_names = _resolve_profile_names(args.config, args.profile)
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1

        output_dir = Path(args.out) if args.out else (
            Path(args.out_root) / Path(args.video).stem
        )
        failures: list[tuple[str, Exception]] = []
        last_total = 0

        for profile_name in profile_names:
            label = profile_name or "default"
            try:
                config = load_config(args.config, profile=profile_name)
                _apply_run_overrides(config, args)
                if not _has_api_key(config):
                    raise RuntimeError(
                        "LLM API key required. Set in config, environment variable, or --llm-api-key"
                    )
                if len(profile_names) > 1:
                    print(f"\n[profile] Running {label}...")
                results = run_pipeline(
                    Path(args.video),
                    output_dir,
                    config,
                    config_path=args.config,
                )
                last_total = sum(len(v) for v in results.values())
            except Exception as exc:
                failures.append((label, exc))
                print(f"[error] Profile {label} failed: {exc}")

        if failures:
            print("\nFailed profiles:")
            for label, exc in failures:
                print(f"  - {label}: {exc}")
            return 1

        print(f"\nDone! Found {last_total} clips in: {output_dir}")
        return 0

    if args.command == "batch-run":
        from .batch import run_batch

        try:
            profile_names = _resolve_profile_names(args.config, args.profile)
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1

        failures: list[tuple[str, Exception]] = []
        extensions = None
        if args.extensions:
            extensions = {item.strip() for item in args.extensions.split(",") if item.strip()}

        all_runs: list[dict] = []
        for profile_name in profile_names:
            label = profile_name or "default"
            try:
                config = load_config(args.config, profile=profile_name)
                _apply_output_overrides(config, args)
                if args.concat:
                    config["output"]["concat_videos"] = True
                if not _has_api_key(config):
                    raise RuntimeError(
                        "LLM API key required. Set in config, environment variable, or --llm-api-key"
                    )
                if len(profile_names) > 1:
                    print(f"\n[profile] Running batch for {label}...")
                runs = run_batch(
                    args.input_root,
                    args.result_root,
                    args.work_root,
                    config,
                    marker_name=args.marker,
                    extensions=extensions,
                    config_path=args.config,
                )
                all_runs.extend(runs)
                print(f"Profile {label}: {len(runs)} run records.")
            except Exception as exc:
                failures.append((label, exc))
                print(f"[error] Profile {label} failed: {exc}")

        if failures:
            print("\nFailed profiles:")
            for label, exc in failures:
                print(f"  - {label}: {exc}")
            return 1

        print("\nDone! Batch run completed.")

        # Cut-copy post-processing
        cut_copy_conf = args.cut_copy_conf
        # Auto-detect from config if not specified via CLI
        if not cut_copy_conf:
            try:
                main_config = load_config(args.config)
                cc_cfg = main_config.get("cut_copy", {})
                if cc_cfg.get("enabled", False):
                    cut_copy_conf = cc_cfg.get("conf_path", "cut_copy.conf")
                    # Resolve relative path against config file directory
                    config_dir = Path(args.config).parent
                    cut_copy_conf = str(config_dir / cut_copy_conf)
                    print(f"[cut-copy] Auto-detected from config: {cut_copy_conf}")
            except Exception:
                pass  # Ignore config loading errors here

        if cut_copy_conf:
            from .cut_copy import load_cut_copy_config, run_batch_cut_copy
            try:
                cc_config = load_cut_copy_config(cut_copy_conf)
                rc = run_batch_cut_copy(cc_config, all_runs, no_shutdown=False)
                if rc != 0:
                    print("[error] Cut-copy post-processing failed.")
                    return rc
            except Exception as exc:
                print(f"[error] Cut-copy post-processing failed: {exc}")
                return 1

        return 0

    if args.command == "manual-cut":
        from .manual import manual_cut

        config = load_config(args.config)
        _apply_output_overrides(config, args)
        results = manual_cut(
            args.run_dir,
            config,
            csv_path=args.csv,
            input_video=args.video,
            output_dir=args.out,
            content_type=args.content_type,
            config_path=args.config,
        )
        print(f"\nDone! Manual cut produced {len(results)} clips.")
        return 0

    if args.command == "post-merge":
        from .post_merge import PostMergeError, post_merge_from_context

        try:
            result = post_merge_from_context(args.context, args.file1, args.file2)
        except PostMergeError as exc:
            print(f"Error: {exc}")
            return 1
        print("Post-merge recut complete:")
        print(f"  Output: {result['output_path']}")
        print(f"  Range: {result['start']:.3f}s - {result['end']:.3f}s")
        return 0

    if args.command == "manual-cut-context":
        from .manual_cut_context import manual_cut_from_context, ManualCutContextError

        try:
            result = manual_cut_from_context(args.context, args.start, args.end, args.filename)
        except ManualCutContextError as exc:
            print(f"Error: {exc}")
            return 1
        print("Manual cut complete:")
        print(f"  Output: {result['output_path']}")
        print(f"  Range: {result['start']} - {result['end']}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
