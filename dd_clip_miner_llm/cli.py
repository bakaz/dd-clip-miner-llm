from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    DEFAULT_CONFIG,
    PROFILE_ALL,
    _load_yaml_with_includes,
    list_profile_names,
    load_config,
)
from .ffmpeg import detect_ffmpeg_environment

_CUT_COPY_CONF_FROM_CONFIG = object()


def _cut_copy_conf_path_from_config(config_path: str | Path | None) -> str | None:
    if not config_path:
        return None
    main_config = load_config(config_path)
    cc_cfg = main_config.get("cut_copy", {})
    conf_path = cc_cfg.get("conf_path", "cut_copy.conf")
    return str(Path(config_path).resolve().parent / conf_path)


def resolve_batch_cut_copy_conf(
    config_path: str | Path | None,
    cut_copy_conf_arg: object,
) -> str | None:
    """Resolve cut_copy.conf for batch-run post-processing."""
    if cut_copy_conf_arg is _CUT_COPY_CONF_FROM_CONFIG:
        return _cut_copy_conf_path_from_config(config_path)

    if cut_copy_conf_arg is not None:
        return str(cut_copy_conf_arg)

    if not config_path:
        return None
    main_config = load_config(config_path)
    cc_cfg = main_config.get("cut_copy", {})
    if not cc_cfg.get("enabled", False):
        return None
    return _cut_copy_conf_path_from_config(config_path)


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
    batch_parser.add_argument(
        "--cut-copy-conf",
        nargs="?",
        const=_CUT_COPY_CONF_FROM_CONFIG,
        default=None,
        help="batch-run 完成后执行 cut_copy；省略路径时从 config 的 cut_copy.conf_path 读取",
    )

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

    # post-merge 命令：拖拽两个及以上已导出片段后，从原始输入重新切为一个片段
    post_merge_parser = subparsers.add_parser("post-merge", help="从两个及以上已导出歌曲片段反查 ASR 并重新切为一个片段")
    post_merge_parser.add_argument("files", nargs="+", help="两个及以上已导出 MP4/MP3")
    post_merge_parser.add_argument("--context", required=True, help="merge_recut_context.json 路径")

    refresh_portable_parser = subparsers.add_parser(
        "refresh-portable",
        help="重装并校验 NAS run 的 _tools/miner 便携包，并同步 song 导出目录中的 bat",
    )
    refresh_portable_parser.add_argument("run_dir", help="运行输出目录（run root）")

    # manual-cut-context 命令：从 context JSON 读取视频路径，手动切片到同目录
    mcc_parser = subparsers.add_parser("manual-cut-context", help="从 context JSON 手动切片到同目录")
    mcc_parser.add_argument("--context", required=True, help="manual_cut_context.json 路径")
    mcc_parser.add_argument("--start", required=True, help="开始时间 (如 10:30 或 630)")
    mcc_parser.add_argument("--end", required=True, help="结束时间 (如 15:45 或 945)")
    mcc_parser.add_argument("--filename", default=None, help="输出文件名（不含扩展名，留空自动生成）")

    cleanup_parser = subparsers.add_parser(
        "cleanup-source",
        help="清理 run 内源视频、concat/concat.mp4 与同级 sus/ 文件夹",
    )
    cleanup_parser.add_argument("--context", required=True, help="merge_recut_context.json 路径")
    cleanup_parser.add_argument("--dry-run", action="store_true", help="只列出将删除的路径，不执行")
    cleanup_parser.add_argument("--yes", action="store_true", help="跳过交互确认")

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

    task_parser = subparsers.add_parser(
        "cut-copy-task",
        help="等待 SMB 路径就绪后执行 batch-run（Windows 计划任务用）",
    )
    task_parser.add_argument("--conf", required=True, help="cut_copy.conf 路径")
    task_parser.add_argument("--project-root", default=".", help="项目根目录")
    task_parser.add_argument("--input-root", default="", help="覆盖 source.path")
    task_parser.add_argument(
        "--network-wait-minutes", type=int, default=45,
        help="等待 SMB 就绪的最长时间（分钟）",
    )
    task_parser.add_argument(
        "--network-poll-seconds", type=int, default=30,
        help="就绪探测间隔（秒）",
    )
    task_parser.add_argument(
        "--log-file", default="cut_copy_task.log",
        help="启动器日志（相对 project-root）",
    )
    task_parser.add_argument(
        "--resolve-json", action="store_true",
        help="输出解析后的路径 JSON 后退出",
    )
    task_parser.add_argument(
        "--resolve-json-file", default="",
        help="将路径 JSON 写入 UTF-8 文件后退出",
    )
    task_parser.add_argument(
        "--probe-json", action="store_true",
        help="探测路径就绪状态并输出 JSON（未就绪时 exit 1）",
    )

    return parser


def _config_example_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "example" / "main.yaml"


def _generate_config_yaml() -> str:
    """Return the shipped config template (config/example/main.yaml)."""
    template = _config_example_path()
    if not template.is_file():
        raise FileNotFoundError(
            f"Config template not found: {template}. "
            "Copy config/example/main.yaml manually if running outside the project tree."
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
    config_path = Path(path)
    raw_text = config_path.read_text(encoding="utf-8")
    if "!include" in raw_text:
        return _load_yaml_with_includes(config_path)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install PyYAML") from exc
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
    loaded = _load_raw_yaml_config(config_path)
    names = list_profile_names(loaded, config_dir=Path(config_path).parent)
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

    if args.command == "cut-copy-task":
        from .cut_copy_task import main as cut_copy_task_main

        return cut_copy_task_main([
            "--conf", args.conf,
            "--project-root", args.project_root,
            "--input-root", args.input_root,
            "--network-wait-minutes", str(args.network_wait_minutes),
            "--network-poll-seconds", str(args.network_poll_seconds),
            "--log-file", args.log_file,
            *([] if not args.resolve_json else ["--resolve-json"]),
            *([] if not args.resolve_json_file else ["--resolve-json-file", args.resolve_json_file]),
            *([] if not args.probe_json else ["--probe-json"]),
        ])

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

        try:
            cut_copy_conf = resolve_batch_cut_copy_conf(args.config, args.cut_copy_conf)
        except Exception as exc:
            print(f"[error] Failed to resolve cut_copy config: {exc}")
            return 1

        if cut_copy_conf:
            print(f"[cut-copy] Using config: {cut_copy_conf}")
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
            result = post_merge_from_context(args.context, *args.files)
        except PostMergeError as exc:
            print(f"Error: {exc}")
            return 1
        print("Post-merge recut complete:")
        print(f"  Output: {result['output_path']}")
        print(f"  Range: {result['start']:.3f}s - {result['end']:.3f}s")
        print(f"  Files merged: {len(args.files)}")
        return 0

    if args.command == "refresh-portable":
        from .portable_bundle import PortableBundleError, refresh_portable_bundle

        try:
            bundle_root = refresh_portable_bundle(args.run_dir)
        except (PortableBundleError, FileNotFoundError, OSError) as exc:
            print(f"Error: {exc}")
            return 1
        print("Portable bundle refreshed:")
        print(f"  Bundle: {bundle_root}")
        print(f"  Run dir: {args.run_dir}")
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

    if args.command == "cleanup-source":
        from .cleanup_context import CleanupContextError, cleanup_from_context

        try:
            result = cleanup_from_context(
                args.context,
                dry_run=args.dry_run,
                yes=args.yes,
            )
        except CleanupContextError as exc:
            print(f"Error: {exc}")
            return 1
        print("Cleanup complete:" if not result.get("dry_run") else "Dry-run cleanup plan:")
        print(f"  Run root: {result.get('run_dir')}")
        for path in result.get("deleted_files", []):
            print(f"  Deleted file: {path}")
        for path in result.get("deleted_dirs", []):
            print(f"  Deleted dir: {path}")
        for item in result.get("skipped", []):
            print(f"  Skipped: {item}")
        for warning in result.get("warnings", []):
            print(f"  Warning: {warning}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
