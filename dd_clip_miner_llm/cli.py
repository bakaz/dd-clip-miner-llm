from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config
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
    batch_parser.add_argument("--content-types", default=None, help="要识别的内容类型，逗号分隔")
    batch_parser.add_argument("--marker", default=".dd_clip_miner_done.json", help="完成标记文件")
    batch_parser.add_argument("--extensions", default=None, help="视频扩展名，逗号分隔")
    batch_parser.add_argument("--concat", action="store_true", help="合并目录下的多个视频后再处理")
    batch_parser.add_argument("--video-codec", default=None, help="视频编码器")
    batch_parser.add_argument("--audio-bitrate-kbps", type=int, default=None, help="音频码率")

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

    # init-config 命令
    init_parser = subparsers.add_parser("init-config", help="生成默认配置文件")
    init_parser.add_argument("--out", default="config.yaml", help="输出路径")

    # ffmpeg-info 命令
    info_parser = subparsers.add_parser("ffmpeg-info", help="显示 GPU 和 FFmpeg 编码器信息")
    info_parser.add_argument("--ffmpeg", default=None, help="FFmpeg 路径")

    return parser


def _generate_config_yaml() -> str:
    lines = [
        "# dd-clip-miner-llm 配置文件",
        "",
        "# 音频预处理",
        "audio:",
        "  sample_rate: 16000",
        "  channels: 1",
        "",
        "# ASR 配置",
        "asr:",
        "  model: small",
        "  device: auto",
        "  compute_type: default",
        "  language: null",
        "  beam_size: 5",
        "  vad_filter: true",
        "  initial_prompt: null",
        "",
        "# LLM 配置",
        "llm:",
        "  api_key: null",
        "  api_key_env: LLM_API_KEY",
        "  base_url: null",
        "  model: gpt-4o",
        "  temperature: 0.1",
        "  max_tokens: 8192",
        "  max_completion_tokens: null",
        "  retry_empty_with_reasoning: true",
        "  reasoning_followup_rounds: 5",
        "  reasoning_followup_max_tokens: 32768",
        "  batch_size: null",
        "  use_tools: true",
        "  verify_with_search: true",
        "  json_fix_rounds: 3",
        "  fallbacks: []",
        "",
        "# 时间 padding（兼容旧项目配置）",
        "padding:",
        "  before_seconds: 15.0",
        "  after_seconds: 15.0",
        "  after_next_asr_end_guard_seconds: 2.0",
        "  min_song_seconds: 75.0",
        "  merge_gap_seconds: 20.0",
        "",
        "# 要识别的内容类型（true/false 控制启用/禁用）",
        "content_types:",
        "  song: true",
        "  dialogue: true",
        "  highlight: true",
        "  funny: true",
        "  cringe: true",
        "  daily_summary: false",
        "",
        "# 歌曲识别配置",
        "song:",
        "  enabled: true",
        "  padding:",
        "    before_seconds: 15.0",
        "    after_seconds: 15.0",
        "    after_next_asr_end_guard_seconds: 2.0",
        "    min_song_seconds: 75.0",
        "    merge_gap_seconds: 20.0",
        "  missed_recheck:",
        "    enabled: true",
        "    batch_size: 500",
        "    min_gap_segments: 1",
        "",
        "# 对话识别配置",
        "dialogue:",
        "  enabled: true",
        "  min_duration: 10.0",
        "  max_duration: 300.0",
        "  min_confidence: 0.6",
        "  merge_gap_seconds: 10.0",
        "  tags:",
        "    - 搞笑",
        "    - 吐槽",
        "    - 名场面",
        "    - 金句",
        "    - 互动",
        "    - 高能",
        "",
        "# 高能时刻配置",
        "highlight:",
        "  enabled: true",
        "  min_duration: 5.0",
        "  max_duration: 120.0",
        "  min_confidence: 0.6",
        "  merge_gap_seconds: 15.0",
        "",
        "# 搞笑片段配置",
        "funny:",
        "  enabled: true",
        "  min_duration: 5.0",
        "  max_duration: 180.0",
        "  min_confidence: 0.6",
        "  merge_gap_seconds: 15.0",
        "",
        "# 下头对话配置",
        "cringe:",
        "  enabled: true",
        "  min_duration: 5.0",
        "  max_duration: 120.0",
        "  min_confidence: 0.6",
        "  merge_gap_seconds: 15.0",
        "",
        "# 当天直播结构化总结配置",
        "daily_summary:",
        "  enabled: false",
        "  summary_only: true",
        "  language: zh-CN",
        "  title: 当天直播内容总结",
        "  max_level1_items: 6",
        "  max_level2_per_level1: 5",
        "  max_level3_per_level2: 4",
        "  include_timeline: true",
        "  include_quotes: true",
        "  include_open_questions: true",
        "",
        "# 输出配置",
        "output:",
        "  video_clips: true",
        "  audio_segments: true",
        "  audio_extension: mp3",
        "  audio_bitrate_kbps: 320",
        "  video_extension: mp4",
        "  video_codec: copy",
        "  match_context_segments: 10",
        "  concat_videos: false  # 合并目录下的多个视频后再处理",
    ]
    return "\n".join(lines) + "\n"


def _apply_run_overrides(config: dict, args: argparse.Namespace) -> None:
    if args.content_types:
        # 将逗号分隔的列表转换为字典格式
        types_list = [ct.strip() for ct in args.content_types.split(",") if ct.strip()]
        config["content_types"] = {ct: True for ct in types_list}
    if args.asr_model:
        config["asr"]["model"] = args.asr_model
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
    api_key = config["llm"].get("api_key")
    api_key_env = config["llm"].get("api_key_env")
    if not api_key and api_key_env:
        import os
        api_key = os.environ.get(str(api_key_env), "")
    return bool(api_key)


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
        print(f"Wrote config: {out_path}")
        return 0

    if args.command == "ffmpeg-info":
        _print_ffmpeg_info(args.ffmpeg)
        return 0

    if args.command == "run":
        from .pipeline import run_pipeline

        config = load_config(args.config)
        _apply_run_overrides(config, args)

        if not _has_api_key(config):
            print("Error: LLM API key required. Set in config or --llm-api-key")
            return 1

        output_dir = Path(args.out) if args.out else (
            Path(args.out_root) / Path(args.video).stem
        )

        results = run_pipeline(Path(args.video), output_dir, config)
        total = sum(len(v) for v in results.values())
        print(f"\nDone! Found {total} clips in: {output_dir}")
        return 0

    if args.command == "batch-run":
        from .batch import run_batch

        config = load_config(args.config)
        _apply_output_overrides(config, args)
        
        # 应用 --concat 参数
        if args.concat:
            config["output"]["concat_videos"] = True
        
        if not _has_api_key(config):
            print("Error: LLM API key required. Set in config or environment")
            return 1
        extensions = None
        if args.extensions:
            extensions = {item.strip() for item in args.extensions.split(",") if item.strip()}
        runs = run_batch(
            args.input_root,
            args.result_root,
            args.work_root,
            config,
            marker_name=args.marker,
            extensions=extensions,
        )
        print(f"\nDone! Batch produced {len(runs)} run records.")
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
        )
        print(f"\nDone! Manual cut produced {len(results)} clips.")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
