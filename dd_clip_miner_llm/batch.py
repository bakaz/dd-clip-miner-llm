from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .ffmpeg import concat_videos
from .paths import VIDEO_EXTENSIONS, iter_video_files, safe_path_part
from .pipeline import run_pipeline


def run_batch(
    input_root: str | Path,
    result_root: str | Path,
    work_root: str | Path,
    config: dict[str, Any],
    marker_name: str = ".dd_clip_miner_done.json",
    extensions: set[str] | None = None,
    config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = Path(input_root).expanduser()
    results_root = Path(result_root)
    work = Path(work_root)
    results_root.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    videos = iter_video_files(root, extensions or VIDEO_EXTENSIONS)
    by_folder: dict[Path, list[Path]] = {}
    for video in videos:
        by_folder.setdefault(video.parent, []).append(video)

    # 检查是否启用视频拼接
    concat_enabled = config.get("output", {}).get("concat_videos", False)
    video_codec = config.get("output", {}).get("video_codec", "copy")
    audio_bitrate = int(config.get("output", {}).get("audio_bitrate_kbps", 320))
    single_file_policy = str(config.get("output", {}).get("single_file_policy", "copy"))
    force_normalize = bool(config.get("output", {}).get("concat_force_normalize", False))

    runs: list[dict[str, Any]] = []
    for folder in sorted(by_folder):
        marker = folder / marker_name
        completed_videos = _load_marker(marker)
        folder_videos = sorted(by_folder[folder])

        folder_runs: list[dict[str, Any]] = []
        folder_ok = True
        has_work = False

        # 如果启用拼接且有多个视频，先拼接
        if concat_enabled and (len(folder_videos) > 1 or single_file_policy != "copy"):
            folder_runs, folder_ok, has_work = _process_folder_concat(
                folder, folder_videos, completed_videos, marker,
                root, results_root, work, config,
                video_codec, audio_bitrate, marker_name,
                single_file_policy=single_file_policy,
                force_normalize=force_normalize,
                config_path=config_path,
            )
            runs.extend(folder_runs)
        else:
            # 正常逐个处理
            for video in folder_videos:
                video_key = str(video.resolve())

                if video_key in completed_videos and completed_videos[video_key].get("status") == "success":
                    print(f"[skip] Already processed: {video}")
                    folder_runs.append(completed_videos[video_key])
                    runs.append(completed_videos[video_key])
                    continue

                has_work = True
                rel_folder = _relative_folder(root, folder)
                run_name = safe_path_part(video.stem)
                run_dir = work / rel_folder / run_name
                result_dir = results_root / rel_folder / run_name

                print(f"[run] {video}")
                try:
                    results = run_pipeline(video, run_dir, config, config_path=config_path)
                    if run_dir.resolve() != result_dir.resolve():
                        shutil.copytree(run_dir, result_dir, dirs_exist_ok=True)
                    total_count = sum(len(v) for v in results.values()) if isinstance(results, dict) else len(results)
                    item = {
                        "video": str(video),
                        "video_key": video_key,
                        "work_dir": str(run_dir),
                        "result_dir": str(result_dir),
                        "song_count": total_count,
                        "content_counts": {k: len(v) for k, v in results.items()} if isinstance(results, dict) else {"song": len(results)},
                        "status": "success",
                    }
                    folder_runs.append(item)
                    runs.append(item)
                    completed_videos[video_key] = item
                except Exception as exc:
                    folder_ok = False
                    item = {
                        "video": str(video),
                        "video_key": video_key,
                        "work_dir": str(run_dir),
                        "result_dir": str(result_dir),
                        "error": str(exc),
                        "status": "failed",
                    }
                    folder_runs.append(item)
                    runs.append(item)
                    completed_videos[video_key] = item
                    print(f"[error] {video}: {exc}")

                _write_marker(marker, completed_videos)

        if has_work and folder_ok:
            print(f"[done] All videos processed in: {folder}")
        elif has_work:
            print(f"[warn] Some videos failed in: {folder}, will retry on next run")

    return runs


def _process_folder_concat(
    folder: Path,
    folder_videos: list[Path],
    completed_videos: dict[str, dict[str, Any]],
    marker: Path,
    root: Path,
    results_root: Path,
    work: Path,
    config: dict[str, Any],
    video_codec: str,
    audio_bitrate: int,
    marker_name: str,
    *,
    single_file_policy: str = "copy",
    force_normalize: bool = False,
    config_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], bool, bool]:
    """处理文件夹内的视频拼接"""
    folder_runs = []
    folder_ok = True
    has_work = False
    
    # 检查是否已经处理过这个文件夹（使用第一个视频作为 key）
    folder_key = f"concat:{','.join(str(v.resolve()) for v in folder_videos)}"
    if folder_key in completed_videos and completed_videos[folder_key].get("status") == "success":
        print(f"[skip] Already processed concat folder: {folder}")
        folder_runs.append(completed_videos[folder_key])
        return folder_runs, True, False
    
    has_work = True
    rel_folder = _relative_folder(root, folder)
    run_name = safe_path_part(folder.name) + "_concat"
    run_dir = work / rel_folder / run_name
    result_dir = results_root / rel_folder / run_name
    concat_dir = run_dir / "concat"
    concat_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[concat] {len(folder_videos)} videos in {folder}")
    try:
        # 拼接视频
        concat_output = concat_dir / "concat.mp4"
        concat_videos(
            folder_videos,
            concat_output,
            video_codec=video_codec,
            audio_bitrate_kbps=audio_bitrate,
            single_file_policy=single_file_policy,
            force_normalize=force_normalize,
        )

        # 处理拼接后的视频
        print(f"[run] Processing concatenated video...")
        results = run_pipeline(concat_output, run_dir, config, config_path=config_path)

        # 清理合并前的临时文件（保留拼接后的视频）
        _cleanup_concat_source(concat_dir)

        # 复制结果
        if run_dir.resolve() != result_dir.resolve():
            shutil.copytree(run_dir, result_dir, dirs_exist_ok=True)

        total_count = sum(len(v) for v in results.values()) if isinstance(results, dict) else len(results)
        item = {
            "video": str(folder),
            "video_key": folder_key,
            "work_dir": str(run_dir),
            "result_dir": str(result_dir),
            "video_count": len(folder_videos),
            "song_count": total_count,
            "content_counts": {k: len(v) for k, v in results.items()} if isinstance(results, dict) else {"song": len(results)},
            "status": "success",
        }
        folder_runs.append(item)
        completed_videos[folder_key] = item
    except Exception as exc:
        folder_ok = False
        item = {
            "video": str(folder),
            "video_key": folder_key,
            "work_dir": str(run_dir),
            "result_dir": str(result_dir),
            "video_count": len(folder_videos),
            "error": str(exc),
            "status": "failed",
        }
        folder_runs.append(item)
        completed_videos[folder_key] = item
        print(f"[error] {folder}: {exc}")
    
    _write_marker(marker, completed_videos)
    return folder_runs, folder_ok, has_work


def _cleanup_concat_source(concat_dir: Path) -> None:
    """清理 concat 中间文件；最终输入保留在 pipeline 的 00_input 目录。
    （新 ConcatPipeline 下，完整 attempt 日志会留在 concat_attempts/ 供调试。）
    """
    try:
        for path in concat_dir.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)
    except OSError:
        pass


def _relative_folder(root: Path, folder: Path) -> Path:
    try:
        rel = folder.relative_to(root)
    except ValueError:
        rel = Path(safe_path_part(folder.name))
    if str(rel) == ".":
        return Path("_root")
    return Path(*[safe_path_part(part) for part in rel.parts])


def _load_marker(marker: Path) -> dict[str, dict[str, Any]]:
    if not marker.exists():
        return {}
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "videos" in data:
            return data["videos"]
        return {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_marker(marker: Path, completed_videos: dict[str, dict[str, Any]]) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "videos": completed_videos,
    }
    marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
