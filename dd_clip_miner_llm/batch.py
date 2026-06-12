from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .ffmpeg import concat_videos
from .paths import VIDEO_EXTENSIONS, iter_video_files, safe_path_part
from .pipeline import run_pipeline


def _is_generated_concat_output(path: Path) -> bool:
    stem = path.stem.lower()
    return stem == "merged_output" or stem.startswith("merged_")


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

    videos = [
        video for video in iter_video_files(root, extensions or VIDEO_EXTENSIONS)
        if not _is_generated_concat_output(video)
    ]
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
    profile_name = str(config.get("_profile_name") or "")
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
                marker_key = _profile_marker_key(video_key, profile_name)

                if marker_key in completed_videos and completed_videos[marker_key].get("status") == "success":
                    print(f"[skip] Already processed: {video}")
                    folder_runs.append(completed_videos[marker_key])
                    runs.append(completed_videos[marker_key])
                    continue

                has_work = True
                rel_folder = _relative_folder(root, folder)
                run_name = safe_path_part(video.stem)
                run_dir = work / rel_folder / run_name
                result_dir = results_root / rel_folder / run_name
                if profile_name:
                    run_dir = result_dir

                print(f"[run] {video}")
                try:
                    results = run_pipeline(video, run_dir, config, config_path=config_path)
                    if run_dir.resolve() != result_dir.resolve():
                        shutil.copytree(run_dir, result_dir, dirs_exist_ok=True)
                        shutil.rmtree(run_dir, ignore_errors=True)
                        print(f"  [cleanup] Removed work dir: {run_dir}")
                    total_count = sum(len(v) for v in results.values()) if isinstance(results, dict) else len(results)
                    item = {
                        "video": str(video),
                        "video_key": video_key,
                        "profile": profile_name or None,
                        "work_dir": str(result_dir),  # 指向实际存在的目录
                        "result_dir": str(result_dir),
                        "song_count": total_count,
                        "content_counts": {k: len(v) for k, v in results.items()} if isinstance(results, dict) else {"song": len(results)},
                        "status": "success",
                    }
                    folder_runs.append(item)
                    runs.append(item)
                    completed_videos[marker_key] = item
                except Exception as exc:
                    folder_ok = False
                    item = {
                        "video": str(video),
                        "video_key": video_key,
                        "profile": profile_name or None,
                        "work_dir": str(run_dir),
                        "result_dir": str(result_dir),
                        "error": str(exc),
                        "status": "failed",
                    }
                    folder_runs.append(item)
                    runs.append(item)
                    completed_videos[marker_key] = item
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
    profile_name = str(config.get("_profile_name") or "")
    marker_key = _profile_marker_key(folder_key, profile_name)
    if marker_key in completed_videos and completed_videos[marker_key].get("status") == "success":
        print(f"[skip] Already processed concat folder: {folder}")
        folder_runs.append(completed_videos[marker_key])
        return folder_runs, True, False
    
    has_work = True
    rel_folder = _relative_folder(root, folder)
    run_name = safe_path_part(folder.name) + "_concat"
    run_dir = work / rel_folder / run_name
    result_dir = results_root / rel_folder / run_name
    if profile_name:
        run_dir = result_dir
    concat_dir = run_dir / "concat"
    
    # 基于目录状态检测是否已完成（不依赖 marker 文件编码）
    concat_output = concat_dir / "concat.mp4"
    progress_path = run_dir / "progress.json"
    skip_reason = (
        None
        if profile_name
        else _check_concat_already_done(concat_output, progress_path)
    )
    if skip_reason:
        print(f"[skip] {skip_reason}: {folder}")
        # 尝试加载已有结果
        existing = _load_existing_concat_result(folder, folder_key, run_dir, result_dir, len(folder_videos))
        if existing:
            folder_runs.append(existing)
            completed_videos[marker_key] = existing
            _write_marker(marker, completed_videos)
            return folder_runs, True, False
    
    concat_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[concat] {len(folder_videos)} videos in {folder}")
    try:
        # 拼接视频
        if concat_output.exists():
            print(f"[concat] Reusing existing concatenated video: {concat_output}")
        else:
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

        # 复制结果
        if run_dir.resolve() != result_dir.resolve():
            shutil.copytree(run_dir, result_dir, dirs_exist_ok=True)
            # 清理合并前的临时文件（复制成功后再清理，避免复制失败时丢失数据）
            _cleanup_concat_source(concat_dir)
            shutil.rmtree(run_dir, ignore_errors=True)
            print(f"  [cleanup] Removed work dir: {run_dir}")

        total_count = sum(len(v) for v in results.values()) if isinstance(results, dict) else len(results)
        item = {
            "video": str(folder),
            "video_key": folder_key,
            "profile": profile_name or None,
            "work_dir": str(run_dir),
            "result_dir": str(result_dir),
            "video_count": len(folder_videos),
            "song_count": total_count,
            "content_counts": {k: len(v) for k, v in results.items()} if isinstance(results, dict) else {"song": len(results)},
            "status": "success",
        }
        folder_runs.append(item)
        completed_videos[marker_key] = item
    except Exception as exc:
        folder_ok = False
        item = {
            "video": str(folder),
            "video_key": folder_key,
            "profile": profile_name or None,
            "work_dir": str(run_dir),
            "result_dir": str(result_dir),
            "video_count": len(folder_videos),
            "error": str(exc),
            "status": "failed",
        }
        folder_runs.append(item)
        completed_videos[marker_key] = item
        print(f"[error] {folder}: {exc}")
    
    _write_marker(marker, completed_videos)
    return folder_runs, folder_ok, has_work


def _check_concat_already_done(concat_output: Path, progress_path: Path) -> str | None:
    """检查 concat 是否已完成，返回跳过原因或 None"""
    if not concat_output.exists():
        return None
    
    # 检查 progress.json
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            last_step = progress.get("last_completed_step", "")
            if last_step == "done":
                return "Pipeline fully completed"
            if last_step in ("audio", "asr", "llm", "export"):
                return f"Pipeline partially done (step: {last_step}), concat.mp4 exists"
        except (json.JSONDecodeError, OSError):
            pass
    
    # concat.mp4 存在但没有 progress.json，可能是之前的运行
    # 检查文件大小是否合理（> 100MB）
    if concat_output.stat().st_size > 100 * 1024 * 1024:
        return "concat.mp4 exists (>100MB)"
    
    return None


def _load_existing_concat_result(
    folder: Path,
    folder_key: str,
    run_dir: Path,
    result_dir: Path,
    video_count: int,
) -> dict[str, Any] | None:
    """加载已有的 concat 结果"""
    progress_path = run_dir / "progress.json"
    last_step = ""
    content_counts: dict[str, int] = {}
    
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            last_step = progress.get("last_completed_step", "")
        except (json.JSONDecodeError, OSError):
            pass
    
    # 尝试从 manifest.json 获取更多信息
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            content_counts = manifest.get("content_types", {})
        except (json.JSONDecodeError, OSError):
            pass
    
    total_count = sum(content_counts.values()) if content_counts else 0
    
    return {
        "video": str(folder),
        "video_key": folder_key,
        "work_dir": str(run_dir),
        "result_dir": str(result_dir),
        "video_count": video_count,
        "song_count": total_count,
        "content_counts": content_counts,
        "status": "success",
        "last_step": last_step,
    }


def _cleanup_concat_source(concat_dir: Path) -> None:
    """清理 concat 中间文件，保留最终 concat.mp4 供 resume / manual-cut 使用。"""
    preserve = {"concat.mp4"}
    try:
        for path in concat_dir.iterdir():
            if path.is_file() and path.name not in preserve:
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


def _profile_marker_key(video_key: str, profile_name: str) -> str:
    if not profile_name:
        return video_key
    return f"{video_key}|profile:{profile_name}"


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
