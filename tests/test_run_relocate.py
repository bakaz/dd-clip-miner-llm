"""Tests for run artifact path relocation after cut_copy."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dd_clip_miner_llm.run_relocate import relocate_run_artifacts


def _write_local_run(local_root: Path) -> dict[str, Path]:
    """Create a minimal run tree with absolute local-machine paths in JSON."""
    run = local_root / "11_58_41_streamer_fix"
    source = run / "00_input" / "input_ca438f064eea.mp4"
    transcript_path = run / "02_asr" / "transcript.json"
    matches_path = run / "02_asr" / "llm" / "song" / "matches.json"
    reports_path = run / "04_reports" / "song" / "songs.json"
    video_dir = run / "03_clips" / "video" / "song"
    audio_dir = run / "03_clips" / "audio" / "song"
    manifest_path = run / "manifest.json"

    for path in (
        source.parent,
        transcript_path.parent,
        matches_path.parent,
        reports_path.parent,
        video_dir,
        audio_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    source.write_bytes(b"fake video")
    transcript_path.write_text("[]", encoding="utf-8")
    matches_path.write_text("[]", encoding="utf-8")
    (video_dir / "clip1.mp4").write_bytes(b"clip")
    (audio_dir / "clip1.mp3").write_bytes(b"clip")
    reports_path.write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "title": "Song A",
                    "video_path": str(video_dir / "clip1.mp4"),
                    "audio_path": str(audio_dir / "clip1.mp3"),
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps({"input_video": str(source), "total_duration": 3600.0}, indent=2),
        encoding="utf-8",
    )

    context = {
        "run_dir": str(run),
        "content_type": "song",
        "manifest_path": str(manifest_path),
        "reports_path": str(reports_path),
        "llm_dir": str(matches_path.parent),
        "matches_path": str(matches_path),
        "transcript_path": str(transcript_path),
        "input_video": str(source),
        "total_duration": 3600.0,
        "python_executable": r"D:\missing\python.exe",
        "project_root": r"D:\missing\dd-clip-miner-llm",
        "config": {"output": {"video_codec": "copy"}},
    }
    (video_dir / "merge_recut_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (video_dir / "manual_cut_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "run": run,
        "source": source,
        "manifest": manifest_path,
        "reports": reports_path,
        "video_dir": video_dir,
        "merge_context": video_dir / "merge_recut_context.json",
        "manual_context": video_dir / "manual_cut_context.json",
    }


def test_relocate_run_artifacts_rewrites_paths_to_destination(tmp_path):
    local = tmp_path / "local" / "runs" / "batch" / "2026_06_24"
    nas = tmp_path / "nas" / "260624_2026_06_24"

    fixture = _write_local_run(local)
    dest = nas / fixture["run"].name
    shutil.copytree(fixture["run"], dest)

    relocate_run_artifacts(dest)

    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert Path(manifest["input_video"]) == Path("00_input/input_ca438f064eea.mp4")

    songs = json.loads((dest / "04_reports" / "song" / "songs.json").read_text(encoding="utf-8"))
    assert Path(songs[0]["video_path"]) == Path("03_clips/video/song/clip1.mp4")
    assert Path(songs[0]["audio_path"]) == Path("03_clips/audio/song/clip1.mp3")

    for context_name in ("merge_recut_context.json", "manual_cut_context.json"):
        context = json.loads((dest / "03_clips" / "video" / "song" / context_name).read_text(encoding="utf-8"))
        assert context["run_dir"] == "."
        assert Path(context["input_video"]) == Path("00_input/input_ca438f064eea.mp4")
        assert Path(context["manifest_path"]) == Path("manifest.json")
        assert Path(context["reports_path"]) == Path("04_reports/song/songs.json")
        assert "python_executable" not in context
        assert "project_root" not in context


def test_relocate_run_artifacts_missing_run_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run root not found"):
        relocate_run_artifacts(tmp_path / "missing")