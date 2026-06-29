from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from dd_clip_miner_llm.cleanup_context import CleanupContextError, cleanup_from_context


def _write_cleanup_fixture(tmp_path: Path) -> dict[str, Path]:
    run = tmp_path / "demo_fix"
    source = run / "00_input" / "input.mp4"
    video_dir = run / "03_clips" / "video" / "song"
    sus_dir = video_dir / "sus"
    manifest = run / "manifest.json"
    for path in (source.parent, video_dir, sus_dir):
        path.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"input video")
    (sus_dir / "unknown_sus.mp4").write_bytes(b"sus clip")
    manifest.write_text(
        json.dumps({"input_video": "00_input/input.mp4", "total_duration": 90.0}),
        encoding="utf-8",
    )
    context = {
        "run_dir": ".",
        "content_type": "song",
        "manifest_path": "manifest.json",
        "input_video": "00_input/input.mp4",
        "total_duration": 90.0,
        "config": {"output": {"video_codec": "copy"}},
    }
    context_path = video_dir / "merge_recut_context.json"
    context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
    return {
        "run": run,
        "source": source,
        "context": context_path,
        "sus_dir": sus_dir,
    }


def test_cleanup_deletes_input_video_and_sus(tmp_path):
    fixture = _write_cleanup_fixture(tmp_path)

    result = cleanup_from_context(fixture["context"], yes=True)

    assert not fixture["source"].exists()
    assert not fixture["sus_dir"].exists()
    assert str(fixture["source"]) in result["deleted_files"]
    assert str(fixture["sus_dir"]) in result["deleted_dirs"]


def test_cleanup_deletes_concat_video_when_present(tmp_path):
    fixture = _write_cleanup_fixture(tmp_path)
    concat_video = fixture["run"] / "concat" / "concat.mp4"
    concat_video.parent.mkdir(parents=True, exist_ok=True)
    concat_video.write_bytes(b"concat video")

    result = cleanup_from_context(fixture["context"], yes=True)

    assert not fixture["source"].exists()
    assert not concat_video.exists()
    assert str(concat_video) in result["deleted_files"]


def test_cleanup_prefers_run_local_copy_over_stale_absolute_path(tmp_path):
    fixture = _write_cleanup_fixture(tmp_path)
    stale_run = tmp_path / "stale_machine" / "results" / fixture["run"].name
    stale_source = stale_run / "00_input" / "input.mp4"
    stale_source.parent.mkdir(parents=True, exist_ok=True)
    stale_source.write_bytes(b"remote original")

    context = json.loads(fixture["context"].read_text(encoding="utf-8"))
    context["run_dir"] = str(stale_run)
    context["input_video"] = str(stale_source)
    context["manifest_path"] = str(stale_run / "manifest.json")
    fixture["context"].write_text(json.dumps(context, indent=2), encoding="utf-8")

    result = cleanup_from_context(fixture["context"], yes=True)

    assert not fixture["source"].exists()
    assert stale_source.exists()
    assert str(fixture["source"]) in result["deleted_files"]


def test_cleanup_skips_source_outside_run_but_deletes_sus(tmp_path):
    fixture = _write_cleanup_fixture(tmp_path)
    external_source = tmp_path / "nas_recordings" / "live.mp4"
    external_source.parent.mkdir(parents=True, exist_ok=True)
    external_source.write_bytes(b"nas original")
    fixture["source"].unlink()

    context = json.loads(fixture["context"].read_text(encoding="utf-8"))
    context["input_video"] = str(external_source)
    context["manifest_path"] = str(fixture["run"] / "missing_manifest.json")
    fixture["context"].write_text(json.dumps(context, indent=2), encoding="utf-8")

    result = cleanup_from_context(fixture["context"], yes=True)

    assert external_source.exists()
    assert not fixture["sus_dir"].exists()
    assert "source_video" in result["skipped"]
    assert any("outside run root" in warning for warning in result["warnings"])


def test_cleanup_dry_run_keeps_files(tmp_path):
    fixture = _write_cleanup_fixture(tmp_path)

    result = cleanup_from_context(fixture["context"], dry_run=True, yes=True)

    assert fixture["source"].exists()
    assert fixture["sus_dir"].exists()
    assert result["dry_run"] is True
    assert str(fixture["source"]) in result["deleted_files"]
    assert str(fixture["sus_dir"]) in result["deleted_dirs"]


def test_cleanup_cancelled_without_yes(tmp_path):
    fixture = _write_cleanup_fixture(tmp_path)

    with pytest.raises(CleanupContextError, match="cancelled"):
        cleanup_from_context(
            fixture["context"],
            input_stream=StringIO("n\n"),
            output_stream=StringIO(),
        )

    assert fixture["source"].exists()
    assert fixture["sus_dir"].exists()


def test_export_writes_cleanup_source_bat(tmp_path, monkeypatch):
    from dd_clip_miner_llm.models import ContentResult
    from dd_clip_miner_llm.pipeline import export as export_module

    run_dir = tmp_path / "run"
    clips_dir = run_dir / "03_clips"
    llm_dir = run_dir / "02_asr" / "llm" / "song"
    reports_dir = run_dir / "04_reports" / "song"
    transcript_path = run_dir / "02_asr" / "transcript.json"
    manifest_path = run_dir / "manifest.json"
    result = ContentResult(
        index=1,
        content_type="song",
        title="Song",
        start=1.0,
        end=5.0,
        duration=4.0,
        transcript="lyrics",
        confidence=0.9,
    )

    input_path = run_dir / "00_input" / "input.mp4"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"input")

    def fake_cut_audio(_input, target, *_args, **_kwargs):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(b"audio")

    def fake_cut_video(_input, target, *_args, **_kwargs):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(b"video")

    monkeypatch.setattr(export_module, "cut_audio", fake_cut_audio)
    monkeypatch.setattr(export_module, "cut_video", fake_cut_video)

    export_module._export_results(
        [result],
        input_path,
        clips_dir,
        {"output": {"video_codec": "copy", "audio_bitrate_kbps": 320}},
        "song",
        run_dir=run_dir,
        llm_dir=llm_dir,
        reports_dir=reports_dir,
        transcript_path=transcript_path,
        manifest_path=manifest_path,
        total_duration=10.0,
    )

    for target_dir in (clips_dir / "audio" / "song", clips_dir / "video" / "song"):
        assert (target_dir / "cleanup_source.bat").is_file()
        bat_content = (target_dir / "cleanup_source.bat").read_text(encoding="utf-8")
        assert 'call "%~dp0_resolve_env.bat"' in bat_content
        assert "cleanup-source" in bat_content