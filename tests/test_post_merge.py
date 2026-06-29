from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from dd_clip_miner_llm.config import DEFAULT_CONFIG, deep_merge
from dd_clip_miner_llm.merger import build_content_results
from dd_clip_miner_llm.models import ContentMatch, ContentResult, TranscriptSegment


def _post_merge_config() -> dict:
    return deep_merge(DEFAULT_CONFIG, {
        "padding": {
            "before_seconds": 1.0,
            "after_seconds": 1.0,
            "after_next_asr_end_guard_seconds": 0.0,
            "adaptive_silence_padding": False,
            "min_song_seconds": 0.0,
            "max_song_seconds": 360.0,
            "merge_gap_seconds": 5.0,
        },
        "song": {
            "padding": {
                "before_seconds": 1.0,
                "after_seconds": 1.0,
                "after_next_asr_end_guard_seconds": 0.0,
                "adaptive_silence_padding": False,
                "min_song_seconds": 0.0,
                "max_song_seconds": 360.0,
                "merge_gap_seconds": 5.0,
            },
        },
        "output": {
            "video_codec": "copy",
            "audio_bitrate_kbps": 192,
        },
    })


def _write_fixture_run(tmp_path: Path) -> dict:
    run = tmp_path / "run"
    source = run / "00_input" / "input.mp4"
    transcript_path = run / "02_asr" / "transcript.json"
    matches_path = run / "02_asr" / "llm" / "song" / "matches.json"
    reports_path = run / "04_reports" / "song" / "songs.json"
    video_dir = run / "03_clips" / "video" / "song"
    audio_dir = run / "03_clips" / "audio" / "song"
    manifest_path = run / "manifest.json"
    context_path = video_dir / "merge_recut_context.json"

    for path in (source.parent, transcript_path.parent, matches_path.parent, reports_path.parent, video_dir, audio_dir):
        path.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"fake video")

    segments = [
        TranscriptSegment(0.0, 2.0, "intro"),
        TranscriptSegment(10.0, 20.0, "song one"),
        TranscriptSegment(20.0, 22.0, "talk"),
        TranscriptSegment(60.0, 70.0, "song two"),
        TranscriptSegment(80.0, 90.0, "outro"),
    ]
    matches = [
        ContentMatch("song", "Song A", [1], 0.9, artist="Singer"),
        ContentMatch("song", "Song B", [3], 0.8, artist="Singer"),
    ]
    config = _post_merge_config()
    results = build_content_results(segments, matches, 90.0, config, "song")
    assert len(results) == 2
    for result in results:
        result.video_path = video_dir / f"clip{result.index}.mp4"
        result.audio_path = audio_dir / f"clip{result.index}.mp3"
        result.video_path.write_bytes(b"video")
        result.audio_path.write_bytes(b"audio")

    transcript_path.write_text(json.dumps([s.to_dict() for s in segments], indent=2), encoding="utf-8")
    matches_path.write_text(json.dumps([m.to_dict() for m in matches], indent=2), encoding="utf-8")
    reports_path.write_text(json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps({"input_video": str(source), "total_duration": 90.0}, indent=2),
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
        "total_duration": 90.0,
        "config": config,
    }
    context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
    return {
        "context": context_path,
        "source": source,
        "video_files": [r.video_path for r in results],
        "audio_files": [r.audio_path for r in results],
    }


def test_post_merge_recuts_two_mp4_files(tmp_path, monkeypatch):
    from dd_clip_miner_llm import post_merge

    fixture = _write_fixture_run(tmp_path)
    calls = []

    def fake_cut_video(source, target, start, end, **kwargs):
        calls.append(("video", Path(source), Path(target), start, end, kwargs))
        Path(target).write_bytes(b"merged video")

    def fake_cut_audio(source, target, start, end, **kwargs):
        calls.append(("audio", Path(source), Path(target), start, end, kwargs))
        Path(target).write_bytes(b"merged audio")

    monkeypatch.setattr(post_merge, "cut_video", fake_cut_video)
    monkeypatch.setattr(post_merge, "cut_audio", fake_cut_audio)

    result = post_merge.post_merge_from_context(
        fixture["context"],
        fixture["video_files"][0],
        fixture["video_files"][1],
    )

    assert Path(result["video_path"]).exists()
    assert result["audio_path"] is None
    assert result["output_path"] == result["video_path"]
    assert result["output_type"] == "mp4"
    assert Path(result["video_path"]).parent == fixture["video_files"][0].parent
    assert [(kind, start, end) for kind, _, _, start, end, _ in calls] == [
        ("video", 9.0, 71.0),
    ]


def test_post_merge_recuts_two_mp3_files(tmp_path, monkeypatch):
    from dd_clip_miner_llm import post_merge

    fixture = _write_fixture_run(tmp_path)

    def fake_cut_video(_source, target, *_args, **_kwargs):
        Path(target).write_bytes(b"merged video")

    def fake_cut_audio(_source, target, *_args, **_kwargs):
        Path(target).write_bytes(b"merged audio")

    monkeypatch.setattr(post_merge, "cut_video", fake_cut_video)
    monkeypatch.setattr(post_merge, "cut_audio", fake_cut_audio)

    result = post_merge.post_merge_from_context(
        fixture["context"],
        fixture["audio_files"][0],
        fixture["audio_files"][1],
    )

    assert result["video_path"] is None
    assert Path(result["audio_path"]).suffix == ".mp3"
    assert result["output_path"] == result["audio_path"]
    assert result["output_type"] == "mp3"
    assert Path(result["audio_path"]).parent == fixture["audio_files"][0].parent


def test_post_merge_recuts_three_mp4_files(tmp_path, monkeypatch):
    from dd_clip_miner_llm import post_merge

    fixture = _write_fixture_run(tmp_path)
    reports_path = fixture["context"].parent.parent.parent.parent / "04_reports" / "song" / "songs.json"
    data = json.loads(reports_path.read_text(encoding="utf-8"))
    third_video = fixture["video_files"][0].parent / "clip3.mp4"
    third_video.write_bytes(b"video")
    data.append(
        {
            **data[0],
            "index": 3,
            "title": "Song C",
            "start": 80.0,
            "end": 90.0,
            "duration": 10.0,
            "video_path": str(third_video),
            "audio_path": str(fixture["audio_files"][0].parent / "clip3.mp3"),
        }
    )
    reports_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    matches_path = fixture["context"].parent.parent.parent.parent / "02_asr" / "llm" / "song" / "matches.json"
    matches = json.loads(matches_path.read_text(encoding="utf-8"))
    matches.append(
        {
            "content_type": "song",
            "title": "Song C",
            "artist": "Singer",
            "segment_indices": [4],
            "confidence": 0.7,
            "tags": [],
            "lyrics_snippet": "",
        }
    )
    matches_path.write_text(json.dumps(matches, indent=2), encoding="utf-8")

    calls = []

    def fake_cut_video(source, target, start, end, **kwargs):
        calls.append((Path(source), Path(target), start, end, kwargs))
        Path(target).write_bytes(b"merged video")

    monkeypatch.setattr(post_merge, "cut_video", fake_cut_video)

    result = post_merge.post_merge_from_context(
        fixture["context"],
        fixture["video_files"][0],
        fixture["video_files"][1],
        third_video,
    )

    assert Path(result["video_path"]).exists()
    assert result["start"] == 9.0
    assert result["end"] == 90.0
    assert len(calls) == 1


def test_post_merge_requires_at_least_two_files(tmp_path):
    from dd_clip_miner_llm.post_merge import PostMergeError, post_merge_from_context

    fixture = _write_fixture_run(tmp_path)

    with pytest.raises(PostMergeError, match="at least two"):
        post_merge_from_context(fixture["context"], fixture["video_files"][0])


def test_post_merge_rejects_mixed_extensions(tmp_path):
    from dd_clip_miner_llm.post_merge import PostMergeError, post_merge_from_context

    fixture = _write_fixture_run(tmp_path)

    with pytest.raises(PostMergeError, match="same extension"):
        post_merge_from_context(
            fixture["context"],
            fixture["video_files"][0],
            fixture["audio_files"][1],
        )


def test_post_merge_falls_back_from_mojibake_report_paths(tmp_path, monkeypatch):
    from dd_clip_miner_llm import post_merge

    fixture = _write_fixture_run(tmp_path)
    reports_path = fixture["context"].parent.parent.parent.parent / "04_reports" / "song" / "songs.json"
    data = json.loads(reports_path.read_text(encoding="utf-8"))
    data[0]["video_path"] = r"results_v2\bad_mojibake\03_clips\video\song\bad_name.mp4"
    data[1]["video_path"] = r"results_v2\bad_mojibake\03_clips\video\song\bad_name_002.mp4"
    data[0]["title"] = "mojibake-title"
    data[1]["title"] = "mojibake-title"
    reports_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    actual_dir = fixture["video_files"][0].parent
    first = actual_dir / "正确中文名.mp4"
    second = actual_dir / "正确中文名_002.mp4"
    first.write_bytes(b"video")
    second.write_bytes(b"video")

    def fake_cut_video(_source, target, *_args, **_kwargs):
        Path(target).write_bytes(b"merged video")

    monkeypatch.setattr(post_merge, "cut_video", fake_cut_video)

    result = post_merge.post_merge_from_context(fixture["context"], first, second)

    assert Path(result["video_path"]).exists()
    assert result["segment_indices"] == [1, 2, 3]


def test_post_merge_relocates_context_paths_from_current_output_dir(tmp_path, monkeypatch):
    from dd_clip_miner_llm import post_merge

    fixture = _write_fixture_run(tmp_path)
    context_path = fixture["context"]
    actual_run = context_path.parents[3]
    stale_run = tmp_path / "stale_machine" / "results" / actual_run.name
    context = json.loads(context_path.read_text(encoding="utf-8"))
    for key in ("run_dir", "manifest_path", "reports_path", "matches_path", "transcript_path", "input_video"):
        context[key] = str(stale_run / Path(context[key]).relative_to(actual_run))
    context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")

    calls = []

    def fake_cut_video(source, target, start, end, **kwargs):
        calls.append((Path(source), Path(target), start, end, kwargs))
        Path(target).write_bytes(b"merged video")

    monkeypatch.setattr(post_merge, "cut_video", fake_cut_video)

    result = post_merge.post_merge_from_context(
        context_path,
        fixture["video_files"][0],
        fixture["video_files"][1],
    )

    assert Path(result["source_video"]) == fixture["source"]
    assert calls[0][0] == fixture["source"]
    assert Path(result["video_path"]).exists()


def test_post_merge_matches_clip_naming_paths_from_relocated_run(tmp_path, monkeypatch):
    from dd_clip_miner_llm import post_merge

    project_root = tmp_path / "project"
    fixture = _write_fixture_run(project_root)
    run = project_root / "run"
    reports_path = run / "04_reports" / "song" / "songs.json"
    data = json.loads(reports_path.read_text(encoding="utf-8"))
    clip_name = "【Streamer】014-未知歌曲_我一个人在家里看了半天-260624.mp4"
    data[0]["video_path"] = (
        "runs\\batch\\2026_06_24\\demo_fix\\03_clips\\kv_optimized\\video\\song\\" + clip_name
    )
    data[1]["video_path"] = (
        "runs\\batch\\2026_06_24\\demo_fix\\03_clips\\kv_optimized\\video\\song\\"
        "【Streamer】013-未知歌曲_嘿,未-260624.mp4"
    )
    reports_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    output_dir = tmp_path / "nas" / "result" / "demo_fix" / "03_clips" / "kv_optimized" / "video" / "song"
    output_dir.mkdir(parents=True, exist_ok=True)
    first = output_dir / clip_name
    second = output_dir / "【Streamer】013-未知歌曲_嘿,未-260624.mp4"
    first.write_bytes(b"video")
    second.write_bytes(b"video")

    context_path = output_dir / "merge_recut_context.json"
    stale_run = tmp_path / "stale" / "runs" / "batch" / "2026_06_24" / "demo_fix"
    context = {
        "run_dir": str(stale_run),
        "content_type": "song",
        "manifest_path": str(stale_run / "manifest.json"),
        "reports_path": str(stale_run / "04_reports" / "song" / "songs.json"),
        "llm_dir": str(stale_run / "02_asr" / "llm" / "song"),
        "matches_path": str(stale_run / "02_asr" / "llm" / "song" / "matches.json"),
        "transcript_path": str(stale_run / "02_asr" / "transcript.json"),
        "input_video": str(stale_run / "00_input" / "input.mp4"),
        "total_duration": 90.0,
        "config": _post_merge_config(),
    }
    context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")

    for src, dest in (
        (run / "manifest.json", stale_run / "manifest.json"),
        (reports_path, stale_run / "04_reports" / "song" / "songs.json"),
        (run / "02_asr" / "transcript.json", stale_run / "02_asr" / "transcript.json"),
        (run / "02_asr" / "llm" / "song" / "matches.json", stale_run / "02_asr" / "llm" / "song" / "matches.json"),
        (fixture["source"], stale_run / "00_input" / "input.mp4"),
    ):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    def fake_cut_video(_source, target, *_args, **_kwargs):
        Path(target).write_bytes(b"merged video")

    monkeypatch.setattr(post_merge, "cut_video", fake_cut_video)

    result = post_merge.post_merge_from_context(context_path, first, second)

    assert Path(result["video_path"]).exists()


def test_post_merge_matches_project_relative_report_paths(tmp_path, monkeypatch):
    from dd_clip_miner_llm import post_merge

    project_root = tmp_path / "project"
    fixture = _write_fixture_run(project_root)
    run = project_root / "run"
    reports_path = run / "04_reports" / "song" / "songs.json"
    data = json.loads(reports_path.read_text(encoding="utf-8"))
    for item in data:
        if item.get("video_path"):
            item["video_path"] = str(Path(item["video_path"]).relative_to(project_root))
        if item.get("audio_path"):
            item["audio_path"] = str(Path(item["audio_path"]).relative_to(project_root))
    reports_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def fake_cut_video(_source, target, *_args, **_kwargs):
        Path(target).write_bytes(b"merged video")

    monkeypatch.setattr(post_merge, "cut_video", fake_cut_video)

    result = post_merge.post_merge_from_context(
        fixture["context"],
        fixture["video_files"][0],
        fixture["video_files"][1],
    )

    assert Path(result["video_path"]).exists()


def test_post_merge_errors_when_dragged_file_is_not_in_report(tmp_path):
    from dd_clip_miner_llm.post_merge import PostMergeError, post_merge_from_context

    fixture = _write_fixture_run(tmp_path)
    unknown = tmp_path / "unknown.mp4"
    unknown.write_bytes(b"unknown")

    with pytest.raises(PostMergeError, match="not listed"):
        post_merge_from_context(fixture["context"], unknown, fixture["video_files"][1])


def test_manual_cut_context_recuts_with_relocated_source_video(tmp_path, monkeypatch):
    from dd_clip_miner_llm import manual_cut_context

    fixture = _write_fixture_run(tmp_path)
    context_path = fixture["context"].with_name("manual_cut_context.json")
    context_path.write_text(fixture["context"].read_text(encoding="utf-8"), encoding="utf-8")
    actual_run = context_path.parents[3]
    stale_run = tmp_path / "stale_machine" / "results" / actual_run.name
    context = json.loads(context_path.read_text(encoding="utf-8"))
    for key in ("run_dir", "manifest_path", "input_video"):
        context[key] = str(stale_run / Path(context[key]).relative_to(actual_run))
    context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")

    calls = []

    def fake_cut_video(source, target, start, end, **kwargs):
        calls.append((Path(source), Path(target), start, end, kwargs))
        Path(target).write_bytes(b"manual video")

    monkeypatch.setattr(manual_cut_context, "cut_video", fake_cut_video)

    result = manual_cut_context.manual_cut_from_context(context_path, "10", "12", "manual")

    assert Path(result["input_video"]) == fixture["source"]
    assert Path(result["output_path"]).exists()
    assert calls == [(fixture["source"], context_path.parent / "manual.mp4", 10.0, 12.0, {"video_codec": "copy"})]


def test_manual_cut_context_wraps_shared_path_errors(tmp_path):
    from dd_clip_miner_llm.manual_cut_context import ManualCutContextError, manual_cut_from_context

    missing_context = tmp_path / "missing" / "manual_cut_context.json"

    with pytest.raises(ManualCutContextError, match="Required file not found"):
        manual_cut_from_context(missing_context, "10", "12")


def test_manual_cut_context_rejects_bad_time_as_manual_error(tmp_path):
    from dd_clip_miner_llm.manual_cut_context import ManualCutContextError, manual_cut_from_context

    fixture = _write_fixture_run(tmp_path)
    context_path = fixture["context"].with_name("manual_cut_context.json")
    context_path.write_text(fixture["context"].read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ManualCutContextError, match="Invalid time format"):
        manual_cut_from_context(context_path, "bad", "12")


def test_export_results_writes_merge_recut_assets(tmp_path, monkeypatch):
    from dd_clip_miner_llm.pipeline import export

    def fake_cut_audio(_input, target, *_args, **_kwargs):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text("audio", encoding="utf-8")

    def fake_cut_video(_input, target, *_args, **_kwargs):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text("video", encoding="utf-8")

    monkeypatch.setattr(export, "cut_audio", fake_cut_audio)
    monkeypatch.setattr(export, "cut_video", fake_cut_video)

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

    export._export_results(
        [result],
        input_path,
        clips_dir,
        _post_merge_config(),
        "song",
        run_dir=run_dir,
        llm_dir=llm_dir,
        reports_dir=reports_dir,
        transcript_path=transcript_path,
        manifest_path=manifest_path,
        total_duration=10.0,
    )

    for target_dir in (clips_dir / "audio" / "song", clips_dir / "video" / "song"):
        bat_content = (target_dir / "merge_mp4.bat").read_text(encoding="utf-8")
        assert 'call "%~dp0_resolve_env.bat"' in bat_content
        assert (target_dir / "_resolve_env.bat").is_file()
        assert (target_dir / "manual_cut.bat").is_file()
        context = json.loads((target_dir / "merge_recut_context.json").read_text(encoding="utf-8"))
        assert context["content_type"] == "song"
        assert context["run_dir"] == "."
        assert Path(context["input_video"]) == Path("00_input/input.mp4")
        assert "python_executable" not in context
        assert "project_root" not in context
        assert context["config"]["output"]["video_codec"] == "copy"
        assert Path(result.video_path) == Path("03_clips/video/song/001-Song.mp4")
