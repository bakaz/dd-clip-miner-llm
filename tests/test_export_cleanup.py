from __future__ import annotations

from pathlib import Path

from dd_clip_miner_llm.config import DEFAULT_CONFIG, deep_merge
from dd_clip_miner_llm.models import ContentResult


def test_export_results_clears_stale_content_type_files(tmp_path, monkeypatch):
    from dd_clip_miner_llm import pipeline

    clips_dir = tmp_path / "03_clips"
    audio_dir = clips_dir / "audio" / "song"
    video_dir = clips_dir / "video" / "song"
    audio_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    (audio_dir / "stale.mp3").write_text("old", encoding="utf-8")
    (video_dir / "stale.mp4").write_text("old", encoding="utf-8")

    def fake_cut_audio(_input, target, *_args, **_kwargs):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text("new audio", encoding="utf-8")

    def fake_cut_video(_input, target, *_args, **_kwargs):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text("new video", encoding="utf-8")

    monkeypatch.setattr(pipeline, "cut_audio", fake_cut_audio)
    monkeypatch.setattr(pipeline, "cut_video", fake_cut_video)

    result = ContentResult(
        index=1,
        content_type="song",
        title="New Song",
        start=1.0,
        end=5.0,
        duration=4.0,
        transcript="lyrics",
        confidence=0.9,
    )
    duplicate = ContentResult(
        index=2,
        content_type="song",
        title="New Song",
        start=6.0,
        end=10.0,
        duration=4.0,
        transcript="lyrics again",
        confidence=0.9,
    )
    config = deep_merge(DEFAULT_CONFIG, {
        "output": {
            "audio_segments": True,
            "video_clips": True,
            "audio_extension": "mp3",
            "video_extension": "mp4",
            "video_codec": "copy",
            "max_export_workers": 1,
        },
    })

    pipeline._export_results([result, duplicate], tmp_path / "input.mp4", clips_dir, config, "song")

    assert not (audio_dir / "stale.mp3").exists()
    assert not (video_dir / "stale.mp4").exists()
    assert len(list(audio_dir.glob("*.mp3"))) == 2
    assert len(list(video_dir.glob("*.mp4"))) == 2
