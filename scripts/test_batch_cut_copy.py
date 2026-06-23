"""Mock test for run_batch_cut_copy integration.

Creates a fake batch-run result and verifies the post-processing logic
without actually running the pipeline or copying to SMB.
"""

import json
import shutil
import tempfile
from pathlib import Path

# Ensure we can import the module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dd_clip_miner_llm.cut_copy import (
    load_cut_copy_config,
    run_batch_cut_copy,
    _format_folder,
    _log,
)


def create_mock_batch_results(base_dir: Path) -> list[dict]:
    """Create mock batch-run results with fake result directories."""
    results = []
    for i, name in enumerate(["video1", "video2", "video3"]):
        result_dir = base_dir / "results" / name
        result_dir.mkdir(parents=True, exist_ok=True)

        # Create some fake output files
        (result_dir / "manifest.json").write_text(
            json.dumps({"content_types": {"song": i + 1}}),
            encoding="utf-8",
        )
        (result_dir / "04_reports").mkdir(exist_ok=True)
        (result_dir / "04_reports" / "songs.csv").write_text(
            "start,end,title\n0,100,TestSong\n",
            encoding="utf-8",
        )

        # Create a fake source video path
        video_path = base_dir / "input" / f"{name}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"\x00" * 1024)  # 1KB dummy

        results.append({
            "video": str(video_path),
            "video_key": str(video_path.resolve()),
            "profile": None,
            "work_dir": str(result_dir),
            "result_dir": str(result_dir),
            "song_count": i + 1,
            "content_counts": {"song": i + 1},
            "status": "success",
        })

    # Add one failed result
    failed_dir = base_dir / "results" / "failed_video"
    failed_dir.mkdir(parents=True, exist_ok=True)
    results.append({
        "video": str(base_dir / "input" / "failed_video.mp4"),
        "video_key": str((base_dir / "input" / "failed_video.mp4").resolve()),
        "profile": None,
        "work_dir": str(failed_dir),
        "result_dir": str(failed_dir),
        "error": "Pipeline failed",
        "status": "failed",
    })

    return results


def create_mock_config(base_dir: Path, dest_dir: Path) -> dict:
    """Create a mock cut_copy config."""
    return {
        "enabled": True,
        "source": {
            "path": str(base_dir / "input"),
            "pattern": "*.mp4",
            "done_marker": ".done.json",
        },
        "destination": {
            "path": str(dest_dir),
            "username": "",
            "password": "",
            "folder_format": "{date}_{streamer}",
        },
        "processing": {
            "config_path": "config.yaml",
            "skip_on_failure": True,
        },
        "behavior": {
            "shutdown_after": False,  # Don't actually shutdown
            "shutdown_delay": 60,
            "delete_source_after_copy": False,  # Don't delete in test
            "delete_work_dir": False,  # Don't delete in test
            "log_file": "test_cut_copy.log",
            "max_files": 0,
            "max_runtime": 0,
        },
    }


def test_enabled_false():
    """Test that disabled config skips processing."""
    print("=== Test: enabled=false ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        dest = base / "dest"
        dest.mkdir()

        config = create_mock_config(base, dest)
        config["enabled"] = False

        runs = create_mock_batch_results(base)
        rc = run_batch_cut_copy(config, runs, no_shutdown=True)

        assert rc == 0, f"Expected 0, got {rc}"
        # Verify nothing was copied
        assert not any(dest.iterdir()), "Dest should be empty when disabled"
        print("  PASS: disabled config returns 0, no files copied")


def test_no_successful_runs():
    """Test with only failed runs."""
    print("=== Test: no successful runs ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        dest = base / "dest"
        dest.mkdir()

        config = create_mock_config(base, dest)
        runs = [
            {"video": "fake.mp4", "result_dir": "/nonexistent", "status": "failed"},
        ]

        rc = run_batch_cut_copy(config, runs, no_shutdown=True)
        assert rc == 0, f"Expected 0, got {rc}"
        print("  PASS: no successful runs returns 0")


def test_copy_successful():
    """Test successful copy of batch results."""
    print("=== Test: successful copy ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        dest = base / "dest"
        dest.mkdir()

        config = create_mock_config(base, dest)
        runs = create_mock_batch_results(base)

        rc = run_batch_cut_copy(config, runs, no_shutdown=True)
        assert rc == 0, f"Expected 0, got {rc}"

        # Verify files were copied
        copied_dirs = list(dest.iterdir())
        print(f"  Copied {len(copied_dirs)} directories to dest")
        assert len(copied_dirs) > 0, "Expected at least one copied directory"

        # Verify content
        for d in copied_dirs:
            manifest = d / "manifest.json"
            if manifest.exists():
                data = json.loads(manifest.read_text(encoding="utf-8"))
                print(f"    {d.name}: {data}")

        print("  PASS: files copied successfully")


def test_format_folder():
    """Test _format_folder with batch video path."""
    print("=== Test: _format_folder ===")

    # Without batch video path (original behavior)
    config = {}
    result = _format_folder("{date}_{streamer}", Path("/some/path/result_dir"), config)
    print(f"  Without batch path: {result}")

    # With batch video path
    config = {"_batch_video_path": Path("/ddtv/2026_06_07_streamer/video.mp4")}
    result = _format_folder("{date}_{streamer}", Path("/some/path/result_dir"), config)
    print(f"  With batch path: {result}")
    assert "2026_06_07_streamer" in result, f"Expected streamer in result: {result}"

    print("  PASS: format_folder works correctly")


def test_skip_missing_result_dir():
    """Test that missing result_dir is skipped."""
    print("=== Test: skip missing result_dir ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        dest = base / "dest"
        dest.mkdir()

        config = create_mock_config(base, dest)
        runs = [
            {
                "video": "/fake/video.mp4",
                "result_dir": "/nonexistent/path",
                "status": "success",
            },
        ]

        rc = run_batch_cut_copy(config, runs, no_shutdown=True)
        assert rc == 0, f"Expected 0, got {rc}"
        assert not any(dest.iterdir()), "Dest should be empty when result_dir missing"
        print("  PASS: missing result_dir skipped")


if __name__ == "__main__":
    print("Running mock tests for run_batch_cut_copy...\n")

    test_enabled_false()
    test_no_successful_runs()
    test_format_folder()
    test_skip_missing_result_dir()
    test_copy_successful()

    print("\n=== All tests passed! ===")
