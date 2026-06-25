"""Baseline characterization tests for config/local/ gitignore behavior."""

from pathlib import Path
import subprocess


def test_config_local_gitignored():
    """Given config/local/ is in .gitignore, when a file is created there, git check-ignore returns 0."""
    test_file = Path("config/local/test_secret.txt")
    test_file.write_text("secret", encoding="utf-8")
    try:
        result = subprocess.run(
            ["git", "check-ignore", str(test_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Expected config/local/ files to be ignored, got: {result.stderr}"
        # git check-ignore may quote and escape paths on Windows; normalize both sides
        stdout_normalized = result.stdout.strip().strip('"').replace("\\\\", "/")
        assert test_file.as_posix() in stdout_normalized, (
            f"Expected path in output, got: {result.stdout!r}"
        )
    finally:
        test_file.unlink(missing_ok=True)


def test_config_example_not_gitignored():
    """Given config/example/ is NOT in .gitignore, when checked, git check-ignore returns 1."""
    # Use --no-index so check-ignore works even if the file doesn't exist on disk
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "config/example/audio.yaml"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, (
        f"Expected config/example/ files to NOT be ignored, but they are: {result.stdout}"
    )


def test_gitkeep_is_tracked():
    """config/local/.gitkeep should exist and NOT be gitignored (it's the sole tracked file in that dir)."""
    gitkeep = Path("config/local/.gitkeep")
    assert gitkeep.exists(), ".gitkeep is missing"
    result = subprocess.run(
        ["git", "check-ignore", str(gitkeep)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, (
        f"Expected .gitkeep to be tracked (not ignored), but it is ignored: {result.stdout}"
    )


def test_wol_webhook_remains_gitignored():
    """wol_webhook.py at root should still be ignored per .gitignore."""
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "wol_webhook.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Expected wol_webhook.py to still be ignored, got exit {result.returncode}: {result.stderr}"
    )
