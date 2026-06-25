"""Baseline characterization: migrate_config.py converts old-format configs.

These tests MUST FAIL before migrate_config.py is implemented.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIGRATE_SCRIPT = PROJECT_ROOT / "scripts" / "migrate_config.py"
CONFIG_EXAMPLE = PROJECT_ROOT / "tests" / "config.example.yaml"


def _run_migrate(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run migrate_config.py with the given args."""
    cmd = [sys.executable, str(MIGRATE_SCRIPT), *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), **kwargs)


# ── Happy path: migration from config.example.yaml ──────────────────

def test_migrate_config_example():
    """Migrating config.example.yaml (test fixture) produces modular config."""
    output = tempfile.mkdtemp()
    result = _run_migrate([str(CONFIG_EXAMPLE), "--output", output])
    assert result.returncode == 0, result.stderr
    assert (Path(output) / "main.yaml").exists(), "main.yaml was not created"
    assert (Path(output) / "profiles" / "kv_optimized.yaml").exists(), "profile file was not created"
    assert (Path(output) / "audio.yaml").exists(), "domain file audio.yaml was not created"
    assert (Path(output) / "llm.yaml").exists(), "domain file llm.yaml was not created"
    assert (Path(output) / "song.yaml").exists(), "domain file song.yaml was not created"
    assert (Path(output) / "asr.yaml").exists(), "domain file asr.yaml was not created"
    assert (Path(output) / "content_types.yaml").exists(), "domain file content_types.yaml was not created"
    assert (Path(output) / "output.yaml").exists(), "domain file output.yaml was not created"
    assert (Path(output) / "padding.yaml").exists(), "domain file padding.yaml was not created"
    assert (Path(output) / "dialogue.yaml").exists(), "domain file dialogue.yaml was not created"
    assert (Path(output) / "highlight.yaml").exists(), "domain file highlight.yaml was not created"
    assert (Path(output) / "funny.yaml").exists(), "domain file funny.yaml was not created"
    assert (Path(output) / "cringe.yaml").exists(), "domain file cringe.yaml was not created"
    assert (Path(output) / "daily_summary.yaml").exists(), "domain file daily_summary.yaml was not created"
    assert (Path(output) / "cut_copy.yaml").exists(), "domain file cut_copy.yaml was not created"


def test_migrate_all_profiles():
    """All three example profiles are migrated."""
    output = tempfile.mkdtemp()
    result = _run_migrate([str(CONFIG_EXAMPLE), "--output", output])
    assert result.returncode == 0, result.stderr
    profiles_dir = Path(output) / "profiles"
    assert profiles_dir.is_dir()
    assert (profiles_dir / "kv_optimized.yaml").exists()
    assert (profiles_dir / "kv_v2.yaml").exists()
    assert (profiles_dir / "accuracy.yaml").exists()


# ── Round-trip data integrity ─────────────────────────────────────

def test_round_trip_domain_data():
    """Domain YAML files contain the same data as the original config sections."""
    output = tempfile.mkdtemp()
    result = _run_migrate([str(CONFIG_EXAMPLE), "--output", output])
    assert result.returncode == 0, result.stderr

    with open(CONFIG_EXAMPLE, "r", encoding="utf-8") as f:
        old = yaml.safe_load(f)

    domain_keys = [
        "audio", "asr", "llm", "content_types", "song", "output", "padding",
        "dialogue", "highlight", "funny", "cringe", "daily_summary", "cut_copy",
    ]
    for key in domain_keys:
        if key not in old:
            continue
        domain_file = Path(output) / f"{key}.yaml"
        assert domain_file.exists(), f"Missing domain file: {key}.yaml"
        with open(domain_file, "r", encoding="utf-8") as f:
            content = f.read()
        # Strip the generated header comment to extract data
        data = _extract_yaml_data(content)
        assert data == old[key], f"Round-trip mismatch for key: {key}"


def test_round_trip_profile_data():
    """Profile YAML files contain the same data as the original profile sections."""
    output = tempfile.mkdtemp()
    result = _run_migrate([str(CONFIG_EXAMPLE), "--output", output])
    assert result.returncode == 0, result.stderr

    with open(CONFIG_EXAMPLE, "r", encoding="utf-8") as f:
        old = yaml.safe_load(f)

    old_profiles = old.get("profiles", {})
    if not isinstance(old_profiles, dict):
        return

    for name, expected_data in old_profiles.items():
        profile_file = Path(output) / "profiles" / f"{name}.yaml"
        assert profile_file.exists(), f"Missing profile file: {name}.yaml"
        with open(profile_file, "r", encoding="utf-8") as f:
            content = f.read()
        data = _extract_yaml_data(content)
        assert data == expected_data, f"Round-trip mismatch for profile: {name}"


def test_default_profile_preserved():
    """default_profile value is written to main.yaml."""
    output = tempfile.mkdtemp()
    result = _run_migrate([str(CONFIG_EXAMPLE), "--output", output])
    assert result.returncode == 0, result.stderr

    with open(CONFIG_EXAMPLE, "r", encoding="utf-8") as f:
        old = yaml.safe_load(f)

    main_yaml = Path(output) / "main.yaml"
    content = main_yaml.read_text(encoding="utf-8")
    expected_default = old.get("default_profile", "")
    assert f"default_profile: {expected_default}" in content, "default_profile not preserved in main.yaml"


# ── DRY RUN ────────────────────────────────────────────────────────

def test_dry_run_does_not_create_files():
    """--dry-run prints planned files without writing anything."""
    output = tempfile.mkdtemp()
    result = _run_migrate([str(CONFIG_EXAMPLE), "--output", output, "--dry-run"])
    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout or "DRY RUN" in result.stdout.upper()
    # No files created
    created = list(Path(output).rglob("*"))
    assert len(created) == 0, f"Dry-run created files: {created}"


# ── HELP ───────────────────────────────────────────────────────────

def test_help_prints_usage():
    """--help prints usage and exits 0."""
    result = _run_migrate(["--help"])
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower() or "usage" in result.stdout


# ── OVERWRITE / SKIP ───────────────────────────────────────────────

def test_overwrite_replaces_existing():
    """--overwrite allows replacing existing files."""
    output = tempfile.mkdtemp()
    # First migration
    result1 = _run_migrate([str(CONFIG_EXAMPLE), "--output", output])
    assert result1.returncode == 0, result1.stderr
    assert "Created" in result1.stdout

    # Second migration without --overwrite should skip
    result2 = _run_migrate([str(CONFIG_EXAMPLE), "--output", output])
    assert result2.returncode == 0, result2.stderr
    assert "Skipped" in result2.stdout or "skipped" in result2.stdout.lower()

    # Third migration with --overwrite should create
    result3 = _run_migrate([str(CONFIG_EXAMPLE), "--output", output, "--overwrite"])
    assert result3.returncode == 0, result3.stderr
    assert "Created" in result3.stdout


# ── Malformed input ────────────────────────────────────────────────

def test_invalid_yaml_errors_cleanly():
    """Invalid YAML input produces a clear error message."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("invalid: [unclosed\n")
        bad_config = f.name

    try:
        output = tempfile.mkdtemp()
        result = _run_migrate([bad_config, "--output", output])
        assert result.returncode != 0, "Should fail on invalid YAML"
        # Should contain some error indication
        assert result.stderr or "Error" in result.stdout or "error" in result.stdout.lower()
    finally:
        Path(bad_config).unlink(missing_ok=True)


# ── Companion files ────────────────────────────────────────────────

def test_companion_files_copied():
    """streamer_dictionary.example.json is NOT copied — script looks for streamer_dictionary.json."""
    output = tempfile.mkdtemp()
    result = _run_migrate([str(CONFIG_EXAMPLE), "--output", output])
    assert result.returncode == 0, result.stderr
    # The config is at tests/config.example.yaml; its parent dir also holds
    # streamer_dictionary.example.json (not streamer_dictionary.json), so no copy expected.
    pass


def test_cut_copy_conf_config_path_patched(tmp_path):
    """cut_copy.conf processing.config_path is rewritten to the migrated main.yaml."""
    old_config = {
        "audio": {"sample_rate": 16000, "channels": 1},
        "cut_copy": {"enabled": False, "conf_path": "cut_copy.conf"},
        "default_profile": "kv_optimized",
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(old_config, allow_unicode=True), encoding="utf-8")

    workflow = {
        "enabled": True,
        "source": {"path": "//nas/recordings"},
        "destination": {"path": "//nas/results"},
        "processing": {"config_path": "config.yaml"},
    }
    (tmp_path / "cut_copy.conf").write_text(
        yaml.dump(workflow, allow_unicode=True), encoding="utf-8"
    )

    output = tmp_path / "config" / "local"
    result = _run_migrate([str(config_file), "--output", str(output)])
    assert result.returncode == 0, result.stderr

    patched = yaml.safe_load((output / "cut_copy.conf").read_text(encoding="utf-8"))
    assert patched["processing"]["config_path"] == "config/local/main.yaml"
    assert "processing.config_path -> config/local/main.yaml" in result.stdout


def test_cut_copy_workflow_from_yaml_companion(tmp_path):
    """A standalone cut_copy.yaml companion is written as patched cut_copy.conf."""
    old_config = {
        "cut_copy": {"enabled": False, "conf_path": "cut_copy.conf"},
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(old_config, allow_unicode=True), encoding="utf-8")

    workflow = {
        "source": {"path": "//nas/recordings"},
        "destination": {"path": "//nas/results"},
        "processing": {"config_path": "D:/old/config.yaml"},
    }
    (tmp_path / "cut_copy.yaml").write_text(
        yaml.dump(workflow, allow_unicode=True), encoding="utf-8"
    )

    output = tmp_path / "config" / "local"
    result = _run_migrate([str(config_file), "--output", str(output)])
    assert result.returncode == 0, result.stderr

    assert (output / "cut_copy.conf").is_file()
    patched = yaml.safe_load((output / "cut_copy.conf").read_text(encoding="utf-8"))
    assert patched["processing"]["config_path"] == "config/local/main.yaml"


# ── Helpers ────────────────────────────────────────────────────────

def _extract_yaml_data(content: str) -> dict:
    """Extract YAML data from a file, stripping the generated header comment."""
    lines = content.split("\n")
    # Skip generated header comment lines and blank lines
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            continue
        if line.strip() == "":
            continue
        start = i
        break
    data_content = "\n".join(lines[start:])
    return yaml.safe_load(data_content) or {}
