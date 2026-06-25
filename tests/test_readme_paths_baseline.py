"""Baseline: README.md must reference new modular config paths, not old flat-file paths.

These tests are expected to FAIL before the README is updated.
Once the README is updated to point to config/example/ and config/local/,
these tests will PASS.
"""

from pathlib import Path


def _readme_text() -> str:
    return Path("README.md").read_text(encoding="utf-8")


def test_readme_uses_new_main_config_path():
    """README must reference config/example/main.yaml."""
    text = _readme_text()
    assert "config/example/main.yaml" in text, (
        "READ ME: update README to reference config/example/main.yaml "
        "instead of config.example.yaml"
    )


def test_readme_uses_new_local_config_dir():
    """README must reference config/local/ directory."""
    text = _readme_text()
    assert "config/local/" in text, (
        "READ ME: update README quick-start to reference config/local/ directory"
    )


def test_readme_mentions_migration_doc():
    """README must link to docs/MIGRATION.md for existing users."""
    text = _readme_text()
    assert "docs/MIGRATION.md" in text, (
        "READ ME: add migration link to docs/MIGRATION.md for users on old flat-file format"
    )


def test_old_main_config_not_barely_present():
    """config.example.yaml must not appear unless in a legacy/migration note."""
    text = _readme_text()
    if "config.example.yaml" in text:
        assert "old format" in text.lower() or "旧" in text, (
            "READ ME: old config.example.yaml reference found outside a migration/legacy note"
        )
