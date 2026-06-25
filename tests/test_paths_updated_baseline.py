"""Baseline characterization: ensure NO old config paths remain in tests."""
from pathlib import Path


def test_no_old_config_paths_in_tests():
    """Verify tests/ contains zero references to deprecated root-level config files."""
    old_paths = [
        '"config.example.yaml"',
        '"config.daily-summary.example.yaml"',
        '"cut_copy.example.conf"',
        '"streamer_dictionary.example.json"',
    ]
    for p in Path('tests').rglob('*.py'):
        if p.name in ('test_paths_updated_baseline.py', 'test_migration_script_baseline.py', 'test_readme_paths_baseline.py'):
            continue  # self, migration, and readme tests — these intentionally reference old paths
        text = p.read_text(encoding='utf-8')
        for old in old_paths:
            assert old not in text, f"{p} still references {old}"
