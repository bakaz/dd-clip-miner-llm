"""测试配置"""
from __future__ import annotations

import os

import pytest
from pathlib import Path


def pytest_configure(config):
    """Windows 上 AppData\\Local\\Temp 可能无写权限，改用项目内临时目录。"""
    if config.option.basetemp is None:
        project_tmp = Path(__file__).resolve().parent.parent / ".tmp" / "pytest"
    else:
        project_tmp = Path(config.option.basetemp)
    project_tmp.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(project_tmp.resolve())


def pytest_collection_modifyitems(config, items):
    """GitHub CI 只跑离线单元测试，跳过需要网络或密钥的用例。"""
    if not os.environ.get("DD_CLIP_MINER_LLM_CI"):
        return
    skip = pytest.mark.skip(reason="skipped in CI (network/secrets)")
    for item in items:
        if "network" in item.keywords or "secrets" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def sample_segments():
    """示例 ASR 转写片段"""
    from dd_clip_miner_llm.models import TranscriptSegment
    return [
        TranscriptSegment(start=0.0, end=2.0, text="大家好"),
        TranscriptSegment(start=2.0, end=4.0, text="欢迎来到直播间"),
        TranscriptSegment(start=4.0, end=8.0, text="今天给大家唱一首歌"),
        TranscriptSegment(start=8.0, end=12.0, text="歌词第一句"),
        TranscriptSegment(start=12.0, end=16.0, text="歌词第二句"),
        TranscriptSegment(start=16.0, end=20.0, text="歌词第三句"),
        TranscriptSegment(start=20.0, end=24.0, text="谢谢大家"),
    ]


@pytest.fixture
def sample_config():
    """示例配置"""
    from dd_clip_miner_llm.config import DEFAULT_CONFIG
    from copy import deepcopy
    return deepcopy(DEFAULT_CONFIG)


@pytest.fixture
def sample_matches():
    """示例歌曲匹配结果"""
    from dd_clip_miner_llm.models import ContentMatch
    return [
        ContentMatch(
            content_type="song",
            title="测试歌曲",
            segment_indices=[3, 4, 5],
            confidence=0.9,
            artist="测试歌手",
        ),
    ]


@pytest.fixture
def tmp_output_dir(tmp_path):
    """临时输出目录"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
