from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContentMatch:
    """通用内容片段匹配结果"""
    content_type: str        # "song", "dialogue", "highlight", "funny"
    title: str               # 片段标题/描述
    segment_indices: list[int]
    confidence: float
    tags: list[str] = field(default_factory=list)
    description: str = ""
    artist: str = ""         # 仅歌曲类型使用
    lyrics_snippet: str = ""  # 仅歌曲类型使用

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContentResult:
    """通用内容片段结果"""
    index: int
    content_type: str
    title: str
    start: float
    end: float
    duration: float
    transcript: str
    confidence: float
    tags: list[str] = field(default_factory=list)
    description: str = ""
    artist: str = ""
    audio_path: Path | None = None
    video_path: Path | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["audio_path"] = str(self.audio_path) if self.audio_path else None
        data["video_path"] = str(self.video_path) if self.video_path else None
        return data


def parse_cringe_description(description: str) -> dict[str, Any]:
    """解析 cringe 识别器的 description 字段，提取 severity 和 scenario
    
    示例输入: "[严重程度:3/5][场景:B] 观众对主播开黄腔，主播尴尬制止"
    返回: {"severity": 3, "scenario": "B", "description": "观众对主播开黄腔，主播尴尬制止"}
    """
    import re
    
    result = {"severity": 0, "scenario": "C", "description": description}
    
    # 提取 severity
    severity_match = re.search(r'\[严重程度:(\d+)/5\]', description)
    if severity_match:
        result["severity"] = int(severity_match.group(1))
    
    # 提取 scenario
    scenario_match = re.search(r'\[场景:([ABC])\]', description)
    if scenario_match:
        result["scenario"] = scenario_match.group(1)
    
    # 提取纯描述（去掉前缀）
    clean_desc = re.sub(r'\[严重程度:\d+/5\]\s*', '', description)
    clean_desc = re.sub(r'\[场景:[ABC]\]\s*', '', clean_desc)
    result["description"] = clean_desc.strip()
    
    return result


# 兼容旧项目 dd-song-miner-llm 的类型别名
SongMatch = ContentMatch
SongResult = ContentResult


def create_song_match(
    title: str,
    artist: str = "",
    lyrics_snippet: str = "",
    segment_indices: list[int] | None = None,
    confidence: float = 0.5,
) -> ContentMatch:
    """创建歌曲匹配结果（兼容旧项目）"""
    return ContentMatch(
        content_type="song",
        title=title,
        segment_indices=segment_indices or [],
        confidence=confidence,
        tags=[],
        description="",
        artist=artist,
        lyrics_snippet=lyrics_snippet,
    )


def create_song_result(
    index: int,
    title: str,
    artist: str = "",
    start: float = 0.0,
    end: float = 0.0,
    duration: float = 0.0,
    lyrics_snippet: str = "",
    confidence: float = 0.5,
    audio_path: Path | None = None,
    video_path: Path | None = None,
    transcript: str = "",
    errors: list[str] | None = None,
) -> ContentResult:
    """创建歌曲结果（兼容旧项目）"""
    return ContentResult(
        index=index,
        content_type="song",
        title=title,
        start=start,
        end=end,
        duration=duration,
        transcript=transcript,
        confidence=confidence,
        tags=[],
        description="",
        artist=artist,
        audio_path=audio_path,
        video_path=video_path,
        errors=errors or [],
    )
