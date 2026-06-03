"""识别器抽象基类"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models import ContentMatch, TranscriptSegment


class BaseRecognizer(ABC):
    """内容识别器基类
    
    所有内容类型（歌曲、对话、高能时刻、搞笑片段）的识别器都应继承此类。
    
    使用方式：
        1. 继承 BaseRecognizer
        2. 实现 name 属性（返回内容类型名称，如 "song", "dialogue"）
        3. 实现 build_prompt() 方法（构建 LLM 提示词）
        4. 可选覆盖 parse_response() 方法（自定义响应解析）
        5. 可选覆盖 get_tools() 方法（提供 LLM 工具）
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """内容类型名称，如 "song", "dialogue", "highlight", "funny"""
        ...
    
    @property
    def default_config(self) -> dict[str, Any]:
        """该识别器的默认配置"""
        return {
            "enabled": True,
            "min_duration": 10.0,
            "max_duration": 300.0,
            "min_confidence": 0.6,
            "merge_gap_seconds": 10.0,
        }
    
    @abstractmethod
    def build_prompt(
        self,
        segments: list[TranscriptSegment],
        batch_start: int,
        config: dict[str, Any],
    ) -> str:
        """构建 LLM 提示词
        
        Args:
            segments: ASR 转写片段列表
            batch_start: 当前批次的起始索引
            config: 完整配置字典
            
        Returns:
            发送给 LLM 的提示词
        """
        ...
    
    def get_tools(self, config: dict[str, Any]) -> list[dict[str, Any]] | None:
        """获取 LLM 工具列表（可选）
        
        Args:
            config: 完整配置字典
            
        Returns:
            工具列表，或 None 表示不使用工具
        """
        return None
    
    def parse_response(
        self,
        items: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[ContentMatch]:
        """解析 LLM 响应为 ContentMatch 列表
        
        默认实现，子类可覆盖以自定义解析逻辑。
        
        Args:
            items: LLM 返回的 JSON 数组中的对象列表
            config: 完整配置字典
            
        Returns:
            ContentMatch 列表
        """
        matches = []
        for item in items:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            
            match = ContentMatch(
                content_type=item.get("content_type", self.name),
                title=title,
                segment_indices=self._parse_segment_indices(item.get("segment_indices", [])),
                confidence=self._parse_confidence(item.get("confidence", 0.5)),
                tags=item.get("tags", []),
                description=item.get("description", ""),
                artist=str(item.get("artist", "")),
                lyrics_snippet=str(item.get("lyrics_snippet", "")),
            )
            if match.segment_indices:
                matches.append(match)
        return matches
    
    @staticmethod
    def _parse_segment_indices(value: Any) -> list[int]:
        """解析 segment_indices"""
        if not isinstance(value, list):
            return []
        indices: list[int] = []
        seen: set[int] = set()
        for item in value:
            if isinstance(item, bool):
                continue
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if idx not in seen:
                indices.append(idx)
                seen.add(idx)
        return indices
    
    @staticmethod
    def _parse_confidence(value: Any) -> float:
        """解析 confidence"""
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, confidence))
    
    def get_merge_gap(self, config: dict[str, Any]) -> float:
        """获取合并间隔"""
        type_config = config.get(self.name, {})
        return float(type_config.get("merge_gap_seconds", self.default_config.get("merge_gap_seconds", 10.0)))
    
    def get_min_duration(self, config: dict[str, Any]) -> float:
        """获取最小持续时间"""
        type_config = config.get(self.name, {})
        return float(type_config.get("min_duration", self.default_config.get("min_duration", 10.0)))
