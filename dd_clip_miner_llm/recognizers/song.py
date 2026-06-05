"""歌曲识别器"""
from __future__ import annotations

from typing import Any

from . import register
from .base import BaseRecognizer
from ..models import ContentMatch, TranscriptSegment
from ..search_tools import get_tools


@register
class SongRecognizer(BaseRecognizer):
    """歌曲片段识别器"""
    
    @property
    def name(self) -> str:
        return "song"
    
    @property
    def default_config(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "padding": {
                "before_seconds": 15.0,
                "after_seconds": 15.0,
                "after_next_asr_end_guard_seconds": 2.0,
                "min_song_seconds": 75.0,
                "max_song_seconds": 360.0,
                "merge_gap_seconds": 20.0,
            },
            "missed_recheck": {
                "enabled": True,
                "batch_size": 500,
                "min_gap_segments": 1,
                "context_segments": 10,
            },
        }
    
    def build_prompt(
        self,
        segments: list[TranscriptSegment],
        batch_start: int,
        config: dict[str, Any],
    ) -> str:
        lines = []
        for i, seg in enumerate(segments):
            idx = batch_start + i
            lines.append(f"[{idx}] ({seg.start:.1f}s-{seg.end:.1f}s) {seg.text}")

        transcript_text = "\n".join(lines)

        return f"""你是一个面向演唱会、直播和长视频的歌曲识别专家。
下面是一整段视频的 Whisper ASR 转写片段，每行格式为 [序号] (开始秒-结束秒) 文本。

任务：从完整上下文中识别所有演唱片段，返回纯 JSON 数组。
每个对象必须包含以下字段：
- content_type: "song"
- title: 歌名。能识别出原曲时填写准确歌名；无法确认时填写"未知歌曲："加最有代表性的一句歌词。
- artist: 原唱或演唱者。无法判断时填空字符串。
- segment_indices: 属于同一首歌的 ASR 段落序号数组，必须只使用输入中出现的序号，按升序排列。
- confidence: 0 到 1 的置信度。
- tags: 空数组。
- description: 空字符串。

识别原则：
1. 只要是**在唱歌**的段落都应识别出来，即使无法确定歌名。
2. 同一首歌的连续演唱段落必须合并成一个对象。不要把主歌、副歌、桥段拆成多首。
3. 明显的说话、聊天、感谢、报幕、互动、口播不要放进 segment_indices。
4. 不要因为短于 2 分钟就丢弃。只要是在唱歌就标出来。

Whisper ASR 转写特性（重要）：
- 歌词转写可能有错字、漏字、同音字替换，不要因为歌词不完全匹配就否定是歌
- 日语、英语歌词可能被误识别为中文谐音，如"息が止まるの"可能写成"息卡止まるの"
- 语气词、感叹词（啊、呀、啦、哦）可能缺失或被错误转写
- 同一首歌的歌词可能在不同段落重复出现，这是正常的（副歌重复）
- 歌手名字可能不完整或有误，需要结合歌词内容综合判断
- ASR 可能将快速说唱段落合并成一行，也可能将一句歌词拆成多行
- 标点符号基本缺失，需要根据语义判断断句

判断是否在唱歌的线索：
- 歌词有押韵、节奏感、重复结构
- 与前后文内容有明显切换（从说话变成唱歌）
- 同一段歌词反复出现（副歌）
- 歌词内容明显是某首已知歌曲

可以使用 search_lyrics 工具搜索歌词确认歌名，最多搜索2次，然后必须返回结果。
宁可返回"未知歌曲"也不要漏掉任何演唱片段。

输出要求：
- 只返回 JSON 数组，不要 Markdown，不要解释，不要代码块。
- 不要输出输入中不存在的 segment index。
- 示例：[{{"content_type": "song", "title": "歌曲名", "artist": "歌手名", "segment_indices": [12, 13, 14], "confidence": 0.86, "tags": [], "description": ""}}]

完整 ASR 转写片段：
{transcript_text}"""
    
    def get_tools(self, config: dict[str, Any]) -> list[dict[str, Any]] | None:
        if config.get("llm", {}).get("use_tools", True):
            return get_tools()
        return None
    
    def get_merge_gap(self, config: dict[str, Any]) -> float:
        padding_config = config.get("song", {}).get("padding", config.get("padding", {}))
        return float(padding_config.get("merge_gap_seconds", 20.0))
    
    def get_min_duration(self, config: dict[str, Any]) -> float:
        padding_config = config.get("song", {}).get("padding", config.get("padding", {}))
        return float(padding_config.get("min_song_seconds", 75.0))
