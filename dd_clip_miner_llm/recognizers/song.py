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
                "adaptive_silence_padding": True,
                "adaptive_silence_gap_threshold_seconds": 25.0,
                "adaptive_silence_gap_ratio": 0.95,
                "adaptive_max_before_seconds": 45.0,
                "adaptive_max_after_seconds": 45.0,
                "min_song_seconds": 75.0,
                "max_song_seconds": 360.0,
                "merge_gap_seconds": 20.0,
            },
            "missed_recheck": {
                "enabled": True,
                "strategy": "windowed",
                "fallback_strategy": "windowed_on_structural_failure",
                "batch_size": 500,
                "min_gap_segments": 1,
                "context_segments": 10,
                "max_completion_tokens": 4096,
                "max_tool_rounds": 1,
                "adaptive": {
                    "mode": "cost_estimate",
                    "full_transcript_max_segments": 3500,
                    "windowed_min_target_ranges": 19,
                },
            },
            "review": {
                "enabled": False,
                "transcript_scope": "local",
                "context_segments": 10,
                "max_window_segments": 500,
                "nearby_title_conflict_gap_segments": 2,
                "max_completion_tokens": 4096,
                "max_tool_rounds": 1,
                "fallback": "local_best",
                "adaptive": {
                    "mode": "cost_estimate",
                    "local_max_clusters": 3,
                    "full_min_clusters": 6,
                    "full_min_segments": 2000,
                },
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
        compact_ranges = bool(
            config.get("llm", {}).get("cache_friendly_prompt_layout", False)
            or config.get("llm", {}).get("compact_segment_ranges", False)
        )
        if compact_ranges:
            segment_field = """- segment_ranges: 属于同一首歌的连续 ASR 序号区间，格式为 [[开始,结束], ...]，起止均包含。单段写成 [[12,12]]。必须只使用输入中出现的序号。
- 不要输出 segment_indices。区间必须精确，不得包含中间的聊天、感谢、报幕或口播。不同歌曲的区间严禁重叠。"""
            coverage_instruction = """
完整性检查（必须执行）：
- 在生成 JSON 前，从第一个 segment 到最后一个 segment 按时间顺序检查一遍，先找出全部演唱区间，再识别歌名。
- 搜索工具只用于确认歌名，不能因为未搜索、搜索失败或无法确认歌名而删除演唱区间。
- 外语谐音、ASR 乱码、只能听出零碎歌词的演唱也必须输出；无法命名时使用"未知歌曲："。
- 输出歌曲数量不设上限。最终数组必须覆盖你判断为演唱的每一个连续区间，不能只返回能确认歌名的歌曲。
"""
            output_example = (
                '[{"content_type": "song", "title": "歌曲名", "artist": "歌手名", '
                '"segment_ranges": [[12, 26], [30, 35]], "confidence": 0.86, '
                '"tags": [], "description": ""}]'
            )
        else:
            coverage_instruction = ""
            segment_field = (
                "- segment_indices: 属于同一首歌的 ASR 段落序号数组，"
                "必须只使用输入中出现的序号，按升序排列。"
            )
            output_example = (
                '[{"content_type": "song", "title": "歌曲名", "artist": "歌手名", '
                '"segment_indices": [12, 13, 14], "confidence": 0.86, '
                '"tags": [], "description": ""}]'
            )

        return f"""你是一个面向演唱会、直播和长视频的歌曲识别专家。
下面是一整段视频的 Whisper ASR 转写片段，每行格式为 [序号] (开始秒-结束秒) 文本。

任务：从完整上下文中识别所有演唱片段，返回纯 JSON 数组。
每个对象必须包含以下字段：
- content_type: "song"
- title: 歌名。能识别出原曲时填写准确歌名；无法确认时填写"未知歌曲："加最有代表性的一句歌词。
- artist: 原唱或演唱者。无法判断时填空字符串。
{segment_field}
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
{coverage_instruction}

输出要求：
- 只返回 JSON 数组，不要 Markdown，不要解释，不要代码块。
- 不要输出输入中不存在的 segment index。
- 示例：{output_example}

完整 ASR 转写片段：
{transcript_text}"""

    def parse_response(
        self,
        items: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[ContentMatch]:
        normalized = []
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized_item = dict(item)
            if "segment_ranges" in normalized_item:
                normalized_item["segment_indices"] = _expand_segment_ranges(
                    normalized_item.get("segment_ranges")
                )
            normalized.append(normalized_item)
        return super().parse_response(normalized, config)
    
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


def _expand_segment_ranges(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []

    indices: list[int] = []
    seen: set[int] = set()
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            continue
        try:
            start = int(item[0])
            end = int(item[1])
        except (TypeError, ValueError):
            continue
        if start < 0 or end < start:
            continue
        for index in range(start, end + 1):
            if index not in seen:
                indices.append(index)
                seen.add(index)
    return indices
