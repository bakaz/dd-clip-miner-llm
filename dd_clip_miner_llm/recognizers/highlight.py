"""高能时刻识别器"""
from __future__ import annotations

from typing import Any

from . import register
from .base import BaseRecognizer
from ..models import TranscriptSegment


@register
class HighlightRecognizer(BaseRecognizer):
    """高能时刻识别器
    
    识别直播/视频中的高能时刻，如：
    - 气氛高涨的时刻
    - 观众反应激烈的时刻
    - 精彩表演片段
    - 重要转折点
    """
    
    @property
    def name(self) -> str:
        return "highlight"
    
    @property
    def default_config(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "min_duration": 5.0,
            "max_duration": 120.0,
            "min_confidence": 0.6,
            "merge_gap_seconds": 15.0,
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
        
        type_config = config.get("highlight", {})
        min_dur = type_config.get("min_duration", self.default_config["min_duration"])
        max_dur = type_config.get("max_duration", self.default_config["max_duration"])

        return f"""你是一个直播/视频内容分析专家。
下面是 Whisper ASR 转写片段，每行格式为 [序号] (开始秒-结束秒) 文本。

任务：从完整上下文中识别所有**高能时刻**，返回纯 JSON 数组。

什么是高能时刻：
- 气氛突然高涨、情绪激动的时刻
- 观众弹幕/评论爆发的时刻（可通过主播回应推断）
- 精彩的表演、才艺展示
- 出人意料的转折或惊喜
- 引发强烈反响的内容（感动、震惊、兴奋）

每个对象必须包含：
- content_type: "highlight"
- title: 高能时刻标题（简短概括，10字以内）
- segment_indices: ASR 段落序号数组
- confidence: 0-1 置信度
- tags: 标签数组，可选值：["气氛高涨", "精彩表演", "意外转折", "观众互动", "感动时刻", "爆笑时刻"]
- description: 详细描述（为什么是高能，20字以内）

识别原则：
1. 关注**氛围和情绪**的变化
2. 前后文要有明显的氛围转变（从平淡到高涨）
3. 片段时长建议 {min_dur}秒-{max_dur}秒
4. 宁可多选不要漏选
5. 单纯的唱歌不算高能（除非有特殊互动）

Whisper ASR 特性：
- 可能有错字、漏字
- 语气词、感叹词可能缺失
- 需要结合上下文理解情绪变化

输出要求：
- 只返回 JSON 数组，不要 Markdown，不要解释，不要代码块。
- 不要输出输入中不存在的 segment index。
- 示例：[{{"content_type": "highlight", "title": "主播收到大额礼物激动", "segment_indices": [23,24,25], "confidence": 0.9, "tags": ["气氛高涨", "观众互动"], "description": "主播收到豪华礼物后情绪激动"}}]

完整 ASR 转写片段：
{transcript_text}"""
