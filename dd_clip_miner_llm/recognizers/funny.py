"""搞笑片段识别器"""
from __future__ import annotations

from typing import Any

from . import register
from .base import BaseRecognizer
from ..models import TranscriptSegment


@register
class FunnyRecognizer(BaseRecognizer):
    """搞笑片段识别器
    
    识别直播/视频中的搞笑内容，如：
    - 幽默对话
    - 搞笑互动
    - 意外失误（节目效果）
    - 梗和笑点
    """
    
    @property
    def name(self) -> str:
        return "funny"
    
    @property
    def default_config(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "min_duration": 5.0,
            "max_duration": 180.0,
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
        
        type_config = config.get("funny", {})
        min_dur = type_config.get("min_duration", self.default_config["min_duration"])
        max_dur = type_config.get("max_duration", self.default_config["max_duration"])

        return f"""你是一个直播/视频内容分析专家，擅长识别幽默搞笑内容。
下面是 Whisper ASR 转写片段，每行格式为 [序号] (开始秒-结束秒) 文本。

任务：从完整上下文中识别所有**搞笑片段**，返回纯 JSON 数组。

什么是搞笑片段：
- 幽默的对话和吐槽
- 主播和观众的搞笑互动
- 意外失误造成的节目效果
- 梗和笑点
- 让人发笑的内容
- 调侃、自嘲、反差萌

每个对象必须包含：
- content_type: "funny"
- title: 搞笑片段标题（简短概括，10字以内）
- segment_indices: ASR 段落序号数组
- confidence: 0-1 置信度
- tags: 标签数组，可选值：["幽默吐槽", "搞笑互动", "意外失误", "梗和笑点", "反差萌", "自嘲"]
- description: 详细描述（为什么搞笑，20字以内）

识别原则：
1. 关注**幽默和笑点**
2. 前后文要有一定的上下文（笑点需要铺垫）
3. 片段时长建议 {min_dur}秒-{max_dur}秒
4. 宁可多选不要漏选
5. 单纯的说话不算搞笑，需要有明显的笑点

Whisper ASR 特性：
- 可能有错字、漏字
- 语气词可能缺失
- 需要结合上下文理解笑点

输出要求：
- 只返回 JSON 数组，不要 Markdown，不要解释，不要代码块。
- 不要输出输入中不存在的 segment index。
- 示例：[{{"content_type": "funny", "title": "主播口误引发爆笑", "segment_indices": [34,35,36], "confidence": 0.85, "tags": ["意外失误", "梗和笑点"], "description": "主播口误后自嘲引发弹幕爆笑"}}]

完整 ASR 转写片段：
{transcript_text}"""
