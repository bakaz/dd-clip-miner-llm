"""对话识别器"""
from __future__ import annotations

from typing import Any

from . import register
from .base import BaseRecognizer
from ..models import TranscriptSegment


@register
class DialogueRecognizer(BaseRecognizer):
    """有趣对话片段识别器"""
    
    @property
    def name(self) -> str:
        return "dialogue"
    
    @property
    def default_config(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "min_duration": 10.0,
            "max_duration": 300.0,
            "min_confidence": 0.6,
            "merge_gap_seconds": 10.0,
            "tags": ["搞笑", "吐槽", "名场面", "金句", "互动", "高能"],
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
        
        # 获取配置的标签
        type_config = config.get("dialogue", {})
        tags = type_config.get("tags", self.default_config["tags"])
        tags_str = "、".join(tags)

        return f"""你是一个直播/视频内容分析专家。
下面是 Whisper ASR 转写片段，每行格式为 [序号] (开始秒-结束秒) 文本。

任务：从完整上下文中识别所有**有趣的对话片段**，返回纯 JSON 数组。

识别类型（tags）：
{chr(10).join(f"- {tag}" for tag in tags)}

每个对象必须包含：
- content_type: "dialogue"
- title: 片段标题（简短概括，10字以内）
- segment_indices: ASR 段落序号数组
- confidence: 0-1 置信度
- tags: 标签数组，从上述类型中选择
- description: 详细描述（为什么有趣，20字以内）

识别原则：
1. 关注**对话内容**而非歌曲
2. 前后文要有一定连贯性，是一个完整的对话或互动
3. 片段时长建议 {type_config.get('min_duration', 10)}秒-{type_config.get('max_duration', 300)}秒
4. 宁可多选不要漏选
5. 单纯的问候、感谢、报幕不算有趣对话

Whisper ASR 特性：
- 可能有错字、漏字
- 说话内容可能不完整
- 需要结合上下文理解语义

输出要求：
- 只返回 JSON 数组，不要 Markdown，不要解释，不要代码块。
- 不要输出输入中不存在的 segment index。
- 示例：[{{"content_type": "dialogue", "title": "主播回应弹幕质疑", "segment_indices": [45,46,47], "confidence": 0.85, "tags": ["互动", "高能"], "description": "主播犀利回应观众质疑"}}]

完整 ASR 转写片段：
{transcript_text}"""
