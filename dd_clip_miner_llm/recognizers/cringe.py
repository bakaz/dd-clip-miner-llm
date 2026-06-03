"""下头对话识别器"""
from __future__ import annotations

from typing import Any

from . import register
from .base import BaseRecognizer
from ..models import ContentMatch, TranscriptSegment


MAX_CRINGE_TITLE_CHARS = 19


@register
class CringeRecognizer(BaseRecognizer):
    """下头对话识别器
    
    识别直播/VTuber 中让人"下头"的内容。
    区分：主播主动说、主播回应、无明显下头事件。
    """
    
    @property
    def name(self) -> str:
        return "cringe"
    
    @property
    def default_config(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "min_duration": 5.0,
            "max_duration": 120.0,
            "min_confidence": 0.5,
            "merge_gap_seconds": 15.0,
        }
    
    def parse_response(
        self,
        items: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[ContentMatch]:
        """解析 LLM 响应，处理 severity 和 scenario 字段
        
        过滤掉 scenario=C（无明显下头事件）和 severity=0 的结果
        """
        matches = []
        for item in items:
            # 获取 severity 和 scenario
            severity = item.get("severity", 0)
            scenario = item.get("scenario", "C")
            description = item.get("description", "")
            title = _short_cringe_title(item.get("title", ""), description)
            if not title:
                continue
            
            # 过滤掉无明显下头事件
            if scenario == "C" or severity == 0:
                continue
            
            # 构建增强的描述
            enhanced_desc = f"[严重程度:{severity}/5][场景:{scenario}] {description}"
            
            match = ContentMatch(
                content_type="cringe",
                title=title,
                segment_indices=self._parse_segment_indices(item.get("segment_indices", [])),
                confidence=self._parse_confidence(item.get("confidence", 0.5)),
                tags=item.get("tags", []),
                description=enhanced_desc,
                artist="",
                lyrics_snippet="",
            )
            if match.segment_indices:
                matches.append(match)
        return matches
    
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
        
        type_config = config.get("cringe", {})
        min_dur = type_config.get("min_duration", self.default_config["min_duration"])
        max_dur = type_config.get("max_duration", self.default_config["max_duration"])

        return f"""你是一个直播切片内容分析助手。你的任务是阅读 Whisper 转写的 VTuber / 主播直播片段，判断其中是否出现"下头事件"。

这里的"下头"不是单指色情内容，而是指让人突然失去好感、感到尴尬、反感、不适、油腻、越界、冒犯、恶心或扫兴的发言/互动。

━━━ 请重点区分以下两种情况 ━━━

【A. 主播主动说了下头话】
主播本人说出令人不适、油腻、越界、擦边、性暗示、冒犯、歧视、攻击、情绪勒索、过度媚宅、恶俗玩笑等内容。

【B. 主播在回应下头话】
主播本身不一定有问题，而是在看到弹幕、SC、评论、连麦、游戏语音或其他人的发言后，表现出尴尬、无语、反感、制止、冷场、转移话题、提醒对方别说了等反应。

━━━ 注意事项 ━━━

- Whisper 转写可能有错别字、断句错误、语气词遗漏，请结合上下文判断。
- 不要因为出现"喜欢""老婆""亲亲""可爱"等词就直接判定为下头，要看是否越界、油腻、冒犯或让主播明显不适。
- 不要因为主播开玩笑、节目效果、角色扮演就过度判定。
- 如果主播是在批评、制止或反应别人的不当发言，不要把责任归到主播身上。

━━━ 严重程度（severity: 1-5）━━━

1 = 轻微尴尬或轻微扫兴
2 = 明显油腻、尴尬、让人不舒服
3 = 明显越界、擦边、恶俗、冒犯
4 = 严重冒犯、骚扰、攻击、恶意引导
5 = 极严重，涉及明显骚扰、仇恨、隐私侵犯、开盒、严重恶意攻击等

━━━━━━━━━━━━━━━━━━━━━━━━━━

每个对象必须包含：
- content_type: "cringe"
- title: 这段下头发言/互动的总结，用作输出文件标题；必须是中文短句，少于20个中文字，不要照抄原话，不要加标点。
- segment_indices: 下头内容的 ASR 段落序号数组（只包含真正下头的段落，不要包含前后正常内容）
- confidence: 0-1 置信度
- severity: 严重程度（1-5）
- scenario: 情况分类，"A"=主播主动说，"B"=主播在回应
- tags: 标签数组，可选值：["性骚扰", "油腻发言", "恶意攻击", "情绪勒索", "价值观问题", "擦边", "恶俗玩笑", "歧视", "其他"]
- description: 详细描述（发生了什么，为什么下头，20字以内）

输出要求：
- 只返回 JSON 数组，不要 Markdown，不要解释，不要代码块。
- 不要输出输入中不存在的 segment index。
- segment_indices 只包含真正下头的段落序号，例如：如果只有第4段下头，就返回 [4]，不要返回 [4, 22]
- title 会直接用于音视频文件名，必须总结核心问题且少于20个中文字。
- 示例：
  [
    {{"content_type": "cringe", "title": "观众开黄腔", "segment_indices": [4], "confidence": 0.9, "severity": 3, "scenario": "B", "tags": ["性骚扰"], "description": "观众对主播开黄腔"}},
    {{"content_type": "cringe", "title": "主播油腻发言", "segment_indices": [15, 16], "confidence": 0.8, "severity": 2, "scenario": "A", "tags": ["油腻发言"], "description": "主播连续说了两句油腻的话"}}
  ]

完整 ASR 转写片段：
{transcript_text}"""


def _short_cringe_title(title: Any, description: Any = "") -> str:
    text = str(title or "").strip()
    if not text:
        text = str(description or "").strip()
    text = " ".join(text.split())
    text = text.strip(" ，。！？；：,.!?;:\"'“”‘’《》【】[]()（）")
    if len(text) > MAX_CRINGE_TITLE_CHARS:
        text = text[:MAX_CRINGE_TITLE_CHARS].rstrip(" ，。！？；：,.!?;:")
    return text
