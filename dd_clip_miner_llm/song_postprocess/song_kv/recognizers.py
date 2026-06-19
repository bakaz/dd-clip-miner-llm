from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ...models import TranscriptSegment
from ...recognizers.base import BaseRecognizer


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class _KVRecognizer(BaseRecognizer):
    stage = "kv"
    transcript_include_timestamps = False

    @property
    def name(self) -> str:
        return "song"

    def task_instructions(self, config: dict[str, Any]) -> str:
        raise NotImplementedError

    def build_prompt(
        self,
        segments: list[TranscriptSegment],
        batch_start: int,
        config: dict[str, Any],
    ) -> str:
        instructions = self.task_instructions(config)
        if self.stage == "kv_discovery":
            expected_last_segment = batch_start + len(segments) - 1
            instructions = instructions.replace(
                "<输入最后一个segment index>",
                str(expected_last_segment),
            )
            instructions = (
                f"{instructions}\n\n"
                f"本次输入的最后一个 segment index 是 {expected_last_segment}。"
                "complete_through_segment 表示你已经检查过的最后一行输入索引，"
                "与该行是否为演唱无关。只有检查到该索引后才能设置 scan_complete=true；"
                "完整响应必须令 complete_through_segment 等于该索引。"
            )
        return (
            f"{instructions}\n\n"
            "只返回一个 JSON object，不要 Markdown、解释或代码块。\n\n"
            f"完整 ASR 转写片段：\n{self._format_transcript(segments, batch_start)}"
        )


class _PrecisionDiscoveryRecognizer(_KVRecognizer):
    stage = "kv_discovery"

    def task_instructions(self, config: dict[str, Any]) -> str:
        return """你是一个面向演唱会、直播和长视频的歌曲分段专家。

下面是一整段视频的 Whisper ASR 转写片段，每行格式为：
[序号] ASR文本

任务：按时间顺序完整扫描 ASR，找出所有可能是连续演唱的歌曲片段，返回一个纯 JSON object。

尽量覆盖同一首歌或一次演唱的完整片段，边界可以略宽，但不要跨到明显聊天、换歌、报幕或另一首歌。

Whisper ASR 可能存在错字、漏字、同音字替换、外语误识别、断句错误、重复切分等问题。判断时不要只依赖文本是否像标准歌词，而要结合上下文连续性、重复结构、押韵/节奏感、旋律化表达、外语段落、说唱结构、哼唱/拟声等线索综合判断。

识别原则：

1. 只要是在唱歌的段落都应识别出来，即使无法确定歌名。
2. 同一首歌的连续演唱段落必须合并成一个候选，不要把主歌、副歌、桥段拆成多首。
3. 不要把单句歌词、点歌讨论、歌曲介绍、聊天、感谢、报幕、直播结束口播单独当作歌曲候选。

此阶段不识别歌名、不搜索歌词。

协议：
{"candidates":[{"segment_ranges":[[241,271]],"confidence":0.82,"anchor_text":"我终于鼓起勇气"}],"scan_complete":true,"complete_through_segment":<输入最后一个segment index>}

字段限制：candidate 只能包含 segment_ranges、confidence、anchor_text。segment_ranges 起止均包含并且必须来自输入。没有候选时 candidates 为 []。

输出要求：
- 只返回一个 JSON object，不要 Markdown，不要解释，不要代码块。
"""


@dataclass
class _RecallAuditRecognizer(_KVRecognizer):
    targets: list[dict[str, Any]]
    stage = "kv_recall_audit"

    def task_instructions(self, config: dict[str, Any]) -> str:
        return f"""你负责歌曲分段 V3 的第二轮 Recall Audit。
第一轮已经确定的歌曲不能修改。你只检查下列未覆盖目标区间，寻找可能漏掉的演唱证据，并只返回短 evidence_ranges；不要推测整首歌边界，不要识别歌名，不要搜索歌词。

目标区间：{_compact_json(self.targets)}

普通聊天、感谢、报幕、歌曲讨论、点歌和单个感叹不能成为 anchor。没有证据的 target 不需要输出。

协议：
{{"anchors":[{{"target_id":"U003","evidence_ranges":[[655,660]],"confidence":0.76,"anchor_text":"执念的雨"}}],"audit_complete":true}}

每个 anchor 只能包含 target_id、evidence_ranges、confidence、anchor_text；target_id 必须来自目标区间，evidence_ranges 必须位于对应目标区间内。"""


@dataclass
class _SegmentationAdjudicationRecognizer(_KVRecognizer):
    candidates: list[dict[str, Any]]
    allow_final_discovery: bool
    stage = "kv_adjudication"

    def task_instructions(self, config: dict[str, Any]) -> str:
        additions = (
            "允许有限 additions，但每项必须给出精确 segment_ranges、evidence_ranges、至少两段歌词或约 10 秒连续演唱证据，并标记 final_discovery=true。"
            if self.allow_final_discovery
            else "additions 必须为空数组。"
        )
        return f"""你负责歌曲分段 V3 的第三轮 Segmentation Adjudication。
结合完整 ASR，统一裁决第一轮候选 P 和第二轮证据锚点 R。输入：{_compact_json(self.candidates)}

每个输入 ID 必须且只能被一个 decision 处理。action 只能是 accept、reject、adjust、split、merge：
- accept/adjust/split 通常处理一个 ID；merge 必须处理两个或更多 ID。
- reject 的 segment_ranges 必须为空。
- 其他 action 必须返回最终精确 segment_ranges，不跨过聊天、感谢、报幕或另一首歌。
- split 可返回多个互不重叠区间；merge 适用于以下情况：
  1. 两个或多个候选属于同一首歌（即使中间有聊天、间奏、互动）
  2. anchor_text 包含相似歌词、时间间隔合理（< 3 分钟且不是另一首歌）
{additions}
此阶段不识别歌名、不搜索歌词，分段完整性优先。

协议：
{{"decisions":[{{"candidate_ids":["P003","R002"],"action":"merge","segment_ranges":[[655,693]],"confidence":0.84}}],"additions":[],"adjudication_complete":true}}

decision 只能包含 candidate_ids、action、segment_ranges、confidence。addition 只能包含 segment_ranges、evidence_ranges、confidence、anchor_text、final_discovery。"""
