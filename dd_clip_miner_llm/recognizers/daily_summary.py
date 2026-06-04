"""Daily livestream summary recognizer."""
from __future__ import annotations

from typing import Any

from . import register
from .base import BaseRecognizer
from ..models import TranscriptSegment
from ..report import _format_timecode


@register
class DailySummaryRecognizer(BaseRecognizer):
    """Build a structured three-level pyramid summary from ASR text."""

    @property
    def name(self) -> str:
        return "daily_summary"

    @property
    def default_config(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "summary_only": True,
            "language": "zh-CN",
            "title": "当天直播内容总结",
            "max_level1_items": 6,
            "max_level2_per_level1": 5,
            "max_level3_per_level2": 4,
            "include_timeline": True,
            "include_quotes": True,
            "include_open_questions": True,
            "max_segment_indices_per_item": 5,
            "max_timeline_items": 8,
            "max_quote_chars": 40,
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
            start_tc = _format_timecode(seg.start)
            end_tc = _format_timecode(seg.end)
            lines.append(f"[{idx}] ({start_tc}-{end_tc}) {seg.text}")

        transcript_text = "\n".join(lines)
        type_config = config.get("daily_summary", {})
        title = type_config.get("title", self.default_config["title"])
        language = type_config.get("language", self.default_config["language"])
        max_l1 = int(type_config.get("max_level1_items", self.default_config["max_level1_items"]))
        max_l2 = int(type_config.get("max_level2_per_level1", self.default_config["max_level2_per_level1"]))
        max_l3 = int(type_config.get("max_level3_per_level2", self.default_config["max_level3_per_level2"]))
        include_timeline = bool(type_config.get("include_timeline", True))
        include_quotes = bool(type_config.get("include_quotes", True))
        include_open_questions = bool(type_config.get("include_open_questions", True))
        max_indices = int(type_config.get(
            "max_segment_indices_per_item",
            self.default_config["max_segment_indices_per_item"],
        ))
        max_timeline = int(type_config.get(
            "max_timeline_items",
            self.default_config["max_timeline_items"],
        ))
        max_quote_chars = int(type_config.get(
            "max_quote_chars",
            self.default_config["max_quote_chars"],
        ))

        return f"""你是一个直播内容分析师。请基于下面的 Whisper ASR 转写，总结当天直播内容。

任务目标：
1. 使用“金字塔结构”做总分式梳理，必须分成三层。
2. 第一层是全场级别的总览/主结论，回答“今天主要讲了什么”。
3. 第二层是支撑主结论的主题、事件或讨论模块。
4. 第三层是每个模块下面的具体事实、例子、观点、互动或证据。
5. 只基于 ASR 内容总结，不要编造未出现的信息。ASR 可能有错字，要结合上下文温和纠错。
6. 保留关键时间点和 segment index，方便回看原直播。

输出语言：{language}
标题：{title}
数量约束：
- level_1 最多 {max_l1} 个
- 每个 level_1 下 level_2 最多 {max_l2} 个
- 每个 level_2 下 level_3 最多 {max_l3} 个
- timeline: {"需要" if include_timeline else "不需要"}
- evidence_quotes: {"需要" if include_quotes else "不需要"}
- open_questions: {"需要" if include_open_questions else "不需要"}
- 所有 segment_indices 数组最多 {max_indices} 个代表性编号；严禁列出连续长数组，长范围只保留起点/中点/终点等代表编号。
- timeline 最多 {max_timeline} 条，只保留关键阶段。
- evidence_quote 最多 {max_quote_chars} 个中文字符；不要复制长段 ASR。
- 输出必须紧凑，目标总长度小于 6000 tokens；优先总结，不要堆砌索引。

输出必须是纯 JSON object，不要 Markdown，不要代码块，不要解释。结构必须严格如下：
{{
  "content_type": "daily_summary",
  "title": "{title}",
  "one_sentence_summary": "一句话总结整场直播",
  "overall": {{
    "summary": "全场总览，先总后分，100-200字",
    "main_topics": ["主题1", "主题2"],
    "tone": "直播整体氛围",
    "audience_interaction": "观众互动概况"
  }},
  "level_1": [
    {{
      "title": "一级总览标题",
      "summary": "这个一级主题的总述",
      "time_range": {{
        "start_segment": 0,
        "end_segment": 10,
        "start_time": "00:00:00",
        "end_time": "00:10:00"
      }},
      "level_2": [
        {{
          "title": "二级分论点标题",
          "summary": "二级分论点说明",
          "segment_indices": [0, 1, 2],
          "level_3": [
            {{
              "point": "三级具体事实/例子/证据",
              "segment_indices": [0],
              "evidence_quote": "ASR 中能支撑该点的短句；没有就留空",
              "importance": "high|medium|low"
            }}
          ]
        }}
      ]
    }}
  ],
  "timeline": [
    {{
      "time": "00:00:00",
      "title": "阶段标题",
      "summary": "这一阶段发生了什么",
      "segment_indices": [0, 1]
    }}
  ],
  "key_takeaways": ["最重要结论1", "最重要结论2"],
  "open_questions": ["仍不明确或需要回看的问题"],
  "tags": ["主题标签"]
}}

ASR 转写：
{transcript_text}"""

    def format_summary_markdown(self, summary: dict[str, Any], config: dict[str, Any]) -> str:
        title = str(summary.get("title") or config.get("daily_summary", {}).get("title") or "当天直播内容总结")
        lines = [f"# {title}", ""]

        one_sentence = str(summary.get("one_sentence_summary") or "").strip()
        if one_sentence:
            lines.extend([one_sentence, ""])

        overall = summary.get("overall") if isinstance(summary.get("overall"), dict) else {}
        if overall:
            lines.extend(["## 总览", ""])
            summary_text = str(overall.get("summary") or "").strip()
            if summary_text:
                lines.extend([summary_text, ""])
            for key, label in (
                ("main_topics", "主要主题"),
                ("tone", "整体氛围"),
                ("audience_interaction", "观众互动"),
            ):
                value = overall.get(key)
                if isinstance(value, list):
                    value_text = "、".join(str(item) for item in value if str(item).strip())
                else:
                    value_text = str(value or "").strip()
                if value_text:
                    lines.append(f"- {label}: {value_text}")
            lines.append("")

        level_1 = summary.get("level_1") if isinstance(summary.get("level_1"), list) else []
        if level_1:
            lines.extend(["## 三层金字塔", ""])
            for i, item in enumerate(level_1, start=1):
                if not isinstance(item, dict):
                    continue
                lines.append(f"### {i}. {item.get('title', '未命名主题')}")
                item_summary = str(item.get("summary") or "").strip()
                if item_summary:
                    lines.extend(["", item_summary])
                time_range = item.get("time_range") if isinstance(item.get("time_range"), dict) else {}
                start_time = str(time_range.get("start_time") or "").strip()
                end_time = str(time_range.get("end_time") or "").strip()
                if start_time or end_time:
                    lines.append("")
                    lines.append(f"时间范围: {start_time or '?'} - {end_time or '?'}")
                lines.append("")

                level_2 = item.get("level_2") if isinstance(item.get("level_2"), list) else []
                for j, subitem in enumerate(level_2, start=1):
                    if not isinstance(subitem, dict):
                        continue
                    lines.append(f"**{i}.{j} {subitem.get('title', '未命名分论点')}**")
                    sub_summary = str(subitem.get("summary") or "").strip()
                    if sub_summary:
                        lines.append(sub_summary)
                    level_3 = subitem.get("level_3") if isinstance(subitem.get("level_3"), list) else []
                    for k, detail in enumerate(level_3, start=1):
                        if not isinstance(detail, dict):
                            continue
                        point = str(detail.get("point") or "").strip()
                        quote = str(detail.get("evidence_quote") or "").strip()
                        indices = detail.get("segment_indices")
                        indices_text = _format_indices(indices)
                        suffix = f" {indices_text}" if indices_text else ""
                        if point:
                            lines.append(f"- {i}.{j}.{k} {point}{suffix}")
                        if quote:
                            lines.append(f"  证据: {quote}")
                    lines.append("")

        timeline = summary.get("timeline") if isinstance(summary.get("timeline"), list) else []
        if timeline:
            lines.extend(["## 时间线", ""])
            for item in timeline:
                if not isinstance(item, dict):
                    continue
                time = str(item.get("time") or "").strip()
                title_text = str(item.get("title") or "").strip()
                body = str(item.get("summary") or "").strip()
                label = f"{time} {title_text}".strip()
                if label:
                    lines.append(f"- {label}: {body}" if body else f"- {label}")
            lines.append("")

        for key, label in (("key_takeaways", "关键结论"), ("open_questions", "待回看问题"), ("tags", "标签")):
            values = summary.get(key)
            if isinstance(values, list) and values:
                lines.extend([f"## {label}", ""])
                for value in values:
                    value_text = str(value).strip()
                    if value_text:
                        lines.append(f"- {value_text}")
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"


def _format_indices(indices: Any) -> str:
    if not isinstance(indices, list):
        return ""
    clean = []
    for value in indices:
        try:
            clean.append(str(int(value)))
        except (TypeError, ValueError):
            continue
    return f"(segments: {', '.join(clean)})" if clean else ""
