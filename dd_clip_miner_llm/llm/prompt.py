"""Prompt 构建模块

构建发送给 LLM 的消息列表。
"""
from __future__ import annotations

from typing import Any

from ..config import get_llm_config
from ..llm_debug import _extract_task_instructions, _format_transcript_for_cache
from ..models import TranscriptSegment
from ..recognizers.base import BaseRecognizer


_CACHE_SYSTEM_PROMPT = (
    "你将先收到一份带全局序号和时间范围的 ASR 转写，再收到具体分析任务。"
    "必须只依据该转写完成任务，不得使用输入中不存在的 segment index。"
)


def build_llm_messages(
    recognizer: BaseRecognizer,
    segments: list[TranscriptSegment],
    batch_start: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """构建请求消息；缓存友好模式把可复用 ASR 长文本放在任务指令之前。"""
    prompt = recognizer.build_prompt(segments, batch_start, config)
    llm_config = get_llm_config(config)
    if not llm_config.get("cache_friendly_prompt_layout", False):
        system_prompt = recognizer.build_system_prompt(config)
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    instructions = _extract_task_instructions(prompt)
    if not instructions:
        return [{"role": "user", "content": prompt}]

    recognizer_system_prompt = recognizer.build_system_prompt(config)
    if recognizer_system_prompt:
        instructions = f"{recognizer_system_prompt}\n\n{instructions}"

    transcript = _format_transcript_for_cache(segments, batch_start, recognizer)
    return [
        {"role": "system", "content": _CACHE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"ASR 转写开始：\n{transcript}\nASR 转写结束。\n\n"
                f"{instructions}\n\n请基于上面的完整 ASR 转写执行任务。"
            ),
        },
    ]
