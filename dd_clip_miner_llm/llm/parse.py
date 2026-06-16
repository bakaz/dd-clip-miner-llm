"""响应解析模块

JSON 解析和 LLM 响应解析。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_llm_json(text: str) -> Any:
    """解析 LLM 响应为 JSON，兼容代码块和前后解释文字。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates_with_start: list[tuple[int, str]] = []
    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start != -1 and object_end > object_start:
        candidates_with_start.append((object_start, text[object_start:object_end + 1]))

    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end > array_start:
        candidates_with_start.append((array_start, text[array_start:array_end + 1]))

    for _, candidate in sorted(candidates_with_start, key=lambda item: item[0]):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def parse_llm_response(text: str) -> list[dict[str, Any]]:
    """解析 LLM 响应为 JSON 数组"""
    items, _ = parse_llm_response_with_status(text)
    return items


def parse_llm_response_with_status(text: str) -> tuple[list[dict[str, Any]], bool]:
    """解析 JSON 数组，并区分合法空数组与解析失败。"""
    result = parse_llm_json(text)
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)], True
    return [], False


def write_llm_debug(debug_dir: Path, batch_start: int, payload: dict[str, Any]) -> None:
    """写入 LLM 调试信息"""
    target = debug_dir / f"llm_batch_{batch_start:06d}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
