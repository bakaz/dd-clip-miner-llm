"""JSON 修复和 Reasoning Followup 模块"""
from __future__ import annotations

import json
from typing import Any

from .parse import parse_llm_json, parse_llm_response_with_status
from .provider import LLMProvider
from .transport import call_llm
from ..config import get_llm_config
from ..llm_debug import _record_usage, llm_response_debug
from ..profile_state import _fingerprint_payload


def _extract_complete_json_objects(text: str) -> list[dict[str, Any]]:
    """Extract complete top-level objects from a possibly truncated JSON array."""
    objects: list[dict[str, Any]] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    value = json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    value = None
                if isinstance(value, dict):
                    objects.append(value)
                start = None
    return objects


def _continuation_item_key(item: dict[str, Any]) -> str:
    if item.get("content_type") == "scan_checkpoint":
        return f"checkpoint:{item.get('scan_id', '')}"
    ranges = item.get("segment_ranges")
    if not isinstance(ranges, list):
        ranges = item.get("segment_indices")
    return _fingerprint_payload({
        "content_type": item.get("content_type"),
        "scan_id": item.get("scan_id"),
        "title": item.get("title"),
        "ranges": ranges,
    })


def _merge_continuation_items(
    target: list[dict[str, Any]],
    seen: set[str],
    content: str,
) -> bool:
    items, valid = parse_llm_response_with_status(content)
    if not valid:
        items = _extract_complete_json_objects(content)
    for item in items:
        key = _continuation_item_key(item)
        if key in seen:
            continue
        seen.add(key)
        target.append(item)
    return valid


def _continue_truncated_json_array(
    client: Any,
    provider: LLMProvider,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    content: str,
    finish_reason: str | None,
    batch_debug: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int | None = None,
) -> str:
    """Continue a truncated JSON array while preserving the original request prefix."""
    llm_config = get_llm_config(config)
    if finish_reason != "length" or not llm_config.get("continuation_on_length", False):
        return content

    max_rounds = max(0, int(llm_config.get("max_continuation_rounds", 0) or 0))
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    _merge_continuation_items(merged, seen, content)
    current_reason = finish_reason
    continuation_debug = batch_debug.setdefault("continuation_rounds", [])

    for round_index in range(max_rounds):
        checkpoints = [
            str(item.get("scan_id"))
            for item in merged
            if item.get("content_type") == "scan_checkpoint" and item.get("scan_id")
        ]
        completed = [
            {
                "scan_id": item.get("scan_id"),
                "title": item.get("title"),
                "segment_ranges": item.get("segment_ranges"),
                "segment_indices": item.get("segment_indices"),
            }
            for item in merged
            if item.get("content_type") != "scan_checkpoint"
        ]
        continuation_prompt = (
            "上一轮 JSON 数组因为输出长度限制而截断。请继续完成同一个任务，只返回尚未输出的对象，"
            "不要重复下列已完成对象。仍然只返回纯 JSON 数组。"
            f"\n已完成 scan checkpoint: {json.dumps(checkpoints, ensure_ascii=False, separators=(',', ':'))}"
            f"\n已完成对象: {json.dumps(completed[-200:], ensure_ascii=False, separators=(',', ':'))}"
        )
        response = call_llm(
            client, provider,
            [*messages, {"role": "user", "content": continuation_prompt}],
            tools=tools,
            tool_choice="none" if tools else None,
            max_tokens_override=max_tokens,
        )
        debug = llm_response_debug(response)
        _record_usage(batch_debug, "continuation", debug, round=round_index + 1)
        continuation_content = debug["content"] or debug["reasoning_content"]
        valid = _merge_continuation_items(merged, seen, continuation_content)
        current_reason = debug["finish_reason"]
        continuation_debug.append({
            "round": round_index + 1,
            "finish_reason": current_reason,
            "parse_valid": valid,
            "content": continuation_content,
            "usage": debug["usage"],
        })
        if current_reason != "length" and valid:
            batch_debug["continuation_complete"] = True
            batch_debug["finish_reason"] = current_reason
            return json.dumps(merged, ensure_ascii=False, separators=(",", ":"))

    batch_debug["scan_incomplete"] = True
    batch_debug["continuation_complete"] = False
    batch_debug["finish_reason"] = current_reason
    if merged:
        return json.dumps(merged, ensure_ascii=False, separators=(",", ":"))
    return content


def _continue_truncated_json_object(
    client: Any,
    provider: LLMProvider,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    content: str,
    finish_reason: str | None,
    batch_debug: dict[str, Any],
    *,
    max_tokens: int | None = None,
) -> str:
    """Continue a truncated JSON object by appending raw continuation text."""
    llm_config = get_llm_config(config)
    if finish_reason != "length" or not llm_config.get("continuation_on_length", False):
        return content

    parsed = parse_llm_json(content)
    if isinstance(parsed, dict) and parsed:
        return content

    max_rounds = max(0, int(llm_config.get("max_continuation_rounds", 0) or 0))
    current = content
    current_reason = finish_reason
    continuation_debug = batch_debug.setdefault("structured_continuation_rounds", [])

    for round_index in range(max_rounds):
        continuation_prompt = (
            "The previous assistant message is a single JSON object that was cut off. "
            "Continue from the exact next character after the previous message. "
            "Do not repeat any earlier text. Do not wrap in Markdown. "
            "Do not explain. Output only the remaining JSON characters needed to "
            "complete that same object."
        )
        response = call_llm(
            client,
            provider,
            [
                *messages,
                {"role": "assistant", "content": current},
                {"role": "user", "content": continuation_prompt},
            ],
            max_tokens_override=max_tokens,
        )
        debug = llm_response_debug(response)
        _record_usage(batch_debug, "structured_continuation", debug, round=round_index + 1)
        continuation_content = debug["content"] or debug["reasoning_content"]
        current_reason = debug["finish_reason"]

        if continuation_content.strip():
            continuation_parsed = parse_llm_json(continuation_content)
            if isinstance(continuation_parsed, dict) and continuation_parsed:
                current = continuation_content
            else:
                current = current + continuation_content

        parsed = parse_llm_json(current)
        parse_valid = isinstance(parsed, dict) and bool(parsed)
        continuation_debug.append({
            "round": round_index + 1,
            "finish_reason": current_reason,
            "parse_valid": parse_valid,
            "content": continuation_content[:500],
            "usage": debug["usage"],
        })
        if parse_valid and current_reason != "length":
            batch_debug["structured_continuation_complete"] = True
            batch_debug["finish_reason"] = current_reason
            return current

        if not continuation_content.strip() and current_reason != "length":
            break

    batch_debug["structured_continuation_complete"] = False
    batch_debug["finish_reason"] = current_reason
    return current


def build_reasoning_followup_prompt(reasoning_content: str, partial_content: str = "") -> str:
    """构建 reasoning followup 提示词"""
    partial_block = (
        f"\n\n上一轮已经生成但可能被截断或格式不完整的内容：\n{partial_content}"
        if partial_content.strip()
        else ""
    )
    return f"""下面是上一轮模型对内容识别任务的分析内容。它可能是不完整的，但里面已经包含了内容边界判断。

不要继续分析，不要解释，不要输出思考过程。请只把分析中已经确定的内容整理成 JSON 数组。

输出必须是纯 JSON 数组，不要 Markdown，不要代码块，不要额外文字。

上一轮分析内容：
{reasoning_content}{partial_block}"""


def reasoning_followup_settings(config: dict[str, Any]) -> tuple[bool, int, int | None]:
    """获取 reasoning followup 配置"""
    llm_config = config["llm"]
    enabled = bool(llm_config.get("retry_empty_with_reasoning", True))
    rounds = int(llm_config.get("reasoning_followup_rounds", 2))
    tokens_value = llm_config.get("reasoning_followup_max_tokens", 8192)
    tokens = int(tokens_value) if tokens_value not in (None, "") else None
    return enabled, max(0, rounds), tokens


def run_reasoning_followups(
    client: Any,
    provider: LLMProvider,
    config: dict[str, Any],
    reasoning_content: str,
    partial_content: str,
    batch_debug: dict[str, Any],
) -> str:
    """运行 reasoning followup 轮次"""
    retry_reasoning, followup_rounds, followup_tokens = reasoning_followup_settings(config)
    if not retry_reasoning:
        return ""

    content = ""
    material = reasoning_content
    partial = partial_content
    for _ in range(followup_rounds):
        if not material.strip() and not partial.strip():
            break

        followup_prompt = build_reasoning_followup_prompt(material, partial)
        try:
            followup_response = call_llm(
                client, provider,
                [{"role": "user", "content": followup_prompt}],
                max_tokens_override=followup_tokens,
            )
            followup_debug = llm_response_debug(followup_response)
            _record_usage(
                batch_debug, "reasoning_followup", followup_debug,
                round=len(batch_debug["reasoning_followups"]) + 1,
            )
            content = followup_debug["content"]
            content = _continue_truncated_json_array(
                client, provider, config,
                [{"role": "user", "content": followup_prompt}],
                content or followup_debug["reasoning_content"],
                followup_debug["finish_reason"],
                batch_debug, max_tokens=followup_tokens,
            )
            batch_debug["reasoning_followups"].append({
                "round": len(batch_debug["reasoning_followups"]) + 1,
                "content": content[:500],
                "reasoning_content": followup_debug["reasoning_content"][:500],
                "usage": followup_debug["usage"],
            })
            batch_debug["raw_response"] = content
        except Exception as exc:
            batch_debug["reasoning_followups"].append({
                "round": len(batch_debug["reasoning_followups"]) + 1,
                "error": str(exc),
            })
            return ""

        _, is_valid_array = parse_llm_response_with_status(content)
        if content.strip() and is_valid_array:
            return content

        material = str(followup_debug.get("reasoning_content") or "")
        partial = content

    return content


def fix_json_with_llm(
    client: Any,
    provider: LLMProvider,
    config: dict[str, Any],
    raw_content: str,
    content_type: str,
    batch_debug: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """当 LLM 返回非 JSON 时，让它把内容转换成 JSON 格式"""
    max_rounds = int(config["llm"].get("json_fix_rounds", 3))
    if max_rounds <= 0:
        return [], raw_content

    fix_prompt = f"""下面是之前对{content_type}识别任务的回复，但它不是纯JSON格式。
请把其中的信息提取出来，转换成纯JSON数组。

输出必须是纯JSON数组，不要Markdown，不要代码块，不要额外文字。

之前的回复：
{raw_content}"""

    content = raw_content
    for round_num in range(max_rounds):
        try:
            response = call_llm(
                client, provider,
                [{"role": "user", "content": fix_prompt}],
                max_tokens_override=(
                    provider.max_completion_tokens
                    if provider.max_completion_tokens is not None
                    else provider.max_tokens
                ),
            )
            debug = llm_response_debug(response)
            _record_usage(batch_debug, "json_fix", debug, round=round_num + 1)
            new_content = debug["content"] or debug["reasoning_content"]
            new_content = _continue_truncated_json_array(
                client, provider, config,
                [{"role": "user", "content": fix_prompt}],
                new_content, debug["finish_reason"], batch_debug,
                max_tokens=(
                    provider.max_completion_tokens
                    if provider.max_completion_tokens is not None
                    else provider.max_tokens
                ),
            )
            batch_debug.setdefault("json_fix_rounds", []).append({
                "round": round_num + 1,
                "content": new_content[:500],
                "finish_reason": debug["finish_reason"],
                "usage": debug["usage"],
            })

            items, is_valid_array = parse_llm_response_with_status(new_content)
            if is_valid_array:
                return items, new_content

            if new_content.strip():
                content = new_content
                fix_prompt = f"""下面的回复仍然不是纯JSON格式。请直接返回纯JSON数组，不要任何其他文字。

{new_content}"""
        except Exception as exc:
            batch_debug.setdefault("json_fix_rounds", []).append({
                "round": round_num + 1,
                "error": str(exc),
            })
            break

    return [], content


def fix_structured_json_with_llm(
    client: Any,
    provider: LLMProvider,
    config: dict[str, Any],
    raw_content: str,
    content_type: str,
    batch_debug: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """当 LLM 返回非 JSON 时，让它把内容转换成 JSON object。"""
    max_rounds = int(config["llm"].get("json_fix_rounds", 3))
    if max_rounds <= 0:
        return {
            "content_type": content_type,
            "title": config.get(content_type, {}).get("title", content_type),
            "error": "LLM JSON repair disabled",
            "raw_response": raw_content,
        }, raw_content

    fix_prompt = f"""下面是之前对{content_type}任务的回复，但它不是纯JSON object。
请把其中的信息提取出来，转换成纯JSON object。

输出必须是纯JSON object，不要Markdown，不要代码块，不要额外文字。

之前的回复：
{raw_content}"""

    content = raw_content
    for round_num in range(max_rounds):
        try:
            response = call_llm(
                client, provider,
                [{"role": "user", "content": fix_prompt}],
                max_tokens_override=(
                    provider.max_completion_tokens
                    if provider.max_completion_tokens is not None
                    else provider.max_tokens
                ),
            )
            debug = llm_response_debug(response)
            _record_usage(batch_debug, "json_fix", debug, round=round_num + 1)
            new_content = debug["content"] or debug["reasoning_content"]
            batch_debug.setdefault("json_fix_rounds", []).append({
                "round": round_num + 1,
                "content": new_content[:500],
                "finish_reason": debug["finish_reason"],
                "usage": debug["usage"],
            })

            parsed = parse_llm_json(new_content)
            if isinstance(parsed, dict) and parsed:
                return parsed, new_content

            if new_content.strip():
                content = new_content
                fix_prompt = f"""下面的回复仍然不是纯JSON object。请直接返回纯JSON object，不要任何其他文字。

{new_content}"""
        except Exception as exc:
            batch_debug.setdefault("json_fix_rounds", []).append({
                "round": round_num + 1,
                "error": str(exc),
            })
            break

    return {
        "content_type": content_type,
        "title": config.get(content_type, {}).get("title", content_type),
        "error": "LLM JSON repair failed",
        "raw_response": content,
    }, content
