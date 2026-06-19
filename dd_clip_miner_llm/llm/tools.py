"""工具调用模块

LLM 工具调用逻辑。
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .provider import LLMProvider
from .transport import call_llm
from .parse import parse_llm_response_with_status
from ..llm_debug import _record_usage, llm_response_debug


def _build_correction_message(reason: str, validation: dict[str, Any]) -> str | None:
    """Build a correction message for invalid content_validator results."""
    if reason == "invalid_json_array":
        return (
            "你返回的不是 JSON 数组。请只返回一个 JSON 数组，不要 Markdown、解释或代码块。"
            "示例：[{\"content_type\":\"song\",\"title\":\"歌名\",\"segment_ranges\":[[1,2]],\"confidence\":0.8}]"
        )
    if reason == "empty_song_match_array":
        return (
            "你返回了空数组 []，但 ASR 中有明显演唱证据。"
            "请重新完整扫描全部 ASR 段落，找出所有演唱区间，返回歌曲识别结果 JSON 数组。"
            "无法确认歌名时使用\"未知歌曲：...\"加代表性歌词，不要因为无法确认歌名而省略。"
        )
    if reason == "invalid_song_match_schema":
        return (
            "你返回的数组中包含非对象元素。每个元素必须是一个 JSON object，包含 content_type、title、segment_ranges、confidence 字段。"
            "不要返回字符串数组或数字数组。"
        )
    if reason == "missing_song_segments":
        return (
            "你返回的对象缺少 segment_ranges 或 segment_indices 字段。"
            "每个歌曲对象必须包含 segment_ranges（格式 [[start,end],...]）或 segment_indices（格式 [1,2,3]）。"
        )
    if reason == "zero_parsed_song_matches":
        return (
            "你返回的 JSON 数组中没有有效的歌曲匹配。"
            "请确保每个对象包含正确的 segment_ranges 格式，且至少有一个有效的演唱区间。"
        )
    return None


def _is_tool_role_unsupported_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".casefold()
    if "role" not in text:
        return False
    if "param incorrect" in text and "not supported" in text:
        return True
    return any(
        marker in text
        for marker in (
            "role is not supported",
            "role not supported",
            "unsupported role",
            "invalid role",
            "not support role",
        )
    )


def _flatten_tool_role_messages(
    messages: list[dict[str, Any]],
    final_instruction: str | None,
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    tool_results: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            content = str(message.get("content", ""))
            tool_call_id = str(message.get("tool_call_id", "")).strip()
            prefix = f"tool_call_id={tool_call_id}\n" if tool_call_id else ""
            tool_results.append(f"{prefix}{content}")
            continue
        if message.get("tool_calls"):
            continue
        flattened.append(message)
    if tool_results:
        flattened.append({
            "role": "user",
            "content": (
                "以下是前面工具调用得到的结果。当前 provider 不支持 role=tool，"
                "因此这些结果以普通消息提供。\n\n"
                + "\n\n--- 工具结果 ---\n\n".join(tool_results)
                + "\n\n"
                + (
                    final_instruction
                    or "请根据这些工具结果和原始 ASR，直接返回识别结果 JSON 数组。不要再调用工具，不要解释。"
                )
            ),
        })
    return flattened


def run_llm_with_tools(
    client: Any,
    provider: LLMProvider,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_executor: Any,
    batch_debug: dict[str, Any],
    max_tool_rounds: int = 2,
    final_max_tokens: int | None = None,
    force_final_round: bool = False,
    final_instruction: str | None = None,
    config: dict[str, Any] | None = None,
    initial_tool_choice: str | None = None,
    content_validator: Callable[[str], tuple[bool, dict[str, Any]]] | None = None,
) -> str:
    """调用 LLM，处理 tool calls"""
    tool_role_compat_active = False
    for tool_round in range(max_tool_rounds + 1):
        is_last = (tool_round == max_tool_rounds)

        call_tools = tools
        if tool_round == 0 and initial_tool_choice is not None:
            tool_choice = initial_tool_choice
        else:
            tool_choice = "none" if is_last else "auto"
        last_round_tokens = final_max_tokens if is_last else None
        if is_last and tool_round > 0:
            messages = messages + [{
                "role": "user",
                "content": final_instruction or (
                    "搜索已完成。现在请根据已有的搜索结果，直接返回识别结果的JSON数组。"
                    "不要再调用任何工具。只返回JSON数组，不要其他文字。"
                ),
            }]

        try:
            response = call_llm(
                client, provider, messages,
                max_tokens_override=last_round_tokens,
                tools=call_tools, tool_choice=tool_choice,
            )
        except Exception as exc:
            if (
                any(message.get("role") == "tool" for message in messages)
                and _is_tool_role_unsupported_error(exc)
            ):
                tool_role_compat_active = True
                messages = _flatten_tool_role_messages(messages, final_instruction)
                batch_debug.setdefault("tool_strategy_events", []).append({
                    "round": tool_round + 1,
                    "reason": "tool_role_unsupported",
                    "action": "retry_with_user_tool_results",
                    "error": str(exc),
                })
                print(
                    f"  LLM tool round {tool_round + 1}: "
                    "provider rejected role=tool; retrying with user tool results",
                    flush=True,
                )
                response = call_llm(
                    client, provider, messages,
                    max_tokens_override=last_round_tokens,
                    tools=None, tool_choice=None,
                )
            else:
                raise
        debug = llm_response_debug(response)
        _record_usage(batch_debug, "tool", debug, round=tool_round + 1)
        batch_debug["finish_reason"] = debug["finish_reason"]
        batch_debug.setdefault("tool_rounds", []).append({
            "round": tool_round + 1,
            "content": debug["content"][:200],
            "reasoning_content": debug["reasoning_content"][:200],
            "finish_reason": debug["finish_reason"],
            "has_tool_calls": bool(debug.get("tool_calls")),
            "usage": debug["usage"],
        })

        content = debug["content"]
        tool_calls_data = debug.get("tool_calls")

        if not tool_calls_data:
            if not content.strip() and debug["reasoning_content"].strip():
                content = debug["reasoning_content"]
            if not is_last and force_final_round:
                _, is_valid_array = parse_llm_response_with_status(content)
                if not is_valid_array:
                    continue
            if config is not None:
                from .repair import _continue_truncated_json_array
                content = _continue_truncated_json_array(
                    client, provider, config, messages, content,
                    debug["finish_reason"], batch_debug, tools=tools,
                    max_tokens=final_max_tokens,
                )
            validation: dict[str, Any] | None = None
            if content_validator is not None:
                is_content_valid, validation = content_validator(content)
                batch_debug.setdefault("content_validation", []).append({
                    "round": tool_round + 1,
                    "valid": is_content_valid,
                    **validation,
                })
                if not is_content_valid and not is_last:
                    reason = validation.get("reason", "invalid_content")
                    raw_count = validation.get("raw_item_count")
                    parsed_count = validation.get("parsed_match_count")
                    detail = ""
                    if raw_count is not None and parsed_count is not None:
                        detail = f" ({raw_count} raw items, {parsed_count} parsed matches)"
                    batch_debug.setdefault("tool_strategy_events", []).append({
                        "round": tool_round + 1,
                        "initial_tool_choice": initial_tool_choice,
                        "reason": reason,
                        "action": "continue_with_auto_tools",
                    })
                    print(
                        f"  LLM tool round {tool_round + 1}: "
                        f"{reason}{detail}, continuing with tools",
                        flush=True,
                    )
                    correction = _build_correction_message(reason, validation)
                    if correction:
                        messages = messages + [{"role": "user", "content": correction}]
                    continue
            if (
                tool_round == 0
                and initial_tool_choice == "none"
                and not is_last
                and content_validator is None
            ):
                items, is_valid_array = parse_llm_response_with_status(content)
                if not is_valid_array or not items:
                    reason = "empty" if is_valid_array else "invalid_json"
                    batch_debug.setdefault("tool_strategy_events", []).append({
                        "round": tool_round + 1,
                        "initial_tool_choice": initial_tool_choice,
                        "reason": reason,
                        "action": "continue_with_auto_tools",
                    })
                    print(
                        f"  LLM tool round {tool_round + 1}: "
                        f"initial no-tools returned {reason}, continuing with tools",
                        flush=True,
                    )
                    continue
            return content

        if is_last:
            if not content.strip() and debug["reasoning_content"].strip():
                content = debug["reasoning_content"]
            if config is not None:
                from .repair import _continue_truncated_json_array
                content = _continue_truncated_json_array(
                    client, provider, config, messages, content,
                    debug["finish_reason"], batch_debug, tools=tools,
                    max_tokens=final_max_tokens,
                )
            if content_validator is not None:
                is_content_valid, validation = content_validator(content)
                batch_debug.setdefault("content_validation", []).append({
                    "round": tool_round + 1,
                    "valid": is_content_valid,
                    **validation,
                })
            return content

        choice = response.choices[0] if response.choices else None
        message = choice.message if choice is not None else None
        if not message or not message.tool_calls:
            return content

        if not tool_role_compat_active:
            assistant_message = message.model_dump()
            assistant_message.setdefault("role", "assistant")
            messages.append(assistant_message)
        compatible_tool_results: list[str] = []
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = tool_executor(tc.function.name, args)
            tool_log: dict[str, Any] = {
                "round": tool_round + 1,
                "function": tc.function.name,
                "arguments": args,
                "result_preview": result[:1000],
                "result_length": len(result),
            }
            if tc.function.name == "search_lyrics":
                try:
                    search_payload = json.loads(result)
                except (TypeError, json.JSONDecodeError):
                    search_payload = None
                if isinstance(search_payload, dict):
                    tool_log["result_summary"] = {
                        "query": search_payload.get("query", ""),
                        "results": [
                            {
                                "title": item.get("title", ""),
                                "snippet": str(item.get("snippet", ""))[:240],
                                "url": item.get("url", ""),
                            }
                            for item in search_payload.get("results", [])[:3]
                            if isinstance(item, dict)
                        ],
                        "lyrics_hints": [
                            str(item)[:240]
                            for item in search_payload.get("lyrics_hints", [])[:2]
                        ],
                    }
            batch_debug.setdefault("tool_calls_log", []).append(tool_log)
            if tool_role_compat_active:
                compatible_tool_results.append(
                    f"{tc.function.name}({json.dumps(args, ensure_ascii=False)}):\n{result}"
                )
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        if compatible_tool_results:
            messages.append({
                "role": "user",
                "content": (
                    "以下是工具调用结果。请根据这些结果继续，并在完成后只返回 JSON 数组。\n\n"
                    + "\n\n--- 工具结果 ---\n\n".join(compatible_tool_results)
                ),
            })

    return ""
