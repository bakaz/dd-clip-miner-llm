"""工具调用模块

LLM 工具调用逻辑。
"""
from __future__ import annotations

import json
from typing import Any

from .provider import LLMProvider
from .transport import call_llm
from .parse import parse_llm_response_with_status
from ..llm_debug import _record_usage, llm_response_debug


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
) -> str:
    """调用 LLM，处理 tool calls"""
    for tool_round in range(max_tool_rounds + 1):
        is_last = (tool_round == max_tool_rounds)

        call_tools = tools
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

        response = call_llm(
            client, provider, messages,
            max_tokens_override=last_round_tokens,
            tools=call_tools, tool_choice=tool_choice,
        )
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
            return content

        choice = response.choices[0] if response.choices else None
        message = choice.message if choice is not None else None
        if not message or not message.tool_calls:
            return content

        messages.append(message.model_dump())
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
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return ""
