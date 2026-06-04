"""LLM 调用工具

提供与 OpenAI 兼容 API 的调用逻辑，包括：
- Provider 管理（多 key、fallback）
- 工具调用
- Reasoning followup
- JSON 修复
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ContentMatch, TranscriptSegment
from .recognizers.base import BaseRecognizer


@dataclass
class LLMProvider:
    api_key: str
    base_url: str | None = None
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 4096
    max_completion_tokens: int | None = None


# ============ Provider 管理 ============

def build_providers(config: dict[str, Any]) -> list[LLMProvider]:
    """构建 LLM provider 列表"""
    llm_config = config["llm"]
    api_keys = llm_config.get("api_key", "")
    api_key_env = llm_config.get("api_key_env")
    if not api_keys and api_key_env:
        api_keys = os.environ.get(str(api_key_env), "")

    base_url = llm_config.get("base_url")
    model = llm_config.get("model", "gpt-4o")
    temperature = float(llm_config.get("temperature", 0.3))
    max_tokens = int(llm_config.get("max_tokens", 4096))
    max_completion_tokens_value = llm_config.get("max_completion_tokens")
    max_completion_tokens = (
        int(max_completion_tokens_value)
        if max_completion_tokens_value not in (None, "")
        else None
    )

    if api_keys is None:
        api_keys = []
    elif isinstance(api_keys, str):
        api_keys = [k.strip() for k in api_keys.split(",") if k.strip()]

    providers: list[LLMProvider] = []
    for i, api_key in enumerate(api_keys):
        if not api_key:
            continue
        providers.append(LLMProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
        ))

    fallbacks = llm_config.get("fallbacks", [])
    for fb in fallbacks:
        fallback_api_key = fb.get("api_key", "")
        fallback_api_key_env = fb.get("api_key_env")
        if not fallback_api_key and fallback_api_key_env:
            fallback_api_key = os.environ.get(str(fallback_api_key_env), "")
        if fallback_api_key:
            providers.append(LLMProvider(
                api_key=fallback_api_key,
                base_url=fb.get("base_url", base_url),
                model=fb.get("model", model),
                temperature=float(fb.get("temperature", temperature)),
                max_tokens=int(fb.get("max_tokens", max_tokens)),
                max_completion_tokens=(
                    int(fb["max_completion_tokens"])
                    if fb.get("max_completion_tokens") not in (None, "")
                    else max_completion_tokens
                ),
            ))

    return providers


# ============ LLM 调用 ============

def call_llm(
    client: Any,
    provider: LLMProvider,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens_override: int | None = None,
) -> Any:
    """调用 LLM，返回完整的 response 对象"""
    token_args = (
        {"max_completion_tokens": max_tokens_override}
        if max_tokens_override is not None
        else (
            {"max_completion_tokens": provider.max_completion_tokens}
            if provider.max_completion_tokens is not None
            else {"max_tokens": provider.max_tokens}
        )
    )

    kwargs: dict[str, Any] = {
        "model": provider.model,
        "messages": messages,
        "temperature": provider.temperature,
        **token_args,
    }

    if tools:
        kwargs["tools"] = tools

    return client.chat.completions.create(**kwargs)


def llm_response_debug(response: Any) -> dict[str, Any]:
    """提取 LLM response 的调试信息"""
    choice = response.choices[0] if response.choices else None
    message = choice.message if choice is not None else None
    message_data = message.model_dump() if message is not None else {}
    usage = response.usage.model_dump() if getattr(response, "usage", None) else None
    content = message_data.get("content") or ""
    reasoning = message_data.get("reasoning_content") or ""
    return {
        "model": getattr(response, "model", None),
        "finish_reason": getattr(choice, "finish_reason", None) if choice is not None else None,
        "content": content,
        "content_length": len(content),
        "reasoning_content": reasoning,
        "reasoning_content_length": len(reasoning),
        "message_keys": list(message_data.keys()),
        "usage": usage,
        "tool_calls": message_data.get("tool_calls"),
    }


# ============ 工具调用 ============

def run_llm_with_tools(
    client: Any,
    provider: LLMProvider,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_executor: Any,
    batch_debug: dict[str, Any],
    max_tool_rounds: int = 2,
) -> str:
    """调用 LLM，处理 tool calls
    
    Args:
        client: OpenAI 客户端
        provider: LLM provider
        messages: 消息列表
        tools: 工具定义
        tool_executor: 工具执行器，接受 (name, arguments) 返回结果字符串
        batch_debug: 调试信息字典
        max_tool_rounds: 最大工具调用轮数
    """
    for tool_round in range(max_tool_rounds + 1):
        is_last = (tool_round == max_tool_rounds)

        call_tools = None if is_last else tools
        last_round_tokens = 16384 if is_last else None
        if is_last and tool_round > 0:
            messages = messages + [{
                "role": "user",
                "content": "搜索已完成。现在请根据已有的搜索结果，直接返回识别结果的JSON数组。不要再调用任何工具。只返回JSON数组，不要其他文字。",
            }]

        response = call_llm(client, provider, messages, max_tokens_override=last_round_tokens, tools=call_tools)
        debug = llm_response_debug(response)
        batch_debug.setdefault("tool_rounds", []).append({
            "round": tool_round + 1,
            "content": debug["content"][:200],
            "reasoning_content": debug["reasoning_content"][:200],
            "finish_reason": debug["finish_reason"],
            "has_tool_calls": bool(debug.get("tool_calls")),
        })

        content = debug["content"]
        tool_calls_data = debug.get("tool_calls")

        if not tool_calls_data:
            if not content.strip() and debug["reasoning_content"].strip():
                content = debug["reasoning_content"]
            return content

        if is_last:
            if not content.strip() and debug["reasoning_content"].strip():
                content = debug["reasoning_content"]
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
            batch_debug.setdefault("tool_calls_log", []).append({
                "round": tool_round + 1,
                "function": tc.function.name,
                "arguments": args,
                "result_preview": result[:200],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return ""


# ============ Reasoning Followup ============

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
                client,
                provider,
                [{"role": "user", "content": followup_prompt}],
                max_tokens_override=followup_tokens,
            )
            followup_debug = llm_response_debug(followup_response)
            content = followup_debug["content"]
            batch_debug["reasoning_followups"].append({
                "round": len(batch_debug["reasoning_followups"]) + 1,
                "content": content[:500],
                "reasoning_content": followup_debug["reasoning_content"][:500],
            })
            batch_debug["raw_response"] = content
        except Exception as exc:
            batch_debug["reasoning_followups"].append({
                "round": len(batch_debug["reasoning_followups"]) + 1,
                "error": str(exc),
            })
            return ""

        if content.strip() and parse_llm_response(content):
            return content

        material = str(followup_debug.get("reasoning_content") or "")
        partial = content

    return content


# ============ JSON 修复 ============

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
                max_tokens_override=16384,
            )
            debug = llm_response_debug(response)
            new_content = debug["content"] or debug["reasoning_content"]
            batch_debug.setdefault("json_fix_rounds", []).append({
                "round": round_num + 1,
                "content": new_content[:500],
                "finish_reason": debug["finish_reason"],
            })

            items = parse_llm_response(new_content)
            if items:
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
                max_tokens_override=16384,
            )
            debug = llm_response_debug(response)
            new_content = debug["content"] or debug["reasoning_content"]
            batch_debug.setdefault("json_fix_rounds", []).append({
                "round": round_num + 1,
                "content": new_content[:500],
                "finish_reason": debug["finish_reason"],
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


# ============ 响应解析 ============

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
    result = parse_llm_json(text)
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    return []


# ============ 调试工具 ============

def write_llm_debug(debug_dir: Path, batch_start: int, payload: dict[str, Any]) -> None:
    """写入 LLM 调试信息"""
    target = debug_dir / f"llm_batch_{batch_start:06d}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ============ 兼容旧接口 ============

def identify_songs(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    debug_dir: Path | None = None,
) -> list[ContentMatch]:
    """识别歌曲片段（兼容旧接口）"""
    from .recognizers import get_recognizer
    recognizer = get_recognizer("song")
    if recognizer is None:
        raise RuntimeError("Song recognizer not found")
    return identify_content(segments, config, recognizer, debug_dir)


def identify_dialogues(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    debug_dir: Path | None = None,
) -> list[ContentMatch]:
    """识别对话片段（兼容旧接口）"""
    from .recognizers import get_recognizer
    recognizer = get_recognizer("dialogue")
    if recognizer is None:
        raise RuntimeError("Dialogue recognizer not found")
    return identify_content(segments, config, recognizer, debug_dir)


def identify_structured_content(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: BaseRecognizer,
    debug_dir: Path | None = None,
) -> dict[str, Any]:
    """通用结构化内容生成，返回 JSON object。"""
    from openai import OpenAI

    content_type = recognizer.name

    providers = build_providers(config)
    if not providers:
        raise RuntimeError("LLM API key not configured. Set llm.api_key in config.")

    debug_path = Path(debug_dir) if debug_dir is not None else None
    if debug_path is not None:
        debug_path.mkdir(parents=True, exist_ok=True)

    prompt = recognizer.build_prompt(segments, 0, config)
    batch_debug: dict[str, Any] = {
        "batch_start": 0,
        "batch_end": len(segments) - 1,
        "segment_count": len(segments),
        "provider": None,
        "raw_response": None,
        "parsed_json": None,
        "json_fix_rounds": [],
        "error": None,
    }

    content = None
    last_error = None
    client = None
    provider = None

    for candidate in providers:
        if not candidate.api_key:
            continue

        try:
            client_kwargs: dict[str, Any] = {"api_key": candidate.api_key}
            if candidate.base_url:
                client_kwargs["base_url"] = candidate.base_url

            client = OpenAI(**client_kwargs)
            provider = candidate
            batch_debug["provider"] = {
                "base_url": candidate.base_url or "openai",
                "model": candidate.model,
            }
            response = call_llm(client, candidate, [{"role": "user", "content": prompt}])
            debug = llm_response_debug(response)
            content = debug["content"]
            if not content.strip() and debug["reasoning_content"].strip():
                content = debug["reasoning_content"]
            batch_debug["raw_response"] = content
            break
        except Exception as exc:
            last_error = exc
            print(f"  [warn] Provider {candidate.model} failed: {exc}")
            continue

    if content is None:
        batch_debug["error"] = str(last_error)
        if debug_path is not None:
            write_llm_debug(debug_path, 0, batch_debug)
        print(f"  [error] All LLM providers failed for {content_type}. Last error: {last_error}")
        return {
            "content_type": content_type,
            "title": config.get(content_type, {}).get("title", content_type),
            "error": str(last_error),
        }

    parsed = parse_llm_json(content)
    if not isinstance(parsed, dict) and content.strip() and client is not None and provider is not None:
        parsed, content = fix_structured_json_with_llm(
            client, provider, config, content, content_type, batch_debug
        )

    if not isinstance(parsed, dict) or not parsed:
        parsed = {
            "content_type": content_type,
            "title": config.get(content_type, {}).get("title", content_type),
            "error": "LLM did not return a JSON object",
            "raw_response": content,
        }

    batch_debug["parsed_json"] = parsed
    batch_debug["raw_response"] = content
    if debug_path is not None:
        write_llm_debug(debug_path, 0, batch_debug)

    return parsed


# ============ 核心识别逻辑 ============

def identify_content(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    recognizer: BaseRecognizer,
    debug_dir: Path | None = None,
) -> list[ContentMatch]:
    """通用内容识别
    
    Args:
        segments: ASR 转写片段列表
        config: 完整配置字典
        recognizer: 识别器实例
        debug_dir: 调试信息输出目录
    """
    from openai import OpenAI
    
    content_type = recognizer.name

    providers = build_providers(config)
    if not providers:
        raise RuntimeError("LLM API key not configured. Set llm.api_key in config.")

    batch_size = config["llm"].get("batch_size")
    if batch_size in (None, "", 0, "0"):
        batches = [(0, segments)]
    else:
        batch_size = int(batch_size)
        batches = [
            (batch_start, segments[batch_start:batch_start + batch_size])
            for batch_start in range(0, len(segments), batch_size)
        ]

    all_matches: list[ContentMatch] = []
    debug_path = Path(debug_dir) if debug_dir is not None else None
    if debug_path is not None:
        debug_path.mkdir(parents=True, exist_ok=True)

    tools = recognizer.get_tools(config)

    for batch_start, batch_segments in batches:
        prompt = recognizer.build_prompt(batch_segments, batch_start, config)
        batch_debug: dict[str, Any] = {
            "batch_start": batch_start,
            "batch_end": batch_start + len(batch_segments) - 1,
            "segment_count": len(batch_segments),
            "provider": None,
            "raw_response": None,
            "parsed_items": [],
            "tool_calls_log": [],
            "tool_rounds": [],
            "reasoning_followups": [],
            "json_fix_rounds": [],
            "error": None,
        }

        content = None
        last_error = None

        for provider in providers:
            if not provider.api_key:
                continue

            try:
                client_kwargs: dict[str, Any] = {"api_key": provider.api_key}
                if provider.base_url:
                    client_kwargs["base_url"] = provider.base_url

                client = OpenAI(**client_kwargs)
                messages: list[dict[str, Any]] = [
                    {"role": "user", "content": prompt}
                ]

                batch_debug["provider"] = {
                    "base_url": provider.base_url or "openai",
                    "model": provider.model,
                }

                if tools and content_type == "song":
                    from .search_tools import execute_tool
                    content = run_llm_with_tools(
                        client, provider, messages, tools, execute_tool, batch_debug
                    )
                else:
                    response = call_llm(client, provider, messages)
                    debug = llm_response_debug(response)
                    content = debug["content"]
                    if not content.strip() and debug["reasoning_content"].strip():
                        content = debug["reasoning_content"]

                batch_debug["raw_response"] = content
                break
            except Exception as exc:
                last_error = exc
                print(f"  [warn] Provider {provider.model} failed: {exc}")
                continue

        if content is None:
            batch_debug["error"] = str(last_error)
            if debug_path is not None:
                write_llm_debug(debug_path, batch_start, batch_debug)
            print(f"  [error] All LLM providers failed for batch {batch_start}. Last error: {last_error}")
            continue

        # content 为空时尝试 reasoning followup
        if not content.strip():
            reasoning_content = ""
            if batch_debug.get("tool_rounds"):
                for tr in batch_debug["tool_rounds"]:
                    if tr.get("reasoning_content"):
                        reasoning_content = tr["reasoning_content"]
                        break
            content = run_reasoning_followups(
                client, provider, config, reasoning_content, "", batch_debug
            )
            if not content.strip():
                batch_debug["error"] = "LLM returned empty content"
                if debug_path is not None:
                    write_llm_debug(debug_path, batch_start, batch_debug)
                print(f"  [warn] LLM returned empty response for batch {batch_start}.")
                continue

        # 尝试解析 JSON
        items = parse_llm_response(content)
        if not items:
            # 从 reasoning 中提取
            if batch_debug.get("tool_rounds"):
                for tr in batch_debug["tool_rounds"]:
                    rc = tr.get("reasoning_content", "")
                    if rc.strip():
                        items = parse_llm_response(rc)
                        if items:
                            break
            # reasoning followup 兜底
            if not items:
                reasoning_content = ""
                if batch_debug.get("tool_rounds"):
                    for tr in batch_debug["tool_rounds"]:
                        if tr.get("reasoning_content"):
                            reasoning_content = tr["reasoning_content"]
                            break
                if reasoning_content.strip():
                    followup_content = run_reasoning_followups(
                        client, provider, config, reasoning_content, content, batch_debug
                    )
                    if followup_content.strip():
                        content = followup_content
                        items = parse_llm_response(content)

        # JSON 修复
        if not items and content.strip():
            items, content = fix_json_with_llm(
                client, provider, config, content, content_type, batch_debug
            )

        batch_debug["parsed_items"] = items
        batch_debug["raw_response"] = content
        if debug_path is not None:
            write_llm_debug(debug_path, batch_start, batch_debug)

        # 使用识别器解析响应
        matches = recognizer.parse_response(items, config)
        all_matches.extend(matches)

    return all_matches
