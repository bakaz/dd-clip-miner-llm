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


_CACHE_SYSTEM_PROMPT = (
    "你将先收到一份带全局序号和时间范围的 ASR 转写，再收到具体分析任务。"
    "必须只依据该转写完成任务，不得使用输入中不存在的 segment index。"
)


@dataclass
class LLMProvider:
    api_key: str
    base_url: str | None = None
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 4096
    max_completion_tokens: int | None = None
    thinking: str | None = None


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
    thinking = llm_config.get("thinking")

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
            thinking=str(thinking) if thinking not in (None, "") else None,
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
                thinking=(
                    str(fb["thinking"])
                    if fb.get("thinking") not in (None, "")
                    else (str(thinking) if thinking not in (None, "") else None)
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
    max_retries: int = 3,
    tool_choice: Any = None,
) -> Any:
    """调用 LLM，返回完整的 response 对象
    
    Args:
        client: OpenAI 客户端
        provider: LLM provider 配置
        messages: 消息列表
        tools: 工具定义
        tool_choice: 工具选择策略
        max_tokens_override: 最大 token 数覆盖
        max_retries: 最大重试次数（指数退避）
    """
    import time
    
    token_limit = (
        max_tokens_override
        if max_tokens_override is not None
        else (
            provider.max_completion_tokens
            if provider.max_completion_tokens is not None
            else provider.max_tokens
        )
    )
    uses_deepseek_api = "deepseek.com" in str(provider.base_url or "").casefold()
    token_args = (
        {"max_tokens": token_limit}
        if uses_deepseek_api or provider.max_completion_tokens is None
        else {"max_completion_tokens": token_limit}
    )

    kwargs: dict[str, Any] = {
        "model": provider.model,
        "messages": messages,
        "temperature": provider.temperature,
        **token_args,
    }
    if uses_deepseek_api and provider.thinking:
        kwargs["extra_body"] = {"thinking": {"type": provider.thinking}}

    if tools:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

    last_exc = None
    for retry in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            if retry < max_retries - 1:
                # 指数退避：1s, 2s, 4s
                wait_time = 2 ** retry
                print(f"  [llm] API call failed (retry {retry + 1}/{max_retries}, wait {wait_time}s): {exc}")
                time.sleep(wait_time)
            continue
    
    raise last_exc


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


def _record_usage(
    batch_debug: dict[str, Any],
    phase: str,
    debug: dict[str, Any],
    **details: Any,
) -> None:
    usage = debug.get("usage")
    if not isinstance(usage, dict):
        return
    batch_debug.setdefault("usage", []).append({
        "phase": phase,
        **details,
        **usage,
    })


def _cache_usage_summary(batch_debug: dict[str, Any]) -> str | None:
    hit_tokens = 0
    miss_tokens = 0
    for usage in batch_debug.get("usage", []):
        if not isinstance(usage, dict):
            continue
        hit_tokens += int(usage.get("prompt_cache_hit_tokens") or 0)
        miss_tokens += int(usage.get("prompt_cache_miss_tokens") or 0)
    total = hit_tokens + miss_tokens
    if total <= 0:
        return None
    return (
        f"KV cache hit {hit_tokens}/{total} input tokens "
        f"({hit_tokens / total:.1%})"
    )


def _format_transcript_for_cache(
    segments: list[TranscriptSegment],
    batch_start: int,
    recognizer: BaseRecognizer,
) -> str:
    index_start = batch_start
    resolve_start = getattr(recognizer, "transcript_index_start", None)
    if callable(resolve_start):
        index_start = int(resolve_start(batch_start))
    return "\n".join(
        f"[{index_start + i}] ({seg.start:.1f}s-{seg.end:.1f}s) {seg.text}"
        for i, seg in enumerate(segments)
    )


def _extract_task_instructions(prompt: str) -> str | None:
    for marker in ("\n完整 ASR 转写片段：\n", "\nASR 转写：\n"):
        if marker in prompt:
            instructions, _ = prompt.rsplit(marker, 1)
            return instructions.strip()
    return None


def build_llm_messages(
    recognizer: BaseRecognizer,
    segments: list[TranscriptSegment],
    batch_start: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """构建请求消息；缓存友好模式把可复用 ASR 长文本放在任务指令之前。"""
    prompt = recognizer.build_prompt(segments, batch_start, config)
    llm_config = config.get("llm", {})
    if not llm_config.get("cache_friendly_prompt_layout", True):
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


# ============ 工具调用 ============

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
            client,
            provider,
            messages,
            max_tokens_override=last_round_tokens,
            tools=call_tools,
            tool_choice=tool_choice,
        )
        debug = llm_response_debug(response)
        _record_usage(batch_debug, "tool", debug, round=tool_round + 1)
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
            _record_usage(
                batch_debug,
                "reasoning_followup",
                followup_debug,
                round=len(batch_debug["reasoning_followups"]) + 1,
            )
            content = followup_debug["content"]
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
            _record_usage(batch_debug, "json_fix", debug, round=round_num + 1)
            new_content = debug["content"] or debug["reasoning_content"]
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
                max_tokens_override=16384,
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
    items, _ = parse_llm_response_with_status(text)
    return items


def parse_llm_response_with_status(text: str) -> tuple[list[dict[str, Any]], bool]:
    """解析 JSON 数组，并区分合法空数组与解析失败。"""
    result = parse_llm_json(text)
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)], True
    return [], False


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

    # 预创建客户端（复用连接）
    clients: dict[str, Any] = {}
    for candidate in providers:
        if not candidate.api_key:
            continue
        if candidate.api_key not in clients:
            client_kwargs: dict[str, Any] = {"api_key": candidate.api_key}
            if candidate.base_url:
                client_kwargs["base_url"] = candidate.base_url
            clients[candidate.api_key] = OpenAI(**client_kwargs)

    debug_path = Path(debug_dir) if debug_dir is not None else None
    if debug_path is not None:
        debug_path.mkdir(parents=True, exist_ok=True)

    batch_debug: dict[str, Any] = {
        "batch_start": 0,
        "batch_end": len(segments) - 1,
        "segment_count": len(segments),
        "provider": None,
        "raw_response": None,
        "parsed_json": None,
        "json_fix_rounds": [],
        "usage": [],
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
            client = clients[candidate.api_key]
            provider = candidate
            batch_debug["provider"] = {
                "base_url": candidate.base_url or "openai",
                "model": candidate.model,
            }
            
            messages = build_llm_messages(recognizer, segments, 0, config)
            batch_debug["request_messages"] = messages
            
            response = call_llm(client, candidate, messages)
            debug = llm_response_debug(response)
            _record_usage(batch_debug, "initial", debug)
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
    cache_summary = _cache_usage_summary(batch_debug)
    if cache_summary:
        print(f"  LLM {cache_summary}")

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

    # 预创建客户端（复用连接，减少 TCP 握手开销）
    clients: dict[str, Any] = {}
    for provider in providers:
        if not provider.api_key:
            continue
        if provider.api_key not in clients:
            client_kwargs: dict[str, Any] = {"api_key": provider.api_key}
            if provider.base_url:
                client_kwargs["base_url"] = provider.base_url
            clients[provider.api_key] = OpenAI(**client_kwargs)

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

    for batch_idx, (batch_start, batch_segments) in enumerate(batches, 1):
        print(f"  LLM batch {batch_idx}/{len(batches)}: segments {batch_start}-{batch_start + len(batch_segments) - 1} (total {len(segments)})...")
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
            "usage": [],
            "error": None,
        }

        content = None
        last_error = None

        for provider in providers:
            if not provider.api_key:
                continue

            try:
                client = clients[provider.api_key]
                messages = build_llm_messages(
                    recognizer,
                    batch_segments,
                    batch_start,
                    config,
                )
                batch_debug["request_messages"] = messages

                batch_debug["provider"] = {
                    "base_url": provider.base_url or "openai",
                    "model": provider.model,
                }

                if tools and content_type == "song":
                    from .search_tools import execute_tool
                    llm_config = config.get("llm", {})
                    content = run_llm_with_tools(
                        client,
                        provider,
                        messages,
                        tools,
                        execute_tool,
                        batch_debug,
                        max_tool_rounds=int(llm_config.get("max_tool_rounds", 2) or 0),
                        final_max_tokens=(
                            int(llm_config["final_tool_max_tokens"])
                            if llm_config.get("final_tool_max_tokens") not in (None, "")
                            else (
                                provider.max_completion_tokens
                                if provider.max_completion_tokens is not None
                                else provider.max_tokens
                            )
                        ),
                        force_final_round=bool(
                            llm_config.get("force_final_tool_round", False)
                        ),
                        final_instruction=(
                            str(llm_config["final_tool_instruction"])
                            if llm_config.get("final_tool_instruction")
                            else None
                        ),
                    )
                else:
                    response = call_llm(client, provider, messages)
                    debug = llm_response_debug(response)
                    _record_usage(batch_debug, "initial", debug)
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
        items, is_valid_array = parse_llm_response_with_status(content)
        if not is_valid_array:
            # 从 reasoning 中提取
            if batch_debug.get("tool_rounds"):
                for tr in batch_debug["tool_rounds"]:
                    rc = tr.get("reasoning_content", "")
                    if rc.strip():
                        items, is_valid_array = parse_llm_response_with_status(rc)
                        if is_valid_array:
                            break
            # reasoning followup 兜底
            if not is_valid_array:
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
                        items, is_valid_array = parse_llm_response_with_status(content)

        # JSON 修复
        if not is_valid_array and content.strip():
            items, content = fix_json_with_llm(
                client, provider, config, content, content_type, batch_debug
            )
            _, is_valid_array = parse_llm_response_with_status(content)

        batch_debug["parsed_items"] = items
        batch_debug["parse_valid"] = is_valid_array
        batch_debug["raw_response"] = content
        if debug_path is not None:
            write_llm_debug(debug_path, batch_start, batch_debug)

        # 使用识别器解析响应
        matches = recognizer.parse_response(items, config)
        all_matches.extend(matches)
        cache_summary = _cache_usage_summary(batch_debug)
        cache_suffix = f", {cache_summary}" if cache_summary else ""
        print(
            f"  LLM batch {batch_idx}/{len(batches)}: done, "
            f"found {len(matches)} match(es){cache_suffix}"
        )

    return all_matches
