"""歌词搜索工具 - DuckDuckGo + SearxNG fallback"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote_plus

import httpx


def _clean_lyrics_query(title: str, artist: str = "") -> str:
    """构建歌词搜索查询"""
    query = f"{title} lyrics"
    if artist:
        query = f"{artist} {title} lyrics"
    return query


def search_duckduckgo(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """使用DuckDuckGo搜索"""
    try:
        from ddgs import DDGS
    except ImportError:
        return _search_duckduckgo_html(query, max_results)
    
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                })
    except Exception:
        return _search_duckduckgo_html(query, max_results)
    return results


def _search_duckduckgo_html(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """DuckDuckGo HTML fallback"""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    results = []
    try:
        with httpx.Client(timeout=10, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                text = resp.text
                # 简单解析结果
                snippets = re.findall(
                    r'class="result__snippet"[^>]*>(.*?)</(?:a|span|div)',
                    text, re.DOTALL
                )
                titles = re.findall(
                    r'class="result__a"[^>]*>(.*?)</a>',
                    text, re.DOTALL
                )
                urls = re.findall(
                    r'class="result__url"[^>]*>(.*?)</a>',
                    text, re.DOTALL
                )
                for i in range(min(len(snippets), max_results)):
                    results.append({
                        "title": _strip_html(titles[i]) if i < len(titles) else "",
                        "snippet": _strip_html(snippets[i]),
                        "url": urls[i].strip() if i < len(urls) else "",
                    })
    except Exception:
        pass
    return results


def search_searxng(query: str, instance: str = "https://searx.be", max_results: int = 5) -> list[dict[str, str]]:
    """使用SearxNG搜索"""
    params = {
        "q": query,
        "format": "json",
        "categories": "general",
    }
    results = []
    try:
        with httpx.Client(timeout=10, follow_redirects=True) as client:
            resp = client.get(f"{instance}/search", params=params)
            if resp.status_code == 200:
                data = resp.json()
                for r in data.get("results", [])[:max_results]:
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("content", ""),
                        "url": r.get("url", ""),
                    })
    except Exception:
        pass
    return results


def _strip_html(text: str) -> str:
    """移除HTML标签"""
    return re.sub(r"<[^>]+>", "", text).strip()


def extract_lyrics_from_snippet(snippet: str) -> str | None:
    """从搜索结果片段中提取歌词"""
    # 查找引号内的歌词
    quoted = re.findall(r'"([^"]{10,})"', snippet)
    if quoted:
        return quoted[0]
    
    # 查找看起来像歌词的内容（押韵、重复结构）
    lines = snippet.split("\n")
    lyric_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 5 and not line.startswith(("http", "www", "©")):
            lyric_lines.append(line)
    if lyric_lines:
        return "\n".join(lyric_lines[:5])
    return None


def search_lyrics(title: str, artist: str = "") -> dict[str, Any]:
    """搜索歌词，返回结果"""
    query = _clean_lyrics_query(title, artist)
    
    # 尝试DuckDuckGo
    results = search_duckduckgo(query)
    
    # Fallback到SearxNG
    if not results:
        results = search_searxng(query)
    
    # 提取可能的歌词
    lyrics_hints = []
    for r in results:
        snippet = r.get("snippet", "")
        lyrics = extract_lyrics_from_snippet(snippet)
        if lyrics:
            lyrics_hints.append(lyrics)
    
    return {
        "query": query,
        "results": results[:3],
        "lyrics_hints": lyrics_hints[:2],
    }


# Tool definition for OpenAI function calling
LYRICS_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_lyrics",
        "description": "搜索歌曲歌词。当无法确定歌名或需要验证歌词时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "歌曲名称或歌词片段"
                },
                "artist": {
                    "type": "string",
                    "description": "歌手名称（可选）"
                }
            },
            "required": ["title"]
        }
    }
}


def get_tools() -> list[dict[str, Any]]:
    """获取可用工具列表"""
    return [LYRICS_SEARCH_TOOL]


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """执行工具调用"""
    if name == "search_lyrics":
        result = search_lyrics(
            title=arguments.get("title", ""),
            artist=arguments.get("artist", ""),
        )
        return str(result)
    return f"Unknown tool: {name}"
