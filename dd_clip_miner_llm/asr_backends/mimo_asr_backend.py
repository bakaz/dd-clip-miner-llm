"""Xiaomi MiMo-V2.5-ASR 后端

通过 OpenAI 兼容 API (chat/completions + input_audio) 调用 MiMo-V2.5-ASR，支持：
- 中文方言、歌词识别、噪音环境
- 自动语言检测
- 5 秒分段处理（获取细粒度时间戳）
- 并发处理 + 重试机制
"""
from __future__ import annotations

import base64
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..models import TranscriptSegment
from .base import ASRBackend

# 默认 5 秒一个 chunk，用于细粒度时间戳
_DEFAULT_TIMESTAMP_CHUNK = 5
# API 单次最大 10MB，5 秒 mp3 ≈ 80KB，远低于限制
_MAX_AUDIO_MB = 10
# 默认并发数
_DEFAULT_MAX_WORKERS = 4
# 默认重试次数
_DEFAULT_MAX_RETRIES = 3


class MimoASRBackend(ASRBackend):
    def __init__(self, settings: dict[str, Any]) -> None:
        super().__init__(settings)
        self._client: Any = None

    @property
    def mimo_settings(self) -> dict[str, Any]:
        value = self.settings.get("mimo", {})
        return value if isinstance(value, dict) else {}

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai not installed. pip install openai") from exc

        cfg = self.mimo_settings
        base_url = str(cfg.get("base_url", "https://token-plan-cn.xiaomimimo.com/v1"))
        api_key = str(cfg.get("api_key", "sk-placeholder"))

        api_key_env = cfg.get("api_key_env")
        if api_key_env:
            import os
            env_key = os.environ.get(str(api_key_env), "")
            if env_key:
                api_key = env_key

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        return self._client

    def transcribe(self, audio_path: str | Path) -> list[TranscriptSegment]:
        from ..ffmpeg import get_duration

        audio_path = Path(audio_path)
        cfg = self.mimo_settings
        chunk_seconds = int(cfg.get("timestamp_chunk_seconds", _DEFAULT_TIMESTAMP_CHUNK))

        duration = get_duration(audio_path)
        total_chunks = int(duration // chunk_seconds) + (1 if duration % chunk_seconds > 0 else 0)

        # 短音频直接处理
        if total_chunks <= 1:
            return self._transcribe_chunk(audio_path, 0.0, cfg)

        print(f"[asr] Audio {duration:.0f}s -> {total_chunks} chunks of {chunk_seconds}s")
        return self._transcribe_chunked(audio_path, duration, chunk_seconds, cfg)

    def _transcribe_chunk(
        self,
        audio_path: Path,
        time_offset: float,
        cfg: dict[str, Any],
    ) -> list[TranscriptSegment]:
        from ..ffmpeg import get_duration

        client = self._get_client()
        model = str(cfg.get("model", "mimo-v2.5-asr"))
        max_retries = int(cfg.get("max_retries", _DEFAULT_MAX_RETRIES))

        # 读取音频并编码为 data URL
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        suffix = audio_path.suffix.lower().lstrip(".")
        mime_map = {
            "wav": "audio/wav", "mp3": "audio/mp3",
            "m4a": "audio/mp4", "aac": "audio/aac", "ogg": "audio/ogg",
        }
        mime = mime_map.get(suffix, "audio/wav")
        data_url = f"data:{mime};base64,{audio_b64}"
        fmt = suffix if suffix in mime_map else "wav"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": data_url, "format": fmt},
                    }
                ],
            }
        ]

        # 重试机制
        last_exc = None
        for retry in range(max_retries):
            try:
                response = client.chat.completions.create(model=model, messages=messages)
                text = response.choices[0].message.content if response.choices else ""
                break
            except Exception as exc:
                last_exc = exc
                if retry < max_retries - 1:
                    print(f"  [asr] MiMo API error (retry {retry + 1}/{max_retries}): {exc}")
                continue
        else:
            print(f"  [asr] MiMo API failed after {max_retries} retries: {last_exc}")
            raise last_exc

        text = (text or "").strip()
        if not text:
            return []

        try:
            duration = get_duration(audio_path)
        except Exception:
            duration = 30.0

        return [TranscriptSegment(
            start=time_offset,
            end=time_offset + duration,
            text=text,
        )]

    def _transcribe_chunked(
        self,
        audio_path: Path,
        total_duration: float,
        chunk_seconds: int,
        cfg: dict[str, Any],
    ) -> list[TranscriptSegment]:
        from ..ffmpeg import cut_audio

        max_workers = int(cfg.get("max_workers", _DEFAULT_MAX_WORKERS))

        # 准备所有 chunk
        chunks: list[tuple[int, Path, float]] = []
        chunk_start = 0.0
        chunk_index = 0

        with tempfile.TemporaryDirectory(prefix="mimo_asr_") as tmp_dir:
            while chunk_start < total_duration:
                chunk_end = min(chunk_start + chunk_seconds, total_duration)

                # 切割为 mp3（比 wav 小约 4 倍）
                chunk_path = Path(tmp_dir) / f"chunk_{chunk_index:04d}.mp3"
                cut_audio(audio_path, chunk_path, chunk_start, chunk_end)
                chunks.append((chunk_index, chunk_path, chunk_start))

                chunk_index += 1
                chunk_start = chunk_end

            # 并发处理
            all_segments: list[tuple[int, list[TranscriptSegment]]] = []

            def process_chunk(item: tuple[int, Path, float]) -> tuple[int, list[TranscriptSegment]]:
                idx, path, offset = item
                segs = self._transcribe_chunk(path, offset, cfg)
                return idx, segs

            print(f"  [asr] Processing {len(chunks)} chunks with {max_workers} workers...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_chunk, item): item[0]
                    for item in chunks
                }
                for future in as_completed(futures):
                    idx, segs = future.result()
                    all_segments.append((idx, segs))
                    print(f"  [asr] Chunk {idx + 1}/{len(chunks)} done, found {len(segs)} segment(s)")

        # 按 chunk 顺序排列
        all_segments.sort(key=lambda x: x[0])
        result: list[TranscriptSegment] = []
        for _, segs in all_segments:
            result.extend(segs)

        return result
