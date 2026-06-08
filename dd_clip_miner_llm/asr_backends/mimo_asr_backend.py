"""Xiaomi MiMo-V2.5-ASR 后端

通过 OpenAI 兼容 API (chat/completions + input_audio) 调用 MiMo-V2.5-ASR，支持：
- 中文方言、歌词识别、噪音环境
- 自动语言检测
- 5 秒分段处理（获取细粒度时间戳）
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any

from ..models import TranscriptSegment
from .base import ASRBackend

# 默认 5 秒一个 chunk，用于细粒度时间戳
_DEFAULT_TIMESTAMP_CHUNK = 5
# API 单次最大 10MB，5 秒 mp3 ≈ 80KB，远低于限制
_MAX_AUDIO_MB = 10


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

        try:
            response = client.chat.completions.create(model=model, messages=messages)
            text = response.choices[0].message.content if response.choices else ""
        except Exception as exc:
            print(f"  [asr] MiMo API error: {exc}")
            raise

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

        all_segments: list[TranscriptSegment] = []

        with tempfile.TemporaryDirectory(prefix="mimo_asr_") as tmp_dir:
            chunk_start = 0.0
            chunk_index = 0

            while chunk_start < total_duration:
                chunk_end = min(chunk_start + chunk_seconds, total_duration)

                # 切割为 mp3（比 wav 小约 4 倍）
                chunk_path = Path(tmp_dir) / f"chunk_{chunk_index:04d}.mp3"
                cut_audio(audio_path, chunk_path, chunk_start, chunk_end)

                chunk_segments = self._transcribe_chunk(chunk_path, chunk_start, cfg)
                all_segments.extend(chunk_segments)

                print(f"  [asr] Chunk {chunk_index + 1}: {chunk_start:.0f}s - {chunk_end:.0f}s -> {len(chunk_segments)} segments")

                chunk_index += 1
                chunk_start = chunk_end

        return all_segments
