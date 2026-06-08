"""FunASR 后端（支持 SenseVoiceSmall / Paraformer）

功能：
- 自动分段（timestamp_chunk_seconds 控制粒度，默认 5 秒）
- 并发处理多个 chunk（max_workers 控制并发数）
- 支持 SenseVoiceSmall、Paraformer 等 FunASR 模型
"""
from __future__ import annotations

import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..models import TranscriptSegment
from .base import ASRBackend

# SenseVoice 特殊标签模式
_SENSEVOICE_TAG_RE = re.compile(r"<\|[^|]*\|>")

# 默认配置
_DEFAULT_TIMESTAMP_CHUNK = 5  # 5 秒一个 chunk，用于细粒度时间戳
_DEFAULT_MAX_WORKERS = 4      # 默认并发数


class FunASRBackend(ASRBackend):
    def __init__(self, settings: dict[str, Any]) -> None:
        super().__init__(settings)
        self._model: Any = None
        self._model_lock = threading.Lock()

    @property
    def funasr_settings(self) -> dict[str, Any]:
        """获取 funasr 配置，兼容新旧格式。"""
        value = self.settings.get("funasr", {})
        if isinstance(value, dict) and value:
            return value
        return self.settings

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError("funasr not installed. pip install funasr") from exc

        cfg = self.funasr_settings
        model_name = str(cfg.get("model", self.settings.get("qwen3_model", "Qwen/Qwen3-ASR-0.6B")))
        kwargs: dict[str, Any] = {
            "model": model_name,
            "device": _resolve_device(str(cfg.get("device", self.settings.get("device", "auto")))),
        }
        for key in ("hub", "trust_remote_code", "vad_model", "punc_model", "spk_model"):
            if key in cfg and cfg[key] is not None:
                kwargs[key] = cfg[key]
        for key in ("vad_kwargs", "punc_kwargs", "spk_kwargs", "model_kwargs"):
            if isinstance(cfg.get(key), dict):
                kwargs[key] = cfg[key]

        self._model = AutoModel(**kwargs)
        return self._model

    def transcribe(self, audio_path: str | Path) -> list[TranscriptSegment]:
        from ..ffmpeg import get_duration

        audio_path = Path(audio_path)
        cfg = self.funasr_settings
        chunk_seconds = int(cfg.get("timestamp_chunk_seconds", _DEFAULT_TIMESTAMP_CHUNK))
        max_workers = int(cfg.get("max_workers", _DEFAULT_MAX_WORKERS))

        duration = get_duration(audio_path)
        total_chunks = int(duration // chunk_seconds) + (1 if duration % chunk_seconds > 0 else 0)

        if total_chunks <= 1:
            return self._transcribe_chunk(audio_path, 0.0, cfg)

        print(f"[asr] Audio {duration:.0f}s -> {total_chunks} chunks of {chunk_seconds}s (max_workers={max_workers})")
        return self._transcribe_chunked(audio_path, duration, chunk_seconds, cfg, max_workers)

    def _transcribe_chunk(
        self,
        audio_path: Path,
        time_offset: float,
        cfg: dict[str, Any],
    ) -> list[TranscriptSegment]:
        model = self._load_model()
        generate_kwargs: dict[str, Any] = {
            "input": str(audio_path),
            "batch_size": int(cfg.get("batch_size", 1)),
        }
        language = cfg.get("language", self.settings.get("language"))
        if language:
            generate_kwargs["language"] = language
        extra = cfg.get("generate_kwargs", {})
        if isinstance(extra, dict):
            generate_kwargs.update(extra)

        with self._model_lock:
            result = model.generate(**generate_kwargs)
        segments = funasr_result_to_segments(result, audio_path)

        # 添加时间偏移
        if time_offset > 0:
            segments = [
                TranscriptSegment(
                    start=seg.start + time_offset,
                    end=seg.end + time_offset,
                    text=seg.text,
                )
                for seg in segments
            ]

        return segments

    def _transcribe_chunked(
        self,
        audio_path: Path,
        total_duration: float,
        chunk_seconds: int,
        cfg: dict[str, Any],
        max_workers: int = 4,
    ) -> list[TranscriptSegment]:
        from ..ffmpeg import cut_audio

        # 准备所有 chunk 的切割任务
        chunks: list[tuple[int, float, float]] = []  # (index, start, end)
        chunk_start = 0.0
        chunk_index = 0
        while chunk_start < total_duration:
            chunk_end = min(chunk_start + chunk_seconds, total_duration)
            chunks.append((chunk_index, chunk_start, chunk_end))
            chunk_index += 1
            chunk_start = chunk_end

        # 切割所有 chunk 到临时目录
        with tempfile.TemporaryDirectory(prefix="funasr_chunk_") as tmp_dir:
            chunk_paths: list[tuple[int, Path, float]] = []
            for idx, start, end in chunks:
                chunk_path = Path(tmp_dir) / f"chunk_{idx:04d}.wav"
                cut_audio(audio_path, chunk_path, start, end)
                chunk_paths.append((idx, chunk_path, start))

            # 并发处理
            all_segments: list[tuple[int, list[TranscriptSegment]]] = []

            def process_chunk(item: tuple[int, Path, float]) -> tuple[int, list[TranscriptSegment]]:
                idx, path, offset = item
                segs = self._transcribe_chunk(path, offset, cfg)
                return idx, segs

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_chunk, item): item[0]
                    for item in chunk_paths
                }
                for future in as_completed(futures):
                    idx, segs = future.result()
                    all_segments.append((idx, segs))
                    print(f"  [asr] Chunk {idx + 1}/{len(chunks)}: {len(segs)} segments")

        # 按 chunk 顺序排列
        all_segments.sort(key=lambda x: x[0])
        result: list[TranscriptSegment] = []
        for _, segs in all_segments:
            result.extend(segs)

        return result


def funasr_result_to_segments(result: Any, audio_path: str | Path) -> list[TranscriptSegment]:
    items = result if isinstance(result, list) else [result]
    segments: list[TranscriptSegment] = []
    fallback_texts: list[str] = []

    for item in items:
        text = _value(item, "text", "sentence", "transcript")
        if text:
            text = _clean_sensevoice_text(str(text))
            fallback_texts.append(text.strip())
        timestamps = _value(item, "timestamp", "timestamps", "time_stamps", "sentence_info", "segments")
        segments.extend(_timestamps_to_segments(timestamps, fallback_text=str(text or "")))

    if segments:
        return segments

    text = " ".join(t for t in fallback_texts if t).strip()
    if not text:
        return []
    try:
        from ..ffmpeg import get_duration
        duration = get_duration(audio_path)
    except Exception:
        duration = 0.0
    return [TranscriptSegment(start=0.0, end=float(duration), text=text)]


def _clean_sensevoice_text(text: str) -> str:
    """清理 SenseVoice 输出的特殊标签"""
    return _SENSEVOICE_TAG_RE.sub("", text).strip()


def _timestamps_to_segments(timestamps: Any, fallback_text: str = "") -> list[TranscriptSegment]:
    if not timestamps:
        return []
    if isinstance(timestamps, dict):
        timestamps = timestamps.get("segments") or timestamps.get("items") or [timestamps]

    segments: list[TranscriptSegment] = []
    for index, item in enumerate(timestamps):
        start, end, text = _parse_timestamp_item(item)
        if start is None or end is None:
            continue
        clean_text = str(text or "").strip()
        if not clean_text and index == 0:
            clean_text = fallback_text.strip()
        if not clean_text:
            continue
        segments.append(TranscriptSegment(start=float(start), end=float(end), text=clean_text))
    return segments


def _parse_timestamp_item(item: Any) -> tuple[float | None, float | None, str | None]:
    if isinstance(item, (list, tuple)):
        if len(item) >= 3:
            return _normalize_time(item[0]), _normalize_time(item[1]), str(item[2])
        if len(item) >= 2:
            return _normalize_time(item[0]), _normalize_time(item[1]), None

    start = _value(item, "start", "start_time", "begin", "begin_time")
    end = _value(item, "end", "end_time", "stop", "stop_time")
    text = _value(item, "text", "word", "sentence", "transcript")
    return _normalize_time(start), _normalize_time(end), None if text is None else str(text)


def _normalize_time(value: Any) -> float | None:
    number = _float_or_none(value)
    if number is None:
        return None
    if number > 1000:
        return number / 1000.0
    return number


def _value(item: Any, *names: str) -> Any:
    if item is None:
        return None
    if isinstance(item, dict):
        for name in names:
            if name in item:
                return item[name]
        return None
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_device(device: str) -> str:
    value = (device or "auto").lower()
    if value == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return value
