from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import TranscriptSegment
from .base import ASRBackend


class FunASRBackend(ASRBackend):
    def __init__(self, settings: dict[str, Any]) -> None:
        super().__init__(settings)
        self._model: Any = None

    @property
    def funasr_settings(self) -> dict[str, Any]:
        value = self.settings.get("funasr", {})
        return value if isinstance(value, dict) else {}

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
        model = self._load_model()
        cfg = self.funasr_settings
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

        result = model.generate(**generate_kwargs)
        return funasr_result_to_segments(result, audio_path)


def funasr_result_to_segments(result: Any, audio_path: str | Path) -> list[TranscriptSegment]:
    items = result if isinstance(result, list) else [result]
    segments: list[TranscriptSegment] = []
    fallback_texts: list[str] = []

    for item in items:
        text = _value(item, "text", "sentence", "transcript")
        if text:
            fallback_texts.append(str(text).strip())
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
