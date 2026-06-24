"""FunASR 后端（支持 SenseVoiceSmall / Paraformer）

功能：
- 自动分段（timestamp_chunk_seconds 控制粒度，默认 5 秒）
- 并发处理多个 chunk（max_workers 控制并发数）
- 支持 SenseVoiceSmall、Paraformer 等 FunASR 模型
"""
from __future__ import annotations

import re
import threading
import unicodedata
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
_DEFAULT_QWEN3_FORCED_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"


class FunASRBackend(ASRBackend):
    def __init__(self, settings: dict[str, Any], runtime_context: dict[str, Any] | None = None) -> None:
        super().__init__(settings, runtime_context=runtime_context)
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

        with self._model_lock:
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
            for key in (
                "hub",
                "trust_remote_code",
                "vad_model",
                "punc_model",
                "spk_model",
                "dtype",
                "model_revision",
                "disable_update",
                "forced_aligner",
                "max_inference_batch_size",
                "max_new_tokens",
            ):
                if key in cfg and cfg[key] is not None:
                    kwargs[key] = cfg[key]
            if (
                _is_qwen3_model(model_name)
                and _wants_timestamps(cfg)
                and not kwargs.get("forced_aligner")
            ):
                kwargs["forced_aligner"] = _DEFAULT_QWEN3_FORCED_ALIGNER
            for key in ("vad_kwargs", "punc_kwargs", "spk_kwargs", "model_kwargs", "forced_aligner_kwargs"):
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
        generate_kwargs: dict[str, Any] = {"input": str(audio_path)}

        if cfg.get("batch_size") is not None:
            generate_kwargs["batch_size"] = int(cfg.get("batch_size", 1))
        language = cfg.get("language", self.settings.get("language"))
        if language:
            generate_kwargs["language"] = language
        for key in ("return_time_stamps", "output_timestamp"):
            if key in cfg and cfg[key] is not None:
                generate_kwargs[key] = bool(cfg[key])
        extra = cfg.get("generate_kwargs", {})
        if isinstance(extra, dict):
            generate_kwargs.update(extra)

        with self._model_lock:
            result = model.generate(**generate_kwargs)
        if _wants_timestamps(cfg) and _missing_requested_timestamps(result):
            raise RuntimeError(
                "Qwen3-ASR timestamp output was requested, but FunASR returned text without timestamps. "
                "Configure a working funasr.forced_aligner (default: "
                f"{_DEFAULT_QWEN3_FORCED_ALIGNER}) and keep generate_kwargs.return_time_stamps=true "
                "or generate_kwargs.output_timestamp=true."
            )
        segments = funasr_result_to_segments(result, audio_path, cfg=cfg)

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

        chunk_dir = self._resolve_chunk_dir(audio_path, cfg)
        from ..ffmpeg.fsutil import safe_rmtree

        safe_rmtree(chunk_dir)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        keep_chunk_audio = bool(cfg.get("keep_chunk_audio", False))
        try:
            chunk_paths: list[tuple[int, Path, float]] = []
            for idx, start, end in chunks:
                chunk_path = chunk_dir / f"chunk_{idx:04d}.wav"
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
        except Exception:
            print(f"  [asr] FunASR chunk audio preserved for debugging: {chunk_dir}")
            raise
        else:
            if not keep_chunk_audio:
                safe_rmtree(chunk_dir)

        # 按 chunk 顺序排列
        all_segments.sort(key=lambda x: x[0])
        result: list[TranscriptSegment] = []
        for _, segs in all_segments:
            result.extend(segs)

        return result

    def _resolve_chunk_dir(self, audio_path: Path, cfg: dict[str, Any]) -> Path:
        configured = cfg.get("chunk_dir")
        if configured:
            base = Path(str(configured)).expanduser()
        else:
            asr_dir = self.runtime_context.get("asr_dir")
            if asr_dir:
                base = Path(asr_dir) / "funasr_chunks"
            else:
                base = Path(audio_path).parent / "funasr_chunks"
        return base / audio_path.stem


def is_punctuation(char: str) -> bool:
    """判断是否是标点符号"""
    if unicodedata.category(char).startswith('P'):
        return True
    if char in '。！？.!?\n；;，,':
        return True
    return False


def align_text_with_timestamps(
    text: str,
    timestamps: list,
) -> list[tuple[str, int, int]]:
    """将含标点的文本与不含标点的时间戳对齐

    标点不消耗时间戳，复用前一个字符的结束时间。
    正常字符每个消耗一个时间戳。
    """
    result: list[tuple[str, int, int]] = []
    ts_idx = 0

    for char in text:
        if is_punctuation(char):
            if result:
                prev_end = result[-1][2]
                result.append((char, prev_end, prev_end))
            else:
                result.append((char, 0, 0))
        else:
            if ts_idx < len(timestamps):
                start, end = timestamps[ts_idx]
                result.append((char, int(start), int(end)))
                ts_idx += 1
            else:
                if result:
                    prev_end = result[-1][2]
                    result.append((char, prev_end, prev_end))

    return result


def merge_to_sentences(
    aligned: list[tuple[str, int, int]],
    punctuation: str = '。！？.!?\n',
    max_duration_ms: int = 5000,
) -> list[dict[str, Any]]:
    """将对齐后的字符合并为句子

    遇到标点或超过 max_duration_ms 时断句。
    """
    sentences: list[dict[str, Any]] = []
    current_text = ""
    current_start: int | None = None
    current_end = 0

    for char, start, end in aligned:
        if current_start is None:
            current_start = start

        current_text += char
        current_end = end

        if char in punctuation:
            sentences.append({
                "text": current_text.strip(),
                "start": current_start,
                "end": current_end,
            })
            current_text = ""
            current_start = None
        elif current_end - current_start >= max_duration_ms:
            sentences.append({
                "text": current_text.strip(),
                "start": current_start,
                "end": current_end,
            })
            current_text = ""
            current_start = None

    if current_text.strip() and current_start is not None:
        sentences.append({
            "text": current_text.strip(),
            "start": current_start,
            "end": current_end,
        })

    return sentences


def repair_qwen3_zero_duration_segments(
    segments: list[TranscriptSegment],
) -> tuple[list[TranscriptSegment], dict[str, int]]:
    """Merge zero-duration Qwen3 segments into adjacent segments (scheme C)."""
    sorted_segments = sorted(segments, key=lambda item: (item.start, item.end))
    result: list[TranscriptSegment] = []
    pending_prefix = ""
    stats = {"merged_count": 0, "dropped_empty_count": 0, "dropped_orphan_count": 0}

    for segment in sorted_segments:
        text = segment.text
        stripped = text.strip()
        is_zero_duration = segment.end <= segment.start

        if not is_zero_duration:
            if pending_prefix:
                text = pending_prefix + text
                pending_prefix = ""
                stats["merged_count"] += 1
            if stripped:
                result.append(TranscriptSegment(segment.start, segment.end, text))
            elif text:
                stats["dropped_empty_count"] += 1
            continue

        if not stripped:
            stats["dropped_empty_count"] += 1
            continue

        if result:
            previous = result[-1]
            result[-1] = TranscriptSegment(previous.start, previous.end, previous.text + text)
            stats["merged_count"] += 1
        else:
            pending_prefix += text

    if pending_prefix:
        stats["dropped_orphan_count"] += 1

    return result, stats


def postprocess_qwen3_asr(
    text: str,
    timestamps: list,
    max_sentence_duration_ms: int = 5000,
) -> list[TranscriptSegment]:
    """将 Qwen3-ASR 输出转换为句子级别 TranscriptSegment

    完整后处理流水线：对齐文本+时间戳 → 合并句子 → 转为 TranscriptSegment。
    """
    aligned = align_text_with_timestamps(text, timestamps)
    sentences = merge_to_sentences(aligned, max_duration_ms=max_sentence_duration_ms)

    segments = [
        TranscriptSegment(
            start=s["start"] / 1000.0,
            end=s["end"] / 1000.0,
            text=s["text"],
        )
        for s in sentences
        if s.get("text", "").strip()
    ]
    repaired, _ = repair_qwen3_zero_duration_segments(segments)
    return repaired


def funasr_result_to_segments(
    result: Any,
    audio_path: str | Path,
    cfg: dict[str, Any] | None = None,
) -> list[TranscriptSegment]:
    items = result if isinstance(result, list) else [result]
    segments: list[TranscriptSegment] = []
    fallback_texts: list[str] = []

    # Extract lyrics config
    lyrics_cfg: dict[str, Any] = {}
    if isinstance(cfg, dict) and isinstance(cfg.get("lyrics"), dict):
        lyrics_cfg = cfg["lyrics"]
    lyrics_enabled = bool(lyrics_cfg.get("enabled", False))
    qwen3_processed = False

    for item in items:
        text = _value(item, "text", "sentence", "transcript")
        if text:
            text = _clean_sensevoice_text(str(text))
            fallback_texts.append(text.strip())

        # Check for structured timestamps first
        timestamps = _value(item, "timestamp", "timestamps", "time_stamps", "sentence_info", "segments")

        # Detect Qwen3-ASR format: text + timestamp (list of [start, end] pairs)
        if text and timestamps and isinstance(timestamps, list) and len(timestamps) > 0:
            first_ts = timestamps[0]
            if isinstance(first_ts, (list, tuple)) and len(first_ts) == 2:
                # Qwen3-ASR format: align text with timestamps
                max_duration_ms = int(lyrics_cfg.get("max_sentence_duration_ms", 5000))
                timestamp_ms = _timestamp_pairs_to_milliseconds(timestamps, audio_path)
                aligned_segments = postprocess_qwen3_asr(str(text), timestamp_ms, max_duration_ms)

                if lyrics_enabled:
                    # Apply lyrics splitting to each segment
                    max_line_chars = int(lyrics_cfg.get("max_line_chars", 24))
                    sentence_punctuation = str(lyrics_cfg.get("sentence_punctuation", "。！？.!?\n"))
                    for seg in aligned_segments:
                        lyrics_text = split_lyrics_text(seg.text, max_line_chars, sentence_punctuation)
                        # Split lyrics_text by newlines and create segments
                        lines = lyrics_text.split("\n")
                        line_duration = (seg.end - seg.start) / len(lines) if lines else 0
                        for i, line in enumerate(lines):
                            if line.strip():
                                segments.append(TranscriptSegment(
                                    start=seg.start + i * line_duration,
                                    end=seg.start + (i + 1) * line_duration,
                                    text=line.strip(),
                                ))
                else:
                    segments.extend(aligned_segments)
                qwen3_processed = True
                continue

        # Existing logic for other formats (SenseVoice/Paraformer)
        segments.extend(_timestamps_to_segments(timestamps, fallback_text=str(text or "")))

    if qwen3_processed and segments:
        segments, stats = repair_qwen3_zero_duration_segments(segments)
        if any(stats.values()):
            print(
                "  [asr] Qwen3 zero-duration repair:"
                f" merged={stats['merged_count']},"
                f" dropped_empty={stats['dropped_empty_count']},"
                f" dropped_orphan={stats['dropped_orphan_count']}",
            )

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


def _is_qwen3_model(model_name: str) -> bool:
    return "qwen3-asr" in str(model_name).lower()


def _wants_timestamps(cfg: dict[str, Any]) -> bool:
    if bool(cfg.get("return_time_stamps") or cfg.get("output_timestamp")):
        return True
    extra = cfg.get("generate_kwargs", {})
    if not isinstance(extra, dict):
        return False
    return bool(extra.get("return_time_stamps") or extra.get("output_timestamp"))


def _missing_requested_timestamps(result: Any) -> bool:
    items = result if isinstance(result, list) else [result]
    saw_text = False
    for item in items:
        text = _value(item, "text", "sentence", "transcript")
        if not text or not str(text).strip():
            continue
        saw_text = True
        timestamps = _value(item, "timestamp", "timestamps", "time_stamps", "sentence_info", "segments")
        if not timestamps:
            return True
    return False


def _timestamp_pairs_to_milliseconds(timestamps: list, audio_path: str | Path) -> list[tuple[int, int]]:
    pairs: list[tuple[float, float]] = []
    for item in timestamps:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start = _float_or_none(item[0])
        end = _float_or_none(item[1])
        if start is None or end is None:
            continue
        pairs.append((start, end))
    if not pairs:
        return []

    max_end = max(end for _, end in pairs)
    duration = _duration_or_none(audio_path)
    # Qwen forced aligner in qwen-asr 0.0.6 returns seconds through FunASR;
    # other FunASR timestamp formats commonly use milliseconds.
    if duration is not None and max_end <= duration + 1.0:
        return [(int(round(start * 1000)), int(round(end * 1000))) for start, end in pairs]
    return [(int(round(start)), int(round(end))) for start, end in pairs]


def _duration_or_none(audio_path: str | Path) -> float | None:
    try:
        from ..ffmpeg import get_duration
        return float(get_duration(audio_path))
    except Exception:
        return None


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


# ---------------------------------------------------------------------------
# 歌词文本切分工具
# ---------------------------------------------------------------------------


def _split_by_sentence_punctuation(text: str, punctuation: str) -> list[str]:
    """按句末标点切分文本，保留标点与前文。

    连续标点（如 ``啊！？``）保持在同一段。
    小数点 ``.`` 在 ``v2.5`` / ``3.14`` 等数字上下文中不作为句末标点。
    """
    # Separate '.' from other punctuation to handle v2.5, 3.14 etc.
    has_dot = "." in punctuation
    other_punct = punctuation.replace(".", "")

    if has_dot and other_punct:
        other_escaped = re.escape(other_punct)
        pattern = f"([{other_escaped}]+|(?<!\\d)\\.(?!\\d))"
    elif has_dot:
        pattern = r"((?<!\d)\.(?!\d))"
    else:
        pattern = f"([{re.escape(punctuation)}]+)"

    parts = re.split(pattern, text)

    # Recombine: merge punctuation with preceding text
    result: list[str] = []
    current = ""
    for part in parts:
        current += part
        if part and any(p in part for p in punctuation):
            stripped = current.strip()
            if stripped:
                result.append(stripped)
            current = ""
    stripped = current.strip()
    if stripped:
        result.append(stripped)

    return result


def _soft_wrap_lyric_line(line: str, max_line_chars: int) -> list[str]:
    """软切分歌词行：中文按字数、英文按空格断行。"""
    if len(line) <= max_line_chars:
        return [line]

    # Check if line contains CJK characters
    has_cjk = any(0x4E00 <= ord(c) <= 0x9FFF for c in line)

    if has_cjk:
        # Chinese: split by character count
        lines: list[str] = []
        current = ""
        for char in line:
            current += char
            if len(current) >= max_line_chars:
                lines.append(current)
                current = ""
        if current:
            lines.append(current)
        return lines
    else:
        # English: split by spaces
        words = line.split()
        lines = []
        current = ""
        for word in words:
            if current and len(current) + len(word) + 1 > max_line_chars:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            lines.append(current)
        return lines


def split_lyrics_text(
    text: str,
    max_line_chars: int = 24,
    sentence_punctuation: str = "。！？.!?\n",
) -> str:
    """将歌词文本切分为多行，适合字幕/歌词显示。

    1. 文本已含换行 → 保留原换行
    2. 否则按句末标点切分
    3. 超长行软切分（中文按字数、英文按空格）
    """
    if not text:
        return ""

    # If text already has newlines, respect them
    if "\n" in text:
        lines = text.split("\n")
    else:
        # Split by sentence punctuation
        lines = _split_by_sentence_punctuation(text, sentence_punctuation)

    # Soft wrap each line
    result_lines: list[str] = []
    for line in lines:
        if len(line) > max_line_chars:
            result_lines.extend(_soft_wrap_lyric_line(line, max_line_chars))
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def _resolve_device(device: str) -> str:
    value = (device or "auto").lower()
    if value == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return value
