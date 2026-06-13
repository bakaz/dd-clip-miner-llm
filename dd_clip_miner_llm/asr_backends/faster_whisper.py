from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from ..config import get_asr_inference_mode
from ..models import TranscriptSegment
from .base import ASRBackend


_DLL_DIRECTORY_HANDLES: list[Any] = []
_REGISTERED_DLL_DIRECTORIES: set[str] = set()


def _register_dll_directory(path: Path) -> bool:
    resolved = str(path.resolve())
    if resolved in _REGISTERED_DLL_DIRECTORIES:
        return False
    try:
        handle = os.add_dll_directory(resolved)
    except OSError:
        return False
    _DLL_DIRECTORY_HANDLES.append(handle)
    _REGISTERED_DLL_DIRECTORIES.add(resolved)
    return True


def _add_nvidia_dll_directories() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    nvidia_paths = []

    for package_name in ("nvidia.cublas", "nvidia.cuda_runtime", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            continue

        package_paths = getattr(package, "__path__", [])
        for package_dir in package_paths:
            package_dir = Path(package_dir).resolve()
            candidates = {package_dir, package_dir / "bin", package_dir / "lib"}
            candidates.update(path.parent for path in package_dir.rglob("*.dll"))

            for candidate in candidates:
                if candidate.is_dir():
                    if _register_dll_directory(candidate):
                        nvidia_paths.append(str(candidate.resolve()))

    cuda_path = os.environ.get("CUDA_PATH")
    system_cuda_paths = [Path(cuda_path)] if cuda_path else []
    toolkit_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit_base.is_dir():
        system_cuda_paths.extend(sorted(toolkit_base.iterdir(), reverse=True))

    for cuda_dir in system_cuda_paths:
        bin_dir = cuda_dir / "bin"
        if bin_dir.is_dir() and any(bin_dir.glob("cublas*.dll")):
            if _register_dll_directory(bin_dir):
                nvidia_paths.append(str(bin_dir.resolve()))

    if nvidia_paths:
        current_entries = os.environ.get("PATH", "").split(os.pathsep)
        additions = [path for path in nvidia_paths if path not in current_entries]
        if additions:
            os.environ["PATH"] = os.pathsep.join([*additions, *current_entries])


_add_nvidia_dll_directories()


class FasterWhisperBackend(ASRBackend):
    def __init__(self, settings: dict[str, Any]) -> None:
        super().__init__(settings)
        self._model: Any = None
        self._batched_model: Any = None
        self.inference_mode = get_asr_inference_mode(settings)

    def _load_model(
        self,
        device_override: str | None = None,
        compute_type_override: str | None = None,
    ) -> Any:
        if self._model is not None:
            return self._model

        _add_nvidia_dll_directories()

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper not installed. pip install faster-whisper") from exc

        model_name = str(self.settings.get("model", "small"))
        device = device_override or str(self.settings.get("device", "auto"))
        compute_type = compute_type_override or str(self.settings.get("compute_type", "default"))
        cpu_threads = int(self.settings.get("cpu_threads", 4))
        num_workers = int(self.settings.get("num_workers", 1))
        kwargs: dict[str, Any] = {"device": device}
        if compute_type != "default":
            kwargs["compute_type"] = compute_type
        if cpu_threads > 0:
            kwargs["cpu_threads"] = cpu_threads
        if num_workers > 1:
            kwargs["num_workers"] = num_workers
        self._model = WhisperModel(model_name, **kwargs)
        self._batched_model = None
        return self._model

    def _reset_model(self) -> None:
        self._model = None
        self._batched_model = None

    def _get_batched_model(self) -> Any:
        if self._batched_model is not None:
            return self._batched_model
        from faster_whisper import BatchedInferencePipeline
        model = self._load_model()
        self._batched_model = BatchedInferencePipeline(model)
        return self._batched_model

    def transcribe(self, audio_path: str | Path) -> list[TranscriptSegment]:
        if self.inference_mode == "batched":
            batch_size = int(self.settings.get("batch_size", 8))
            return self._transcribe_batched(audio_path, batch_size)
        return self._transcribe_standard(audio_path)

    def _transcribe_batched(self, audio_path: str | Path, batch_size: int) -> list[TranscriptSegment]:
        def run() -> list[TranscriptSegment]:
            batched_model = self._get_batched_model()
            segments, _info = batched_model.transcribe(
                str(audio_path),
                language=self.settings.get("language"),
                beam_size=int(self.settings.get("beam_size", 5)),
                vad_filter=bool(self.settings.get("vad_filter", True)),
                initial_prompt=self.settings.get("initial_prompt"),
                batch_size=batch_size,
                without_timestamps=False,
                word_timestamps=bool(self.settings.get("word_timestamps", True)),
            )
            return self._segments_to_results(
                segments,
                split_word_gaps=bool(
                    self.settings.get("split_on_word_gaps", True)
                ),
            )

        return self._run_with_cuda_fallback(run)

    def _transcribe_standard(self, audio_path: str | Path) -> list[TranscriptSegment]:
        def run() -> list[TranscriptSegment]:
            model = self._load_model()
            segments, _info = self._transcribe_with_model(model, audio_path)
            return self._segments_to_results(segments, split_word_gaps=False)

        return self._run_with_cuda_fallback(run)

    def _run_with_cuda_fallback(self, operation: Any) -> list[TranscriptSegment]:
        try:
            return operation()
        except RuntimeError as exc:
            device = str(self.settings.get("device", "auto")).lower()
            if device == "cpu" or not _is_cuda_runtime_error(exc):
                raise
            print(f"CUDA ASR failed ({exc}); retrying on CPU with int8 compute.")
            self._reset_model()
            self._load_model(device_override="cpu", compute_type_override="int8")
            return operation()

    def _segments_to_results(
        self,
        segments: Any,
        *,
        split_word_gaps: bool,
    ) -> list[TranscriptSegment]:
        results: list[TranscriptSegment] = []
        for seg in segments:
            if split_word_gaps and getattr(seg, "words", None):
                results.extend(self._split_segment_words(seg.words))
                continue
            text = seg.text.strip()
            if text:
                results.append(TranscriptSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=text,
                ))
        return results

    def _split_segment_words(self, words: Any) -> list[TranscriptSegment]:
        max_gap = float(self.settings.get("word_gap_seconds", 2.0))
        max_duration = float(self.settings.get("max_segment_seconds", 15.0))
        groups: list[list[Any]] = []
        current: list[Any] = []

        for word in words:
            if word.start is None or word.end is None or not str(word.word).strip():
                continue
            if current:
                gap = float(word.start) - float(current[-1].end)
                duration = float(word.end) - float(current[0].start)
                if gap > max_gap or (max_duration > 0 and duration > max_duration):
                    groups.append(current)
                    current = []
            current.append(word)
        if current:
            groups.append(current)

        results: list[TranscriptSegment] = []
        for group in groups:
            text = "".join(str(word.word) for word in group).strip()
            if text:
                results.append(TranscriptSegment(
                    start=float(group[0].start),
                    end=float(group[-1].end),
                    text=text,
                ))
        return results

    def _transcribe_with_model(self, model: Any, audio_path: str | Path) -> Any:
        return model.transcribe(
            str(audio_path),
            language=self.settings.get("language"),
            beam_size=int(self.settings.get("beam_size", 5)),
            vad_filter=bool(self.settings.get("vad_filter", True)),
            initial_prompt=self.settings.get("initial_prompt"),
            condition_on_previous_text=True,
        )


def _is_cuda_runtime_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "cuda",
            "cublas",
            "cudnn",
            "ctranslate2",
            "dll is not found",
            "dll is not found or cannot be loaded",
        )
    )
