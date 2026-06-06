from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from ..models import TranscriptSegment
from .base import ASRBackend


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
                    try:
                        os.add_dll_directory(str(candidate))
                    except OSError:
                        pass
                    nvidia_paths.append(str(candidate))

    cuda_path = os.environ.get("CUDA_PATH")
    system_cuda_paths = [Path(cuda_path)] if cuda_path else []
    toolkit_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit_base.is_dir():
        system_cuda_paths.extend(sorted(toolkit_base.iterdir(), reverse=True))

    for cuda_dir in system_cuda_paths:
        bin_dir = cuda_dir / "bin"
        if bin_dir.is_dir() and any(bin_dir.glob("cublas*.dll")):
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass
            nvidia_paths.append(str(bin_dir))

    if nvidia_paths:
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(nvidia_paths) + os.pathsep + current_path


_add_nvidia_dll_directories()


class FasterWhisperBackend(ASRBackend):
    def __init__(self, settings: dict[str, Any]) -> None:
        super().__init__(settings)
        self._model: Any = None

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
        kwargs: dict[str, Any] = {"device": device}
        if compute_type != "default":
            kwargs["compute_type"] = compute_type
        self._model = WhisperModel(model_name, **kwargs)
        return self._model

    def _reset_model(self) -> None:
        self._model = None

    def transcribe(self, audio_path: str | Path) -> list[TranscriptSegment]:
        model = self._load_model()
        try:
            segments, _info = self._transcribe_with_model(model, audio_path)
        except RuntimeError as exc:
            device = str(self.settings.get("device", "auto")).lower()
            if device == "cpu" or not _is_cuda_runtime_error(exc):
                raise
            print(f"CUDA ASR failed ({exc}); retrying on CPU with int8 compute.")
            print(f"[debug] Current PATH (first 5): {os.environ['PATH'][:500]}")
            self._reset_model()
            model = self._load_model(device_override="cpu", compute_type_override="int8")
            segments, _info = self._transcribe_with_model(model, audio_path)

        results: list[TranscriptSegment] = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                results.append(TranscriptSegment(
                    start=float(seg.start),
                    end=float(seg.end),
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
