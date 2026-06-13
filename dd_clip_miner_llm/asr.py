from __future__ import annotations

from pathlib import Path
from typing import Any

from .asr_backends import build_asr_backend
from .config import get_asr_inference_mode
from .models import TranscriptSegment


class Transcriber:
    def __init__(self, config: dict[str, Any]) -> None:
        self.settings = config["asr"]
        self._backend = build_asr_backend(self.settings)
        self.inference_mode = get_asr_inference_mode(self.settings)

    def transcribe(self, audio_path: str | Path) -> list[TranscriptSegment]:
        return self._backend.transcribe(audio_path)
