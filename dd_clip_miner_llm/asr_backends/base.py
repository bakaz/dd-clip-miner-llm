from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models import TranscriptSegment


class ASRBackend(ABC):
    def __init__(self, settings: dict[str, Any], runtime_context: dict[str, Any] | None = None) -> None:
        self.settings = settings
        self.runtime_context = runtime_context or {}

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> list[TranscriptSegment]:
        ...
