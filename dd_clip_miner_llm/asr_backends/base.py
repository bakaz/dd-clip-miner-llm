from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models import TranscriptSegment


class ASRBackend(ABC):
    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> list[TranscriptSegment]:
        ...
