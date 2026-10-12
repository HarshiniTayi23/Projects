from __future__ import annotations

from ..config import Settings
from .base import LLMError, LLMProvider, LLMRequest, LLMResult
from .nebius import NebiusProvider
from .offline import OfflineProvider


def build_provider(settings: Settings) -> LLMProvider:
    if settings.live:
        return NebiusProvider(settings)
    return OfflineProvider()


__all__ = ["LLMError", "LLMProvider", "LLMRequest", "LLMResult", "NebiusProvider", "OfflineProvider", "build_provider"]
