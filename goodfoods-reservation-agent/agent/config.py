"""Central configuration loaded from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    api_key: str = os.getenv("LLM_API_KEY", "")
    model: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    timezone: str = os.getenv("APP_TIMEZONE", "Asia/Kolkata")
    request_timeout: float = float(os.getenv("LLM_TIMEOUT", "30"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    max_tool_iterations: int = int(os.getenv("AGENT_MAX_STEPS", "6"))
    history_turns: int = int(os.getenv("AGENT_HISTORY_TURNS", "12"))


settings = Settings()
