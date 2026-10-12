"""Runtime configuration, read from environment variables.

Clerk runs in one of two modes:

* ``live``    - every model call goes to Nebius Token Factory and web research
                goes to Tavily. Requires NEBIUS_API_KEY and the model IDs.
* ``offline`` - a deterministic simulator stands in for the models so the app,
                the tests and the eval harness run without network or keys.
                The UI labels this mode clearly; it is never presented as
                real model output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw is not None else default


def _env_float(name: str) -> float | None:
    raw = _env(name)
    return float(raw) if raw is not None else None


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens. None means 'not configured' (cost shown as unknown)."""

    input_per_m: float | None = None
    output_per_m: float | None = None

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        if self.input_per_m is None or self.output_per_m is None:
            return None
        return (prompt_tokens * self.input_per_m + completion_tokens * self.output_per_m) / 1_000_000


@dataclass(frozen=True)
class Settings:
    mode: str
    nebius_api_key: str | None
    nebius_base_url: str
    tavily_api_key: str | None
    model_super: str | None
    model_nano: str | None
    model_vision: str | None
    model_embed: str | None
    prices: dict[str, ModelPrice] = field(default_factory=dict)
    db_path: Path = Path("clerk.db")
    static_dir: Path | None = None
    max_upload_bytes: int = 2 * 1024 * 1024
    max_document_chars: int = 20_000
    audits_per_workspace_per_hour: int = 20
    daily_llm_call_limit: int = 3000
    llm_timeout_seconds: float = 90.0
    allowed_origins: tuple[str, ...] = ()

    @property
    def live(self) -> bool:
        return self.mode == "live"

    def missing_live_settings(self) -> list[str]:
        missing = []
        if not self.nebius_api_key:
            missing.append("NEBIUS_API_KEY")
        if not self.model_super:
            missing.append("CLERK_MODEL_SUPER")
        if not self.model_nano:
            missing.append("CLERK_MODEL_NANO")
        return missing

    def price_for(self, model: str) -> ModelPrice:
        return self.prices.get(model, ModelPrice())


def load_settings() -> Settings:
    mode = (_env("CLERK_MODE", "offline") or "offline").lower()
    if mode not in {"live", "offline"}:
        raise ValueError(f"CLERK_MODE must be 'live' or 'offline', got {mode!r}")

    model_super = _env("CLERK_MODEL_SUPER")
    model_nano = _env("CLERK_MODEL_NANO")
    model_vision = _env("CLERK_MODEL_VISION")
    model_embed = _env("CLERK_MODEL_EMBED")

    prices: dict[str, ModelPrice] = {}
    for model, prefix in (
        (model_super, "CLERK_PRICE_SUPER"),
        (model_nano, "CLERK_PRICE_NANO"),
        (model_vision, "CLERK_PRICE_VISION"),
        (model_embed, "CLERK_PRICE_EMBED"),
    ):
        if model:
            prices[model] = ModelPrice(_env_float(f"{prefix}_IN"), _env_float(f"{prefix}_OUT"))

    static = _env("CLERK_STATIC_DIR")
    origins = _env("CLERK_ALLOWED_ORIGINS", "") or ""

    settings = Settings(
        mode=mode,
        nebius_api_key=_env("NEBIUS_API_KEY"),
        nebius_base_url=_env("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/") or "",
        tavily_api_key=_env("TAVILY_API_KEY"),
        model_super=model_super,
        model_nano=model_nano,
        model_vision=model_vision,
        model_embed=model_embed,
        prices=prices,
        db_path=Path(_env("CLERK_DB_PATH", "clerk.db") or "clerk.db"),
        static_dir=Path(static) if static else None,
        max_upload_bytes=_env_int("CLERK_MAX_UPLOAD_BYTES", 2 * 1024 * 1024),
        max_document_chars=_env_int("CLERK_MAX_DOCUMENT_CHARS", 20_000),
        audits_per_workspace_per_hour=_env_int("CLERK_AUDITS_PER_HOUR", 20),
        daily_llm_call_limit=_env_int("CLERK_DAILY_LLM_CALL_LIMIT", 3000),
        llm_timeout_seconds=float(_env("CLERK_LLM_TIMEOUT", "90") or 90),
        allowed_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
    )
    if settings.live:
        missing = settings.missing_live_settings()
        if missing:
            raise RuntimeError(
                "CLERK_MODE=live but these settings are missing: "
                + ", ".join(missing)
                + ". Run `python scripts/list_models.py` to find the exact Nemotron model IDs."
            )
    return settings
