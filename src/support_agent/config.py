"""Application settings, loaded from environment variables or a .env file.

Secrets (the OpenAI key, the service API key) are never hard-coded: they come
from the environment and are wrapped in SecretStr so they don't leak into logs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SUPPORT_", extra="ignore")

    # "offline" runs with no API key (keyword classifier + templated replies),
    # which is what tests and CI use. "openai" uses the LLM for every step.
    llm_provider: Literal["offline", "openai"] = "offline"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPPORT_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    llm_timeout_s: float = 20.0
    llm_max_retries: int = 2

    data_dir: Path = PROJECT_ROOT / "data"
    ticket_db_path: Path = PROJECT_ROOT / "tickets.db"

    # Minimum similarity for a knowledge-base hit to count as an answer.
    retrieval_threshold: float = 0.20
    retrieval_top_k: int = 3
    # After this many clarification questions, hand over to a human.
    max_clarifications: int = 1

    # Optional shared key for the HTTP API (sent as the X-API-Key header).
    api_key: SecretStr | None = None
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
