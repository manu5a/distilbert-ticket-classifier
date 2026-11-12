"""App settings, read from environment variables (or a .env file)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    app_name: str = "ticket-classifier"
    environment: str = "local"
    log_level: str = "INFO"

    database_url: str = "postgresql+psycopg://tickets:tickets@localhost:5432/tickets"
    model_dir: str = "models/ticket-classifier"
    model_version: str = "local-dev"

    max_text_length: int = 5000
    max_batch_size: int = 32
    # Predictions below this confidence are flagged for a human to check.
    review_threshold: float = 0.6


@lru_cache
def get_settings() -> Settings:
    return Settings()
