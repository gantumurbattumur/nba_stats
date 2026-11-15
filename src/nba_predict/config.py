"""Settings: env prefix ``NBA_`` (plus unprefixed ``ODDS_API_KEY``), read from ``.env``."""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NBA_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    db_path: str = "data/nba.db"
    windows: tuple[int, ...] = (5, 8, 10, 15)
    edge_threshold: float = 0.03
    current_season: str = "2026-27"
    train_seasons: tuple[str, ...] = ("2023-24", "2024-25", "2025-26")
    nba_delay_s: float = 0.6
    espn_delay_s: float = 1.0
    odds_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("ODDS_API_KEY", "NBA_ODDS_API_KEY")
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
