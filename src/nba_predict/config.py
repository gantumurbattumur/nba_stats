"""Settings: env prefix ``NBA_`` (plus unprefixed ``ODDS_API_KEY``), read from ``<root>/.env``.

``<root>`` is the project directory (the one holding ``pyproject.toml``), found by walking up
from this package's location, then from the cwd. Relative paths (``db_path``) resolve
against it, so running ``nba`` from another folder uses the same DB and key.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def project_root() -> Path:
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for p in (start, *start.parents):
            if (p / "pyproject.toml").is_file():
                return p
    return Path.cwd().resolve()


ROOT = project_root()
ENV_FILE = ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NBA_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
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

    @field_validator("db_path", mode="after")
    @classmethod
    def _resolve_db_path(cls, v: str) -> str:
        p = Path(v).expanduser()
        return str(p if p.is_absolute() else ROOT / p)


@lru_cache
def get_settings() -> Settings:
    return Settings()
