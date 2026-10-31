"""Конфігурація веб-сервісу findex на базі pydantic-settings.

Зчитує налаштування зі змінних оточення (12-Factor App)
або з локального файлу .env у режимі розробки.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Налаштування веб-сервісу findex."""

    index_path: Path = Path("data/index.bin")
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
