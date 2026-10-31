"""Керування життєвим циклом (Lifespan) застосунку FastAPI.

Завантажує пошуковий індекс у пам'ять один раз під час запуску сервера,
а не на кожен вхідний HTTP-запит, та коректно вивільняє ресурси під час зупинки.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from findex.index import Index, load_index
from findex.web.dependencies import get_settings

logger = logging.getLogger("findex.web.lifespan")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Асинхронний контекстний менеджер життєвого циклу веб-сервісу."""
    settings = get_settings()
    app.state.index = None

    if settings.index_path.exists():
        try:
            logger.info("Завантаження індексу з %s...", settings.index_path)
            loaded_index: Index = load_index(settings.index_path)
            app.state.index = loaded_index
            logger.info(
                "Індекс успішно завантажено: %d документів, %d унікальних термів.",
                loaded_index.num_docs,
                len(loaded_index),
            )
        except Exception as exc:  # noqa: BLE001 - перехоплюємо будь-яке пошкодження індексу для переходу в health 503
            logger.error("Не вдалося завантажити індекс з %s: %s", settings.index_path, exc)
            app.state.index = None
    else:
        logger.warning(
            "Файл індексу %s не знайдено. Сервіс запущено, маршрут /health повертатиме 503.",
            settings.index_path,
        )

    try:
        yield
    finally:
        logger.info("Зупинка веб-сервісу findex...")
        app.state.index = None
