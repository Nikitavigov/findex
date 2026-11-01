"""Впровадження залежностей (Dependency Injection) для FastAPI.

Дозволяє маршрутам отримувати налаштування та індекс без прямого імпорту,
що спрощує тестування через dependency_overrides у TestClient.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from findex.index import Index
from findex.semantic import SemanticEngine, SemanticIndex
from findex.web.config import Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Повертає кешований екземпляр налаштувань сервісу."""
    return Settings()


@lru_cache(maxsize=1)
def get_semantic_engine() -> SemanticEngine:
    """Повертає глобальний екземпляр SemanticEngine."""
    return SemanticEngine()


def get_optional_semantic_index(request: Request) -> SemanticIndex | None:
    """Повертає семантичний індекс зі стану застосунку або None, якщо не завантажений."""
    res = getattr(request.app.state, "semantic_index", None)
    return res if isinstance(res, SemanticIndex) else None


def get_optional_index(request: Request) -> Index | None:
    """Повертає пошуковий індекс зі стану застосунку або None, якщо не завантажений."""
    return getattr(request.app.state, "index", None)


def get_index(
    index: Annotated[Index | None, Depends(get_optional_index)],
) -> Index:
    """Повертає завантажений пошуковий індекс.

    Якщо індекс не було завантажено під час запуску або він недоступний,
    повертає HTTP 503 Service Unavailable.
    """
    if index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Пошуковий індекс ще не завантажений або недоступний.",
        )
    return index
