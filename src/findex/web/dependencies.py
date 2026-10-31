"""Впровадження залежностей (Dependency Injection) для FastAPI.

Дозволяє маршрутам отримувати налаштування та індекс без прямого імпорту,
що спрощує тестування через dependency_overrides у TestClient.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from findex.index import Index
from findex.web.config import Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Повертає кешований екземпляр налаштувань сервісу."""
    return Settings()


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
