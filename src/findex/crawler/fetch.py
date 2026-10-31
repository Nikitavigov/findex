import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Final

import httpx

logger = logging.getLogger("findex.crawler.fetch")

DEFAULT_USER_AGENT: Final[str] = (
    "findex-bot/1.0 (+https://github.com/Nikitavigov/findex; bot@findex.local)"
)


@dataclass(slots=True)
class FetchResult:
    """Результат завантаження сторінки через асинхронний клієнт."""

    url: str
    status_code: int
    content: bytes
    elapsed: float
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300 and self.error is None


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    user_agent: str = DEFAULT_USER_AGENT,
) -> FetchResult:
    """
    Асинхронно завантажує вміст за заданим URL з підтримкою таймаутів та повторів.
    Повтори застосовуються виключно для помилок 429 (Rate Limit), 5xx (Server Error)
    та мережевих таймаутів з експоненційним відкатом і джиoffset-ом.
    """
    headers = {"User-Agent": user_agent}

    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(timeout):
                response = await client.get(url, headers=headers, follow_redirects=True)
                elapsed = time.perf_counter() - t0

                # Успішний статус або клієнтська помилка (крім 429)
                if response.status_code < 500 and response.status_code != 429:
                    return FetchResult(
                        url=str(response.url),
                        status_code=response.status_code,
                        content=response.content,
                        elapsed=elapsed,
                    )

                # Обробка 429 Too Many Requests та 5xx Server Error для повторів
                if attempt == max_retries:
                    return FetchResult(
                        url=url,
                        status_code=response.status_code,
                        content=response.content,
                        elapsed=elapsed,
                        error=f"Вичерпано спроби для статусу {response.status_code}",
                    )

                retry_after_val = response.headers.get("Retry-After")
                delay = 0.0
                if retry_after_val:
                    try:
                        delay = float(retry_after_val)
                    except ValueError:
                        delay = min(10.0, 0.5 * (2**attempt)) + random.uniform(0.0, 0.3)
                else:
                    delay = min(10.0, 0.5 * (2**attempt)) + random.uniform(0.0, 0.3)

                logger.warning(
                    "URL %s повернув статус %d. Повтор через %.2f с (спроба %d/%d)",
                    url,
                    response.status_code,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay)

        except (httpx.TimeoutException, TimeoutError) as exc:
            elapsed = time.perf_counter() - t0
            if attempt == max_retries:
                logger.error("Таймаут очікування для URL %s після %d спроб", url, max_retries)
                return FetchResult(
                    url=url,
                    status_code=408,
                    content=b"",
                    elapsed=elapsed,
                    error=f"Таймаут запиту: {exc}",
                )
            delay = min(10.0, 0.5 * (2**attempt)) + random.uniform(0.0, 0.3)
            logger.warning("Таймаут для URL %s. Повтор через %.2f с", url, delay)
            await asyncio.sleep(delay)

        except httpx.HTTPError as exc:
            elapsed = time.perf_counter() - t0
            if attempt == max_retries:
                logger.error("Мережева помилка для URL %s: %s", url, exc)
                return FetchResult(
                    url=url,
                    status_code=599,
                    content=b"",
                    elapsed=elapsed,
                    error=f"Мережева помилка: {exc}",
                )
            delay = min(10.0, 0.5 * (2**attempt)) + random.uniform(0.0, 0.3)
            logger.warning("Мережева помилка для %s: %s. Повтор через %.2f с", url, exc, delay)
            await asyncio.sleep(delay)

    return FetchResult(
        url=url,
        status_code=500,
        content=b"",
        elapsed=0.0,
        error="Невідома помилка під час виконання запиту",
    )
