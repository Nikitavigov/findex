import asyncio
import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import DEFAULT_USER_AGENT

logger = logging.getLogger("findex.crawler.robots")


class RobotsCache:
    """
    Кеш файлів robots.txt для різних хостів.
    Гарантує одноразове асинхронне завантаження та парсинг правил ввічливості.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, RobotFileParser] = {}
        self._lock = asyncio.Lock()

    async def allowed(
        self,
        client: httpx.AsyncClient,
        url: str,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> bool:
        """Перевіряє, чи дозволено завантаження URL згідно з правилами robots.txt."""
        parsed = urlparse(url)
        scheme = parsed.scheme or "https"
        host = parsed.netloc.lower()

        if not host:
            return False

        async with self._lock:
            if host not in self._parsers:
                parser = RobotFileParser()
                robots_url = f"{scheme}://{host}/robots.txt"
                try:
                    resp = await client.get(
                        robots_url,
                        headers={"User-Agent": user_agent},
                        follow_redirects=True,
                        timeout=5.0,
                    )
                    if resp.status_code == 200:
                        parser.parse(resp.text.splitlines())
                    else:
                        # 404 або відсутній robots.txt дозволяє доступ до всіх сторінок
                        parser.allow_all = True  # type: ignore
                except (httpx.HTTPError, OSError, ValueError) as exc:
                    logger.debug("Не вдалося завантажити robots.txt для %s: %s", host, exc)
                    parser.allow_all = True  # type: ignore

                self._parsers[host] = parser

        parser = self._parsers[host]
        return parser.can_fetch(user_agent, url)
