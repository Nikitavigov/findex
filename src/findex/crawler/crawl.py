import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from findex.crawler.fetch import fetch
from findex.crawler.parse import canonicalize_url, extract_links_and_text
from findex.crawler.robots import RobotsCache

logger = logging.getLogger("findex.crawler")


@dataclass(slots=True)
class Page:
    """Збережена сторінка веб-краулера для подальшої індексації."""

    url: str
    title: str
    text: str
    fetched_at: float


async def crawl(
    seeds: list[str] | str,
    *,
    max_pages: int = 100,
    concurrency: int = 10,
    per_host: int = 2,
    per_host_delay: float = 0.1,
    allowed_domains: set[str] | None = None,
    client: httpx.AsyncClient | None = None,
    robots_cache: RobotsCache | None = None,
    timeout: float = 10.0,
    log_callback: Callable[[str, int, int, float, str | None], None] | None = None,
) -> AsyncIterator[Page]:
    """
    Асинхронний генератор сторінок на базі asyncio.Queue та структурованої конкурентності (TaskGroup).
    Потоково повертає об'єкти Page без буферизації всього корпусу в оперативній пам'яті.
    """
    seed_list = [seeds] if isinstance(seeds, str) else list(seeds)
    if not seed_list:
        return

    # Якщо домени не передані явно, за замовчуванням обмежуємося хостами початкових посилань
    if allowed_domains is None:
        allowed_domains = {urlparse(canonicalize_url(s)).netloc.lower() for s in seed_list}

    queue: asyncio.Queue[str] = asyncio.Queue()
    seen: set[str] = set()

    for seed in seed_list:
        canon = canonicalize_url(seed)
        if canon not in seen:
            seen.add(canon)
            await queue.put(canon)

    robots = robots_cache or RobotsCache()
    global_sem = asyncio.Semaphore(concurrency)
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_last_req: dict[str, float] = {}
    host_lock = asyncio.Lock()

    busy_workers = 0
    busy_lock = asyncio.Lock()

    pages_yielded = 0
    stop_event = asyncio.Event()

    # Внутрішня черга сторінок для передачі від фонових воркерів до генератора
    page_channel: asyncio.Queue[Page | None] = asyncio.Queue(maxsize=concurrency * 2)

    async def get_host_semaphore(host: str) -> asyncio.Semaphore:
        async with host_lock:
            if host not in host_sems:
                host_sems[host] = asyncio.Semaphore(per_host)
            return host_sems[host]

    async def worker(worker_id: int, http_client: httpx.AsyncClient) -> None:
        nonlocal pages_yielded, busy_workers
        while not stop_event.is_set():
            try:
                # Очікуємо URL з черги або завершення за таймаутом
                try:
                    async with asyncio.timeout(0.5):
                        url = await queue.get()
                except TimeoutError:
                    async with busy_lock:
                        if queue.empty() and busy_workers == 0:
                            break
                    continue

                if stop_event.is_set():
                    queue.task_done()
                    break

                async with busy_lock:
                    busy_workers += 1

                try:
                    parsed_url = urlparse(url)
                    host = parsed_url.netloc.lower()

                    # 1. Перевірка robots.txt
                    try:
                        is_allowed = await robots.allowed(http_client, url)
                    except (httpx.HTTPError, OSError, ValueError):
                        is_allowed = True

                    if not is_allowed:
                        logger.info("URL %s заблоковано через robots.txt", url)
                        if log_callback:
                            log_callback(url, 403, 0, 0.0, "Robots.txt Disallow")
                        continue

                    # 2. Дотримання ввічливості та лімітів (хостовий семафор + затримка)
                    host_sem = await get_host_semaphore(host)

                    async with global_sem, host_sem:
                        # Мінімальна пауза між запитами до того самого хоста
                        async with host_lock:
                            now = time.perf_counter()
                            last_t = host_last_req.get(host, 0.0)
                            wait_t = per_host_delay - (now - last_t)
                            if wait_t > 0:
                                await asyncio.sleep(wait_t)
                            host_last_req[host] = time.perf_counter()

                        res = await fetch(http_client, url, timeout=timeout)

                    if log_callback:
                        log_callback(
                            url,
                            res.status_code,
                            len(res.content),
                            res.elapsed,
                            res.error,
                        )

                    if res.is_success and res.content:
                        html_text = res.content.decode("utf-8", errors="replace")
                        title, text, links = extract_links_and_text(html_text, url)

                        page = Page(
                            url=url,
                            title=title or url,
                            text=text,
                            fetched_at=time.time(),
                        )
                        await page_channel.put(page)

                        # Додаємо нові знайдені посилання до черги (frontier)
                        for link in links:
                            link_host = urlparse(link).netloc.lower()
                            if link_host in allowed_domains and link not in seen:
                                seen.add(link)
                                await queue.put(link)
                finally:
                    async with busy_lock:
                        busy_workers -= 1
                    queue.task_done()

            except asyncio.CancelledError:
                break
            except (httpx.HTTPError, OSError, ValueError) as exc:
                logger.error("Помилка воркера %d: %s", worker_id, exc)

    # Управління HTTP-клієнтом та TaskGroup
    own_client = client is None
    http_client = client or httpx.AsyncClient(timeout=timeout)

    try:
        async def run_workers() -> None:
            try:
                async with asyncio.TaskGroup() as tg:
                    for i in range(concurrency):
                        tg.create_task(worker(i, http_client))
            finally:
                await page_channel.put(None)

        producer_task = asyncio.create_task(run_workers())

        while True:
            page = await page_channel.get()
            if page is None:
                break
            yield page
            pages_yielded += 1

            if pages_yielded >= max_pages:
                stop_event.set()
                break

        # Скасування залишкових завдань при досягненні ліміту
        producer_task.cancel()
        try:
            await producer_task
        except (asyncio.CancelledError, ExceptionGroup):
            pass

    finally:
        if own_client:
            await http_client.aclose()
