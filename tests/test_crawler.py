from typing import Any

import httpx
import pytest

from findex.crawler.crawl import crawl
from findex.crawler.fetch import fetch
from findex.crawler.parse import canonicalize_url, extract_links_and_text
from findex.crawler.robots import RobotsCache


def test_canonicalize_url() -> None:
    """Перевірка канонізації URL-адрес."""
    # 1. Приведення до нижнього регістру та видалення фрагмента
    assert (
        canonicalize_url("HTTP://EXAMPLE.COM/Path#heading")
        == "http://example.com/Path"
    )

    # 2. Видалення стандартного порту
    assert canonicalize_url("http://example.com:80/page") == "http://example.com/page"
    assert canonicalize_url("https://example.com:443/page") == "https://example.com/page"

    # 3. Нормалізація шляху та подвійних слешів
    assert canonicalize_url("http://example.com") == "http://example.com/"
    assert canonicalize_url("http://example.com//a//b") == "http://example.com/a/b"

    # 4. Сортування query-параметрів
    assert (
        canonicalize_url("http://example.com/search?z=3&a=1&m=2")
        == "http://example.com/search?a=1&m=2&z=3"
    )


def test_extract_links_and_text() -> None:
    """Перевірка вилучення тексту, заголовка та посилань з HTML."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>  Тестова сторінка findex  </title>
        <style>body { color: red; }</style>
        <script>console.log("secret");</script>
    </head>
    <body>
        <h1>Головний заголовок</h1>
        <p>Це перший параграф тексту пошукового рушія.</p>
        <a href="/about">Про нас</a>
        <a href="https://other.com/docs#intro">Документація</a>
        <a href="javascript:void(0)">Ігнорувати</a>
    </body>
    </html>
    """
    base_url = "http://example.com/blog/post"
    title, text, links = extract_links_and_text(html, base_url)

    assert title == "Тестова сторінка findex"
    assert "Головний заголовок" in text
    assert "перший параграф" in text
    assert "body { color: red; }" not in text
    assert "console.log" not in text
    assert "http://example.com/about" in links
    assert "https://other.com/docs" in links


@pytest.mark.asyncio
async def test_fetch_success() -> None:
    """Перевірка успішного асинхронного завантаження 200 OK через MockTransport."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="OK response content")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        res = await fetch(client, "http://example.com/test")
        assert res.is_success
        assert res.status_code == 200
        assert res.content == b"OK response content"
        assert res.error is None


@pytest.mark.asyncio
async def test_fetch_404_no_retry() -> None:
    """Клієнтська помилка 404 Not Found не повинна повторюватися."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(404, text="Not Found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        res = await fetch(client, "http://example.com/missing", max_retries=3)
        assert res.status_code == 404
        assert not res.is_success
        assert call_count == 1  # Жодного повтору!


@pytest.mark.asyncio
async def test_fetch_500_with_retry() -> None:
    """Помилка 500 повинна повторюватися до успіху або вичерпання ліміту."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(500, text="Server Error")
        return httpx.Response(200, text="Recovered OK")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        res = await fetch(client, "http://example.com/flaky", max_retries=3)
        assert res.is_success
        assert res.status_code == 200
        assert attempts == 3


@pytest.mark.asyncio
async def test_fetch_429_retry_after() -> None:
    """Помилка 429 Too Many Requests враховує заголовок Retry-After."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0.01"}, text="Rate limit")
        return httpx.Response(200, text="Success after rate limit")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        res = await fetch(client, "http://example.com/rate-limited", max_retries=2)
        assert res.is_success
        assert res.status_code == 200
        assert attempts == 2


@pytest.mark.asyncio
async def test_robots_cache() -> None:
    """Перевірка кешування та дотримання правил robots.txt."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            robots_content = """
            User-agent: *
            Disallow: /admin/
            Disallow: /private.html
            """
            return httpx.Response(200, text=robots_content)
        return httpx.Response(200, text="Page")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        cache = RobotsCache()

        # Дозволені URL
        assert await cache.allowed(client, "http://example.com/public")
        assert await cache.allowed(client, "http://example.com/blog/post")

        # Заборонені URL
        assert not await cache.allowed(client, "http://example.com/admin/settings")
        assert not await cache.allowed(client, "http://example.com/private.html")


@pytest.mark.asyncio
async def test_crawl_mock_graph() -> None:
    """
    Комплексна перевірка роботи crawl() на зв'язному графі з 3 сторінок.
    Перевіряється:
    - Потокова генерація Page
    - Запобігання повторним візитам (seen set)
    - Дотримання max_pages
    """
    pages_db = {
        "http://example.com/": """
            <html><head><title>Home</title></head><body>
            <p>Welcome to findex home page</p>
            <a href="/page1">Page 1</a>
            <a href="/page2">Page 2</a>
            </body></html>
        """,
        "http://example.com/page1": """
            <html><head><title>Page 1</title></head><body>
            <p>Content of page 1</p>
            <a href="/">Back to home</a>
            <a href="/page2">To page 2</a>
            </body></html>
        """,
        "http://example.com/page2": """
            <html><head><title>Page 2</title></head><body>
            <p>Content of page 2</p>
            <a href="/page1">To page 1</a>
            </body></html>
        """,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        if url_str in pages_db:
            return httpx.Response(200, text=pages_db[url_str])
        return httpx.Response(404, text="Not Found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        crawled_pages: list[Any] = []
        async for page in crawl(
            "http://example.com/",
            max_pages=5,
            concurrency=2,
            per_host=2,
            per_host_delay=0.0,
            client=client,
        ):
            crawled_pages.append(page)

        # Оскільки сторінок всього 3 і є циклічні посилання,
        # краулер має зібрати рівно 3 сторінки без зациклення
        assert len(crawled_pages) == 3
        crawled_urls = {p.url for p in crawled_pages}
        assert "http://example.com/" in crawled_urls
        assert "http://example.com/page1" in crawled_urls
        assert "http://example.com/page2" in crawled_urls
        assert all("findex" in p.text or "Content" in p.text for p in crawled_pages)
