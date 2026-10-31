"""Модуль асинхронного веб-краулера для findex на базі asyncio та httpx."""

from findex.crawler.crawl import Page, crawl
from findex.crawler.fetch import FetchResult, fetch
from findex.crawler.parse import canonicalize_url, extract_links_and_text
from findex.crawler.robots import RobotsCache

__all__ = [
    "FetchResult",
    "Page",
    "RobotsCache",
    "canonicalize_url",
    "crawl",
    "extract_links_and_text",
    "fetch",
]
