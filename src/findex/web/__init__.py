"""Веб-модуль пошукового рушія findex на базі FastAPI.

Включає HTTP API (/search, /stats, /health, /docs/{id})
та веб-інтерфейс на базі шаблонів Jinja2.
"""

from findex.web.app import app, create_app
from findex.web.config import Settings

__all__ = ["Settings", "app", "create_app"]
