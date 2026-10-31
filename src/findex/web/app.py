"""Головний модуль FastAPI-застосунку для пошукового рушія findex.

Містить фабрику create_app(), HTTP-маршрути для API та Web UI,
проміжне ПЗ (middleware) для Request-ID та обробники помилок.
"""

import html
import logging
import math
import re
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from findex.index import Index
from findex.ranking import BM25Scorer, TfIdfScorer, rank_documents
from findex.snippets import generate_snippet
from findex.tokenizer import normalize_text, tokenize
from findex.web.dependencies import get_index, get_optional_index
from findex.web.lifespan import lifespan
from findex.web.schemas import (
    DocumentResponse,
    HealthResponse,
    SearchResponse,
    SearchResult,
    StatsResponse,
)

logger = logging.getLogger("findex.web")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _get_doc_text_and_url(index: Index, doc_id: int) -> tuple[str, str | None]:
    """Отримує повний текст та URL документа за його doc_id."""
    doc_meta = index.doc_meta.get(doc_id)
    if not doc_meta:
        return ("", None)

    url = (
        doc_meta.source_name
        if doc_meta.source_name.startswith(("http://", "https://"))
        else None
    )

    # 1. Пошук у каталозі корпусу
    if index.corpus_dir and index.corpus_dir.is_dir():
        file_path = index.corpus_dir / doc_meta.source_name
        if file_path.is_file():
            try:
                return (file_path.read_text(encoding="utf-8", errors="replace"), url)
            except OSError:
                pass

    # 2. Прямий шлях до файлу
    direct_path = Path(doc_meta.source_name)
    if direct_path.is_file():
        try:
            return (direct_path.read_text(encoding="utf-8", errors="replace"), url)
        except OSError:
            pass

    # 3. Резервний текст, якщо оригінальний файл недоступний
    return (
        f"{doc_meta.title}\n\n(Джерело документа: {doc_meta.source_name}, довжина: {doc_meta.length} токенів)",
        url,
    )


def _generate_html_snippet(text: str, query_terms: list[str]) -> str:
    """Генерує безпечний HTML-сніпет із виділенням слів тегом <mark>."""
    raw = generate_snippet(text, query_terms, highlight=False)
    escaped_raw = html.escape(raw)
    norm_terms = [re.escape(normalize_text(t)) for t in query_terms if t.strip()]
    if not norm_terms or not escaped_raw:
        return escaped_raw

    pattern = re.compile(r"\b(" + "|".join(norm_terms) + r")\b", re.IGNORECASE | re.UNICODE)
    return pattern.sub(r"<mark>\1</mark>", escaped_raw)


def create_app() -> FastAPI:
    """Фабрика для створення екземпляра застосунку FastAPI."""
    app = FastAPI(
        title="findex Search API",
        description="Високопродуктивний повнотекстовий пошуковий рушій на Python (Лаб 7).",
        version="0.7.0",
        lifespan=lifespan,
    )

    # Дозволяємо CORS для відкритих API-запитів
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Підключення статичних файлів (CSS)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Middleware для Request-ID та вимірювання затримки запитів
    @app.middleware("http")
    async def request_id_and_logging_middleware(
        request: Request, call_next: Any
    ) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = req_id
        t0 = time.perf_counter()

        response: Response = await call_next(request)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = req_id
        logger.info(
            "[%s] %s %s -> %d (%.2f ms)",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # Глобальний обробник непередбачуваних помилок 500
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id = getattr(request.state, "request_id", "unknown")
        logger.exception("[%s] Непередбачена внутрішня помилка сервера: %s", req_id, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Внутрішня помилка сервера під час обробки запиту.",
                "request_id": req_id,
            },
            headers={"X-Request-ID": req_id},
        )

    # --- HTTP API Маршрути ---

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Перевірка працездатності сервісу",
        tags=["System"],
    )
    def health_check(
        index: Annotated[Index | None, Depends(get_optional_index)],
    ) -> JSONResponse:
        """Повертає 200, якщо сервіс готовий і індекс завантажений; інакше 503."""
        if index is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unhealthy",
                    "index_loaded": False,
                    "doc_count": 0,
                },
            )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "ok",
                "index_loaded": True,
                "doc_count": index.num_docs,
            },
        )

    @app.get(
        "/stats",
        response_model=StatsResponse,
        summary="Статистика пошукового індексу",
        tags=["System"],
    )
    def stats_endpoint(
        index: Annotated[Index, Depends(get_index)],
    ) -> StatsResponse:
        """Повертає загальні метрики індексу: кількість документів, словник, токени."""
        return StatsResponse(
            doc_count=index.num_docs,
            vocab_size=len(index),
            total_tokens=index.total_tokens,
            avg_doc_len=round(index.avg_doc_length, 2),
        )

    @app.get(
        "/docs/{doc_id}",
        response_model=DocumentResponse,
        summary="Отримання документа за його doc_id",
        tags=["Documents"],
    )
    def get_document(
        doc_id: int,
        index: Annotated[Index, Depends(get_index)],
    ) -> DocumentResponse:
        """Повертає метадані та текст документа або 404, якщо doc_id відсутній."""
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Документ #{doc_id} не знайдено в поточному індексі.",
            )

        text, url = _get_doc_text_and_url(index, doc_id)
        return DocumentResponse(
            doc_id=doc_id,
            title=meta.title,
            url=url,
            text=text,
            length=meta.length,
        )

    @app.get(
        "/search",
        response_model=SearchResponse,
        summary="Ранжований повнотекстовий пошук",
        tags=["Search"],
    )
    def search_endpoint(
        q: Annotated[str, Query(min_length=1, max_length=200, description="Пошуковий запит")],
        index: Annotated[Index, Depends(get_index)],
        k: Annotated[int, Query(ge=1, le=100, description="Кількість результатів на сторінку")] = 10,
        page: Annotated[int, Query(ge=1, description="Номер сторінки результатів")] = 1,
        scorer: Annotated[Literal["bm25", "tfidf"], Query(description="Алгоритм ранжування")] = "bm25",
    ) -> SearchResponse:
        """Повнотекстовий пошук за алгоритмами BM25 або TF-IDF.

        Маршрут визначено як звичайну синхронну функцію def, завдяки чому FastAPI
        виконує CPU-інтенсивний розрахунок ранжування у пулі потоків (threadpool)
        і не блокує головний цикл подій asyncio.
        """
        t0 = time.perf_counter()
        query_terms = list(tokenize(q))
        if not query_terms:
            return SearchResponse(
                query=q,
                total=0,
                page=page,
                page_size=k,
                total_pages=0,
                elapsed_ms=0.0,
                results=[],
            )

        scorer_instance = BM25Scorer() if scorer == "bm25" else TfIdfScorer()
        ranked = rank_documents(query_terms, index, scorer_instance, top_k=index.num_docs)

        total_results = len(ranked)
        total_pages = max(1, math.ceil(total_results / k)) if total_results > 0 else 0
        offset = (page - 1) * k
        page_items = ranked[offset : offset + k]

        results: list[SearchResult] = []
        for cand in page_items:
            meta = index.doc_meta.get(cand.doc_id)
            title = meta.title if meta else f"Документ #{cand.doc_id}"
            text, url = _get_doc_text_and_url(index, cand.doc_id)
            snippet = generate_snippet(text, query_terms, highlight=False)
            results.append(
                SearchResult(
                    doc_id=cand.doc_id,
                    title=title,
                    url=url,
                    score=round(cand.score, 4),
                    snippet=snippet,
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return SearchResponse(
            query=q,
            total=total_results,
            page=page,
            page_size=k,
            total_pages=total_pages,
            elapsed_ms=round(elapsed_ms, 2),
            results=results,
        )

    # --- Маршрут для асинхронного тестування (M2 benchmark: def vs async def) ---

    @app.get(
        "/search/async",
        response_model=SearchResponse,
        summary="Асинхронний ендпоінт для порівняльного бенчмарку",
        include_in_schema=False,
    )
    async def search_endpoint_async(
        q: Annotated[str, Query(min_length=1, max_length=200)],
        index: Annotated[Index, Depends(get_index)],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        page: Annotated[int, Query(ge=1)] = 1,
        scorer: Annotated[Literal["bm25", "tfidf"], Query(description="Алгоритм ранжування")] = "bm25",
    ) -> SearchResponse:
        """Варіант через async def, що виконує CPU-ранжування прямо в циклі подій."""
        t0 = time.perf_counter()
        query_terms = list(tokenize(q))
        if not query_terms:
            return SearchResponse(
                query=q,
                total=0,
                page=page,
                page_size=k,
                total_pages=0,
                elapsed_ms=0.0,
                results=[],
            )

        scorer_instance = BM25Scorer() if scorer == "bm25" else TfIdfScorer()
        ranked = rank_documents(query_terms, index, scorer_instance, top_k=index.num_docs)

        total_results = len(ranked)
        total_pages = max(1, math.ceil(total_results / k)) if total_results > 0 else 0
        offset = (page - 1) * k
        page_items = ranked[offset : offset + k]

        results: list[SearchResult] = []
        for cand in page_items:
            meta = index.doc_meta.get(cand.doc_id)
            title = meta.title if meta else f"Документ #{cand.doc_id}"
            text, url = _get_doc_text_and_url(index, cand.doc_id)
            snippet = generate_snippet(text, query_terms, highlight=False)
            results.append(
                SearchResult(
                    doc_id=cand.doc_id,
                    title=title,
                    url=url,
                    score=round(cand.score, 4),
                    snippet=snippet,
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return SearchResponse(
            query=q,
            total=total_results,
            page=page,
            page_size=k,
            total_pages=total_pages,
            elapsed_ms=round(elapsed_ms, 2),
            results=results,
        )

    # --- Web UI (HTML сторінки Jinja2) ---

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def web_ui_index(
        request: Request,
        index: Annotated[Index | None, Depends(get_optional_index)],
        q: str | None = None,
        k: int = 10,
        page: int = 1,
        scorer: Literal["bm25", "tfidf"] = "bm25",
    ) -> Response:
        """Головна веб-сторінка пошуку."""
        query = (q or "").strip()

        if not query or index is None:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "query": query,
                    "results": [],
                    "total": 0,
                    "page": 1,
                    "page_size": k,
                    "total_pages": 0,
                    "elapsed_ms": 0.0,
                    "scorer": scorer,
                },
            )

        t0 = time.perf_counter()
        query_terms = list(tokenize(query))
        scorer_instance = BM25Scorer() if scorer == "bm25" else TfIdfScorer()
        ranked = rank_documents(query_terms, index, scorer_instance, top_k=index.num_docs)

        total_results = len(ranked)
        total_pages = max(1, math.ceil(total_results / k)) if total_results > 0 else 0
        offset = (page - 1) * k
        page_items = ranked[offset : offset + k]

        results_data: list[dict[str, Any]] = []
        for cand in page_items:
            meta = index.doc_meta.get(cand.doc_id)
            title = meta.title if meta else f"Документ #{cand.doc_id}"
            text, url = _get_doc_text_and_url(index, cand.doc_id)
            html_snippet = _generate_html_snippet(text, query_terms)
            results_data.append(
                {
                    "doc_id": cand.doc_id,
                    "title": title,
                    "url": url,
                    "score": cand.score,
                    "snippet": html_snippet,
                }
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "query": query,
                "results": results_data,
                "total": total_results,
                "page": page,
                "page_size": k,
                "total_pages": total_pages,
                "elapsed_ms": elapsed_ms,
                "scorer": scorer,
            },
        )

    @app.get("/view/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def web_ui_view_document(
        request: Request,
        doc_id: int,
        index: Annotated[Index, Depends(get_index)],
    ) -> Response:
        """Веб-сторінка перегляду вмісту окремого документа."""
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Документ #{doc_id} не знайдено.",
            )

        text, url = _get_doc_text_and_url(index, doc_id)
        doc_data = {
            "doc_id": doc_id,
            "title": meta.title,
            "url": url,
            "text": text,
            "length": meta.length,
        }
        return templates.TemplateResponse(
            request=request,
            name="document.html",
            context={"doc": doc_data},
        )

    return app


app = create_app()
