"""Тести для веб-сервісу findex на базі FastAPI та TestClient.

Перевіряє коректність роботи API, валідацію вхідних параметрів Pydantic (422),
обробку відсутності документа (404), перевірку статусу здоров'я (503),
генерацію сніпетів та рендеринг сторінок Web UI (HTML).
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from findex.index import Index
from findex.models import DocMeta, Posting
from findex.web.app import app
from findex.web.dependencies import get_index, get_optional_index


@pytest.fixture
def mock_index() -> Index:
    """Фікстура синтетичного індексу для ізольованого тестування API без файлів на диску."""
    meta = {
        1: DocMeta(doc_id=1, source_name="doc1.txt", title="Вступ до Python", length=5),
        2: DocMeta(doc_id=2, source_name="doc2.txt", title="Асинхронний веб та asyncio", length=6),
        3: DocMeta(doc_id=3, source_name="doc3.txt", title="Пошуковий рушій findex", length=4),
    }
    postings = {
        "python": [Posting(doc_id=1, term_frequency=2), Posting(doc_id=2, term_frequency=1)],
        "asyncio": [Posting(doc_id=2, term_frequency=3)],
        "findex": [Posting(doc_id=3, term_frequency=2)],
    }
    return Index(postings=postings, doc_meta=meta)


@pytest.fixture
def client(mock_index: Index) -> TestClient:
    """Клієнт тестування з підміненим через dependency_overrides індексом."""
    app.dependency_overrides[get_optional_index] = lambda: mock_index
    app.dependency_overrides[get_index] = lambda: mock_index
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_unloaded() -> TestClient:
    """Клієнт тестування для сценарію, коли індекс ще не завантажений."""
    def _raise_503() -> Index:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Індекс не завантажено.")

    app.dependency_overrides[get_optional_index] = lambda: None
    app.dependency_overrides[get_index] = _raise_503
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_ok(client: TestClient) -> None:
    """Перевірка /health повертає 200, коли індекс завантажено."""
    res = client.get("/health")
    assert res.status_code == status.HTTP_200_OK
    data = res.json()
    assert data["status"] == "ok"
    assert data["index_loaded"] is True
    assert data["doc_count"] == 3
    assert "X-Request-ID" in res.headers


def test_health_unloaded(client_unloaded: TestClient) -> None:
    """Перевірка /health повертає 503, якщо індекс відсутній."""
    res = client_unloaded.get("/health")
    assert res.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    data = res.json()
    assert data["status"] == "unhealthy"
    assert data["index_loaded"] is False
    assert data["doc_count"] == 0


def test_stats_success(client: TestClient) -> None:
    """Перевірка /stats повертає коректні метрики індексу."""
    res = client.get("/stats")
    assert res.status_code == status.HTTP_200_OK
    data = res.json()
    assert data["doc_count"] == 3
    assert data["vocab_size"] == 3
    assert data["total_tokens"] == 15
    assert data["avg_doc_len"] == 5.0


def test_search_success(client: TestClient) -> None:
    """Успішний пошук повертає відповідь SearchResponse та результати."""
    res = client.get("/search?q=python")
    assert res.status_code == status.HTTP_200_OK
    data = res.json()
    assert data["query"] == "python"
    assert data["total"] == 2
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert len(data["results"]) == 2
    assert data["results"][0]["doc_id"] in {1, 2}
    assert "score" in data["results"][0]


def test_search_pagination(client: TestClient) -> None:
    """Перевірка пагінації: запит з page_size=1 повинен повертати 1 результат."""
    res = client.get("/search?q=python&k=1&page=1")
    assert res.status_code == status.HTTP_200_OK
    data = res.json()
    assert data["total"] == 2
    assert data["total_pages"] == 2
    assert len(data["results"]) == 1


def test_search_tfidf_scorer(client: TestClient) -> None:
    """Перевірка вибору скорера tfidf."""
    res = client.get("/search?q=asyncio&scorer=tfidf")
    assert res.status_code == status.HTTP_200_OK
    data = res.json()
    assert data["total"] == 1
    assert data["results"][0]["doc_id"] == 2


def test_search_validation_empty_query(client: TestClient) -> None:
    """Порожній запит q відхиляється валідацією Pydantic з кодом 422."""
    res = client.get("/search?q=")
    assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_search_validation_invalid_k(client: TestClient) -> None:
    """Некоректне значення k (< 1 або > 100) викликає 422."""
    res_zero = client.get("/search?q=python&k=0")
    assert res_zero.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    res_large = client.get("/search?q=python&k=999")
    assert res_large.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_search_validation_invalid_scorer(client: TestClient) -> None:
    """Невідомий скорер відхиляється валідацією з кодом 422."""
    res = client.get("/search?q=python&scorer=unknown_scorer")
    assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_doc_get_success(client: TestClient) -> None:
    """Отримання наявного документа за doc_id."""
    res = client.get("/docs/1")
    assert res.status_code == status.HTTP_200_OK
    data = res.json()
    assert data["doc_id"] == 1
    assert data["title"] == "Вступ до Python"
    assert data["length"] == 5


def test_doc_get_404(client: TestClient) -> None:
    """Запит неіснуючого doc_id повертає 404 Not Found."""
    res = client.get("/docs/9999")
    assert res.status_code == status.HTTP_404_NOT_FOUND


def test_web_ui_home(client: TestClient) -> None:
    """Головна сторінка віддає HTML-інтерфейс пошуку."""
    res = client.get("/")
    assert res.status_code == status.HTTP_200_OK
    assert "text/html" in res.headers["content-type"]
    assert "findex" in res.text
    assert "Знайти" in res.text


def test_web_ui_search_results(client: TestClient) -> None:
    """Пошук через веб-інтерфейс рендерить картки результатів та виділення."""
    res = client.get("/?q=python")
    assert res.status_code == status.HTTP_200_OK
    assert "text/html" in res.headers["content-type"]
    assert "Вступ до Python" in res.text
    assert "Знайдено" in res.text


def test_web_ui_view_document(client: TestClient) -> None:
    """Сторінка перегляду документа віддає HTML з заголовком та текстом."""
    res = client.get("/view/1")
    assert res.status_code == status.HTTP_200_OK
    assert "text/html" in res.headers["content-type"]
    assert "Вступ до Python" in res.text
    assert "Документ #1" in res.text


def test_web_ui_view_document_404(client: TestClient) -> None:
    """Сторінка перегляду неіснуючого документа повертає 404."""
    res = client.get("/view/9999")
    assert res.status_code == status.HTTP_404_NOT_FOUND


def test_unhandled_exception_returns_500_without_stacktrace() -> None:
    """Непередбачуваний виняток повертає 500 JSON без витоку стека помилки назовні."""
    from fastapi import Request

    test_app = app

    @test_app.get("/test-error-endpoint", include_in_schema=False)
    def crash_endpoint(request: Request) -> None:
        raise RuntimeError("Тестова штучна аварія сервісу")

    with TestClient(test_app, raise_server_exceptions=False) as c:
        res = c.get("/test-error-endpoint")
        assert res.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = res.json()
        assert "detail" in data
        assert "request_id" in data
        assert "RuntimeError" not in res.text
        assert "Traceback" not in res.text
