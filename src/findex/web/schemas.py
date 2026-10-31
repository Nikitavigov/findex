"""Pydantic-схеми для вхідних та вихідних даних HTTP API findex.

Використовуються виключно на межі системи (вхідні параметри та HTTP-відповіді),
залишаючи внутрішні структури Index та DocMeta у вигляді легких dataclass.
"""

from pydantic import BaseModel, ConfigDict, Field


class SearchResult(BaseModel):
    """Один результат пошуку у відповіді API."""

    model_config = ConfigDict(frozen=True)

    doc_id: int = Field(description="Унікальний ідентифікатор документа")
    title: str = Field(description="Заголовок або назва джерела документа")
    url: str | None = Field(default=None, description="URL адреса першоджерела (якщо є)")
    score: float = Field(description="Релевантність документа за обраним алгоритмом")
    snippet: str = Field(description="Текстовий фрагмент з контекстом навколо ключових слів")


class SearchResponse(BaseModel):
    """Повна відповідь пошукового ендпоінта з пагінацією."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="Оригінальний текстовий запит")
    total: int = Field(description="Загальна кількість знайдених документів")
    page: int = Field(description="Поточний номер сторінки (починаючи з 1)")
    page_size: int = Field(description="Кількість результатів на одну сторінку")
    total_pages: int = Field(description="Загальна кількість доступних сторінок")
    elapsed_ms: float = Field(description="Час виконання пошуку в мілісекундах")
    results: list[SearchResult] = Field(description="Список знайдених результатів для поточної сторінки")


class DocumentResponse(BaseModel):
    """Повна інформація про документ за його ідентифікатором."""

    model_config = ConfigDict(frozen=True)

    doc_id: int = Field(description="Унікальний ідентифікатор документа")
    title: str = Field(description="Заголовок документа")
    url: str | None = Field(default=None, description="URL адреса документа")
    text: str = Field(description="Повний текстовий вміст документа")
    length: int = Field(description="Кількість токенів у документі")


class StatsResponse(BaseModel):
    """Загальна статистика завантаженого пошукового індексу."""

    model_config = ConfigDict(frozen=True)

    doc_count: int = Field(description="Загальна кількість документів в індексі")
    vocab_size: int = Field(description="Розмір словника унікальних термів")
    total_tokens: int = Field(description="Сумарна кількість токенів у всіх документах")
    avg_doc_len: float = Field(description="Середня довжина документа в токенах")


class HealthResponse(BaseModel):
    """Статус працездатності сервісу для систем моніторингу та балансувальників."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(description="Текстовий статус: ok або unhealthy")
    index_loaded: bool = Field(description="Чи завантажено індекс у пам'ять")
    doc_count: int = Field(description="Кількість документів у доступному індексі")
