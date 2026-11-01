from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import Index, build_index, save_index


@pytest.fixture
def sample_documents() -> list[Document]:
    """Колекція тестових документів для юніт-тестів."""
    return [
        Document(
            doc_id="1",
            text="Python is a high-level programming language with dynamic typing.",
            metadata={"title": "Doc 1", "source": "doc1.txt"},
        ),
        Document(
            doc_id="2",
            text="Natural language processing and information retrieval are fields of computer science.",
            metadata={"title": "Doc 2", "source": "doc2.txt"},
        ),
        Document(
            doc_id="3",
            text="Python makes it easy to write clean and concise search engines.",
            metadata={"title": "Doc 3", "source": "doc3.txt"},
        ),
    ]


@pytest.fixture
def sample_corpus_dir(tmp_path: Path, sample_documents: list[Document]) -> Path:
    """Тимчасовий каталог з текстовими файлами для перевірки індексації."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for doc in sample_documents:
        file_path = corpus_dir / f"{doc.doc_id}.txt"
        file_path.write_text(doc.text, encoding="utf-8")
    return corpus_dir


@pytest.fixture
def sample_index(sample_corpus_dir: Path) -> Index:
    """Зібраний індекс на основі тестового корпусу."""
    return build_index(sample_corpus_dir)


@pytest.fixture
def saved_index_path(tmp_path: Path, sample_index: Index) -> Path:
    """Шлях до збереженого на диску індексу у форматі pickle."""
    idx_path = tmp_path / "index.bin"
    save_index(sample_index, idx_path, fmt="pickle")
    return idx_path


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Сумісність pytest_benchmark з Pytest 9+ (виправлення відсутності _nodeid)."""
    for item in items:
        if not hasattr(item, "_nodeid"):
            item._nodeid = item.nodeid
