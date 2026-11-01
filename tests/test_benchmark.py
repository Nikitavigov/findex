"""Бенчмарк-тести pytest-benchmark для контролю швидкодії пошуку та індексації (Lab 8).

Забезпечує автоматичний контроль регресій швидкодії в CI/CD конвеєрі.
"""

from typing import Any

import pytest

from findex.index import Index
from findex.models import DocMeta, Posting
from findex.numpy_index import NumPyIndex, rank_bm25_numpy
from findex.ranking import BM25Scorer, rank_documents


@pytest.fixture
def benchmark_index() -> tuple[Index, NumPyIndex]:
    """Створює синтетичний індекс середнього розміру для бенчмарків."""
    meta: dict[int, DocMeta] = {}
    postings: dict[str, list[Posting]] = {f"term_{i}": [] for i in range(200)}

    for doc_id in range(1, 301):
        meta[doc_id] = DocMeta(
            doc_id=doc_id,
            source_name=f"doc_{doc_id}.txt",
            title=f"Документ #{doc_id}",
            length=100,
        )
        # Додаємо кілька термів у кожен документ
        for t_id in range(0, 200, 10):
            postings[f"term_{t_id}"].append(Posting(doc_id=doc_id, term_frequency=(doc_id % 5) + 1))

    py_idx = Index(postings=postings, doc_meta=meta)
    np_idx = NumPyIndex.from_index(py_idx)
    return py_idx, np_idx


def test_benchmark_search_python_classic(
    benchmark: Any,
    benchmark_index: tuple[Index, NumPyIndex],
) -> None:
    """Вимірювання швидкодії класичного ранжування BM25 з Лаб 3."""
    py_idx, _ = benchmark_index
    terms = ["term_0", "term_10", "term_20"]
    scorer = BM25Scorer()

    results = benchmark(rank_documents, terms, py_idx, scorer, top_k=10)
    assert len(results) == 10


def test_benchmark_search_numpy_vectorized(
    benchmark: Any,
    benchmark_index: tuple[Index, NumPyIndex],
) -> None:
    """Вимірювання швидкодії векторизованого ранжування BM25 NumPy з Лаб 8."""
    _, np_idx = benchmark_index
    terms = ["term_0", "term_10", "term_20"]

    results = benchmark(rank_bm25_numpy, terms, np_idx, top_k=10)
    assert len(results) == 10
