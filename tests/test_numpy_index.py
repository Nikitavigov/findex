"""Тести для векторизованого індексу NumPyIndex та ранжування над C-масивами.

Перевіряє 100% ідентичність результатів ранжування BM25 та TF-IDF
між класичною реалізацією (Лаб 3) та векторизованими ufuncs NumPy (Лаб 8).
"""

from pathlib import Path

import pytest

from findex.index import Index
from findex.models import DocMeta, Posting
from findex.numpy_index import NumPyIndex, rank_bm25_numpy, rank_tfidf_numpy
from findex.ranking import BM25Scorer, TfIdfScorer, rank_documents


@pytest.fixture
def sample_index() -> Index:
    """Синтетичний тестовий індекс для перевірки математичної еквівалентності."""
    meta = {
        1: DocMeta(doc_id=1, source_name="doc1.txt", title="Вступ до Python", length=10),
        2: DocMeta(doc_id=2, source_name="doc2.txt", title="Паралельні потоки та процеси", length=20),
        3: DocMeta(doc_id=3, source_name="doc3.txt", title="Швидкі масиви NumPy", length=15),
        4: DocMeta(doc_id=4, source_name="doc4.txt", title="Пошуковий рушій на Python", length=8),
        5: DocMeta(doc_id=5, source_name="doc5.txt", title="Алгоритм BM25 та векторний пошук", length=12),
    }
    postings = {
        "python": [
            Posting(doc_id=1, term_frequency=3),
            Posting(doc_id=4, term_frequency=2),
            Posting(doc_id=2, term_frequency=1),
        ],
        "numpy": [
            Posting(doc_id=3, term_frequency=4),
            Posting(doc_id=5, term_frequency=1),
        ],
        "search": [
            Posting(doc_id=4, term_frequency=3),
            Posting(doc_id=5, term_frequency=2),
            Posting(doc_id=1, term_frequency=1),
        ],
        "bm25": [
            Posting(doc_id=5, term_frequency=3),
        ],
    }
    return Index(postings=postings, doc_meta=meta)


def test_numpy_index_conversion(sample_index: Index) -> None:
    """Перевірка коректності конвертації з Index у NumPyIndex."""
    np_index = NumPyIndex.from_index(sample_index)
    assert np_index.total_docs == 5
    assert len(np_index.postings) == 4
    assert len(np_index.doc_lengths) >= 6  # max_doc_id 5 + 1
    assert np_index.doc_lengths[1] == 10
    assert np_index.doc_lengths[2] == 20


def test_bm25_ranking_equivalence(sample_index: Index) -> None:
    """Критичний тест: порядок результатів та бали BM25 NumPy мають 100% збігатися з Лаб 3."""
    np_index = NumPyIndex.from_index(sample_index)
    terms = ["python", "search", "bm25"]

    # Лаб 3 класичний розрахунок
    py_results = rank_documents(terms, sample_index, BM25Scorer(), top_k=5)
    # Лаб 8 NumPy векторизований розрахунок
    np_results = rank_bm25_numpy(terms, np_index, top_k=5)

    assert len(py_results) == len(np_results)
    assert [r.doc_id for r in py_results] == [r.doc_id for r in np_results]

    for py_r, np_r in zip(py_results, np_results, strict=True):
        assert abs(py_r.score - np_r.score) < 1e-5, f"Бал відрізняється: {py_r.score} vs {np_r.score}"


def test_tfidf_ranking_equivalence(sample_index: Index) -> None:
    """Критичний тест: порядок результатів та бали TF-IDF NumPy мають 100% збігатися з Лаб 3."""
    np_index = NumPyIndex.from_index(sample_index)
    terms = ["python", "numpy"]

    py_results = rank_documents(terms, sample_index, TfIdfScorer(), top_k=5)
    np_results = rank_tfidf_numpy(terms, np_index, top_k=5)

    assert len(py_results) == len(np_results)
    assert [r.doc_id for r in py_results] == [r.doc_id for r in np_results]

    for py_r, np_r in zip(py_results, np_results, strict=True):
        assert abs(py_r.score - np_r.score) < 1e-5


def test_numpy_index_save_and_load(sample_index: Index, tmp_path: Path) -> None:
    """Перевірка серіалізації та десеріалізації NumPyIndex у формат .npz."""
    np_index = NumPyIndex.from_index(sample_index)
    out_file = tmp_path / "numpy_idx.npz"

    np_index.save(out_file)
    assert out_file.exists()

    loaded = NumPyIndex.load(out_file)
    assert loaded.total_docs == np_index.total_docs
    assert len(loaded.postings) == len(np_index.postings)

    # Перевірка пошуку на завантаженому індексі
    terms = ["python", "search"]
    res1 = rank_bm25_numpy(terms, np_index, top_k=3)
    res2 = rank_bm25_numpy(terms, loaded, top_k=3)
    assert [r.doc_id for r in res1] == [r.doc_id for r in res2]
