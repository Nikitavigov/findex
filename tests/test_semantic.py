"""Тести для модуля семантичного та гібридного пошуку findex (Lab 8).

Перевіряє роботу розбиття на чанки (chunking), обчислення векторних ембедингів,
косинусної подібності та злиття рейтингів через Reciprocal Rank Fusion (RRF).
"""

from pathlib import Path

import numpy as np

from findex.index import Index
from findex.models import DocMeta, Posting
from findex.numpy_index import NumPyIndex
from findex.semantic import (
    SemanticEngine,
    SemanticIndex,
    chunk_text,
    hybrid_search,
)


def test_chunk_text_basic() -> None:
    """Перевірка розбиття довгого тексту на перекривні фрагменти."""
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)

    chunks = chunk_text(text, max_words=30, overlap=10)
    assert len(chunks) > 1
    # Кожен чанк має містити не більше 30 слів
    for ch in chunks:
        assert len(ch.split()) <= 30


def test_chunk_text_short() -> None:
    """Короткий текст не повинен розбиватися на кілька частин."""
    text = "Короткий документ про мову Python."
    chunks = chunk_text(text, max_words=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_semantic_index_save_and_load(tmp_path: Path) -> None:
    """Перевірка серіалізації та завантаження SemanticIndex."""
    matrix = np.random.randn(5, 16).astype(np.float32)
    chunk_doc_ids = np.array([1, 1, 2, 3, 3], dtype=np.int32)
    chunk_texts = [f"Text chunk #{i}" for i in range(5)]

    idx = SemanticIndex(
        matrix=matrix,
        chunk_doc_ids=chunk_doc_ids,
        chunk_texts=chunk_texts,
        model_name="mock-model",
    )

    out_path = tmp_path / "mock_sem.npz"
    idx.save(out_path)
    assert out_path.exists()

    loaded = SemanticIndex.load(out_path)
    assert loaded.matrix.shape == (5, 16)
    assert np.array_equal(loaded.chunk_doc_ids, chunk_doc_ids)
    assert len(loaded.chunk_texts) == 5
    assert loaded.model_name == "mock-model"


def test_semantic_search_mock() -> None:
    """Перевірка пошуку за штучно заданими векторами косинусної подібності."""
    matrix = np.array(
        [
            [1.0, 0.0],  # Чанк документа 1
            [0.0, 1.0],  # Чанк документа 2
            [0.707, 0.707],  # Чанк документа 3
        ],
        dtype=np.float32,
    )
    chunk_doc_ids = np.array([1, 2, 3], dtype=np.int32)
    chunk_texts = ["Чанк 1", "Чанк 2", "Чанк 3"]

    idx = SemanticIndex(
        matrix=matrix,
        chunk_doc_ids=chunk_doc_ids,
        chunk_texts=chunk_texts,
    )

    doc_meta = {
        1: DocMeta(doc_id=1, source_name="d1.txt", title="Док 1", length=10),
        2: DocMeta(doc_id=2, source_name="d2.txt", title="Док 2", length=10),
        3: DocMeta(doc_id=3, source_name="d3.txt", title="Док 3", length=10),
    }

    engine = SemanticEngine()

    class MockModel:
        def embed(self, texts: list[str]) -> list[list[float]]:
            # Запит векторно близький до [1.0, 0.0] (Документ 1)
            return [[0.99, 0.05]]

    engine._model = MockModel()  # type: ignore

    results = engine.search_semantic("тестовий запит", idx, doc_meta, top_k=3)
    assert len(results) == 3
    # Найближчим має бути Документ 1
    assert results[0].doc_id == 1
    assert results[0].score > results[1].score


def test_hybrid_search_rrf() -> None:
    """Перевірка злиття результатів BM25 та векторного пошуку через RRF."""
    meta = {
        1: DocMeta(doc_id=1, source_name="d1.txt", title="Python Документ", length=10),
        2: DocMeta(doc_id=2, source_name="d2.txt", title="Java Документ", length=10),
        3: DocMeta(doc_id=3, source_name="d3.txt", title="Оптимізація та Швидкодія", length=10),
    }
    postings = {
        "python": [Posting(doc_id=1, term_frequency=5)],
        "java": [Posting(doc_id=2, term_frequency=5)],
    }
    idx = Index(postings=postings, doc_meta=meta)
    np_index = NumPyIndex.from_index(idx)

    # Семантичний індекс дає високий бал документу 3
    matrix = np.array([[0.1, 0.9], [0.1, 0.1], [0.9, 0.1]], dtype=np.float32)
    sem_index = SemanticIndex(
        matrix=matrix,
        chunk_doc_ids=np.array([1, 2, 3], dtype=np.int32),
        chunk_texts=["Python", "Java", "Оптимізація"],
    )

    engine = SemanticEngine()

    class MockModel:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.8, 0.2]]

    engine._model = MockModel()  # type: ignore

    results = hybrid_search("python швидкодія", np_index, sem_index, engine, top_k=3)
    assert len(results) >= 2
    # Документи мають коректно злитися через RRF
    assert results[0].doc_id in {1, 3}
    assert results[0].score > 0.0
