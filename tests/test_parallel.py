from pathlib import Path

import pytest

from findex.index import build_index
from findex.parallel import (
    _collect_corpus_files,
    build_index_parallel,
    build_partial,
    merge_partials,
    split_chunks,
)


def test_split_chunks_logic() -> None:
    """Перевірка правильного розбиття списку елементів на рівномірні чанки."""
    items = list(range(10))

    # 1 чанк
    assert split_chunks(items, 1) == [items]

    # 3 чанки: 10 ділиться як 4, 3, 3
    c3 = split_chunks(items, 3)
    assert len(c3) == 3
    assert sum(len(c) for c in c3) == 10
    assert c3[0] == [0, 1, 2, 3]
    assert c3[1] == [4, 5, 6]
    assert c3[2] == [7, 8, 9]

    # Більше чанків, ніж елементів
    c15 = split_chunks(items, 15)
    assert sum(len(c) for c in c15) == 10
    assert all(len(c) == 1 for c in c15)

    # Порожній список
    assert split_chunks([], 4) == []


def test_build_partial_and_merge(sample_corpus_dir: Path) -> None:
    """
    Перевірка M1: Послідовне об'єднання merge([build_partial(усі_шляхи)])
    дає індекс, ідентичний build_index().
    """
    files = _collect_corpus_files(sample_corpus_dir)
    assert len(files) == 3

    # Серійний шлях через build_index
    serial_idx = build_index(sample_corpus_dir)

    # Шлях через build_partial + merge_partials
    partial = build_partial(files, start_doc_id=0)
    merged_idx = merge_partials([partial], corpus_dir=sample_corpus_dir)

    assert merged_idx.num_docs == serial_idx.num_docs
    assert len(merged_idx) == len(serial_idx)
    assert merged_idx.postings == serial_idx.postings
    assert merged_idx.doc_meta == serial_idx.doc_meta


def test_executors_equivalence(sample_corpus_dir: Path) -> None:
    """
    Перевірка еквівалентності результатів для трьох екзекуторів:
    serial, threads (2 воркери) та processes (2 воркери).
    Усі три варіанти мають повертати побайтово ідентичні індекси.
    """
    idx_serial, _ = build_index_parallel(
        sample_corpus_dir, workers=1, executor="serial"
    )
    idx_threads, _ = build_index_parallel(
        sample_corpus_dir, workers=2, executor="threads"
    )
    idx_procs, _ = build_index_parallel(
        sample_corpus_dir, workers=2, executor="processes"
    )

    # Перевірка кількості документів та словника
    assert idx_serial.num_docs == idx_threads.num_docs == idx_procs.num_docs
    assert len(idx_serial) == len(idx_threads) == len(idx_procs)

    # Повна ідентичність метаданих та постінгів
    assert idx_serial.doc_meta == idx_threads.doc_meta == idx_procs.doc_meta
    assert idx_serial.postings == idx_threads.postings == idx_procs.postings


def test_invalid_executor(sample_corpus_dir: Path) -> None:
    """Перевірка викидання винятку при невідомому екзекуторі."""
    with pytest.raises(ValueError, match="Невідомий тип екзекутора"):
        build_index_parallel(sample_corpus_dir, executor="unknown_executor")  # type: ignore
