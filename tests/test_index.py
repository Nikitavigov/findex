from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from findex.index import build_index, load_index, save_index
from findex.models import DocMeta, Posting


def test_posting_dataclass_contract():
    """Перевірка контракту Posting: slots, frozen, хешування та незмінність."""
    p1 = Posting(doc_id=1, term_frequency=5)
    p2 = Posting(doc_id=1, term_frequency=5)
    p3 = Posting(doc_id=2, term_frequency=1)

    # Рівність та хешованість
    assert p1 == p2
    assert hash(p1) == hash(p2)
    assert p1 != p3

    # Перевірка можливості збереження у множині set
    s = {p1, p2, p3}
    assert len(s) == 2

    # Перевірка незмінності (frozen)
    with pytest.raises(FrozenInstanceError):
        p1.term_frequency = 10  # type: ignore

    # Перевірка slots (відсутність __dict__)
    assert not hasattr(p1, "__dict__")


def test_docmeta_dataclass_contract():
    """Перевірка контракту DocMeta: slots, frozen та хешованість."""
    m1 = DocMeta(doc_id=0, source_name="file.txt", title="Test", length=42)
    m2 = DocMeta(doc_id=0, source_name="file.txt", title="Test", length=42)

    assert m1 == m2
    assert hash(m1) == hash(m2)
    assert not hasattr(m1, "__dict__")

    with pytest.raises(FrozenInstanceError):
        m1.title = "New Title"  # type: ignore


def test_build_index_sorted_postings(tmp_path: Path):
    """Перевірка побудови індексу: списки постінгів мають бути впорядковані за doc_id."""
    (tmp_path / "doc0.txt").write_text("яблуко груша яблуко", encoding="utf-8")
    (tmp_path / "doc1.txt").write_text("груша слива", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("яблуко слива вишня", encoding="utf-8")

    index = build_index(tmp_path)

    assert index.total_docs == 3
    assert index.vocabulary_size == 4

    apple_postings = index.get_postings("яблуко")
    # Має бути в документах 0 та 2 (впорядковано за doc_id)
    doc_ids = [p.doc_id for p in apple_postings]
    assert doc_ids == sorted(doc_ids)
    assert doc_ids == [0, 2]
    # Частота у документі 0: 2, у документі 2: 1
    assert apple_postings[0].term_frequency == 2
    assert apple_postings[1].term_frequency == 1


def test_save_load_pickle(tmp_path: Path):
    """Перевірка серіалізації та десеріалізації через pickle."""
    (tmp_path / "test.txt").write_text("алгоритм пошуку", encoding="utf-8")
    index = build_index(tmp_path)

    pkl_file = tmp_path / "index.bin"
    save_index(index, pkl_file, fmt="pickle")
    loaded = load_index(pkl_file, fmt="pickle")

    assert loaded.total_docs == index.total_docs
    assert loaded.total_tokens == index.total_tokens
    assert loaded.get_postings("алгоритм") == index.get_postings("алгоритм")


def test_save_load_json(tmp_path: Path):
    """Перевірка серіалізації та десеріалізації через JSON."""
    (tmp_path / "test.txt").write_text("тестування json формату", encoding="utf-8")
    index = build_index(tmp_path)

    json_file = tmp_path / "index.json"
    save_index(index, json_file, fmt="json")
    loaded = load_index(json_file, fmt="json")

    assert loaded.total_docs == index.total_docs
    assert loaded.total_tokens == index.total_tokens
    assert loaded.get_postings("тестування") == index.get_postings("тестування")
