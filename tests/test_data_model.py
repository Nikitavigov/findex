from collections.abc import Mapping
from pathlib import Path

import pytest

from findex.index import Index, build_index, open_index
from findex.models import DocMeta, Posting


def test_index_mapping_protocol():
    """Перевірка реалізації протоколу Mapping: len, in, [], iter, repr."""
    postings = {
        "python": [Posting(0, 2), Posting(1, 1)],
        "search": [Posting(0, 1)],
    }
    doc_meta = {
        0: DocMeta(0, "doc0.txt", "Title 0", 10),
        1: DocMeta(1, "doc1.txt", "Title 1", 20),
    }
    index = Index(postings=postings, doc_meta=doc_meta)

    # Перевірка наслідування Mapping
    assert isinstance(index, Mapping)

    # len() -> кількість термінів
    assert len(index) == 2

    # in -> перевірка наявності терміна з нормалізацією
    assert "python" in index
    assert "PYTHON" in index
    assert "java" not in index

    # index[term] -> повертає постінги, викидає KeyError для відсутніх
    assert index["python"] == [Posting(0, 2), Posting(1, 1)]
    with pytest.raises(KeyError):
        _ = index["nonexistent"]

    # ітерація по термінах
    terms = list(iter(index))
    assert set(terms) == {"python", "search"}

    # repr(index)
    assert repr(index) == "Index(terms=2, docs=2)"


def test_index_properties():
    """Перевірка властивостей num_docs, avg_doc_length (cached_property) та df."""
    doc_meta = {
        0: DocMeta(0, "doc0.txt", "T0", 100),
        1: DocMeta(1, "doc1.txt", "T1", 200),
        2: DocMeta(2, "doc2.txt", "T2", 300),
    }
    postings = {"word": [Posting(0, 1), Posting(1, 1)]}
    index = Index(postings=postings, doc_meta=doc_meta)

    assert index.num_docs == 3
    # avg_doc_length = (100 + 200 + 300) / 3 = 200.0
    assert index.avg_doc_length == 200.0

    # df()
    assert index.df("word") == 2
    assert index.df("missing") == 0


def test_open_index_context_manager(tmp_path: Path):
    """Перевірка роботи контекстного менеджера open_index з гарантованим закриттям."""
    (tmp_path / "test.txt").write_text("тест контекстного менеджера", encoding="utf-8")
    idx = build_index(tmp_path)
    idx_file = tmp_path / "index.bin"
    from findex.index import save_index

    save_index(idx, idx_file)

    with open_index(idx_file) as loaded:
        assert isinstance(loaded, Index)
        assert len(loaded) > 0
        assert "контекстного" in loaded
