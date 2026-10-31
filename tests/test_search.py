from findex.index import InvertedIndex
from findex.models import DocMeta, Posting
from findex.search import (
    execute_query,
    merge_and,
    merge_not,
    merge_or,
    set_and,
    set_not,
    set_or,
)


def test_merge_and():
    p1 = [Posting(1, 2), Posting(3, 1), Posting(5, 4), Posting(7, 1)]
    p2 = [Posting(2, 1), Posting(3, 3), Posting(7, 2), Posting(9, 1)]

    res = merge_and(p1, p2)
    assert [p.doc_id for p in res] == [3, 7]
    assert res[0].term_frequency == 4  # 1 + 3
    assert res[1].term_frequency == 3  # 1 + 2


def test_merge_or():
    p1 = [Posting(1, 2), Posting(4, 1)]
    p2 = [Posting(2, 3), Posting(4, 5)]

    res = merge_or(p1, p2)
    assert [p.doc_id for p in res] == [1, 2, 4]
    assert res[2].term_frequency == 6


def test_merge_not():
    p1 = [Posting(1, 1), Posting(2, 1), Posting(3, 1), Posting(4, 1)]
    p2 = [Posting(2, 1), Posting(4, 1)]

    res = merge_not(p1, p2)
    assert [p.doc_id for p in res] == [1, 3]


def test_merge_vs_set_equivalence():
    """Перевірка тотожності результатів двох алгоритмів (двовказівникового та множинного)."""
    p1 = [Posting(1, 2), Posting(3, 4), Posting(6, 1), Posting(8, 2)]
    p2 = [Posting(2, 1), Posting(3, 1), Posting(7, 3), Posting(8, 5)]

    assert merge_and(p1, p2) == set_and(p1, p2)
    assert merge_or(p1, p2) == set_or(p1, p2)
    assert merge_not(p1, p2) == set_not(p1, p2)


def test_execute_query():
    postings = {
        "python": [Posting(0, 3), Posting(1, 1), Posting(2, 2)],
        "rust": [Posting(1, 2), Posting(2, 1)],
        "legacy": [Posting(2, 1)],
    }
    doc_meta = {
        0: DocMeta(0, "doc0.txt", "Intro to Python", 10),
        1: DocMeta(1, "doc1.txt", "Python and Rust", 15),
        2: DocMeta(2, "doc2.txt", "Python Rust Legacy", 20),
    }
    index = InvertedIndex(
        postings=postings, doc_meta=doc_meta, total_docs=3, total_tokens=45
    )

    # AND запит
    res_and = execute_query("python AND rust", index, engine="merge")
    assert [p.doc_id for p in res_and] == [1, 2]

    # OR запит
    res_or = execute_query("rust OR legacy", index, engine="merge")
    assert [p.doc_id for p in res_or] == [1, 2]

    # NOT запит
    res_not = execute_query("python NOT rust", index, engine="merge")
    assert [p.doc_id for p in res_not] == [0]

    # Порівняння результатів рушіїв merge та set
    assert execute_query("python AND rust", index, engine="merge") == execute_query(
        "python AND rust", index, engine="set"
    )
