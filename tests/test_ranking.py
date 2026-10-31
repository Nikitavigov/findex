from findex.index import Index
from findex.models import DocMeta, Posting
from findex.ranking import BM25Scorer, TfIdfScorer, rank_documents


def make_test_index() -> Index:
    """Створює тестовий індекс для перевірки властивостей ранжування."""
    doc_meta = {
        0: DocMeta(0, "short.txt", "Short Doc", 20),
        1: DocMeta(1, "long.txt", "Long Doc", 500),
        2: DocMeta(2, "common.txt", "Doc with common term", 50),
    }
    # "rare" зустрічається тільки в doc 0 (df=1)
    # "common" зустрічається в усіх doc 0, 1, 2 (df=3)
    postings = {
        "rare": [Posting(0, 1)],
        "common": [Posting(0, 1), Posting(1, 1), Posting(2, 1)],
        "repeated": [Posting(0, 1), Posting(1, 20)],
    }
    return Index(postings=postings, doc_meta=doc_meta)


def test_sanity_check_1_rare_beats_frequent():
    """
    Sanity Check 1: Рідкісний термін дає більший внесок у бал, ніж частий термін.
    """
    index = make_test_index()
    bm25 = BM25Scorer()
    tfidf = TfIdfScorer()

    score_rare_bm25 = bm25.score("rare", tf=1, doc_len=20, index=index)
    score_common_bm25 = bm25.score("common", tf=1, doc_len=20, index=index)
    assert score_rare_bm25 > score_common_bm25

    score_rare_tfidf = tfidf.score("rare", tf=1, doc_len=20, index=index)
    score_common_tfidf = tfidf.score("common", tf=1, doc_len=20, index=index)
    assert score_rare_tfidf > score_common_tfidf


def test_sanity_check_2_term_saturation():
    """
    Sanity Check 2: Насичення частоти терміна.
    Різниця між 19-м і 20-м входженням набагато менша, ніж між 0-м і 1-м.
    """
    index = make_test_index()
    bm25 = BM25Scorer(k1=1.5, b=0.75)

    score_1 = bm25.score("repeated", tf=1, doc_len=100, index=index)
    score_2 = bm25.score("repeated", tf=2, doc_len=100, index=index)
    score_19 = bm25.score("repeated", tf=19, doc_len=100, index=index)
    score_20 = bm25.score("repeated", tf=20, doc_len=100, index=index)

    gain_first = score_1
    gain_second = score_2 - score_1
    gain_twentieth = score_20 - score_19

    assert gain_first > gain_second > gain_twentieth
    # Приріст від 20-го входження складає мізерну частку від першого
    assert gain_twentieth < (0.1 * gain_first)


def test_sanity_check_3_length_normalization():
    """
    Sanity Check 3: Нормалізація довжини документа.
    Короткий документ з одним входженням терміна ранжується вище за величезний документ з одним входженням.
    """
    index = make_test_index()
    bm25 = BM25Scorer()

    score_short = bm25.score("rare", tf=1, doc_len=20, index=index)
    score_long = bm25.score("rare", tf=1, doc_len=500, index=index)

    assert score_short > score_long


def test_rank_documents_top_k():
    """Перевірка взяття топ-K документів через heapq.nlargest."""
    index = make_test_index()
    results = rank_documents(["rare", "common"], index, BM25Scorer(), top_k=2)

    assert len(results) <= 2
    # Найвищий бал повинен бути у документа з рідкісним словом
    assert results[0].doc_id == 0
    assert results[0].score >= results[1].score
