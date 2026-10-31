import heapq
import math
from typing import Protocol

from findex.index import Index
from findex.models import SearchResult


class Scorer(Protocol):
    """
    Протокол для функцій оцінки релевантності документів (duck typing).
    Будь-який клас із методом score задовольняє цей інтерфейс без наслідування.
    """

    def score(self, term: str, tf: int, doc_len: int, index: Index) -> float: ...


class TfIdfScorer:
    """
    Класичний скорер TF-IDF з логарифмічним масштабуванням частоти терміна.
    """

    def score(self, term: str, tf: int, doc_len: int, index: Index) -> float:
        if tf <= 0:
            return 0.0
        n = index.num_docs
        df = index.df(term)
        if df == 0 or n == 0:
            return 0.0

        # Логарифмічний TF та класичний smoothing IDF
        tf_weight = 1.0 + math.log(tf)
        idf_weight = math.log(1.0 + (n / df))
        return tf_weight * idf_weight


class BM25Scorer:
    """
    Скорер Okapi BM25 з насиченням частоти терміна (k1) та нормалізацією довжини (b).
    Стандарт галузі для текстового пошуку.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(self, term: str, tf: int, doc_len: int, index: Index) -> float:
        if tf <= 0:
            return 0.0
        n = index.num_docs
        df = index.df(term)
        if df == 0 or n == 0:
            return 0.0

        # Robertson-Spärck Jones IDF
        idf = math.log(1.0 + ((n - df + 0.5) / (df + 0.5)))
        avg_dl = index.avg_doc_length or 1.0

        # Обчислення ваги з насиченням частоти та нормалізацією довжини документа
        len_norm = 1.0 - self.b + self.b * (doc_len / avg_dl)
        tf_component = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)

        return idf * tf_component


def rank_documents(
    query_terms: list[str],
    index: Index,
    scorer: Scorer,
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Ранжує документи за сумарним балом скорера для списку термінів.
    Використовує heapq.nlargest для ефективного взяття топ-K без повного сортування O(N log N).
    """
    doc_scores: dict[int, float] = {}

    for term in query_terms:
        postings = index.get_postings(term)
        for p in postings:
            meta = index.doc_meta.get(p.doc_id)
            doc_len = meta.length if meta else 0
            term_score = scorer.score(term, p.term_frequency, doc_len, index)
            doc_scores[p.doc_id] = doc_scores.get(p.doc_id, 0.0) + term_score

    if not doc_scores:
        return []

    # Створюємо генератор SearchResult об'єктів
    results = (
        SearchResult(
            score=score,
            doc_id=doc_id,
            title=index.doc_meta[doc_id].title
            if doc_id in index.doc_meta
            else f"Doc #{doc_id}",
            source_name=index.doc_meta[doc_id].source_name
            if doc_id in index.doc_meta
            else "",
        )
        for doc_id, score in doc_scores.items()
    )

    # Завдяки @dataclass(order=True) на SearchResult, heapq.nlargest автоматично порівнює за score
    return heapq.nlargest(top_k, results)
