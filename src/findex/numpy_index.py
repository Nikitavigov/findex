"""Векторизований інвертований індекс на базі масивів NumPy.

Замінює списки Python-об'єктів Posting компактними одновимірними C-масивами
doc_ids (int32) та term_freqs (int32). Довжини документів зберігаються
у суцільному масиві doc_lengths (int32).

Ранжування BM25 та TF-IDF виконується через векторизовані операції NumPy (ufuncs),
а вибірка найкращих документів здійснюється за O(N) через np.argpartition.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from findex.index import Index
from findex.models import DocMeta, SearchResult
from findex.tokenizer import normalize_text


@dataclass(slots=True)
class NumPyPostings:
    """Постінги одного терміна у вигляді двох паралельних масивів NumPy int32."""

    doc_ids: np.ndarray  # dtype=int32, відсортований масив doc_id
    term_freqs: np.ndarray  # dtype=int32, відповідні частоти term_frequency


class NumPyIndex:
    """Векторизований пошуковий індекс на базі масивів NumPy."""

    def __init__(
        self,
        postings: dict[str, NumPyPostings],
        doc_lengths: np.ndarray,
        doc_meta: dict[int, DocMeta],
        total_docs: int,
        corpus_dir: Path | None = None,
    ) -> None:
        self.postings = postings
        self.doc_lengths = doc_lengths.astype(np.int32)
        self.doc_meta = doc_meta
        self.total_docs = total_docs
        self.corpus_dir = corpus_dir

        # Середня довжина документа
        if total_docs > 0 and len(doc_lengths) > 0:
            valid_lengths = [m.length for m in doc_meta.values()]
            self.avg_doc_len = float(sum(valid_lengths) / len(valid_lengths)) if valid_lengths else 1.0
        else:
            self.avg_doc_len = 1.0

    @classmethod
    def from_index(cls, index: Index) -> "NumPyIndex":
        """Конвертує стандартний Index з лаб 2–4 у векторизований NumPyIndex."""
        postings_np: dict[str, NumPyPostings] = {}

        for term, plist in index.postings.items():
            if not plist:
                continue
            doc_ids = np.array([p.doc_id for p in plist], dtype=np.int32)
            term_freqs = np.array([p.term_frequency for p in plist], dtype=np.int32)
            postings_np[term] = NumPyPostings(doc_ids=doc_ids, term_freqs=term_freqs)

        max_doc_id = max(index.doc_meta.keys()) if index.doc_meta else 0
        doc_lengths = np.zeros(max_doc_id + 1, dtype=np.int32)
        for doc_id, meta in index.doc_meta.items():
            doc_lengths[doc_id] = meta.length

        return cls(
            postings=postings_np,
            doc_lengths=doc_lengths,
            doc_meta=dict(index.doc_meta),
            total_docs=index.num_docs,
            corpus_dir=index.corpus_dir,
        )

    def save(self, file_path: str | Path) -> None:
        """Зберігає векторизований індекс у стиснутий файл .npz."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays_dict: dict[str, np.ndarray] = {
            "__doc_lengths__": self.doc_lengths,
        }
        meta_items: list[tuple[int, str, str, int]] = []
        for meta in self.doc_meta.values():
            meta_items.append((meta.doc_id, meta.source_name, meta.title, meta.length))

        for term, post in self.postings.items():
            arrays_dict[f"ids_{term}"] = post.doc_ids
            arrays_dict[f"tfs_{term}"] = post.term_freqs

        arrays_dict["__metadata__"] = np.array(meta_items, dtype=object)

        # Зберігаємо масиви та метадані
        np.savez_compressed(path, allow_pickle=True, **arrays_dict)

    @classmethod
    def load(cls, file_path: str | Path) -> "NumPyIndex":
        """Завантажує векторизований індекс із файлу .npz."""
        path = Path(file_path)
        data = np.load(path, allow_pickle=True)

        doc_lengths: np.ndarray = data["__doc_lengths__"]
        metadata_raw = data["__metadata__"]

        doc_meta: dict[int, DocMeta] = {}
        for row in metadata_raw:
            d_id = int(row[0])
            doc_meta[d_id] = DocMeta(
                doc_id=d_id,
                source_name=str(row[1]),
                title=str(row[2]),
                length=int(row[3]),
            )

        postings: dict[str, NumPyPostings] = {}
        for key in data.files:
            if key.startswith("ids_"):
                term = key[4:]
                tfs_key = f"tfs_{term}"
                if tfs_key in data:
                    postings[term] = NumPyPostings(
                        doc_ids=data[key],
                        term_freqs=data[tfs_key],
                    )

        return cls(
            postings=postings,
            doc_lengths=doc_lengths,
            doc_meta=doc_meta,
            total_docs=len(doc_meta),
            corpus_dir=None,
        )


def rank_bm25_numpy(
    query_terms: list[str],
    index: NumPyIndex,
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[SearchResult]:
    """Векторизоване ранжування BM25 над масивами NumPy з np.argpartition для вибірки топ-K.

    Забезпечує 100% ідентичний порядок та оцінки результатів порівняно з BM25Scorer з Лаб 3.
    """
    if not query_terms or index.total_docs == 0:
        return []

    scores = np.zeros(len(index.doc_lengths), dtype=np.float64)
    n = float(index.total_docs)
    avg_dl = float(index.avg_doc_len) or 1.0

    for term in query_terms:
        norm = normalize_text(term)
        post = index.postings.get(norm)
        if post is None or len(post.doc_ids) == 0:
            continue

        df = float(len(post.doc_ids))
        # Robertson-Spärck Jones IDF
        idf = math.log(1.0 + ((n - df + 0.5) / (df + 0.5)))

        doc_ids = post.doc_ids
        tfs = post.term_freqs.astype(np.float64)
        lens = index.doc_lengths[doc_ids].astype(np.float64)

        len_norm = (1.0 - b) + b * (lens / avg_dl)
        tf_component = (tfs * (k1 + 1.0)) / (tfs + k1 * len_norm)

        # Додавання балів до накопичувального масиву
        scores[doc_ids] += (idf * tf_component)

    nonzero_ids = np.nonzero(scores > 0.0)[0]
    if len(nonzero_ids) == 0:
        return []

    candidate_scores = scores[nonzero_ids]

    # Вибірка топ-k через np.argpartition (O(N) замість O(N log N))
    if len(nonzero_ids) <= top_k:
        sorted_within = np.argsort(-candidate_scores)
        top_doc_ids = nonzero_ids[sorted_within]
        top_scores = candidate_scores[sorted_within]
    else:
        part_idx = np.argpartition(candidate_scores, -top_k)[-top_k:]
        sorted_within = part_idx[np.argsort(-candidate_scores[part_idx])]
        top_doc_ids = nonzero_ids[sorted_within]
        top_scores = candidate_scores[sorted_within]

    results: list[SearchResult] = []
    for d_id, sc in zip(top_doc_ids, top_scores, strict=True):
        doc_id_int = int(d_id)
        meta = index.doc_meta.get(doc_id_int)
        results.append(
            SearchResult(
                score=float(sc),
                doc_id=doc_id_int,
                title=meta.title if meta else f"Документ #{doc_id_int}",
                source_name=meta.source_name if meta else "",
                snippet="",
            )
        )

    return results


def rank_tfidf_numpy(
    query_terms: list[str],
    index: NumPyIndex,
    top_k: int = 10,
) -> list[SearchResult]:
    """Векторизоване ранжування TF-IDF над масивами NumPy."""
    if not query_terms or index.total_docs == 0:
        return []

    scores = np.zeros(len(index.doc_lengths), dtype=np.float64)
    n = float(index.total_docs)

    for term in query_terms:
        norm = normalize_text(term)
        post = index.postings.get(norm)
        if post is None or len(post.doc_ids) == 0:
            continue

        df = float(len(post.doc_ids))
        idf_weight = math.log(1.0 + (n / df))

        doc_ids = post.doc_ids
        tfs = post.term_freqs.astype(np.float64)

        tf_weight = 1.0 + np.log(tfs)
        scores[doc_ids] += (tf_weight * idf_weight)

    nonzero_ids = np.nonzero(scores > 0.0)[0]
    if len(nonzero_ids) == 0:
        return []

    candidate_scores = scores[nonzero_ids]

    if len(nonzero_ids) <= top_k:
        sorted_within = np.argsort(-candidate_scores)
        top_doc_ids = nonzero_ids[sorted_within]
        top_scores = candidate_scores[sorted_within]
    else:
        part_idx = np.argpartition(candidate_scores, -top_k)[-top_k:]
        sorted_within = part_idx[np.argsort(-candidate_scores[part_idx])]
        top_doc_ids = nonzero_ids[sorted_within]
        top_scores = candidate_scores[sorted_within]

    results: list[SearchResult] = []
    for d_id, sc in zip(top_doc_ids, top_scores, strict=True):
        doc_id_int = int(d_id)
        meta = index.doc_meta.get(doc_id_int)
        results.append(
            SearchResult(
                score=float(sc),
                doc_id=doc_id_int,
                title=meta.title if meta else f"Документ #{doc_id_int}",
                source_name=meta.source_name if meta else "",
                snippet="",
            )
        )

    return results


def rank_documents_numpy(
    query_terms: list[str],
    index: NumPyIndex,
    scorer_type: Literal["bm25", "tfidf"] = "bm25",
    top_k: int = 10,
) -> list[SearchResult]:
    """Єдина точка входу векторизованого ранжування за типом скорера."""
    if scorer_type == "bm25":
        return rank_bm25_numpy(query_terms, index, top_k=top_k)
    return rank_tfidf_numpy(query_terms, index, top_k=top_k)
