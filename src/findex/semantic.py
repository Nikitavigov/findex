"""Семантичний та гібридний пошук (Semantic & Hybrid Retrieval) для findex.

Використовує нейромережеві векторні ембединги (fastembed / BAAI/bge-small-en-v1.5)
для представлення змісту документів у багатовимірному просторі.

Підтримує:
1. Поділ документів на фрагменти (chunking) та max-pooling по документу.
2. Семантичний пошук через косинусну близькість (скалярний добуток нормалізованих векторів).
3. Гібридний пошук через Reciprocal Rank Fusion (RRF), що об'єднує сигнатуру BM25 та векторну подібність.
"""

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from findex.models import DocMeta, SearchResult
from findex.numpy_index import NumPyIndex, rank_bm25_numpy

logger = logging.getLogger("findex.semantic")

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def chunk_text(text: str, max_words: int = 120, overlap: int = 25) -> list[str]:
    """Розбиває текст документа на перекривні фрагменти (чанкі)."""
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [text.strip()]

    chunks: list[str] = []
    step = max_words - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_words]
        chunks.append(" ".join(chunk_words))
        if i + max_words >= len(words):
            break
    return chunks


@dataclass
class SemanticIndex:
    """Векторний індекс ембедингів фрагментів документів."""

    matrix: np.ndarray  # [N_chunks, embedding_dim], float32, L2-нормалізований
    chunk_doc_ids: np.ndarray  # [N_chunks], int32
    chunk_texts: list[str]
    model_name: str = DEFAULT_MODEL

    def save(self, file_path: str | Path) -> None:
        """Зберігає матрицю ембедингів та метадані у файл .npz."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            matrix=self.matrix,
            chunk_doc_ids=self.chunk_doc_ids,
            chunk_texts=np.array(self.chunk_texts, dtype=object),
            model_name=self.model_name,
        )

    @classmethod
    def load(cls, file_path: str | Path) -> "SemanticIndex":
        """Завантажує векторний індекс із файлу .npz."""
        data = np.load(file_path, allow_pickle=True)
        matrix: np.ndarray = data["matrix"]
        chunk_doc_ids: np.ndarray = data["chunk_doc_ids"]
        chunk_texts = list(data["chunk_texts"])
        model_name = str(data.get("model_name", DEFAULT_MODEL))
        return cls(
            matrix=matrix,
            chunk_doc_ids=chunk_doc_ids,
            chunk_texts=chunk_texts,
            model_name=model_name,
        )


class SemanticEngine:
    """Двигун для обчислення векторних ембедингів та семантичного пошуку."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        """Лінива ініціалізація ONNX-моделі fastembed."""
        if self._model is None:
            logger.info("Ініціалізація моделі ембедингів %s...", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def build_index(
        self,
        documents: Iterable[tuple[int, str]],
        max_words: int = 120,
        overlap: int = 25,
    ) -> SemanticIndex:
        """Будує векторний індекс для ітератора (doc_id, text)."""
        all_chunks: list[str] = []
        chunk_doc_ids: list[int] = []

        for doc_id, text in documents:
            chunks = chunk_text(text, max_words=max_words, overlap=overlap)
            for ch in chunks:
                if ch.strip():
                    all_chunks.append(ch)
                    chunk_doc_ids.append(doc_id)

        if not all_chunks:
            dim = 384
            return SemanticIndex(
                matrix=np.empty((0, dim), dtype=np.float32),
                chunk_doc_ids=np.empty((0,), dtype=np.int32),
                chunk_texts=[],
                model_name=self.model_name,
            )

        logger.info("Обчислення ембедингів для %d чанків...", len(all_chunks))
        # Отримуємо генератор векторів з fastembed
        raw_embs = list(self.model.embed(all_chunks))
        matrix = np.array(raw_embs, dtype=np.float32)

        # L2-нормалізація: тепер косинусна подібність є простим скалярним добутком (dot product)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms

        return SemanticIndex(
            matrix=matrix,
            chunk_doc_ids=np.array(chunk_doc_ids, dtype=np.int32),
            chunk_texts=all_chunks,
            model_name=self.model_name,
        )

    def search_semantic(
        self,
        query: str,
        index: SemanticIndex,
        doc_meta: dict[int, DocMeta],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Семантичний пошук: знаходить документи з найбільшою косинусною подібністю."""
        if not query.strip() or len(index.matrix) == 0:
            return []

        # Ембединг пошукового запиту
        q_raw = next(iter(self.model.embed([query])))
        q_vec = np.array(q_raw, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Скалярний добуток матриці чанків на вектор запиту
        sims: np.ndarray = index.matrix @ q_vec

        # Max-pooling: кожному документу призначається оцінка його найкращого чанка
        best_scores: dict[int, float] = defaultdict(float)
        best_chunks: dict[int, str] = {}

        for score, doc_id, text in zip(sims, index.chunk_doc_ids, index.chunk_texts, strict=True):
            d_id = int(doc_id)
            sc = float(score)
            if sc > best_scores[d_id]:
                best_scores[d_id] = sc
                best_chunks[d_id] = text

        # Сортуємо унікальні документи за оцінкою
        sorted_docs = sorted(best_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results: list[SearchResult] = []
        for d_id, sc in sorted_docs:
            meta = doc_meta.get(d_id)
            title = meta.title if meta else f"Документ #{d_id}"
            source_name = meta.source_name if meta else ""
            snippet = best_chunks.get(d_id, "")[:160] + "..."
            results.append(
                SearchResult(
                    score=round(sc, 4),
                    doc_id=d_id,
                    title=title,
                    source_name=source_name,
                    snippet=snippet,
                )
            )

        return results


def hybrid_search(
    query: str,
    keyword_index: NumPyIndex,
    semantic_index: SemanticIndex,
    semantic_engine: SemanticEngine,
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Гібридний пошук: поєднання BM25 та векторного семантичного пошуку через RRF.

    Формула Reciprocal Rank Fusion:
        RRF(d) = sum_{m in {bm25, semantic}} 1.0 / (rrf_k + rank_m(d))
    """
    from findex.tokenizer import tokenize

    query_terms = list(tokenize(query))

    # 1. Отримуємо ранжовані списки від обох систем
    bm25_list = rank_bm25_numpy(query_terms, keyword_index, top_k=50) if query_terms else []
    sem_list = semantic_engine.search_semantic(
        query, semantic_index, keyword_index.doc_meta, top_k=50
    )

    rrf_scores: dict[int, float] = defaultdict(float)
    doc_snippets: dict[int, str] = {}

    for rank, res in enumerate(bm25_list, start=1):
        rrf_scores[res.doc_id] += 1.0 / (rrf_k + rank)
        if res.snippet:
            doc_snippets[res.doc_id] = res.snippet

    for rank, res in enumerate(sem_list, start=1):
        rrf_scores[res.doc_id] += 1.0 / (rrf_k + rank)
        if res.doc_id not in doc_snippets and res.snippet:
            doc_snippets[res.doc_id] = res.snippet

    # 2. Сортуємо документи за комбінованим RRF-балом
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results: list[SearchResult] = []
    for d_id, sc in sorted_rrf:
        meta = keyword_index.doc_meta.get(d_id)
        title = meta.title if meta else f"Документ #{d_id}"
        source_name = meta.source_name if meta else ""
        snippet = doc_snippets.get(d_id, "")
        results.append(
            SearchResult(
                score=round(sc, 5),
                doc_id=d_id,
                title=title,
                source_name=source_name,
                snippet=snippet,
            )
        )

    return results
