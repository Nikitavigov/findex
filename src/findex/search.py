import argparse
import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from findex.boolean import (
    merge_and,
    merge_not,
    merge_or,
    set_and,
    set_not,
    set_or,
)
from findex.index import Index, load_index
from findex.models import Posting, SearchResult
from findex.query import parse_query
from findex.ranking import BM25Scorer, Scorer, TfIdfScorer, rank_documents
from findex.snippets import generate_snippet
from findex.tokenizer import normalize_text, tokenize

logger = logging.getLogger("findex.search")


# --- Декоратор @timed з functools.wraps ---


def timed(func: Callable) -> Callable:
    """
    Декоратор для вимірювання часу виконання функцій.
    Зберігає ім'я, docstring та сигнатуру оригінальної функції через functools.wraps.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"[{func.__name__}] виконано за {elapsed_ms:.3f} мс")
        return result

    return wrapper


# --- Кешування запитів @lru_cache ---


@functools.lru_cache(maxsize=128)
def _cached_query_postings(
    query_str: str, _index_hash: int, index: Index
) -> tuple[Posting, ...]:
    """
    Кешована версія виконання запиту.
    При повторному виклику з тим самим запитом повертає результат миттєво з кешу.
    """
    logger.info(f"Cache MISS для запиту: '{query_str}'")
    ast = parse_query(query_str)
    postings = ast.evaluate(index)
    return tuple(postings)


# --- Комплексний пошук з ранжуванням та сніпетами ---


@timed
def search_ranked(
    query_str: str,
    index: Index,
    scorer: Scorer,
    top_k: int = 10,
    with_snippets: bool = True,
) -> list[SearchResult]:
    """
    Головна точка входу ранжованого пошуку:
    1. Виділяє ключові слова запиту.
    2. Ранжує відповідні документи за допомогою вибраного скорера (BM25 / TF-IDF).
    3. Додає контекстні сніпети з підсвіткою слів.
    """
    query_terms = list(tokenize(query_str))
    if not query_terms:
        return []

    ranked = rank_documents(query_terms, index, scorer, top_k=top_k)

    if with_snippets and index.corpus_dir:
        # Підтягуємо тексти документів для формування сніпетів
        for result in ranked:
            doc_meta = index.doc_meta.get(result.doc_id)
            if not doc_meta:
                continue
            file_path = index.corpus_dir / doc_meta.source_name
            if file_path.exists():
                try:
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                    result.snippet = generate_snippet(text, query_terms)
                except OSError:
                    result.snippet = ""

    return ranked


@timed
def execute_query(query: str, index: Index, engine: str = "merge") -> list[Posting]:
    """
    Сумісна точка входу для логічних запитів (Boolean search).
    Підтримує 'merge' та 'set'.
    """
    # Спроба отримати результат через кеш для парсера
    try:
        postings = list(_cached_query_postings(query, id(index), index))
        return postings
    except (ValueError, KeyError, TypeError):
        # Fallback до базового розбору якщо запит простий
        tokens = query.strip().split()
        if not tokens:
            return []

        and_func = merge_and if engine == "merge" else set_and
        or_func = merge_or if engine == "merge" else set_or
        not_func = merge_not if engine == "merge" else set_not

        current_results: list[Posting] | None = None
        next_op = "AND"

        i = 0
        while i < len(tokens):
            token = tokens[i]
            upper_token = token.upper()

            if upper_token in {"AND", "OR", "NOT"}:
                next_op = upper_token
                i += 1
                continue

            normalized_word = normalize_text(token)
            term_postings = index.get_postings(normalized_word)

            if current_results is None:
                current_results = list(term_postings)
            else:
                if next_op == "AND":
                    current_results = and_func(current_results, term_postings)
                elif next_op == "OR":
                    current_results = or_func(current_results, term_postings)
                elif next_op == "NOT":
                    current_results = not_func(current_results, term_postings)
                next_op = "AND"

            i += 1

        return current_results if current_results is not None else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Ранжований та логічний пошук findex")
    parser.add_argument("index_path", type=str, help="Шлях до файлу індексу")
    parser.add_argument("query", type=str, help="Пошуковий запит")
    parser.add_argument(
        "--scorer",
        choices=["bm25", "tfidf"],
        default="bm25",
        help="Алгоритм ранжування",
    )
    parser.add_argument("--limit", type=int, default=10, help="Кількість результатів")
    parser.add_argument(
        "--boolean",
        action="store_true",
        help="Виконати чистий логічний пошук без оцінок",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    t0 = time.perf_counter()
    index = load_index(args.index_path)
    load_time = (time.perf_counter() - t0) * 1000

    print(f"\nЗавантажено індекс: {index!r} ({load_time:.2f} мс)")
    print(f"Пошуковий запит: '{args.query}'\n")

    if args.boolean:
        postings = execute_query(args.query, index)
        print(f"Знайдено документів (Boolean): {len(postings):,}")
        for rank, p in enumerate(postings[: args.limit], start=1):
            meta = index.doc_meta.get(p.doc_id)
            title = meta.title if meta else f"Doc #{p.doc_id}"
            print(
                f"  {rank:2d}. [doc_id={p.doc_id}] {title} (частота: {p.term_frequency})"
            )
    else:
        scorer = BM25Scorer() if args.scorer == "bm25" else TfIdfScorer()
        results = search_ranked(args.query, index, scorer, top_k=args.limit)

        print(
            f"Знайдено документів (Ранжований {args.scorer.upper()}): {len(results):,}"
        )
        for rank, r in enumerate(results, start=1):
            print(f"  {rank:2d}. [бал: {r.score:.4f}] {r.title} (doc_id={r.doc_id})")
            if r.snippet:
                print(f"      Сніпет: {r.snippet}")


if __name__ == "__main__":
    main()
