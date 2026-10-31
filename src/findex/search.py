import argparse
import sys
import time
from collections.abc import Callable

from findex.index import InvertedIndex, load_index
from findex.models import Posting
from findex.tokenizer import normalize_text

# --- Двовказівниковий алгоритм злиття (Two-pointer merge) ---


def merge_and(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """
    Перетин двох списків постінгів методом двох вказівників.
    Складність: O(|p1| + |p2|). Не потребує створення проміжних хеш-таблиць.
    """
    result: list[Posting] = []
    i = 0
    j = 0
    len1 = len(p1)
    len2 = len(p2)

    while i < len1 and j < len2:
        doc1 = p1[i].doc_id
        doc2 = p2[j].doc_id

        if doc1 == doc2:
            # Співпадіння: додаємо сумарну частоту входження
            result.append(Posting(doc1, p1[i].term_frequency + p2[j].term_frequency))
            i += 1
            j += 1
        elif doc1 < doc2:
            i += 1
        else:
            j += 1

    return result


def merge_or(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """
    Об'єднання двох списків постінгів методом двох вказівників.
    Складність: O(|p1| + |p2|). Зберігає відсортованість за doc_id.
    """
    result: list[Posting] = []
    i = 0
    j = 0
    len1 = len(p1)
    len2 = len(p2)

    while i < len1 and j < len2:
        doc1 = p1[i].doc_id
        doc2 = p2[j].doc_id

        if doc1 == doc2:
            result.append(Posting(doc1, p1[i].term_frequency + p2[j].term_frequency))
            i += 1
            j += 1
        elif doc1 < doc2:
            result.append(p1[i])
            i += 1
        else:
            result.append(p2[j])
            j += 1

    while i < len1:
        result.append(p1[i])
        i += 1
    while j < len2:
        result.append(p2[j])
        j += 1

    return result


def merge_not(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """
    Різниця списків постінгів: повертає документи з p1, яких немає у p2.
    Складність: O(|p1| + |p2|).
    """
    result: list[Posting] = []
    i = 0
    j = 0
    len1 = len(p1)
    len2 = len(p2)

    while i < len1 and j < len2:
        doc1 = p1[i].doc_id
        doc2 = p2[j].doc_id

        if doc1 == doc2:
            i += 1
            j += 1
        elif doc1 < doc2:
            result.append(p1[i])
            i += 1
        else:
            j += 1

    while i < len1:
        result.append(p1[i])
        i += 1

    return result


# --- Альтернативний рушій на базі множин (Set-based) ---


def set_and(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Перетин списків постінгів через множини set."""
    if not p1 or not p2:
        return []
    map1 = {p.doc_id: p for p in p1}
    map2 = {p.doc_id: p for p in p2}
    common_ids = sorted(map1.keys() & map2.keys())
    return [
        Posting(did, map1[did].term_frequency + map2[did].term_frequency)
        for did in common_ids
    ]


def set_or(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Об'єднання списків постінгів через множини set."""
    map1 = {p.doc_id: p for p in p1}
    map2 = {p.doc_id: p for p in p2}
    all_ids = sorted(map1.keys() | map2.keys())
    res: list[Posting] = []
    for did in all_ids:
        tf1 = map1[did].term_frequency if did in map1 else 0
        tf2 = map2[did].term_frequency if did in map2 else 0
        res.append(Posting(did, tf1 + tf2))
    return res


def set_not(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Різниця списків постінгів через множини set."""
    if not p1:
        return []
    if not p2:
        return p1
    map1 = {p.doc_id: p for p in p1}
    set2 = {p.doc_id for p in p2}
    diff_ids = sorted(map1.keys() - set2)
    return [map1[did] for did in diff_ids]


# --- Обробка та виконання пошукових запитів ---


def execute_query(
    query: str, index: InvertedIndex, engine: str = "merge"
) -> list[Posting]:
    """
    Виконує Boolean-пошук за запитом з підтримкою операторів AND, OR, NOT.
    За замовчуванням кілька термінів без оператора поєднуються як AND.
    Підтримувані рушії: 'merge' (двовказівниковий) та 'set' (множини).
    """
    tokens = query.strip().split()
    if not tokens:
        return []

    and_func: Callable[[list[Posting], list[Posting]], list[Posting]]
    or_func: Callable[[list[Posting], list[Posting]], list[Posting]]
    not_func: Callable[[list[Posting], list[Posting]], list[Posting]]

    if engine == "merge":
        and_func, or_func, not_func = merge_and, merge_or, merge_not
    elif engine == "set":
        and_func, or_func, not_func = set_and, set_or, set_not
    else:
        raise ValueError(f"Невідомий рушій пошуку: {engine}")

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

        # Звичайне слово запиту
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
    parser = argparse.ArgumentParser(description="Пошук документів в індексі findex")
    parser.add_argument(
        "index_path", type=str, help="Шлях до файлу збереженого індексу"
    )
    parser.add_argument(
        "query", type=str, help="Пошуковий запит (напр. 'python AND indexing')"
    )
    parser.add_argument(
        "--engine", choices=["merge", "set"], default="merge", help="Рушій пошуку"
    )
    parser.add_argument("--limit", type=int, default=10, help="Кількість результатів")

    args = parser.parse_args()

    load_start = time.perf_counter()
    index = load_index(args.index_path)
    load_time = (time.perf_counter() - load_start) * 1000

    search_start = time.perf_counter()
    results = execute_query(args.query, index, engine=args.engine)
    search_time = (time.perf_counter() - search_start) * 1000

    print(f"\nЗапит: '{args.query}' (рушій: {args.engine})")
    print(
        f"Знайдено документів: {len(results):,} | Час завантаження індексу: {load_time:.2f} мс | Час пошуку: {search_time:.3f} мс\n"
    )

    if not results:
        print("Нічого не знайдено.")
        sys.exit(0)

    for rank, posting in enumerate(results[: args.limit], start=1):
        meta = index.doc_meta.get(posting.doc_id)
        title = meta.title if meta else f"Doc #{posting.doc_id}"
        source = meta.source_name if meta else "unknown"
        doc_len = meta.length if meta else 0
        print(f"  {rank:2d}. [doc_id={posting.doc_id}] {title}")
        print(
            f"      Джерело: {source} | Частота в документі: {posting.term_frequency} | Довжина: {doc_len} токенів"
        )


if __name__ == "__main__":
    main()
