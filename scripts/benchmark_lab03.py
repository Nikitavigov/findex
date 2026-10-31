import logging
import sys
import time
from pathlib import Path

from findex.index import build_index, open_index, save_index
from findex.ranking import BM25Scorer, TfIdfScorer, rank_documents
from findex.search import execute_query
from findex.tokenizer import tokenize


def run_sanity_checks(index) -> None:
    print("\n" + "=" * 60)
    print("1. ТРИ САНІТАРНІ ПЕРЕВІРКИ РАНЖУВАННЯ (SANITY CHECKS)")
    print("=" * 60)

    bm25 = BM25Scorer()
    tfidf = TfIdfScorer()

    # Перевірка 1: Рідкісний термін проти частого
    rare_term = "frankenstein"
    frequent_term = "the"
    df_rare = index.df(rare_term)
    df_freq = index.df(frequent_term)

    bm25_rare = bm25.score(
        rare_term, tf=1, doc_len=int(index.avg_doc_length), index=index
    )
    bm25_freq = bm25.score(
        frequent_term, tf=1, doc_len=int(index.avg_doc_length), index=index
    )
    tfidf_rare = tfidf.score(
        rare_term, tf=1, doc_len=int(index.avg_doc_length), index=index
    )
    tfidf_freq = tfidf.score(
        frequent_term, tf=1, doc_len=int(index.avg_doc_length), index=index
    )

    print("\n[Sanity Check 1: Рідкісний термін вище частого]")
    print(
        f"  Рідкісний '{rare_term}' (df={df_rare}):  BM25={bm25_rare:.4f}, TF-IDF={tfidf_rare:.4f}"
    )
    print(
        f"  Частий    '{frequent_term}' (df={df_freq}):    BM25={bm25_freq:.4f}, TF-IDF={tfidf_freq:.4f}"
    )
    print(
        f"  -> Рідкісний термін оцінено в {bm25_rare / bm25_freq:.1f}x вище за частий у BM25."
    )

    # Перевірка 2: Насичення частоти терміна (Saturation)
    score_1 = bm25.score(
        "captain", tf=1, doc_len=int(index.avg_doc_length), index=index
    )
    score_2 = bm25.score(
        "captain", tf=2, doc_len=int(index.avg_doc_length), index=index
    )
    score_19 = bm25.score(
        "captain", tf=19, doc_len=int(index.avg_doc_length), index=index
    )
    score_20 = bm25.score(
        "captain", tf=20, doc_len=int(index.avg_doc_length), index=index
    )

    gain_1 = score_1
    gain_2 = score_2 - score_1
    gain_20 = score_20 - score_19

    print("\n[Sanity Check 2: Насичення частоти терміна (BM25 Saturation)]")
    print(f"  Приріст від 1-го входження (tf=1):     +{gain_1:.4f}")
    print(f"  Приріст від 2-го входження (tf=2):     +{gain_2:.4f}")
    print(f"  Приріст від 20-го входження (tf=20):   +{gain_20:.4f}")
    print(
        f"  -> 20-те входження додає лише {(gain_20 / gain_1) * 100:.2f}% від першого."
    )

    # Перевірка 3: Нормалізація довжини документа
    short_len = 100
    long_len = 3000
    score_short = bm25.score("sea", tf=1, doc_len=short_len, index=index)
    score_long = bm25.score("sea", tf=1, doc_len=long_len, index=index)

    print("\n[Sanity Check 3: Нормалізація довжини документа]")
    print(f"  Короткий документ (len={short_len}):  BM25={score_short:.4f}")
    print(f"  Довгий документ   (len={long_len}): BM25={score_long:.4f}")
    print(
        f"  -> Короткий документ отримав на {((score_short - score_long) / score_long) * 100:.1f}% вищий бал."
    )


def run_precision_at_5(index) -> None:
    print("\n" + "=" * 60)
    print("2. ОЦІНКА ЯКОСТІ: ТАБЛИЦЯ PRECISION@5 (TF-IDF vs BM25)")
    print("=" * 60)

    # 10 тестових запитів з очікуваною семантикою (релевантними ключовими словами у назвах чи тексті)
    queries = [
        ("frankenstein monster", {"frankenstein"}),
        ("captain sea ship", {"moby_dick"}),
        ("whale ocean ahab", {"moby_dick"}),
        ("london paris revolution", {"two_cities"}),
        ("king queen prince", {"fairy"}),
        ("victor geneva science", {"frankenstein"}),
        ("pequod harpoon starbuck", {"moby_dick"}),
        ("guillotine prison doctor", {"two_cities"}),
        ("wolf forest grandmother", {"fairy"}),
        ("white whale leviathan", {"moby_dick"}),
    ]

    bm25 = BM25Scorer()
    tfidf = TfIdfScorer()

    results_table = []
    total_p5_bm25 = 0.0
    total_p5_tfidf = 0.0

    for query, relevant_markers in queries:
        terms = list(tokenize(query))

        res_bm25 = rank_documents(terms, index, bm25, top_k=5)
        res_tfidf = rank_documents(terms, index, tfidf, top_k=5)

        def calc_p5(results, markers=relevant_markers):
            if not results:
                return 0.0
            hits = 0
            for r in results[:5]:
                meta = index.doc_meta.get(r.doc_id)
                text_to_check = (
                    (meta.source_name + " " + meta.title).lower() if meta else ""
                )
                if any(marker in text_to_check for marker in markers):
                    hits += 1
            return hits / min(len(results), 5)

        p5_b = calc_p5(res_bm25)
        p5_t = calc_p5(res_tfidf)

        total_p5_bm25 += p5_b
        total_p5_tfidf += p5_t

        results_table.append((query, p5_t, p5_b))

    print("| №  | Запит                       | P@5 (TF-IDF) | P@5 (BM25) |")
    print("|:---|:----------------------------|:-------------|:-----------|")
    for idx, (q, pt, pb) in enumerate(results_table, start=1):
        print(f"| {idx:<2} | {q:<28}| {pt:<13.2f}| {pb:<11.2f}|")

    avg_t = total_p5_tfidf / len(queries)
    avg_b = total_p5_bm25 / len(queries)
    print(
        f"|    | **Середнє (Mean P@5)**      | **{avg_t:.2f}**        | **{avg_b:.2f}**      |"
    )


def demonstrate_caching_and_timing(index) -> None:
    print("\n" + "=" * 60)
    print("3. ДЕМОНСТРАЦІЯ @timed ТА @lru_cache")
    print("=" * 60)

    query = "captain AND sea"

    # Перший запит (Cache MISS)
    t0 = time.perf_counter()
    res1 = execute_query(query, index)
    time_first_ms = (time.perf_counter() - t0) * 1000

    # Повторний запит (Cache HIT)
    t0 = time.perf_counter()
    res2 = execute_query(query, index)
    time_cached_ms = (time.perf_counter() - t0) * 1000

    print(f"Запит: '{query}'")
    print(
        f"  Перший запуск (Cache MISS):  {time_first_ms:.4f} мс (знайдено {len(res1)} docs)"
    )
    print(
        f"  Повторний запуск (Cache HIT): {time_cached_ms:.4f} мс (знайдено {len(res2)} docs)"
    )
    if time_cached_ms > 0:
        print(f"  Прискорення кешування: {time_first_ms / time_cached_ms:.1f}x")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    corpus = Path("data/corpus")
    if not corpus.exists():
        print("Корпус data/corpus не знайдено!", file=sys.stderr)
        sys.exit(1)

    index_file = Path("data/index.bin")
    if not index_file.exists():
        print("Побудова індексу...")
        idx = build_index(corpus)
        save_index(idx, index_file)

    with open_index(index_file) as index:
        run_sanity_checks(index)
        run_precision_at_5(index)
        demonstrate_caching_and_timing(index)
