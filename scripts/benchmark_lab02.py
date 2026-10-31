import gc
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import build_index, load_index, save_index
from findex.models import ArrayPostings, PlainPosting, Posting
from findex.search import execute_query
from findex.tokenizer import tokenize


def benchmark_postings_memory(corpus_path: Path) -> None:
    """Вимірює пікову пам'ять для 3 варіантів збереження постінгів під tracemalloc."""
    print("\n--- 1. БЕНЧМАРК ПАМ'ЯТІ ВАРІАНТІВ ПОСТІНГІВ ---")

    # Варіант 1: Звичайний dataclass (без slots)
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    plain_postings: dict[str, list[PlainPosting]] = defaultdict(list)
    for doc_id, doc in enumerate(iter_documents(corpus_path)):
        for token in tokenize(doc.text):
            plain_postings[token].append(PlainPosting(doc_id, 1))
    _, peak_plain = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del plain_postings
    gc.collect()

    # Варіант 2: Slotted frozen dataclass (наша реалізація)
    tracemalloc.start()
    tracemalloc.reset_peak()
    slotted_postings: dict[str, list[Posting]] = defaultdict(list)
    for doc_id, doc in enumerate(iter_documents(corpus_path)):
        for token in tokenize(doc.text):
            slotted_postings[token].append(Posting(doc_id, 1))
    _, peak_slotted = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del slotted_postings
    gc.collect()

    # Варіант 3: array('I')
    tracemalloc.start()
    tracemalloc.reset_peak()
    array_postings: dict[str, ArrayPostings] = defaultdict(ArrayPostings)
    for doc_id, doc in enumerate(iter_documents(corpus_path)):
        for token in tokenize(doc.text):
            array_postings[token].append(doc_id, 1)
    _, peak_array = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del array_postings
    gc.collect()

    mb_plain = peak_plain / (1024 * 1024)
    mb_slotted = peak_slotted / (1024 * 1024)
    mb_array = peak_array / (1024 * 1024)

    print("| Представлення постінгів | Пікова пам'ять (MB) | Відносний розмір |")
    print("|:------------------------|:--------------------|:-----------------|")
    print(f"| Звичайний dataclass     | {mb_plain:<20.2f}| 1.00x (базовий)  |")
    print(
        f"| slots=True (frozen)     | {mb_slotted:<20.2f}| {mb_slotted / mb_plain:.2f}x           |"
    )
    print(
        f"| array('I')              | {mb_array:<20.2f}| {mb_array / mb_plain:.2f}x           |"
    )


def benchmark_serialization(corpus_path: Path, tmp_dir: Path) -> None:
    """Вимірює розмір файлу та час збереження/завантаження для pickle та json."""
    print("\n--- 2. БЕНЧМАРК ФОРМАТІВ ЗБЕРЕЖЕННЯ (Pickle vs JSON) ---")
    index = build_index(corpus_path)

    pkl_path = tmp_dir / "index.bin"
    json_path = tmp_dir / "index.json"

    # Pickle save
    t0 = time.perf_counter()
    save_index(index, pkl_path, fmt="pickle")
    pickle_save_time = time.perf_counter() - t0
    pickle_size_mb = pkl_path.stat().st_size / (1024 * 1024)

    # Pickle load
    t0 = time.perf_counter()
    _ = load_index(pkl_path, fmt="pickle")
    pickle_load_time = time.perf_counter() - t0

    # JSON save
    t0 = time.perf_counter()
    save_index(index, json_path, fmt="json")
    json_save_time = time.perf_counter() - t0
    json_size_mb = json_path.stat().st_size / (1024 * 1024)

    # JSON load
    t0 = time.perf_counter()
    _ = load_index(json_path, fmt="json")
    json_load_time = time.perf_counter() - t0

    print("| Формат | Розмір файлу (MB) | Час збереження (с) | Час завантаження (с) |")
    print("|:-------|:------------------|:-------------------|:---------------------|")
    print(
        f"| pickle | {pickle_size_mb:<18.2f}| {pickle_save_time:<19.4f}| {pickle_load_time:<21.4f}|"
    )
    print(
        f"| json   | {json_size_mb:<18.2f}| {json_save_time:<19.4f}| {json_load_time:<21.4f}|"
    )


def benchmark_engines(corpus_path: Path) -> None:
    """Порівнює швидкість виконання запитів рушіями merge та set."""
    print("\n--- 3. БЕНЧМАРК РУШІЇВ ПОШУКУ (Merge vs Set) ---")
    index = build_index(corpus_path)

    queries = [
        ("Часті терми", "the AND and"),
        ("Середні терми", "captain AND sea"),
        ("Рідкісні терми", "frankenstein AND monster"),
    ]

    print(
        "| Тип запиту       | Запит                     | Merge (мс) | Set (мс)   | Прискорення merge |"
    )
    print(
        "|:-----------------|:--------------------------|:-----------|:-----------|:------------------|"
    )

    for q_type, q_str in queries:
        # Merge timing (100 повторів)
        t0 = time.perf_counter()
        for _ in range(100):
            execute_query(q_str, index, engine="merge")
        time_merge = ((time.perf_counter() - t0) / 100) * 1000

        # Set timing (100 повторів)
        t0 = time.perf_counter()
        for _ in range(100):
            execute_query(q_str, index, engine="set")
        time_set = ((time.perf_counter() - t0) / 100) * 1000

        speedup = f"{time_set / time_merge:.2f}x" if time_merge > 0 else "N/A"
        print(
            f"| {q_type:<17}| {q_str:<26}| {time_merge:<11.4f}| {time_set:<11.4f}| {speedup:<18}|"
        )


if __name__ == "__main__":
    corpus = Path("data/corpus")
    if not corpus.exists():
        print(f"Корпус {corpus} не знайдено!", file=sys.stderr)
        sys.exit(1)

    benchmark_postings_memory(corpus)
    benchmark_serialization(corpus, Path("data"))
    benchmark_engines(corpus)
