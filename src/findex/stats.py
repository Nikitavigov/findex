import argparse
import sys
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenizer import tokenize


@dataclass
class StatsResult:
    mode: str
    doc_count: int
    token_count: int
    vocab_size: int
    elapsed_sec: float
    peak_memory_mb: float
    top_50: list[tuple[str, int]]


def run_lazy(source: str | Path, limit: int | None = None) -> StatsResult:
    """Streams corpus and tokens using lazy generators (constant memory for stream)."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    start = time.perf_counter()

    doc_count = 0
    token_count = 0
    vocab = Counter()

    for doc in iter_documents(source, limit=limit):
        doc_count += 1
        for token in tokenize(doc.text):
            token_count += 1
            vocab[token] += 1

    elapsed = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return StatsResult(
        mode="lazy (generators)",
        doc_count=doc_count,
        token_count=token_count,
        vocab_size=len(vocab),
        elapsed_sec=elapsed,
        peak_memory_mb=peak_bytes / (1024 * 1024),
        top_50=vocab.most_common(50),
    )


def run_eager(source: str | Path, limit: int | None = None) -> StatsResult:
    """Loads all documents and all tokens into memory lists at once (greedy)."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    start = time.perf_counter()

    # Load all documents into memory
    documents = list(iter_documents(source, limit=limit))

    # Load all tokens for all documents into lists
    all_tokens: list[str] = []
    for doc in documents:
        all_tokens.extend(list(tokenize(doc.text)))

    vocab = Counter(all_tokens)

    elapsed = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return StatsResult(
        mode="eager (lists)",
        doc_count=len(documents),
        token_count=len(all_tokens),
        vocab_size=len(vocab),
        elapsed_sec=elapsed,
        peak_memory_mb=peak_bytes / (1024 * 1024),
        top_50=vocab.most_common(50),
    )


def print_stats(res: StatsResult) -> None:
    print(f"\n[{res.mode.upper()}] Statistics:")
    print(f"  Documents:     {res.doc_count:,}")
    print(f"  Total tokens:  {res.token_count:,}")
    print(f"  Vocabulary:    {res.vocab_size:,} unique terms")
    print(f"  Elapsed time:  {res.elapsed_sec:.4f} s")
    print(f"  Peak memory:   {res.peak_memory_mb:.2f} MB")
    print("\nTop 50 terms:")
    for rank, (term, count) in enumerate(res.top_50, start=1):
        print(f"  {rank:2d}. {term:<20} {count:,}")


def print_comparison(lazy_res: StatsResult, eager_res: StatsResult) -> None:
    print("\n" + "=" * 60)
    print("EAGER vs LAZY COMPARISON")
    print("=" * 60)
    print(
        "| Metric / Mode            | Eager (lists)         | Lazy (generators)     |"
    )
    print(
        "|--------------------------|-----------------------|-----------------------|"
    )
    print(
        f"| Documents processed      | {eager_res.doc_count:<21,}| {lazy_res.doc_count:<21,}|"
    )
    print(
        f"| Total tokens             | {eager_res.token_count:<21,}| {lazy_res.token_count:<21,}|"
    )
    print(
        f"| Vocabulary size          | {eager_res.vocab_size:<21,}| {lazy_res.vocab_size:<21,}|"
    )
    print(
        f"| Elapsed time (s)         | {eager_res.elapsed_sec:<21.4f}| {lazy_res.elapsed_sec:<21.4f}|"
    )
    print(
        f"| Peak memory (MB)         | {eager_res.peak_memory_mb:<21.2f}| {lazy_res.peak_memory_mb:<21.2f}|"
    )
    savings = (
        (
            (eager_res.peak_memory_mb - lazy_res.peak_memory_mb)
            / eager_res.peak_memory_mb
        )
        * 100
        if eager_res.peak_memory_mb
        else 0
    )
    print(f"\nMemory saved with generators: {savings:.1f}%\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="findex corpus statistics and memory benchmark"
    )
    parser.add_argument("source", type=str, help="Path to corpus directory or file")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max documents to process"
    )
    parser.add_argument(
        "--eager", action="store_true", help="Run eager in-memory list version"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Run both lazy and eager and compare"
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: path '{args.source}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.compare:
        print("Running lazy stream benchmark...")
        lazy_res = run_lazy(source_path, limit=args.limit)
        print("Running eager in-memory benchmark...")
        eager_res = run_eager(source_path, limit=args.limit)
        print_stats(lazy_res)
        print_comparison(lazy_res, eager_res)
    elif args.eager:
        res = run_eager(source_path, limit=args.limit)
        print_stats(res)
    else:
        res = run_lazy(source_path, limit=args.limit)
        print_stats(res)


if __name__ == "__main__":
    main()
