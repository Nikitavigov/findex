import argparse
import gzip
import json
import pickle
import time
from collections import Counter, defaultdict
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

from findex.corpus import iter_documents
from findex.models import DocMeta, Posting
from findex.tokenizer import normalize_text, tokenize


class Index(Mapping[str, list[Posting]]):
    """
    Пошуковий індекс, що реалізує протокол collections.abc.Mapping.
    Підтримує len(index), term in index, index[term], ітерацію та repr.
    """

    def __init__(
        self,
        postings: dict[str, list[Posting]] | None = None,
        doc_meta: dict[int, DocMeta] | None = None,
        corpus_dir: str | Path | None = None,
        total_docs: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        self._postings: dict[str, list[Posting]] = (
            postings if postings is not None else {}
        )
        self.doc_meta: dict[int, DocMeta] = doc_meta if doc_meta is not None else {}
        self.corpus_dir: Path | None = Path(corpus_dir) if corpus_dir else None

    @property
    def total_docs(self) -> int:
        """Псевдонім для num_docs для сумісності з попередніми лабами."""
        return len(self.doc_meta)

    @property
    def postings(self) -> dict[str, list[Posting]]:
        """Словник термів та відповідних списків постінгів."""
        return self._postings

    @property
    def total_tokens(self) -> int:
        """Сумарна кількість токенів у всіх документах."""
        return sum(meta.length for meta in self.doc_meta.values())

    # --- Реалізація методів Mapping ---

    def __getitem__(self, term: str) -> list[Posting]:
        norm = normalize_text(term)
        if norm not in self._postings:
            raise KeyError(term)
        return self._postings[norm]

    def __len__(self) -> int:
        return len(self._postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __contains__(self, term: object) -> bool:
        if not isinstance(term, str):
            return False
        return normalize_text(term) in self._postings

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"

    # --- Властивості індексу ---

    @property
    def num_docs(self) -> int:
        """Загальна кількість проіндексованих документів."""
        return len(self.doc_meta)

    @cached_property
    def avg_doc_length(self) -> float:
        """
        Середня довжина документа в токенах.
        Обчислюється один раз ліниво та кешується у властивості.
        """
        if not self.doc_meta:
            return 0.0
        total_len = sum(meta.length for meta in self.doc_meta.values())
        return total_len / len(self.doc_meta)

    def df(self, term: str) -> int:
        """Частота документа (Document Frequency) — у скількох документах зустрічається термін."""
        norm = normalize_text(term)
        return len(self._postings.get(norm, []))

    def get_postings(self, term: str) -> list[Posting]:
        """Повертає постінги або порожній список без викидання KeyError."""
        norm = normalize_text(term)
        return self._postings.get(norm, [])

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)


# Зворотна сумісність
InvertedIndex = Index


def build_index(source: str | Path, limit: int | None = None) -> Index:
    """
    Будує індекс за один прохід по документах.
    Постінги природно впорядковані за зростанням doc_id.
    """
    src_path = Path(source)
    postings: dict[str, list[Posting]] = defaultdict(list)
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(iter_documents(src_path, limit=limit)):
        term_counts: Counter[str] = Counter()
        doc_len = 0
        for token in tokenize(doc.text):
            term_counts[token] += 1
            doc_len += 1

        title = (
            doc.text.split("\n", 1)[0].lstrip("#").strip() if doc.text else doc.doc_id
        )
        doc_meta[doc_id] = DocMeta(
            doc_id=doc_id,
            source_name=doc.doc_id,
            title=title[:100],
            length=doc_len,
        )

        for term, freq in term_counts.items():
            postings[term].append(Posting(doc_id=doc_id, term_frequency=freq))

    return Index(
        postings=dict(postings),
        doc_meta=doc_meta,
        corpus_dir=src_path if src_path.is_dir() else src_path.parent,
    )


def save_index(index: Index, path: str | Path, fmt: str = "pickle") -> None:
    """
    Зберігає індекс на диск.

    ПОПЕРЕДЖЕННЯ З БЕЗПЕКИ: pickle дозволяє виконання довільного коду (RCE).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "pickle":
        data = {
            "postings": index.postings,
            "doc_meta": index.doc_meta,
            "corpus_dir": str(index.corpus_dir) if index.corpus_dir else None,
        }
        with open(out_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif fmt in {"json", "json.gz"}:
        data = {
            "corpus_dir": str(index.corpus_dir) if index.corpus_dir else None,
            "doc_meta": {
                str(k): {
                    "doc_id": v.doc_id,
                    "source_name": v.source_name,
                    "title": v.title,
                    "length": v.length,
                }
                for k, v in index.doc_meta.items()
            },
            "postings": {
                term: [[p.doc_id, p.term_frequency] for p in plist]
                for term, plist in index.postings.items()
            },
        }
        opener = gzip.open if fmt == "json.gz" or out_path.suffix == ".gz" else open
        with opener(out_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)
    else:
        raise ValueError(f"Невідомий формат індексу: {fmt}")


def load_index(path: str | Path, fmt: str | None = None) -> Index:
    """Завантажує індекс з диска."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Файл індексу не знайдено: {in_path}")

    if fmt is None:
        fmt = (
            "json"
            if in_path.suffix in {".json", ".gz"} or in_path.name.endswith(".json.gz")
            else "pickle"
        )

    if fmt == "pickle":
        with open(in_path, "rb") as f:
            data = pickle.load(f)
        return Index(
            postings=data["postings"],
            doc_meta=data["doc_meta"],
            corpus_dir=data.get("corpus_dir"),
        )
    elif fmt in {"json", "json.gz"}:
        opener = (
            gzip.open
            if in_path.suffix == ".gz" or in_path.name.endswith(".json.gz")
            else open
        )
        with opener(in_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        doc_meta = {
            int(k): DocMeta(
                doc_id=v["doc_id"],
                source_name=v["source_name"],
                title=v["title"],
                length=v["length"],
            )
            for k, v in data["doc_meta"].items()
        }
        postings = {
            term: [Posting(doc_id=p[0], term_frequency=p[1]) for p in plist]
            for term, plist in data["postings"].items()
        }
        return Index(
            postings=postings,
            doc_meta=doc_meta,
            corpus_dir=data.get("corpus_dir"),
        )
    else:
        raise ValueError(f"Невідомий формат індексу: {fmt}")


@contextmanager
def open_index(path: str | Path, fmt: str | None = None) -> Generator[Index, None, None]:
    """
    Контекстний менеджер для безпечного завантаження та роботи з індексом.
    Гарантує завершення ресурсів навіть при виникненні винятків.
    """
    index = load_index(path, fmt=fmt)
    try:
        yield index
    finally:
        # Очищення або скидання кешів за потреби
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Побудова та збереження індексу findex"
    )
    parser.add_argument("source", type=str, help="Шлях до каталогу або файлу корпусу")
    parser.add_argument(
        "--out", type=str, default="index.bin", help="Шлях вихідного файлу індексу"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pickle", "json"],
        default="pickle",
        help="Формат",
    )
    parser.add_argument("--limit", type=int, default=None, help="Обмеження документів")

    args = parser.parse_args()

    print(f"Побудова індексу для: {args.source}...")
    t0 = time.perf_counter()
    index = build_index(args.source, limit=args.limit)
    print(f"Індекс побудовано: {index!r} за {time.perf_counter() - t0:.2f} с.")

    t0 = time.perf_counter()
    save_index(index, args.out, fmt=args.format)
    print(f"Збережено у {args.out} за {time.perf_counter() - t0:.2f} с.")


if __name__ == "__main__":
    main()
