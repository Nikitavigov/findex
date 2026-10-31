import argparse
import gzip
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

from findex.corpus import iter_documents
from findex.models import DocMeta, Posting
from findex.tokenizer import tokenize


class InvertedIndex:
    """
    Інвертований індекс для повнотекстового пошуку.
    Зберігає словник термін -> впорядкований список постінгів,
    а також метадані документів.
    """

    def __init__(
        self,
        postings: dict[str, list[Posting]] | None = None,
        doc_meta: dict[int, DocMeta] | None = None,
        total_docs: int = 0,
        total_tokens: int = 0,
    ) -> None:
        self.postings: dict[str, list[Posting]] = (
            postings if postings is not None else {}
        )
        self.doc_meta: dict[int, DocMeta] = doc_meta if doc_meta is not None else {}
        self.total_docs: int = total_docs
        self.total_tokens: int = total_tokens

    def get_postings(self, term: str) -> list[Posting]:
        """Повертає список постінгів для заданого терміна (або порожній список)."""
        return self.postings.get(term.casefold(), [])

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)


def build_index(source: str | Path, limit: int | None = None) -> InvertedIndex:
    """
    Будує інвертований індекс за один прохід через лінивий конвеєр документів.
    Оскільки doc_id зростає послідовно (0, 1, 2...), кожен список постінгів
    виходить відсортованим за doc_id без потреби у додатковому сортуванні.
    """
    postings: dict[str, list[Posting]] = defaultdict(list)
    doc_meta: dict[int, DocMeta] = {}
    total_tokens = 0

    for doc_id, doc in enumerate(iter_documents(source, limit=limit)):
        # Рахуємо частоти термінів у поточному документі
        term_counts: Counter[str] = Counter()
        doc_len = 0
        for token in tokenize(doc.text):
            term_counts[token] += 1
            doc_len += 1

        total_tokens += doc_len

        # Записуємо метадані документа
        title = (
            doc.text.split("\n", 1)[0].lstrip("#").strip() if doc.text else doc.doc_id
        )
        doc_meta[doc_id] = DocMeta(
            doc_id=doc_id,
            source_name=doc.doc_id,
            title=title[:100],
            length=doc_len,
        )

        # Додаємо постінги у відповідні списки
        for term, freq in term_counts.items():
            postings[term].append(Posting(doc_id=doc_id, term_frequency=freq))

    return InvertedIndex(
        postings=dict(postings),
        doc_meta=doc_meta,
        total_docs=len(doc_meta),
        total_tokens=total_tokens,
    )


def save_index(index: InvertedIndex, path: str | Path, fmt: str = "pickle") -> None:
    """
    Зберігає індекс на диск у форматі pickle або json.

    ПОПЕРЕДЖЕННЯ З БЕЗПЕКИ (pickle):
    Модуль pickle не захищений від шкідливих даних. Виклик pickle.load на файлі
    з ненадійного джерела дозволяє виконати довільний код через механізм __reduce__.
    Для публічного обміну файлами використовуйте формат JSON.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "pickle":
        data = {
            "postings": index.postings,
            "doc_meta": index.doc_meta,
            "total_docs": index.total_docs,
            "total_tokens": index.total_tokens,
        }
        with open(out_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif fmt in {"json", "json.gz"}:
        data = {
            "total_docs": index.total_docs,
            "total_tokens": index.total_tokens,
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
        if fmt == "json.gz" or out_path.suffix == ".gz":
            with gzip.open(out_path, "wt", encoding="utf-8") as f:
                json.dump(data, f)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
    else:
        raise ValueError(f"Невідомий формат індексу: {fmt}")


def load_index(path: str | Path, fmt: str | None = None) -> InvertedIndex:
    """
    Завантажує індекс з диска.
    Автоматично визначає формат за розширенням файлу, якщо fmt не вказано явно.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Файл індексу не знайдено: {in_path}")

    if fmt is None:
        if in_path.suffix in {".json", ".gz"} or in_path.name.endswith(".json.gz"):
            fmt = "json"
        else:
            fmt = "pickle"

    if fmt == "pickle":
        with open(in_path, "rb") as f:
            data = pickle.load(f)
        return InvertedIndex(
            postings=data["postings"],
            doc_meta=data["doc_meta"],
            total_docs=data["total_docs"],
            total_tokens=data["total_tokens"],
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
        return InvertedIndex(
            postings=postings,
            doc_meta=doc_meta,
            total_docs=data["total_docs"],
            total_tokens=data["total_tokens"],
        )
    else:
        raise ValueError(f"Невідомий формат індексу: {fmt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Побудова та збереження інвертованого індексу findex"
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
        help="Формат збереження",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Обмеження на кількість документів"
    )

    args = parser.parse_args()

    print(f"Побудова індексу для: {args.source}...")
    start_time = time.perf_counter()
    index = build_index(args.source, limit=args.limit)
    build_time = time.perf_counter() - start_time

    print(f"Індекс побудовано за {build_time:.2f} с.")
    print(f"  Документів: {index.total_docs:,}")
    print(f"  Токенів:    {index.total_tokens:,}")
    print(f"  Словник:    {index.vocabulary_size:,} унікальних слів")

    save_start = time.perf_counter()
    save_index(index, args.out, fmt=args.format)
    save_time = time.perf_counter() - save_start
    file_size_mb = Path(args.out).stat().st_size / (1024 * 1024)

    print(
        f"Збережено у {args.out} ({args.format}) за {save_time:.2f} с. Розмір: {file_size_mb:.2f} MB"
    )


if __name__ == "__main__":
    main()
