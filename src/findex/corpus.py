import gzip
import itertools
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any


def _empty_metadata() -> dict[str, Any]:
    return {}


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


def _read_file_text(file_path: Path) -> str:
    """Читати вміст файлу у кодуванні UTF-8, враховуючи можливе стиснення gzip."""
    if file_path.suffix == ".gz":
        with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
            return f.read()
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _iter_jsonl(file_path: Path) -> Iterator[Document]:
    """Потокове читання документів з файлу JSON-lines (.jsonl або .jsonl.gz)."""

    def _parse_lines(stream: IO[str]) -> Iterator[Document]:
        for idx, line in enumerate(stream, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            data: dict[str, Any] = json.loads(line_str)
            doc_id = str(data.get("id", data.get("doc_id", idx)))
            text = str(data.get("text", data.get("content", data.get("body", ""))))
            meta: dict[str, Any] = {
                k: v
                for k, v in data.items()
                if k not in {"id", "doc_id", "text", "content", "body"}
            }
            yield Document(doc_id=doc_id, text=text, metadata=meta)

    if file_path.suffix == ".gz":
        with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
            yield from _parse_lines(f)
    else:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            yield from _parse_lines(f)


def _iter_directory(dir_path: Path) -> Iterator[Document]:
    """Рекурсивне потокове читання текстових файлів з каталогу."""
    for path in sorted(dir_path.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue

        if path.name.endswith((".jsonl", ".jsonl.gz")):
            yield from _iter_jsonl(path)
        elif path.suffix in {".txt", ".md", ".csv", ".html", ""}:
            try:
                content = _read_file_text(path)
                yield Document(
                    doc_id=path.name,
                    text=content,
                    metadata={"path": str(path), "size": path.stat().st_size},
                )
            except (OSError, UnicodeError):
                continue


def iter_documents(source: str | Path, limit: int | None = None) -> Iterator[Document]:
    """
    Лінивий генератор, що повертає екземпляри Document по одному.
    Не завантажує весь корпус в оперативну пам'ять.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Шлях до корпусу не знайдено: {path}")

    def _generator() -> Iterator[Document]:
        if path.is_file():
            if path.name.endswith((".jsonl", ".jsonl.gz")):
                yield from _iter_jsonl(path)
            else:
                yield Document(
                    doc_id=path.name,
                    text=_read_file_text(path),
                    metadata={"path": str(path), "size": path.stat().st_size},
                )
        elif path.is_dir():
            yield from _iter_directory(path)

    gen = _generator()
    if limit is not None:
        return itertools.islice(gen, limit)
    return gen
