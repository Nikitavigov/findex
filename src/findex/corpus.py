import gzip
import itertools
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _read_file_text(file_path: Path) -> str:
    """Read file content with UTF-8 encoding, handling compression if necessary."""
    if file_path.suffix == ".gz":
        with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
            return f.read()
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _iter_jsonl(file_path: Path) -> Iterator[Document]:
    """Stream documents from a JSON-lines (.jsonl or .jsonl.gz) file."""
    def _parse_lines(stream) -> Iterator[Document]:
        for idx, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            doc_id = str(data.get("id", data.get("doc_id", idx)))
            text = data.get("text", data.get("content", data.get("body", "")))
            meta = {
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
    """Stream text files from directory recursively without loading everything into memory."""
    for path in dir_path.rglob("*"):
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
    Lazy generator that yields Document instances one by one.
    Does not load the corpus into memory.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Corpus path not found: {path}")

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
