import concurrent.futures
import gzip
import json
import logging
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal

from findex.corpus import read_file_text
from findex.index import Index
from findex.models import DocMeta, Posting
from findex.tokenizer import tokenize

logger = logging.getLogger("findex.parallel")


@dataclass(slots=True)
class PartialIndex:
    """
    Частковий індекс, побудований окремим потоком або процесом
    для виділеного діапазону документів.
    """

    postings: dict[str, list[Posting]]
    doc_meta: dict[int, DocMeta]


def _collect_corpus_files(source: Path) -> list[Path]:
    """Збирає всі придатні файли корпусу в детермінованому сортованому порядку."""
    if source.is_file():
        return [source]
    return sorted(
        [
            p
            for p in source.rglob("*")
            if p.is_file() and not p.name.startswith(".")
        ]
    )


def split_chunks[T](items: list[T], n_chunks: int) -> list[list[T]]:
    """
    Розподіляє список елементів на n_chunks рівномірних частин.
    Використовує divmod для збалансованого розміру часток.
    """
    if n_chunks <= 1 or not items:
        return [items] if items else []
    k, m = divmod(len(items), n_chunks)
    chunks: list[list[T]] = []
    start = 0
    for i in range(n_chunks):
        end = start + k + (1 if i < m else 0)
        chunk = items[start:end]
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _index_single_document(
    text: str,
    source_name: str,
    doc_id: int,
    postings: dict[str, list[Posting]],
    doc_meta: dict[int, DocMeta],
) -> None:
    """Індексує текст одного документа та оновлює структури даних."""
    term_counts: Counter[str] = Counter()
    doc_len = 0
    for token in tokenize(text):
        term_counts[token] += 1
        doc_len += 1

    title = text.split("\n", 1)[0].lstrip("#").strip() if text else source_name
    doc_meta[doc_id] = DocMeta(
        doc_id=doc_id,
        source_name=source_name,
        title=title[:100],
        length=doc_len,
    )

    for term, freq in term_counts.items():
        postings[term].append(Posting(doc_id=doc_id, term_frequency=freq))


def build_partial(
    doc_paths: list[Path],
    start_doc_id: int = 0,
) -> PartialIndex:
    """
    Будує частковий індекс для заданого списку файлів.
    Призначає кожному документу унікальний глобальний doc_id, починаючи з start_doc_id.
    Функція визначена на рівні модуля для коректної серіалізації через pickle у процеси.
    """
    postings: dict[str, list[Posting]] = defaultdict(list)
    doc_meta: dict[int, DocMeta] = {}
    current_doc_id = start_doc_id

    for file_path in doc_paths:
        if file_path.name.endswith((".jsonl", ".jsonl.gz")):
            opener = gzip.open if file_path.suffix == ".gz" else open
            with opener(file_path, "rt", encoding="utf-8", errors="replace") as stream:
                stream_typed: IO[str] = stream
                for line in stream_typed:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        data: dict[str, Any] = json.loads(line_str)
                        text = str(
                            data.get(
                                "text", data.get("content", data.get("body", ""))
                            )
                        )
                        src_name = str(data.get("id", data.get("doc_id", file_path.name)))
                        _index_single_document(
                            text, src_name, current_doc_id, postings, doc_meta
                        )
                        current_doc_id += 1
                    except (json.JSONDecodeError, KeyError, ValueError) as exc:
                        logger.debug("Пропуск некоректного рядка JSONL: %s", exc)
                        continue
        else:
            try:
                content = read_file_text(file_path)
            except (OSError, UnicodeError):
                content = ""
            _index_single_document(
                content, file_path.name, current_doc_id, postings, doc_meta
            )
            current_doc_id += 1

    return PartialIndex(postings=dict(postings), doc_meta=doc_meta)


def merge_partials(
    partials: Sequence[PartialIndex],
    corpus_dir: Path | None = None,
) -> Index:
    """
    Об'єднує впорядкований список часткових індексів в єдиний монолітний Index.
    Оскільки діапазони doc_id воркерів суворо впорядковані і не перетинаються,
    постінги кожного терміна зливаються простою конкатенацією без додаткового сортування.
    """
    merged_postings: dict[str, list[Posting]] = defaultdict(list)
    merged_doc_meta: dict[int, DocMeta] = {}

    for partial in partials:
        merged_doc_meta.update(partial.doc_meta)
        for term, plist in partial.postings.items():
            merged_postings[term].extend(plist)

    return Index(
        postings=dict(merged_postings),
        doc_meta=merged_doc_meta,
        corpus_dir=corpus_dir,
    )


def build_index_parallel(
    source: str | Path,
    workers: int = 1,
    executor: Literal["serial", "threads", "processes"] = "processes",
    limit: int | None = None,
) -> tuple[Index, float]:
    """
    Паралельна побудова індексу з підтримкою трьох режимів виконання:
    - serial: звичайний послідовний цикл
    - threads: пулл потоків ThreadPoolExecutor
    - processes: пулл процесів ProcessPoolExecutor

    Повертає кортеж (Index, merge_time_seconds).
    У разі виникнення помилки у воркері, збірка негайно переривається з оригінальним traceback.
    """
    src_path = Path(source)
    if executor not in {"serial", "threads", "processes"}:
        raise ValueError(f"Невідомий тип екзекутора: {executor}")

    all_files = _collect_corpus_files(src_path)
    if limit is not None:
        all_files = all_files[:limit]

    effective_workers = max(1, workers)
    chunks = split_chunks(all_files, effective_workers)
    if not chunks:
        return Index(corpus_dir=src_path), 0.0

    # Розраховуємо точний стартовий doc_id для кожного чанка
    tasks: list[tuple[list[Path], int]] = []
    curr_id = 0
    for chunk in chunks:
        tasks.append((chunk, curr_id))
        curr_id += len(chunk)

    partials: list[PartialIndex] = [
        PartialIndex(postings={}, doc_meta={}) for _ in range(len(tasks))
    ]

    if executor == "serial" or effective_workers == 1 or len(tasks) == 1:
        for idx, (chunk_paths, start_id) in enumerate(tasks):
            partials[idx] = build_partial(chunk_paths, start_id)
    elif executor == "threads":
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=effective_workers
        ) as pool:
            future_to_idx = {
                pool.submit(build_partial, chunk_paths, start_id): idx
                for idx, (chunk_paths, start_id) in enumerate(tasks)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                partials[idx] = future.result()
    elif executor == "processes":
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=effective_workers
        ) as pool:
            future_to_idx = {
                pool.submit(build_partial, chunk_paths, start_id): idx
                for idx, (chunk_paths, start_id) in enumerate(tasks)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                partials[idx] = future.result()
    else:
        raise ValueError(f"Невідомий тип екзекутора: {executor}")

    # Замір часу фази редукції (Merge)
    t_merge_start = time.perf_counter()
    index = merge_partials(partials, corpus_dir=src_path)
    merge_time = time.perf_counter() - t_merge_start

    return index, merge_time
