import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from findex.crawler import crawl
from findex.index import load_index, save_index
from findex.parallel import build_index_parallel
from findex.ranking import BM25Scorer, TfIdfScorer
from findex.search import search_ranked
from findex.stats import run_eager, run_lazy

app = typer.Typer(
    name="findex",
    help="findex — повнотекстовий пошуковий рушій на Python з типізацією strict та Rich CLI.",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


def setup_logging(verbose: int) -> None:
    """Налаштовує логування в stderr залежно від кількості прапорців -v."""
    if verbose >= 2:
        level = logging.DEBUG
        fmt = "[DEBUG] %(name)s: %(message)s"
    elif verbose == 1:
        level = logging.INFO
        fmt = "[INFO] %(message)s"
    else:
        level = logging.WARNING
        fmt = "%(levelname)s: %(message)s"

    logging.basicConfig(level=level, format=fmt, stream=sys.stderr, force=True)


@app.callback()
def main_callback(
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Рівень деталізації логів (-v для INFO, -vv для DEBUG).",
        ),
    ] = 0,
) -> None:
    setup_logging(verbose)


@app.command(name="index")
def index_command(
    source: Annotated[Path, typer.Argument(help="Каталог або файл корпусу для індексації.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Шлях для збереження файлу індексу.")] = Path(
        "index.bin"
    ),
    fmt: Annotated[
        Literal["pickle", "json"],
        typer.Option("--format", "-f", help="Формат збереження індексу (pickle або json)."),
    ] = "pickle",
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Кількість паралельних воркерів."),
    ] = 4,
    executor: Annotated[
        Literal["serial", "threads", "processes"],
        typer.Option("--executor", "-e", help="Модель виконання (serial, threads або processes)."),
    ] = "processes",
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-l", help="Максимальна кількість документів для обробки."),
    ] = None,
) -> None:
    """Побудувати інвертований індекс для заданого корпусу текстів."""
    if not source.exists():
        err_console.print(f"[red]Помилка:[/red] Шлях до корпусу '{source}' не існує.")
        raise typer.Exit(code=1)

    t0 = time.perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task(
            f"Індексація корпусу з {source} ({executor}, воркерів: {workers})...",
            total=None,
        )
        try:
            index, merge_time = build_index_parallel(
                source, workers=workers, executor=executor, limit=limit
            )
        except Exception as exc:
            err_console.print(f"[red]Помилка під час індексації:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        progress.update(task, description="Збереження файлу індексу на диск...")
        save_index(index, out, fmt=fmt)

    elapsed = time.perf_counter() - t0
    size_mb = out.stat().st_size / (1024 * 1024)

    stats_text = (
        f"[green]Індексацію успішно завершено![/green]\n\n"
        f"• Екзекутор / Воркери:        [bold]{executor}[/bold] ({workers} воркерів)\n"
        f"• Документів проіндексовано:   [bold]{index.num_docs:,}[/bold]\n"
        f"• Унікальних слів у словнику:  [bold]{len(index):,}[/bold]\n"
        f"• Сумарно токенів:             [bold]{index.total_tokens:,}[/bold]\n"
        f"• Середня довжина документа:   [bold]{index.avg_doc_length:.1f}[/bold] слів\n"
        f"• Час фази злиття (merge):     [bold]{merge_time:.3f} с[/bold]\n"
        f"• Загальний час (wall time):   [bold]{elapsed:.2f} с[/bold]\n"
        f"• Розмір файлу на диску:       [bold]{size_mb:.2f} MB[/bold] ({fmt})\n"
        f"• Файл збережено у:            [cyan]{out}[/cyan]"
    )
    console.print(Panel(stats_text, title="findex index summary", expand=False))


@app.command(name="search")
def search_command(
    index_path: Annotated[Path, typer.Argument(help="Шлях до файлу збереженого індексу.")],
    query: Annotated[str, typer.Argument(help="Пошуковий запит.")],
    scorer_name: Annotated[
        Literal["bm25", "tfidf"],
        typer.Option("--scorer", "-s", help="Алгоритм оцінки релевантності (bm25 або tfidf)."),
    ] = "bm25",
    limit: Annotated[
        int,
        typer.Option("--limit", "-k", help="Кількість найрелевантніших результатів."),
    ] = 10,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Вивести результати у чистому форматі JSON для конвеєрів (pipes).",
        ),
    ] = False,
) -> None:
    """Виконати повнотекстовий ранжований пошук за запитом."""
    if not index_path.exists():
        err_console.print(f"[red]Помилка:[/red] Файл індексу '{index_path}' не знайдено.")
        raise typer.Exit(code=1)

    try:
        index = load_index(index_path)
    except Exception as exc:
        err_console.print(f"[red]Помилка завантаження індексу:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    scorer = BM25Scorer() if scorer_name == "bm25" else TfIdfScorer()
    t0 = time.perf_counter()
    results = search_ranked(query, index, scorer=scorer, top_k=limit)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if as_json:
        output_data = [
            {
                "rank": rank,
                "score": round(r.score, 4),
                "doc_id": r.doc_id,
                "title": r.title,
                "source": r.source_name,
                "snippet": r.snippet,
            }
            for rank, r in enumerate(results, start=1)
        ]
        print(json.dumps(output_data, ensure_ascii=False, indent=2))
        return

    table = Table(
        title=f"Результати пошуку: '{query}' (Скорер: {scorer_name.upper()}, час: {elapsed_ms:.2f} мс)",
        show_lines=True,
    )
    table.add_column("№", style="cyan", width=4, justify="right")
    table.add_column("Бал", style="green", width=10, justify="right")
    table.add_column("ID", style="dim", width=6, justify="right")
    table.add_column("Назва документа / Джерело", style="bold", width=35)
    table.add_column("Знайдений фрагмент (сніпет)", style="white")

    for rank, r in enumerate(results, start=1):
        doc_label = f"{r.title}\n[dim]{r.source_name}[/dim]" if r.source_name else r.title
        table.add_row(
            str(rank),
            f"{r.score:.4f}",
            str(r.doc_id),
            doc_label,
            r.snippet or "[italic dim]сніпет недоступний[/italic dim]",
        )

    console.print(table)


@app.command(name="stats")
def stats_command(
    source: Annotated[Path, typer.Argument(help="Каталог або файл корпусу.")],
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-l", help="Максимальна кількість документів."),
    ] = None,
    compare: Annotated[
        bool,
        typer.Option("--compare", "-c", help="Порівняти лінивий та жадібний режими."),
    ] = False,
) -> None:
    """Показати статистику корпусу або порівняти використання пам'яті (Eager vs Lazy)."""
    if not source.exists():
        err_console.print(f"[red]Помилка:[/red] Шлях '{source}' не існує.")
        raise typer.Exit(code=1)

    lazy_res = run_lazy(source, limit=limit)

    if compare:
        eager_res = run_eager(source, limit=limit)
        table = Table(title="Порівняння ефективності використання пам'яті (Eager vs Lazy)")
        table.add_column("Метрика", style="bold")
        table.add_column("Жадібний (Eager: списки)", style="red", justify="right")
        table.add_column("Лінивий (Lazy: генератори)", style="green", justify="right")

        table.add_row("Документів", f"{eager_res.doc_count:,}", f"{lazy_res.doc_count:,}")
        table.add_row("Всього токенів", f"{eager_res.token_count:,}", f"{lazy_res.token_count:,}")
        table.add_row("Розмір словника", f"{eager_res.vocab_size:,}", f"{lazy_res.vocab_size:,}")
        table.add_row("Час виконання", f"{eager_res.elapsed_sec:.3f} с", f"{lazy_res.elapsed_sec:.3f} с")
        table.add_row(
            "Пікова пам'ять (RAM)",
            f"{eager_res.peak_memory_mb:.2f} MB",
            f"{lazy_res.peak_memory_mb:.2f} MB",
        )

        console.print(table)
        savings = (
            ((eager_res.peak_memory_mb - lazy_res.peak_memory_mb) / eager_res.peak_memory_mb) * 100
            if eager_res.peak_memory_mb
            else 0
        )
        console.print(f"\n[bold green]Заощаджено пам'яті за допомогою генераторів:[/bold green] [bold]{savings:.1f}%[/bold]\n")
    else:
        table = Table(title=f"Статистика корпусу: {source}")
        table.add_column("Параметр", style="bold")
        table.add_column("Значення", style="cyan", justify="right")
        table.add_row("Оброблено документів", f"{lazy_res.doc_count:,}")
        table.add_row("Загальна кількість токенів", f"{lazy_res.token_count:,}")
        table.add_row("Унікальних слів (словник)", f"{lazy_res.vocab_size:,}")
        table.add_row("Час підрахунку", f"{lazy_res.elapsed_sec:.3f} с")
        table.add_row("Пікове споживання RAM", f"{lazy_res.peak_memory_mb:.2f} MB")
        console.print(table)


@app.command(name="crawl")
def crawl_command(
    seed_url: Annotated[str, typer.Argument(help="Початкова адреса URL для обходу.")],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Шлях до файлу JSONL для збереження сторінок."),
    ] = Path("data/crawl.jsonl"),
    max_pages: Annotated[
        int,
        typer.Option("--max-pages", "-m", help="Максимальна кількість сторінок для завантаження."),
    ] = 100,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-c", help="Кількість одночасних запитів (семафор)."),
    ] = 10,
    per_host: Annotated[
        int,
        typer.Option("--per-host", "-p", help="Максимальна кількість одночасних запитів до хоста."),
    ] = 2,
    delay: Annotated[
        float,
        typer.Option("--delay", "-d", help="Мінімальна пауза між запитами до одного хоста (секунди)."),
    ] = 0.1,
    timeout: Annotated[
        float,
        typer.Option("--timeout", "-t", help="Таймаут завантаження сторінки (секунди)."),
    ] = 10.0,
    log_file: Annotated[
        Path,
        typer.Option("--log", "-l", help="Шлях до файлу логу статусів crawl.log."),
    ] = Path("crawl.log"),
) -> None:
    """Асинхронний обхід веб-сторінок через asyncio та потоковий запис у JSONL."""
    out.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    errors_count = 0
    total_bytes = 0
    pages_count = 0

    with open(log_file, "a", encoding="utf-8") as log_handle, open(out, "w", encoding="utf-8") as jsonl_handle:
        def on_log(url: str, status: int, nbytes: int, elapsed: float, err: str | None) -> None:
            nonlocal errors_count, total_bytes
            total_bytes += nbytes
            if status >= 400 or err:
                errors_count += 1
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            status_str = f"{status}" if not err else f"{status} ({err})"
            log_handle.write(f"[{ts}] {status_str} {url} {nbytes}B {elapsed:.2f}s\n")
            log_handle.flush()

        async def run_crawler() -> None:
            nonlocal pages_count
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=err_console,
            ) as progress:
                task = progress.add_task(
                    f"Обхід {seed_url} (ліміт: {max_pages} сторінок)...", total=max_pages
                )
                async for page in crawl(
                    seed_url,
                    max_pages=max_pages,
                    concurrency=concurrency,
                    per_host=per_host,
                    per_host_delay=delay,
                    timeout=timeout,
                    log_callback=on_log,
                ):
                    pages_count += 1
                    row = {
                        "id": page.url,
                        "url": page.url,
                        "title": page.title,
                        "text": page.text,
                        "fetched_at": page.fetched_at,
                    }
                    jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    jsonl_handle.flush()
                    progress.update(
                        task,
                        completed=pages_count,
                        description=f"Зібрано {pages_count}/{max_pages} сторінок...",
                    )

        t0 = time.perf_counter()
        asyncio.run(run_crawler())
        elapsed = time.perf_counter() - t0

    rate = pages_count / elapsed if elapsed > 0 else 0.0
    summary_text = (
        f"[green]Асинхронний обхід успішно завершено![/green]\n\n"
        f"• Зібрано сторінок:        [bold]{pages_count:,}[/bold] / {max_pages:,}\n"
        f"• Швидкість:              [bold]{rate:.1f}[/bold] стор./сек\n"
        f"• Загальний час:          [bold]{elapsed:.2f} с[/bold]\n"
        f"• Завантажено даних:      [bold]{total_bytes / (1024*1024):.2f} MB[/bold]\n"
        f"• Помилок / блокувань:    [bold]{errors_count}[/bold]\n"
        f"• Результат збережено у:  [cyan]{out}[/cyan]\n"
        f"• Журнал запитів:         [cyan]{log_file}[/cyan]"
    )
    console.print(Panel(summary_text, title="findex crawl summary", expand=False))


if __name__ == "__main__":
    app()
