import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import psutil

from findex.crawler.crawl import crawl

TOTAL_PAGES = 200
LATENCY_SEC = 0.03  # 30 мс симуляції затримки мережі


class MockServerHandler(BaseHTTPRequestHandler):
    """Локальний HTTP-сервер з 250 зв'язаними сторінками та симуляцією помилок."""

    def log_message(self, format: str, *args: Any) -> None:
        # Приглушуємо логування у консоль
        pass

    def do_GET(self) -> None:
        time.sleep(LATENCY_SEC)

        if self.path == "/robots.txt":
            body = "User-agent: *\nDisallow: /admin/\nAllow: /\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            return

        if self.path == "/page_429":
            self.send_response(429)
            self.send_header("Retry-After", "0.05")
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Rate limit exceeded")
            return

        # Генерація зв'язаних сторінок /page_0 ... /page_250
        path_str = self.path.lstrip("/").split("?")[0]
        if path_str == "" or path_str == "page_0":
            page_id = 0
        elif path_str.startswith("page_"):
            try:
                page_id = int(path_str.replace("page_", ""))
            except ValueError:
                page_id = 0
        else:
            self.send_response(404)
            self.end_headers()
            return

        # Посилання на наступні 3 сторінки
        next_links = [f"/page_{(page_id + step) % 250}" for step in (1, 2, 3)]
        links_html = "".join(f'<li><a href="{href}">Посилання {href}</a></li>' for href in next_links)

        html = f"""<!DOCTYPE html>
<html>
<head><title>Документ #{page_id} findex web</title></head>
<body>
    <h1>Сторінка #{page_id}</h1>
    <p>Текстовий контент для повнотекстового пошукового індексу #{page_id}.</p>
    <ul>{links_html}</ul>
</body>
</html>"""
        body_bytes = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)


def run_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), MockServerHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server


async def benchmark_concurrency(
    seed_url: str,
    concurrency: int,
    max_pages: int,
) -> dict[str, Any]:
    proc = psutil.Process()
    errors = 0
    pages_count = 0

    def on_log(url: str, status: int, nbytes: int, elapsed: float, err: str | None) -> None:
        nonlocal errors
        if status >= 400 or err:
            errors += 1

    t0 = time.perf_counter()
    async for _ in crawl(
        seed_url,
        max_pages=max_pages,
        concurrency=concurrency,
        per_host=concurrency,  # Щоб протестувати чисту масштабованість черги
        per_host_delay=0.0,
        timeout=5.0,
        log_callback=on_log,
    ):
        pages_count += 1

    elapsed = time.perf_counter() - t0
    peak_rss = proc.memory_info().rss / (1024 * 1024)
    rate = pages_count / elapsed if elapsed > 0 else 0.0

    return {
        "concurrency": concurrency,
        "pages": pages_count,
        "wall": elapsed,
        "rate": rate,
        "errors": errors,
        "rss": peak_rss,
    }


def main() -> None:
    port = 8765
    server = run_server(port)
    seed_url = f"http://127.0.0.1:{port}/"

    print("=== Лабораторна 6: Бенчмарк асинхронного краулера findex crawl ===")
    print(f"Сервер запущено на {seed_url} (симуляція затримки мережі {LATENCY_SEC*1000:.0f} мс)")
    print(f"Цільова кількість сторінок: {TOTAL_PAGES}\n")

    results: list[dict[str, Any]] = []

    for c in (1, 5, 20):
        print(f"Вимірювання для concurrency={c}...")
        res = asyncio.run(benchmark_concurrency(seed_url, concurrency=c, max_pages=TOTAL_PAGES))
        results.append(res)
        print(
            f"Concurrency {c:2d}: {res['pages']} сторінок за {res['wall']:.2f} с | "
            f"{res['rate']:.1f} стор./сек | RSS: {res['rss']:.1f} MB | Помилок: {res['errors']}"
        )

    server.shutdown()

    # Формування Markdown-таблиці
    print("\n\n### Підсумкова таблиця бенчмарку краулера (200 сторінок):\n")
    print("| Concurrency (одночасних запитів) | Сторінок | Wall time (с) | Сторінок / сек | Помилок | Піковий RSS (MB) | Прискорення |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    base_wall = results[0]["wall"]
    for r in results:
        speedup = base_wall / r["wall"] if r["wall"] > 0 else 1.0
        print(
            f"| **{r['concurrency']}** | {r['pages']} | {r['wall']:.2f} | "
            f"**{r['rate']:.1f}** | {r['errors']} | {r['rss']:.1f} | **{speedup:.2f}x** |"
        )

    # Збереження результатів у docs/benchmark_crawler.json
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    with open(docs_dir / "benchmark_crawler.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nДані збережено у {docs_dir / 'benchmark_crawler.json'}")


if __name__ == "__main__":
    main()
