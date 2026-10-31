import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import psutil

from findex.parallel import build_index_parallel

CORPUS_PATH = Path("data/corpus").resolve()
PYTHON_313T = Path(
    r"C:\Users\Nikita Vigov\AppData\Roaming\uv\python\cpython-3.13+freethreaded-windows-x86_64-none\python.exe"
)


def run_benchmark_cell_312(
    workers: int,
    executor: str,
    repeats: int = 3,
) -> dict[str, float]:
    """Виконує замір для CPython 3.12 (поточний інтерпретатор)."""
    wall_times: list[float] = []
    cpu_times: list[float] = []
    merge_times: list[float] = []
    rss_list: list[float] = []

    process = psutil.Process()

    for _ in range(repeats):
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()

        _idx, m_time = build_index_parallel(
            CORPUS_PATH,
            workers=workers,
            executor=executor,  # type: ignore
        )

        t_wall = time.perf_counter() - t0_wall
        t_cpu = time.process_time() - t0_cpu
        rss_peak = process.memory_info().rss / (1024 * 1024)

        wall_times.append(t_wall)
        cpu_times.append(t_cpu)
        merge_times.append(m_time)
        rss_list.append(rss_peak)

    return {
        "wall": statistics.median(wall_times),
        "cpu": statistics.median(cpu_times),
        "merge": statistics.median(merge_times),
        "rss": statistics.median(rss_list),
    }


def run_benchmark_cell_313t(
    workers: int,
    repeats: int = 3,
) -> dict[str, float]:
    """Виконує замір для free-threaded CPython 3.13t через підпроцес."""
    if not PYTHON_313T.exists():
        return {"wall": 0.0, "cpu": 0.0, "merge": 0.0, "rss": 0.0}

    code = f"""
import time, tracemalloc, statistics, json
from findex.parallel import build_index_parallel
from pathlib import Path

corpus = Path(r'{CORPUS_PATH}')
wall_times, cpu_times, merge_times = [], [], []

for _ in range({repeats}):
    tracemalloc.start()
    t0_w = time.perf_counter()
    t0_c = time.process_time()
    idx, m_t = build_index_parallel(corpus, workers={workers}, executor='threads')
    t_w = time.perf_counter() - t0_w
    t_c = time.process_time() - t0_c
    peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    wall_times.append(t_w)
    cpu_times.append(t_c)
    merge_times.append(m_t)

print(json.dumps({{
    'wall': statistics.median(wall_times),
    'cpu': statistics.median(cpu_times),
    'merge': statistics.median(merge_times),
    'rss': peak_mem
}}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())
    res = subprocess.run(
        [str(PYTHON_313T), "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return json.loads(res.stdout.strip())  # type: ignore


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=== Лабораторна 5: Комплексний бенчмарк паралельної індексації ===")
    cpu_count = os.cpu_count() or 4
    worker_counts = sorted({1, 2, 4, 8, cpu_count})

    print(f"Платформа: {platform.system()} {platform.machine()}")
    print(f"Процесор: {platform.processor()} ({cpu_count} логічних ядер)")
    print(f"Корпус: {CORPUS_PATH}\n")

    # 1. Серійний базовий запуск
    print("Вимірювання serial (1 worker)...")
    base_res = run_benchmark_cell_312(1, "serial")
    base_wall = base_res["wall"]
    print(f"Serial base wall time: {base_wall:.3f} s\n")

    results: list[dict[str, object]] = []

    # Додаємо серійний рядок
    results.append(
        {
            "executor": "serial",
            "workers": 1,
            "runtime": "CPython 3.12 (GIL)",
            "wall": base_wall,
            "cpu": base_res["cpu"],
            "rss": base_res["rss"],
            "merge": base_res["merge"],
            "speedup": 1.0,
        }
    )

    # 2. Потоки (threads) на CPython 3.12
    for w in worker_counts:
        if w == 1:
            continue
        print(f"Вимірювання threads (CPython 3.12, {w} workers)...")
        r = run_benchmark_cell_312(w, "threads")
        speedup = base_wall / r["wall"] if r["wall"] else 1.0
        results.append(
            {
                "executor": "threads",
                "workers": w,
                "runtime": "CPython 3.12 (GIL)",
                "wall": r["wall"],
                "cpu": r["cpu"],
                "rss": r["rss"],
                "merge": r["merge"],
                "speedup": speedup,
            }
        )

    # 3. Процеси (processes) на CPython 3.12
    for w in worker_counts:
        print(f"Вимірювання processes (CPython 3.12, {w} workers)...")
        r = run_benchmark_cell_312(w, "processes")
        speedup = base_wall / r["wall"] if r["wall"] else 1.0
        results.append(
            {
                "executor": "processes",
                "workers": w,
                "runtime": "CPython 3.12 (GIL)",
                "wall": r["wall"],
                "cpu": r["cpu"],
                "rss": r["rss"],
                "merge": r["merge"],
                "speedup": speedup,
            }
        )

    # 4. Потоки (threads) на Free-threaded CPython 3.13t
    if PYTHON_313T.exists():
        for w in worker_counts:
            print(f"Вимірювання threads (CPython 3.13t No-GIL, {w} workers)...")
            r = run_benchmark_cell_313t(w)
            speedup = base_wall / r["wall"] if r["wall"] else 1.0
            results.append(
                {
                    "executor": "threads",
                    "workers": w,
                    "runtime": "CPython 3.13t (No-GIL)",
                    "wall": r["wall"],
                    "cpu": r["cpu"],
                    "rss": r["rss"],
                    "merge": r["merge"],
                    "speedup": speedup,
                }
            )

    # Друк підсумкової Markdown-таблиці
    print("\n\n### Підсумкова таблиця бенчмарку (медіана з 3 повторів):\n")
    print(
        "| Інтерпретатор | Екзекутор | Воркери | Wall time (с) | CPU time (с) | RSS (MB) | Merge time (с) | Прискорення (Speedup) |"
    )
    print(
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|"
    )
    for row in results:
        print(
            f"| {row['runtime']} | `{row['executor']}` | {row['workers']} | "
            f"{row['wall']:.3f} | {row['cpu']:.3f} | {row['rss']:.1f} | "
            f"{row['merge']:.3f} | **{row['speedup']:.2f}x** |"
        )

    # Побудова графіка matplotlib
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    plot_path = docs_dir / "speedup_benchmark.png"

    plt.figure(figsize=(9, 6), dpi=150)

    # Ідеальне лінійне прискорення
    max_w = max(worker_counts)
    plt.plot([1, max_w], [1, max_w], "k--", label="Ідеальне лінійне прискорення ($y = x$)", alpha=0.6)

    # Дані для графіків
    proc_workers = [r["workers"] for r in results if r["executor"] == "processes"]
    proc_speedups = [r["speedup"] for r in results if r["executor"] == "processes"]

    threads_312_w = [1] + [r["workers"] for r in results if r["executor"] == "threads" and "3.12" in str(r["runtime"])]
    threads_312_s = [1.0] + [r["speedup"] for r in results if r["executor"] == "threads" and "3.12" in str(r["runtime"])]

    threads_313t_w = [r["workers"] for r in results if r["executor"] == "threads" and "3.13t" in str(r["runtime"])]
    threads_313t_s = [r["speedup"] for r in results if r["executor"] == "threads" and "3.13t" in str(r["runtime"])]

    plt.plot(proc_workers, proc_speedups, "o-", color="#1f77b4", linewidth=2.5, label="Processes (CPython 3.12)")
    plt.plot(threads_312_w, threads_312_s, "s-", color="#d62728", linewidth=2, label="Threads (CPython 3.12 з GIL)")
    if threads_313t_w:
        plt.plot(threads_313t_w, threads_313t_s, "^-", color="#2ca02c", linewidth=2.5, label="Threads (CPython 3.13t No-GIL)")

    plt.title("Масштабованість паралельної індексації: Закон Амдала та вплив GIL", fontsize=13, fontweight="bold")
    plt.xlabel("Кількість воркерів (ядер)", fontsize=11)
    plt.ylabel("Прискорення (Speedup = T1 / TN)", fontsize=11)
    plt.xticks(worker_counts)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="upper left")
    plt.tight_layout()

    plt.savefig(plot_path)
    print(f"\nГрафік успішно збережено у {plot_path}")

    # Збереження JSON з сирими даними
    json_path = docs_dir / "benchmark_lab05.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Дані бенчмарку збережено у {json_path}")


if __name__ == "__main__":
    main()
