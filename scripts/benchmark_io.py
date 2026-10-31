import concurrent.futures
import time
from pathlib import Path


def io_task(path: Path) -> int:
    # Симуляція I/O-bound затримки читання з диску / мережевого запиту
    time.sleep(0.02)
    return len(path.read_bytes())


def main():
    files = list(Path("data/corpus").glob("*.txt"))[:50]
    if not files:
        files = list(Path("data/corpus").rglob("*"))[:50]

    # 1. Serial
    t0 = time.perf_counter()
    for f in files:
        io_task(f)
    t_serial = time.perf_counter() - t0

    # 2. Threads 10
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(io_task, files))
    t_threads = time.perf_counter() - t0

    print(f"I/O-bound: Serial ({len(files)} files): {t_serial:.3f} s")
    print(f"I/O-bound: Threads (10 workers): {t_threads:.3f} s")
    print(f"I/O-bound Speedup with threads: {t_serial / t_threads:.2f}x")


if __name__ == "__main__":
    main()
