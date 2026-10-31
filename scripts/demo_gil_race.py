import dis
import sys
import threading
import time


def inspect_bytecode() -> None:
    print("=== Байткод операції shared += 1 ===")

    def inc_operation():
        nonlocal shared
        shared += 1

    shared = 0
    dis.dis(inc_operation)
    print("===================================\n")


def test_race_condition(use_lock: bool = False, num_iterations: int = 1_000_000) -> int:
    shared_counter = 0
    lock = threading.Lock()

    def worker():
        nonlocal shared_counter
        for _ in range(num_iterations):
            if use_lock:
                with lock:
                    shared_counter += 1
            else:
                shared_counter += 1

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)

    t0 = time.perf_counter()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    elapsed = time.perf_counter() - t0

    mode = "З Lock" if use_lock else "БЕЗ Lock (гонка)"
    gil_status = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(
        f"[{mode}] Очікувано: {2 * num_iterations:,} | Отримано: {shared_counter:,} | "
        f"Час: {elapsed:.3f} с | GIL enabled: {gil_status}"
    )
    return shared_counter


if __name__ == "__main__":
    inspect_bytecode()
    print("Тест на поточному інтерпретаторі:")
    test_race_condition(use_lock=False)
    test_race_condition(use_lock=True)
