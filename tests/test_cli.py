import json
import re
from pathlib import Path

from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    """Перевірка виклику команди help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    clean_out = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", result.stdout)
    assert "findex" in clean_out
    assert "index" in clean_out
    assert "search" in clean_out
    assert "stats" in clean_out
    assert "crawl" in clean_out


def test_cli_crawl_help() -> None:
    """Перевірка виклику команди findex crawl --help."""
    result = runner.invoke(app, ["crawl", "--help"])
    assert result.exit_code == 0
    clean_out = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", result.stdout)
    assert "seed_url" in clean_out or "URL" in clean_out
    assert "--concurrency" in clean_out
    assert "--max-pages" in clean_out


def test_cli_index_and_search(sample_corpus_dir: Path, tmp_path: Path) -> None:
    """Перевірка побудови індексу та виконання пошуку через CLI."""
    out_index = tmp_path / "cli_idx.bin"

    # 1. Побудова індексу
    res_index = runner.invoke(
        app,
        ["index", str(sample_corpus_dir), "--out", str(out_index), "--format", "pickle"],
    )
    assert res_index.exit_code == 0
    assert out_index.exists()

    # 2. Пошук зі звичайним Rich виведенням
    res_search = runner.invoke(app, ["search", str(out_index), "Python"])
    assert res_search.exit_code == 0
    assert "Python" in res_search.stdout or "Результати пошуку" in res_search.stdout

    # 3. Пошук з прапорцем --json
    res_json = runner.invoke(app, ["search", str(out_index), "Python", "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "doc_id" in data[0]
    assert "score" in data[0]
    assert "snippet" in data[0]


def test_cli_search_tfidf(sample_corpus_dir: Path, tmp_path: Path) -> None:
    """Перевірка пошуку з алгоритмом TF-IDF."""
    out_index = tmp_path / "cli_tfidf.bin"
    runner.invoke(app, ["index", str(sample_corpus_dir), "--out", str(out_index)])

    res = runner.invoke(app, ["search", str(out_index), "language", "--scorer", "tfidf"])
    assert res.exit_code == 0
    assert "TFIDF" in res.stdout


def test_cli_index_parallel_executors(sample_corpus_dir: Path, tmp_path: Path) -> None:
    """Перевірка запуску CLI findex index з потоками та процесами."""
    out_threads = tmp_path / "threads.bin"
    res_th = runner.invoke(
        app,
        [
            "index",
            str(sample_corpus_dir),
            "--out",
            str(out_threads),
            "--executor",
            "threads",
            "--workers",
            "2",
        ],
    )
    assert res_th.exit_code == 0
    assert out_threads.exists()

    out_procs = tmp_path / "procs.bin"
    res_pr = runner.invoke(
        app,
        [
            "index",
            str(sample_corpus_dir),
            "--out",
            str(out_procs),
            "--executor",
            "processes",
            "--workers",
            "2",
        ],
    )
    assert res_pr.exit_code == 0
    assert out_procs.exists()


def test_cli_index_nonexistent_path() -> None:
    """CLI повинно повертати код помилки 1, якщо шлях до корпусу не існує."""
    res = runner.invoke(app, ["index", "non_existent_directory_xyz"])
    assert res.exit_code == 1


def test_cli_search_nonexistent_index() -> None:
    """CLI повинно повертати код помилки 1, якщо файл індексу не знайдено."""
    res = runner.invoke(app, ["search", "non_existent_index.bin", "query"])
    assert res.exit_code == 1


def test_cli_stats(sample_corpus_dir: Path) -> None:
    """Перевірка виведення статистики корпусу."""
    res = runner.invoke(app, ["stats", str(sample_corpus_dir)])
    assert res.exit_code == 0
    assert "Оброблено документів" in res.stdout or "Статистика корпусу" in res.stdout


def test_cli_stats_compare(sample_corpus_dir: Path) -> None:
    """Перевірка порівняння пам'яті (Eager vs Lazy)."""
    res = runner.invoke(app, ["stats", str(sample_corpus_dir), "--compare"])
    assert res.exit_code == 0
    assert "Порівняння ефективності" in res.stdout
