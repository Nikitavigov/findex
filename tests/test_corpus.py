from pathlib import Path

from findex.corpus import iter_documents


def test_iter_documents_is_generator(tmp_path: Path):
    doc_file = tmp_path / "test.txt"
    doc_file.write_text("Hello world", encoding="utf-8")

    gen = iter_documents(tmp_path)
    # The return value from iter_documents is an iterator / generator
    assert hasattr(gen, "__iter__")
    assert hasattr(gen, "__next__")

    docs = list(gen)
    assert len(docs) == 1
    assert docs[0].doc_id == "test.txt"
    assert docs[0].text == "Hello world"


def test_iter_documents_limit(tmp_path: Path):
    for i in range(5):
        (tmp_path / f"doc_{i}.txt").write_text(f"Text for doc {i}", encoding="utf-8")

    docs = list(iter_documents(tmp_path, limit=3))
    assert len(docs) == 3


def test_iter_documents_jsonl(tmp_path: Path):
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text(
        '{"id": "doc1", "text": "First doc"}\n{"id": "doc2", "text": "Second doc"}\n',
        encoding="utf-8",
    )

    docs = list(iter_documents(jsonl_file))
    assert len(docs) == 2
    assert docs[0].doc_id == "doc1"
    assert docs[0].text == "First doc"
    assert docs[1].doc_id == "doc2"
