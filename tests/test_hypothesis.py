import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.boolean import merge_and, merge_or, set_and, set_or
from findex.index import Index, build_index, load_index, save_index
from findex.models import DocMeta, Posting
from findex.tokenizer import normalize_text

# --- Генератор стратегій для постінгів та документів ---

# Список унікальних відсортованих doc_id
doc_ids_strategy = st.lists(
    st.integers(min_value=1, max_value=10_000),
    min_size=0,
    max_size=50,
    unique=True,
).map(sorted)


def make_postings(doc_ids: list[int]) -> list[Posting]:
    return [Posting(doc_id=d, term_frequency=1) for d in doc_ids]


# --- Властивість 1: Еквівалентність булевих операцій (Merge vs Set) ---


@given(list_a=doc_ids_strategy, list_b=doc_ids_strategy)
def test_property_boolean_merge_equals_set(list_a: list[int], list_b: list[int]) -> None:
    """
    Властивість 1: Двовказівниковий алгоритм (merge) та множинні операції (set)
    повинні давати абсолютно ідентичні результати для AND та OR на будь-яких вхідних даних.
    """
    postings_a = make_postings(list_a)
    postings_b = make_postings(list_b)

    # Перевірка AND
    res_merge_and = [p.doc_id for p in merge_and(postings_a, postings_b)]
    res_set_and = [p.doc_id for p in set_and(postings_a, postings_b)]
    assert res_merge_and == res_set_and

    # Перевірка OR
    res_merge_or = [p.doc_id for p in merge_or(postings_a, postings_b)]
    res_set_or = [p.doc_id for p in set_or(postings_a, postings_b)]
    assert res_merge_or == res_set_or


# --- Властивість 2: Інваріант сортованості списків постінгів ---


@given(
    texts=st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=100),
        min_size=1,
        max_size=10,
    )
)
def test_property_postings_are_strictly_sorted(tmp_path_factory: pytest.TempPathFactory, texts: list[str]) -> None:
    """
    Властивість 2: Для будь-якого корпусу документів кожен список постінгів
    в інвертованому індексі повинен бути строго монотонно зростаючим за doc_id.
    """
    tmp_dir = tmp_path_factory.mktemp("hypo_corpus")
    for idx, text in enumerate(texts, start=1):
        (tmp_dir / f"doc_{idx}.txt").write_text(text, encoding="utf-8")

    index = build_index(tmp_dir)

    for term in index:
        postings = index[term]
        doc_ids = [p.doc_id for p in postings]
        # Перевірка строгого зростання: d[i] < d[i+1]
        for i in range(len(doc_ids) - 1):
            assert doc_ids[i] < doc_ids[i + 1], f"Постінги для '{term}' не відсортовані: {doc_ids}"


# --- Властивість 3: Round-trip серіалізації індексу (JSON та Pickle) ---


@given(
    terms=st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=8),
        min_size=1,
        max_size=5,
        unique=True,
    ),
    doc_count=st.integers(min_value=1, max_value=5),
)
def test_property_index_serialization_roundtrip(
    tmp_path_factory: pytest.TempPathFactory,
    terms: list[str],
    doc_count: int,
) -> None:
    """
    Властивість 3: Збереження та відновлення індексу через save_index/load_index
    повинно зберігати всі терми, розмір словника та кількість документів без спотворень.
    """
    tmp_dir = tmp_path_factory.mktemp("roundtrip")
    postings_dict: dict[str, list[Posting]] = {}
    for term in terms:
        postings_dict[term] = [Posting(doc_id=i, term_frequency=i) for i in range(1, doc_count + 1)]

    doc_meta = {
        i: DocMeta(doc_id=i, source_name=f"file_{i}.txt", title=f"Title {i}", length=len(terms))
        for i in range(1, doc_count + 1)
    }

    original_index = Index(postings=postings_dict, doc_meta=doc_meta)

    # Тестуємо збереження у форматі JSON
    json_path = tmp_dir / "index.json"
    save_index(original_index, json_path, fmt="json")
    loaded_json = load_index(json_path)

    assert len(loaded_json) == len(original_index)
    assert loaded_json.num_docs == original_index.num_docs
    for term in terms:
        assert term in loaded_json
        assert [p.doc_id for p in loaded_json[term]] == [p.doc_id for p in original_index[term]]

    # Тестуємо збереження у форматі Pickle
    pkl_path = tmp_dir / "index.pkl"
    save_index(original_index, pkl_path, fmt="pickle")
    loaded_pkl = load_index(pkl_path)

    assert len(loaded_pkl) == len(original_index)
    assert loaded_pkl.num_docs == original_index.num_docs
    for term in terms:
        assert term in loaded_pkl


# --- Властивість 4: Ідемпотентність нормалізації тексту ---


@given(raw_text=st.text(min_size=0, max_size=100))
def test_property_normalization_idempotent(raw_text: str) -> None:
    """
    Властивість 4: Нормалізація тексту є ідемпотентною:
    normalize_text(normalize_text(x)) == normalize_text(x).
    """
    first_pass = normalize_text(raw_text)
    second_pass = normalize_text(first_pass)
    assert first_pass == second_pass
