import inspect
from types import GeneratorType

from findex.tokenizer import tokenize


def test_is_generator():
    """Verify that tokenize is a generator function and returns a generator iterator."""
    assert inspect.isgeneratorfunction(tokenize)
    gen = tokenize("Привіт світ")
    assert isinstance(gen, GeneratorType)
    assert iter(gen) is gen


def test_basic_latin():
    """Tokenize plain English sentence with punctuation."""
    text = "Hello, world! This is a test of the search engine."
    tokens = list(tokenize(text))
    assert tokens == [
        "hello",
        "world",
        "this",
        "is",
        "a",
        "test",
        "of",
        "the",
        "search",
        "engine",
    ]


def test_ukrainian_text():
    """Tokenize standard Ukrainian text."""
    text = "Пошукова система findex написана на мові Python."
    tokens = list(tokenize(text))
    assert tokens == [
        "пошукова",
        "система",
        "findex",
        "написана",
        "на",
        "мові",
        "python",
    ]


def test_ukrainian_apostrophes():
    """
    Verify internal apostrophes (ASCII ', typographical ’, and Ukrainian standard ʼ)
    are preserved as part of the token, while surrounding quotes are stripped.
    """
    text = "«м'яч», 'п’ять', “звʼязок”, don't"
    tokens = list(tokenize(text))
    assert tokens == ["м'яч", "п’ять", "звʼязок", "don't"]


def test_hyphenated_words():
    """Verify compound hyphenated words are preserved, but isolated dashes and punctuation are ignored."""
    text = "Це будь-який state-of-the-art алгоритм — тире не слово, як і --дефіс."
    tokens = list(tokenize(text))
    assert "будь-який" in tokens
    assert "state-of-the-art" in tokens
    assert "—" not in tokens
    assert "--" not in tokens


def test_casefolding():
    """Casefolding handles Unicode lowercase conversions beyond simple ASCII."""
    # German Eszett: 'ß'.casefold() == 'ss', while 'ß'.lower() == 'ß'
    text = "GROSS Straße"
    tokens = list(tokenize(text))
    assert tokens == ["gross", "strasse"]


def test_unicode_nfc_normalization():
    """
    Verify that decomposed Unicode characters (e.g. letter + combining acute accent)
    are normalized to NFC single code points.
    """
    decomposed = "e\u0301"  # 'e' + combining acute accent
    precomposed = "é"  # NFC single character
    assert decomposed != precomposed

    # Tokenizing both forms yields identical NFC normalized tokens
    tokens1 = list(tokenize(f"caf{decomposed}"))
    tokens2 = list(tokenize(f"caf{precomposed}"))
    assert tokens1 == tokens2 == ["café"]


def test_alphanumerics_and_digits():
    """Alphanumeric terms and standalone numbers are indexed as tokens."""
    text = "Python 3.12 і версія v1-release вийшла у 2026 році."
    tokens = list(tokenize(text))
    assert "python" in tokens
    assert "3" in tokens
    assert "12" in tokens
    assert "v1-release" in tokens
    assert "2026" in tokens


def test_empty_and_punctuation_only():
    """Empty strings and punctuation-only text yield zero tokens."""
    assert list(tokenize("")) == []
    assert list(tokenize("   \t\n   ")) == []
    assert list(tokenize("... !!! ??? ,,, ;;; --- \"\" '' ")) == []


def test_streaming_consumption():
    """Calling next() yields tokens on demand without precomputing all."""
    gen = tokenize("один два три")
    assert next(gen) == "один"
    assert next(gen) == "два"
    assert next(gen) == "три"
    try:
        next(gen)
        assert False, "Expected StopIteration"
    except StopIteration:
        pass
