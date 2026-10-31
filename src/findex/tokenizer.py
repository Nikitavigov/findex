import re
import unicodedata
from collections.abc import Iterator

# Matches word tokens:
# - Supports Latin, Cyrillic, digits, and any Unicode letters (\w)
# - Preserves internal apostrophes (', ’, ʼ) e.g. "м'яч", "п'ять", "don't"
# - Preserves internal hyphens (-) e.g. "будь-який", "state-of-the-art"
# - Strips surrounding punctuation, quotes, symbols, and formatting
TOKEN_PATTERN = re.compile(r"[\w]+(?:['’ʼ-][\w]+)*", re.UNICODE)


def normalize_text(text: str) -> str:
    """
    Applies Unicode NFC normalization followed by full casefolding.
    NFC ensures canonical composition (e.g. 'e' + combining acute becomes 'é').
    casefold() eliminates case distinctions for all Unicode scripts (e.g. 'ß' -> 'ss').
    """
    return unicodedata.normalize("NFC", text).casefold()


def tokenize(text: str) -> Iterator[str]:
    """
    Streaming generator that yields tokens from text one by one.
    Does not allocate intermediate token lists in memory.
    """
    normalized = normalize_text(text)
    for match in TOKEN_PATTERN.finditer(normalized):
        yield match.group(0)
