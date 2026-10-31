import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from findex.boolean import merge_and, merge_not, merge_or
from findex.index import Index
from findex.models import Posting
from findex.tokenizer import normalize_text, tokenize

# --- Вузли синтаксичного дерева запиту (AST) ---


class QueryNode(ABC):
    """Базовий клас вузла синтаксичного дерева пошукового запиту."""

    @abstractmethod
    def evaluate(self, index: Index) -> list[Posting]:
        """Обчислює збіги в індексі та повертає впорядкований список постінгів."""
        ...

    def __and__(self, other: "QueryNode") -> "And":
        return And(self, other)

    def __or__(self, other: "QueryNode") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)


@dataclass(frozen=True)
class Term(QueryNode):
    term: str

    def __post_init__(self):
        object.__setattr__(self, "term", normalize_text(self.term))

    def evaluate(self, index: Index) -> list[Posting]:
        return index.get_postings(self.term)

    def __repr__(self) -> str:
        return f"Term('{self.term}')"


@dataclass(frozen=True)
class Phrase(QueryNode):
    terms: tuple[str, ...]

    def __init__(self, *terms: str):
        normalized = tuple(normalize_text(t) for t in terms if t.strip())
        object.__setattr__(self, "terms", normalized)

    def evaluate(self, index: Index) -> list[Posting]:
        if not self.terms:
            return []
        # Базова реалізація фразового пошуку через перетин усіх термінів
        current = index.get_postings(self.terms[0])
        for t in self.terms[1:]:
            current = merge_and(current, index.get_postings(t))
        return current

    def __repr__(self) -> str:
        return f"Phrase({', '.join(repr(t) for t in self.terms)})"


@dataclass(frozen=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> list[Posting]:
        return merge_and(self.left.evaluate(index), self.right.evaluate(index))

    def __repr__(self) -> str:
        return f"And({self.left!r}, {self.right!r})"


@dataclass(frozen=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> list[Posting]:
        return merge_or(self.left.evaluate(index), self.right.evaluate(index))

    def __repr__(self) -> str:
        return f"Or({self.left!r}, {self.right!r})"


@dataclass(frozen=True)
class Not(QueryNode):
    target: QueryNode

    def evaluate(self, index: Index) -> list[Posting]:
        all_docs = [
            Posting(doc_id=doc_id, term_frequency=1)
            for doc_id in sorted(index.doc_meta.keys())
        ]
        return merge_not(all_docs, self.target.evaluate(index))

    def __repr__(self) -> str:
        return f"Not({self.target!r})"


# --- Лексер та Парсер рекурсивного спуску ---


TOKEN_REGEX = re.compile(
    r"""
    (?P<PHRASE>"[^"]*")           | # Фрази в лапках "..."
    (?P<LPAREN>\()                | # Ліва дужка
    (?P<RPAREN>\))                | # Права дужка
    (?P<OP_AND>\bAND\b|\bІ\b)     | # Оператор AND
    (?P<OP_OR>\bOR\b|\bАБО\b)     | # Оператор OR
    (?P<OP_NOT>\bNOT\b|\bНЕ\b)    | # Оператор NOT
    (?P<WORD>[^\s()"]+)             # Звичайне слово
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _tokenize_query(query_str: str) -> list[tuple[str, str]]:
    """Розбиває рядок запиту на список токенів (тип, значення)."""
    tokens = []
    for match in TOKEN_REGEX.finditer(query_str):
        kind = match.lastgroup
        value = match.group(0)
        if kind == "PHRASE":
            clean_phrase = value[1:-1].strip()
            tokens.append(("PHRASE", clean_phrase))
        elif kind == "LPAREN":
            tokens.append(("LPAREN", "("))
        elif kind == "RPAREN":
            tokens.append(("RPAREN", ")"))
        elif kind == "OP_AND":
            tokens.append(("AND", "AND"))
        elif kind == "OP_OR":
            tokens.append(("OR", "OR"))
        elif kind == "OP_NOT":
            tokens.append(("NOT", "NOT"))
        elif kind == "WORD":
            tokens.append(("WORD", value))
    return tokens


class QueryParser:
    """
    Парсер рекурсивного спуску з пріоритетом операторів:
    1. primary: дужки, фрази, терми
    2. NOT
    3. AND (явно або поруч розташовані слова: a b -> a AND b)
    4. OR
    """

    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str, str] | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected_type: str | None = None) -> tuple[str, str]:
        current = self.peek()
        if current is None:
            raise ValueError("Неочікуваний кінець запиту")
        if expected_type and current[0] != expected_type:
            raise ValueError(f"Очікувався токен {expected_type}, отримано {current[0]}")
        self.pos += 1
        return current

    def parse(self) -> QueryNode:
        node = self.parse_or()
        if self.peek() is not None:
            raise ValueError(f"Зайвий символ у запиті: {self.peek()[1]}")
        return node

    def parse_or(self) -> QueryNode:
        left = self.parse_and()
        while self.peek() and self.peek()[0] == "OR":
            self.consume("OR")
            right = self.parse_and()
            left = Or(left, right)
        return left

    def parse_and(self) -> QueryNode:
        left = self.parse_not()
        while self.peek() and self.peek()[0] not in {"OR", "RPAREN"}:
            if self.peek()[0] == "AND":
                self.consume("AND")
            right = self.parse_not()
            left = And(left, right)
        return left

    def parse_not(self) -> QueryNode:
        if self.peek() and self.peek()[0] == "NOT":
            self.consume("NOT")
            target = self.parse_primary()
            return Not(target)
        return self.parse_primary()

    def parse_primary(self) -> QueryNode:
        token = self.peek()
        if token is None:
            raise ValueError("Очікувався термін або дужка, але запит завершився")

        kind, value = token
        if kind == "LPAREN":
            self.consume("LPAREN")
            node = self.parse_or()
            self.consume("RPAREN")
            return node
        elif kind == "PHRASE":
            self.consume("PHRASE")
            terms = list(tokenize(value))
            return Phrase(*terms)
        elif kind == "WORD":
            self.consume("WORD")
            return Term(value)
        else:
            raise ValueError(f"Неочікуваний токен {kind} ('{value}')")


def parse_query(query_str: str) -> QueryNode:
    """Парсить рядок запиту в дерево об'єктів QueryNode."""
    tokens = _tokenize_query(query_str)
    if not tokens:
        raise ValueError("Порожній пошуковий запит")
    return QueryParser(tokens).parse()
