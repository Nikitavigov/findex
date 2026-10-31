from findex.query import And, Not, Or, Phrase, Term, parse_query


def test_operator_overloading():
    """Перевірка перевантаження операторів &, |, ~ для вузлів AST."""
    q1 = Term("python")
    q2 = Term("rust")

    assert (q1 & q2) == And(q1, q2)
    assert (q1 | q2) == Or(q1, q2)
    assert (~q1) == Not(q1)


def test_parse_simple_terms():
    assert parse_query("python") == Term("python")
    assert parse_query("python rust") == And(Term("python"), Term("rust"))


def test_parse_explicit_operators():
    assert parse_query("python AND rust") == And(Term("python"), Term("rust"))
    assert parse_query("python OR rust") == Or(Term("python"), Term("rust"))
    assert parse_query("NOT python") == Not(Term("python"))


def test_parse_phrase():
    assert parse_query('"event loop"') == Phrase("event", "loop")


def test_parse_precedence_and_parentheses():
    """
    Перевірка пріоритетів:
    1. a OR b c розбирається як a OR (b AND c)
    2. a AND (b OR c) враховує дужки
    """
    assert parse_query("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))
    assert parse_query("python AND (async OR await)") == And(
        Term("python"),
        Or(Term("async"), Term("await")),
    )
    assert parse_query("python AND (async OR await) NOT java") == And(
        And(Term("python"), Or(Term("async"), Term("await"))),
        Not(Term("java")),
    )
