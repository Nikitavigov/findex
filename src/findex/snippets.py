import re

from findex.tokenizer import normalize_text


def generate_snippet(
    text: str,
    query_terms: list[str],
    window_chars: int = 160,
    highlight: bool = True,
) -> str:
    """
    Генерує контекстний фрагмент (сніпет) довжиною близько window_chars символів
    навколо найкращого входження слів із запиту з виділенням знайдених термінів.
    """
    if not text:
        return ""

    norm_terms = [normalize_text(t) for t in query_terms if t.strip()]
    if not norm_terms:
        # Якщо термінів немає, повертаємо початок тексту
        clean_text = " ".join(text.split())
        return clean_text[:window_chars] + (
            "..." if len(clean_text) > window_chars else ""
        )

    # Створюємо регулярний вираз для пошуку будь-якого зі слів
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in norm_terms) + r")\b",
        re.IGNORECASE | re.UNICODE,
    )

    match = pattern.search(text)
    if not match:
        clean_text = " ".join(text.split())
        return clean_text[:window_chars] + (
            "..." if len(clean_text) > window_chars else ""
        )

    # Визначаємо вікно ±(window_chars // 2) символів навколо збігу
    half = window_chars // 2
    match_start = match.start()
    match_end = match.end()

    start = max(0, match_start - half)
    end = min(len(text), match_end + half)

    # Намагаємося посунути межі до цілих слів (пробілів)
    if start > 0:
        space_pos = text.rfind(" ", 0, start)
        if space_pos != -1 and space_pos >= start - 20:
            start = space_pos + 1

    if end < len(text):
        space_pos = text.find(" ", end)
        if space_pos != -1 and space_pos <= end + 20:
            end = space_pos

    snippet_raw = text[start:end].replace("\n", " ").strip()

    # Підсвічуємо знайдені терми
    if highlight:
        snippet_highlighted = pattern.sub(r"**\1**", snippet_raw)
    else:
        snippet_highlighted = snippet_raw

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""

    return f"{prefix}{snippet_highlighted}{suffix}"
