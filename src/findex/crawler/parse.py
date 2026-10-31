import re
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

WHITESPACE_REGEX = re.compile(r"\s+")


def canonicalize_url(url: str) -> str:
    """
    Канонізує URL для запобігання повторному завантаженню еквівалентних адрес:
    - Схема та хост приводяться до нижнього регістру.
    - Видаляється фрагмент після '#'.
    - Видаляються стандартні порти (:80 для http, :443 для https).
    - Нормалізується порожній шлях у '/'.
    - Параметри запиту (query params) сортуються за ключем.
    """
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Видалення стандартних портів
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    path = parsed.path or "/"
    # Нормалізація подвійних слешів
    path = re.sub(r"/+", "/", path)

    # Сортування query-параметрів
    query_parts = parse_qsl(parsed.query, keep_blank_values=True)
    sorted_query = urlencode(sorted(query_parts))

    return urlunparse((scheme, netloc, path, "", sorted_query, ""))


class PageContentParser(HTMLParser):
    """Швидкий парсер HTML на базі стандартного html.parser."""

    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self._in_ignored = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower_tag = tag.lower()
        if lower_tag == "title":
            self._in_title = True
        elif lower_tag in {"script", "style", "noscript", "svg"}:
            self._in_ignored = True
        elif lower_tag == "a":
            for attr_name, attr_val in attrs:
                if attr_name.lower() == "href" and attr_val:
                    clean_href = attr_val.strip()
                    if not clean_href.startswith(("javascript:", "mailto:", "tel:")):
                        self.links.append(clean_href)

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        if lower_tag == "title":
            self._in_title = False
        elif lower_tag in {"script", "style", "noscript", "svg"}:
            self._in_ignored = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        elif not self._in_ignored:
            cleaned = data.strip()
            if cleaned:
                self.text_parts.append(cleaned)


def extract_links_and_text(
    html_content: str,
    base_url: str,
) -> tuple[str, str, list[str]]:
    """
    Витягує заголовок сторінки, очищений текст та список абсолютних посилань.
    """
    parser = PageContentParser()
    try:
        parser.feed(html_content)
    except (ValueError, TypeError, UnicodeError):
        pass

    title = " ".join(parser.title_parts).strip()
    text = " ".join(parser.text_parts).strip()
    text = WHITESPACE_REGEX.sub(" ", text)

    absolute_links: list[str] = []
    for link in parser.links:
        try:
            full_url = canonicalize_url(urljoin(base_url, link))
            parsed = urlparse(full_url)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                absolute_links.append(full_url)
        except (ValueError, TypeError):
            continue

    return title, text, absolute_links
