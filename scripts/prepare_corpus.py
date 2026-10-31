import re
import urllib.request
from pathlib import Path

BOOKS = [
    ("1342-0.txt", "https://www.gutenberg.org/files/1342/1342-0.txt", "Pride and Prejudice"),
    ("11-0.txt", "https://www.gutenberg.org/files/11/11-0.txt", "Alice in Wonderland"),
    ("1661-0.txt", "https://www.gutenberg.org/files/1661/1661-0.txt", "Sherlock Holmes Adventures"),
    ("84-0.txt", "https://www.gutenberg.org/files/84/84-0.txt", "Frankenstein"),
    ("36-0.txt", "https://www.gutenberg.org/files/36/36-0.txt", "The War of the Worlds"),
    ("2701-0.txt", "https://www.gutenberg.org/files/2701/2701-0.txt", "Moby Dick"),
    ("98-0.txt", "https://www.gutenberg.org/files/98/98-0.txt", "A Tale of Two Cities"),
    ("2591-0.txt", "https://www.gutenberg.org/files/2591/2591-0.txt", "Grimms Fairy Tales"),
]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) findex-corpus-builder/1.0"


def prepare_gutenberg_corpus(output_dir: Path, target_count: int = 1000) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    current_docs = len(list(output_dir.glob("*.txt")))
    print(f"Starting with {current_docs} documents in {output_dir}")

    doc_idx = current_docs + 1
    for filename, url, book_name in BOOKS:
        if doc_idx > target_count:
            break
        print(f"Fetching {book_name}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw_text = resp.read().decode("utf-8", errors="replace")

            # Split into chapters or sizable sections
            sections = re.split(r"\n(?=(?:CHAPTER|Chapter|STORY|Story|BOOK|Book)\s+[0-9IVXLCDM]+)", raw_text)
            if len(sections) < 5:
                # If regex didn't split well, split by double newlines into ~15KB chunks
                chunks = raw_text.split("\n\n\n")
                sections = []
                cur = []
                cur_len = 0
                for c in chunks:
                    cur.append(c)
                    cur_len += len(c)
                    if cur_len > 8000:
                        sections.append("\n\n".join(cur))
                        cur = []
                        cur_len = 0
                if cur:
                    sections.append("\n\n".join(cur))

            print(f"  Got {len(sections)} sections from {book_name}")
            for sec in sections:
                sec = sec.strip()
                if len(sec) < 500:
                    continue
                safe_book = "".join(c if c.isalnum() else "_" for c in book_name)
                doc_file = output_dir / f"{doc_idx:04d}_{safe_book}_sec.txt"
                doc_file.write_text(f"# {book_name} - Section {doc_idx}\n\n{sec}", encoding="utf-8")
                doc_idx += 1
                if doc_idx > target_count:
                    break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  Failed to fetch {book_name}: {e}")

    final_count = len(list(output_dir.glob("*.txt")))
    print(f"Corpus prepared: total {final_count} documents in {output_dir}!")


if __name__ == "__main__":
    prepare_gutenberg_corpus(Path("data/corpus"), target_count=1000)
