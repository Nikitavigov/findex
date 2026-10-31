import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


def download_wikipedia_corpus(target_dir: Path, target_count: int = 1000) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    user_agent = "findex-bot/1.0 (nikitavigov06@gmail.com)"
    base_url = "https://uk.wikipedia.org/w/api.php"

    saved = len(list(target_dir.glob("*.txt")))
    print(f"Existing files in {target_dir}: {saved}. Target: {target_count}")

    batch_size = 20
    while saved < target_count:
        params = {
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": "0",
            "grnlimit": str(min(batch_size, target_count - saved)),
            "prop": "extracts",
            "explaintext": "1",
            "exlimit": "max",
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    title = page.get("title", f"doc_{saved + 1}")
                    extract = page.get("extract", "").strip()
                    if len(extract) < 100:
                        continue

                    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:80].strip()
                    doc_path = target_dir / f"{saved + 1:04d}_{safe_title}.txt"
                    doc_path.write_text(f"# {title}\n\n{extract}", encoding="utf-8")
                    saved += 1
                    if saved % 50 == 0 or saved >= target_count:
                        print(f"Downloaded {saved}/{target_count} documents...")
                    if saved >= target_count:
                        break
            time.sleep(0.3)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            print(f"Error fetching batch: {e}, retrying in 2s...")
            time.sleep(2)

    print(f"Successfully prepared corpus of {saved} documents in {target_dir}")


if __name__ == "__main__":
    out = Path("data/corpus")
    download_wikipedia_corpus(out, target_count=500)
