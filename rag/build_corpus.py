import os
import re
import csv
import json
import time
import hashlib
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROMPTS_CSV = "bench/prompts.csv"
OUT_JSONL = "rag/corpus.jsonl"
BROKEN_CSV = "rag/broken_urls.csv"

CACHE_DIR = "rag/cache_html"      # raw html cache
CLEAN_DIR = "rag/cache_clean"     # cleaned text cache

# Practical rules
MAX_CHUNKS_PER_URL = 6
CHUNK_CHAR_LEN = 1800            # 1–2k chars recommended
CHUNK_CHAR_OVERLAP = 200         # overlap helps retrieval
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS = 0.25    # be polite / reduce rate-limit risk

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LLM-Benchmark-RAG/1.0; +https://example.com)"
}

def ensure_dirs():
    os.makedirs("rag", exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)

def url_to_id(url: str) -> str:
    """Stable short id for caching."""
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"u_{h}"

def safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:120]

def split_source_urls(source_url_field: str) -> list[str]:
    """
    Supports:
      - single URL
      - ';' delimited URLs (if you later add multiple)
      - empty string
    """
    if not isinstance(source_url_field, str):
        return []
    raw = source_url_field.strip()
    if not raw:
        return []
    # common delimiter in datasets: ';'
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    return parts if parts else []

def fetch_html(url: str) -> str | None:
    """Fetch with caching. Returns HTML string or None if failed."""
    uid = url_to_id(url)
    cache_path = os.path.join(CACHE_DIR, f"{uid}.html")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200 or not r.text:
            return None
        html = r.text
        with open(cache_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(html)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        return html
    except Exception:
        return None

def extract_main_text(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    for tag in soup.find_all(["nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    # Prefer content containers if present
    root = soup.find("article") or soup.find("main") or soup.body or soup

    # Paragraph-first extraction (prevents "missing words" from line filtering)
    paras = []
    for p in root.find_all("p"):
        # remove citation markers like [1], [2]
        for sup in p.find_all("sup"):
            sup.decompose()

        txt = p.get_text(" ", strip=True)  # <- IMPORTANT: space separator keeps sentences intact
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) >= 80:  # keep real paragraphs
            paras.append(txt)

    if paras:
        return "\n\n".join(paras).strip()

    # Fallback if no <p> tags found
    text = root.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def cache_clean_text(url: str, text: str) -> None:
    uid = url_to_id(url)
    path = os.path.join(CLEAN_DIR, f"{uid}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def read_clean_text(url: str) -> str | None:
    uid = url_to_id(url)
    path = os.path.join(CLEAN_DIR, f"{uid}.txt")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return None

def chunk_text(text: str, chunk_len: int, overlap: int) -> list[str]:
    """
    Paragraph-based chunking:
    - splits on blank lines (\\n\\n)
    - packs full paragraphs into chunks up to chunk_len chars
    - uses a light overlap by carrying the tail of the previous chunk
    """
    if not text:
        return []

    # Normalize newlines and split into paragraphs
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    cur = ""

    for p in paras:
        # If a single paragraph is huge, split it on sentence-ish boundaries
        if len(p) > chunk_len:
            # flush current buffer first
            if cur:
                chunks.append(cur.strip())
                if len(chunks) >= MAX_CHUNKS_PER_URL:
                    return chunks
                cur = ""

            # sentence-ish split (period/question/exclamation)
            parts = re.split(r"(?<=[.!?])\s+", p)
            buf = ""
            for part in parts:
                if not part:
                    continue
                if not buf:
                    buf = part
                elif len(buf) + 1 + len(part) <= chunk_len:
                    buf += " " + part
                else:
                    chunks.append(buf.strip())
                    if len(chunks) >= MAX_CHUNKS_PER_URL:
                        return chunks
                    # overlap: carry tail
                    buf = (buf[-overlap:] + " " + part) if overlap > 0 else part

            if buf:
                chunks.append(buf.strip())
                if len(chunks) >= MAX_CHUNKS_PER_URL:
                    return chunks

            continue

        # Normal paragraph packing
        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= chunk_len:
            cur += "\n\n" + p
        else:
            chunks.append(cur.strip())
            if len(chunks) >= MAX_CHUNKS_PER_URL:
                return chunks

            # overlap: carry tail of previous chunk into next
            tail = cur[-overlap:] if overlap > 0 else ""
            cur = (tail + "\n\n" + p).strip() if tail else p

    if cur and len(chunks) < MAX_CHUNKS_PER_URL:
        chunks.append(cur.strip())

    return chunks[:MAX_CHUNKS_PER_URL]

def get_title_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    t = soup.title.get_text(strip=True) if soup.title else ""
    return t[:200]

def log_broken(broken_rows: list[dict]):
    # Append style logging
    file_exists = os.path.exists(BROKEN_CSV)
    with open(BROKEN_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "url", "reason"])
        if not file_exists:
            writer.writeheader()
        for r in broken_rows:
            writer.writerow(r)

def main():
    ensure_dirs()

    prompts = pd.read_csv(PROMPTS_CSV)
    if "source_url" not in prompts.columns:
        raise ValueError("prompts.csv must have a 'source_url' column")

    # Map URL -> list of prompt ids that reference it
    url_to_prompt_ids: dict[str, set[str]] = {}
    for _, row in prompts.iterrows():
        pid = str(row.get("id", "")).strip()
        urls = split_source_urls(str(row.get("source_url", "") or ""))
        for u in urls:
            url_to_prompt_ids.setdefault(u, set()).add(pid)

    all_urls = list(url_to_prompt_ids.keys())
    print(f"Found {len(all_urls)} unique source URLs")

    # Build corpus
    written = 0
    broken = []

    # Start fresh corpus.jsonl each run (safer)
    if os.path.exists(OUT_JSONL):
        os.remove(OUT_JSONL)

    with open(OUT_JSONL, "a", encoding="utf-8") as out:
        for idx, url in enumerate(all_urls, start=1):
            # Use cleaned cache if present to avoid rework
            clean_text = read_clean_text(url)
            title = ""

            if clean_text is None:
                html = fetch_html(url)
                if html is None:
                    broken.append({"prompt_id": ",".join(sorted(url_to_prompt_ids[url])), "url": url, "reason": "fetch_failed"})
                    continue
                title = get_title_from_html(html)
                clean_text = extract_main_text(html, url)
                if len(clean_text) < 300:
                    broken.append({"prompt_id": ",".join(sorted(url_to_prompt_ids[url])), "url": url, "reason": f"too_short_clean_text_len={len(clean_text)}"})
                    continue
                cache_clean_text(url, clean_text)
            else:
                # best-effort title from url
                title = urlparse(url).netloc or ""

            chunks = chunk_text(clean_text, CHUNK_CHAR_LEN, CHUNK_CHAR_OVERLAP)
            if not chunks:
                broken.append({"prompt_id": ",".join(sorted(url_to_prompt_ids[url])), "url": url, "reason": "no_chunks"})
                continue

            uid = url_to_id(url)
            prompt_ids = sorted(url_to_prompt_ids[url])

            # Write up to MAX_CHUNKS_PER_URL chunks
            for j, chunk in enumerate(chunks[:MAX_CHUNKS_PER_URL]):
                doc = {
                    "doc_id": f"{uid}_{j:02d}",
                    "url": url,
                    "title": title,
                    "source": "TruthfulQA_source",
                    "question_ids": prompt_ids,
                    "text": chunk
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                written += 1

            if idx % 10 == 0 or idx == len(all_urls):
                print(f"[{idx}/{len(all_urls)}] processed, corpus_chunks_written={written}")

    if broken:
        log_broken(broken)
        print(f"Logged {len(broken)} broken/skip URLs -> {BROKEN_CSV}")

    print(f"Done. Wrote {written} chunks -> {OUT_JSONL}")

if __name__ == "__main__":
    main()