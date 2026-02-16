import os
import json
import time
import numpy as np
import requests

CORPUS_JSONL = "rag/corpus.jsonl"
OUT_NPZ = "rag/corpus_embeddings.npz"
OUT_META_JSONL = "rag/corpus_meta.jsonl"

MODEL = "text-embedding-3-small"

# Batching / reliability
BATCH_SIZE = 96            # safe default
REQUEST_TIMEOUT = 60
MAX_RETRIES = 6
RETRY_BASE_SECONDS = 1.5

def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment")
    return key

def openai_embed(texts: list[str]) -> list[list[float]]:
    """
    Calls OpenAI embeddings endpoint for a batch of texts.
    Returns list of embedding vectors aligned with input order.
    """
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_api_key()}",
    }
    payload = {"model": MODEL, "input": texts}

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                rows = data.get("data", [])
                # Sort by index to be safe
                rows = sorted(rows, key=lambda x: x.get("index", 0))
                embs = [row["embedding"] for row in rows]
                return embs

            # Handle rate limits / transient failures with backoff
            if r.status_code in (429, 500, 502, 503, 504):
                wait = RETRY_BASE_SECONDS * (2 ** attempt)
                # Respect Retry-After if present
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        wait = max(wait, float(ra))
                    except Exception:
                        pass
                time.sleep(wait)
                continue

            # Non-retryable error
            raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

        except requests.RequestException as e:
            wait = RETRY_BASE_SECONDS * (2 ** attempt)
            time.sleep(wait)

    raise RuntimeError("Failed to embed batch after retries")

def load_existing_doc_ids() -> set[str]:
    """
    If OUT_META_JSONL exists, we treat it as the source of truth for already-embedded doc_ids.
    This supports resuming without duplicating work.
    """
    done = set()
    if not os.path.exists(OUT_META_JSONL):
        return done
    with open(OUT_META_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                did = obj.get("doc_id")
                if did:
                    done.add(did)
            except Exception:
                continue
    return done

def iter_corpus_rows():
    with open(CORPUS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

def main():
    if not os.path.exists(CORPUS_JSONL):
        raise FileNotFoundError(f"Missing {CORPUS_JSONL}. Run rag/build_corpus.py first.")

    os.makedirs("rag", exist_ok=True)

    already = load_existing_doc_ids()
    if already:
        print(f"Resume mode: found {len(already)} already embedded doc_ids in {OUT_META_JSONL}")

    # accumulate embeddings in memory then save once.
    embeddings = []
    meta_rows = []

    batch_texts = []
    batch_meta = []

    total_seen = 0
    total_queued = 0
    total_skipped = 0

    for row in iter_corpus_rows():
        total_seen += 1
        doc_id = row.get("doc_id")
        text = row.get("text", "")

        if not doc_id or not isinstance(text, str) or not text.strip():
            continue

        if doc_id in already:
            total_skipped += 1
            continue

        # queue
        batch_texts.append(text)
        batch_meta.append({
            "doc_id": doc_id,
            "url": row.get("url", ""),
            "title": row.get("title", ""),
            "source": row.get("source", ""),
            "question_ids": row.get("question_ids", []),
            # store a short preview to help debugging; keep it small
            "text_preview": text[:200].replace("\n", " ")
        })
        total_queued += 1

        # process batch
        if len(batch_texts) >= BATCH_SIZE:
            embs = openai_embed(batch_texts)

            # append
            embeddings.extend(embs)
            meta_rows.extend(batch_meta)

            print(f"Embedded {len(meta_rows)} chunks so far (seen={total_seen}, skipped={total_skipped})")

            batch_texts = []
            batch_meta = []

    # final partial batch
    if batch_texts:
        embs = openai_embed(batch_texts)
        embeddings.extend(embs)
        meta_rows.extend(batch_meta)
        print(f"Embedded {len(meta_rows)} chunks total (final batch).")

    if not meta_rows:
        print("No new chunks to embed (everything already done).")
        # Still ensure we can load existing npz if needed; we won't overwrite.
        return

    # Write/append meta jsonl so resume works next time
    mode = "a" if os.path.exists(OUT_META_JSONL) else "w"
    with open(OUT_META_JSONL, mode, encoding="utf-8") as f:
        for m in meta_rows:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Save embeddings: align with meta_rows we just wrote
    E = np.array(embeddings, dtype=np.float32)
    print(f"Embedding matrix shape for this run: {E.shape}")

    # If an existing npz exists, append to it
    if os.path.exists(OUT_NPZ):
        old = np.load(OUT_NPZ)
        E_old = old["embeddings"]
        E = np.vstack([E_old, E]).astype(np.float32)
        print(f"Appended to existing embeddings. New shape: {E.shape}")

    np.savez_compressed(OUT_NPZ, embeddings=E, model=MODEL)
    print(f"Saved embeddings -> {OUT_NPZ}")
    print(f"Saved metadata -> {OUT_META_JSONL}")

if __name__ == "__main__":
    main()