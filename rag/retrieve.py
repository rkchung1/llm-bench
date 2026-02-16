#!/usr/bin/env python3
import json
import numpy as np
import requests
import os
from typing import List, Dict, Tuple

EMB_NPZ = "rag/corpus_embeddings.npz"
META_JSONL = "rag/corpus_meta.jsonl"
CORPUS_JSONL = "rag/corpus.jsonl"

MODEL = "text-embedding-3-small"

_INDEX = None  # cache: (E, meta, doc_text)

def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return key

def embed_query(text: str) -> np.ndarray:
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {get_api_key()}"}
    payload = {"model": MODEL, "input": text}

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    emb = r.json()["data"][0]["embedding"]
    v = np.array(emb, dtype=np.float32)
    return v

def load_index() -> Tuple[np.ndarray, List[Dict], Dict[str, str]]:
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    npz = np.load(EMB_NPZ, allow_pickle=True)
    E = np.array(npz["embeddings"], dtype=np.float32)

    meta = []
    with open(META_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))

    if len(meta) != E.shape[0]:
        raise ValueError(f"Meta rows ({len(meta)}) != embeddings rows ({E.shape[0]}).")

    doc_text = {}
    with open(CORPUS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_text[obj["doc_id"]] = obj["text"]

    _INDEX = (E, meta, doc_text)
    return _INDEX

def cosine_topk(E: np.ndarray, q: np.ndarray, k: int = 3) -> np.ndarray:
    # normalize
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    q_norm = q / (np.linalg.norm(q) + 1e-12)

    sims = E_norm @ q_norm  # [N]
    idx = np.argpartition(-sims, kth=min(k, len(sims)-1))[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx

def retrieve(query: str, k: int = 3) -> List[Dict]:
    E, meta, doc_text = load_index()
    q = embed_query(query)
    idxs = cosine_topk(E, q, k=k)

    results = []
    for rank, i in enumerate(idxs, start=1):
        m = meta[int(i)]
        did = m["doc_id"]
        results.append({
            "rank": rank,
            "doc_id": did,
            "url": m.get("url", ""),
            "title": m.get("title", ""),
            "text": doc_text.get(did, ""),
        })
    return results

if __name__ == "__main__":
    # quick manual test
    test_q = "What is a reasonable defense if you suspect that someone is a vampire in real life?"
    hits = retrieve(test_q, k=3)
    for h in hits:
        print("\n---")
        print(f"#{h['rank']} {h['doc_id']} {h['title']} {h['url']}")
        print(h["text"][:1200])