import os
import json
import time
import argparse
from typing import Dict, Any, List, Optional

import pandas as pd
import requests

# ======================
# Files / Paths
# ======================
PROMPTS_CSV = "bench/prompts.csv"
RESULTS_DIR = "results"
BROKEN_URLS_CSV = "rag/broken_urls.csv"

# RAG retrieval
from rag.retrieve import retrieve  # retrieve(query, k)

# ======================
# Ollama Config
# ======================
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# ======================
# Helpers
# ======================
def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def load_prompts() -> pd.DataFrame:
    df = pd.read_csv(PROMPTS_CSV)
    required = {"id", "question"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prompts.csv missing columns: {missing}")
    return df

def load_broken_prompt_ids() -> set[str]:
    if not os.path.exists(BROKEN_URLS_CSV):
        return set()

    broken_df = pd.read_csv(BROKEN_URLS_CSV)
    if "prompt_id" not in broken_df.columns:
        return set()

    broken_ids: set[str] = set()
    for value in broken_df["prompt_id"].dropna():
        for pid in str(value).split(","):
            pid = pid.strip()
            if pid:
                broken_ids.add(pid)
    return broken_ids

def build_system_prompt(strict: bool) -> str:
    if strict:
        return (
            "You are a careful assistant. Answer truthfully.\n"
            "If unsure, say \"I don't know\".\n"
            "Do not guess or fabricate facts.\n"
        )
    return "You are a helpful assistant.\n"

def build_user_prompt(question: str, context: Optional[str]) -> str:
    if context:
        return (
            "Use the provided context to help answer the question.\n"
            "If the context is irrelevant, say \"I don't know\".\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
        )
    return question

def call_llama(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature},
        "stream": False,
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("message", {}) or {}).get("content", "").strip()

def build_context_from_hits(hits: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    parts = []
    used = 0

    for h in hits:
        header = f"[{h['rank']}] {h.get('title','')} | {h.get('url','')}\n"
        body = (h.get("text") or "").strip()
        block = header + body

        if not body:
            continue

        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 200:
                parts.append(block[:remaining])
            break

        parts.append(block)
        used += len(block) + 2

    return "\n\n".join(parts).strip()

def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in [
        "i don't know",
        "i do not know",
        "cannot answer",
        "insufficient information",
        "not enough information",
        "unknown"
    ])

# ======================
# Benchmark Configs
# ======================
CONFIGS = [
    {"name": "base",         "use_rag": False, "k": 0, "strict": False},
    {"name": "base_strict",  "use_rag": False, "k": 0, "strict": True},
    {"name": "rag3",         "use_rag": True,  "k": 3, "strict": False},
    {"name": "rag3_strict",  "use_rag": True,  "k": 3, "strict": True},
    {"name": "rag6",         "use_rag": True,  "k": 6, "strict": False},
    {"name": "rag6_strict",  "use_rag": True,  "k": 6, "strict": True},
]
CONFIG_BY_NAME = {c["name"]: c for c in CONFIGS}

# ======================
# Runner
# ======================
def run_config(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    limit: Optional[int],
    overwrite: bool,
    broken_prompt_ids: set[str],
):
    out_path = os.path.join(RESULTS_DIR, f"llama_{cfg['name']}.jsonl")

    if os.path.exists(out_path) and not overwrite:
        print(f"Skipping {cfg['name']} (exists) -> {out_path}  (use --overwrite to rerun)")
        return

    run_df = df
    if cfg["use_rag"] and broken_prompt_ids:
        run_df = df[~df["id"].astype(str).isin(broken_prompt_ids)].reset_index(drop=True)
        skipped = len(df) - len(run_df)
        if skipped:
            print(
                f"\n=== Running {cfg['name']} → {out_path} "
                f"(skipping {skipped} broken-url prompt(s)) ==="
            )
        else:
            print(f"\n=== Running {cfg['name']} → {out_path} ===")
    else:
        print(f"\n=== Running {cfg['name']} → {out_path} ===")

    n = len(run_df) if limit is None else min(limit, len(run_df))

    with open(out_path, "w", encoding="utf-8") as out:
        for i in range(n):
            row = run_df.iloc[i]
            pid = str(row["id"])
            question = str(row["question"])

            hits = []
            context = None

            if cfg["use_rag"]:
                hits = retrieve(question, k=cfg["k"])
                context = build_context_from_hits(hits)

            system_prompt = build_system_prompt(cfg["strict"])
            user_prompt = build_user_prompt(question, context)

            try:
                answer = call_llama(system_prompt, user_prompt)

                record = {
                    "config": cfg["name"],
                    "model": OLLAMA_MODEL,
                    "prompt_id": pid,
                    "question": question,
                    "use_rag": cfg["use_rag"],
                    "k": cfg["k"],
                    "strict": cfg["strict"],
                    "context_chars": len(context) if context else 0,
                    "retrieved_docs": [
                        {
                            "rank": h["rank"],
                            "doc_id": h["doc_id"],
                            "title": h.get("title", ""),
                            "url": h.get("url", ""),
                        }
                        for h in hits
                    ],
                    "answer": answer,
                    "refusal": is_refusal(answer),
                    "timestamp": time.time(),
                }

            except Exception as e:
                record = {
                    "config": cfg["name"],
                    "prompt_id": pid,
                    "error": str(e),
                    "timestamp": time.time(),
                }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0 or (i + 1) == n:
                print(f"[{i+1}/{n}] done")

# ======================
# Main
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit prompts for quick test")
    parser.add_argument(
        "--config",
        action="append",
        help=(
            "Run only selected config(s) by name. "
            "Repeatable: --config rag3 --config rag3_strict. "
            "If omitted, runs all."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results file(s) for selected configs.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Print available config names and exit.",
    )
    args = parser.parse_args()

    if args.list_configs:
        print("Available configs:")
        for c in CONFIGS:
            print(f"  - {c['name']}")
        return

    ensure_dirs()
    df = load_prompts()
    broken_prompt_ids = load_broken_prompt_ids()

    # Connectivity check
    try:
        print("Testing Ollama connection...")
        _ = call_llama("You are a test.", "Say OK.")
    except Exception as e:
        raise RuntimeError(f"Ollama not responding at {OLLAMA_URL}: {e}")

    # Pick configs to run
    if args.config:
        selected = []
        for name in args.config:
            if name not in CONFIG_BY_NAME:
                raise ValueError(
                    f"Unknown config '{name}'. Use --list-configs to see valid names."
                )
            selected.append(CONFIG_BY_NAME[name])
    else:
        selected = CONFIGS

    for cfg in selected:
        run_config(
            df,
            cfg,
            args.limit,
            overwrite=args.overwrite,
            broken_prompt_ids=broken_prompt_ids,
        )

    print("\nDone. Results saved to results/*.jsonl")

if __name__ == "__main__":
    main()
