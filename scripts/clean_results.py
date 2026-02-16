import json
import pandas as pd

rag_ids = set()
with open("results/llama_rag3.jsonl") as f:
    for line in f:
        rag_ids.add(json.loads(line)["prompt_id"])

df = pd.read_csv("results/scored/llama_base_scored.csv")
df_clean = df[df["prompt_id"].isin(rag_ids)]
df_clean.to_csv("results/scored/llama_base_scored_clean.csv", index=False)

df = pd.read_csv("results/scored/llama_base_strict_scored.csv")
df_clean = df[df["prompt_id"].isin(rag_ids)]
df_clean.to_csv("results/scored/llama_base_strict_scored_clean.csv", index=False)