import os
import pandas as pd
from datasets import load_dataset

DATASET_NAME = "domenicrosati/TruthfulQA"
OUT_PATH = "bench/prompts.csv"
GOLD_PATH = "bench/prompts_gold.csv"

N_ADV = 60
N_NONADV = 60

def main():
    os.makedirs("bench", exist_ok=True)

    ds = load_dataset(DATASET_NAME)
    df = ds["train"].to_pandas()

    adv = df[df["Type"] == "Adversarial"]
    nonadv = df[df["Type"] == "Non-Adversarial"]

    if len(adv) < N_ADV:
        raise ValueError(f"Need {N_ADV} adversarial rows, found {len(adv)}")
    if len(nonadv) < N_NONADV:
        raise ValueError(f"Need {N_NONADV} non-adversarial rows, found {len(nonadv)}")

    seed = 42
    adv_s = adv.sample(n=N_ADV, random_state=seed)
    nonadv_s = nonadv.sample(n=N_NONADV, random_state=seed)

    df_out = pd.concat([adv_s, nonadv_s], ignore_index=True)

    records = []
    gold_records = []

    for i, (_, r) in enumerate(df_out.iterrows(), start=1):
        pid = f"{r['Type'].lower().replace('-', '')}_{i:04d}"

        records.append({
            "id": pid,
            "question": r["Question"],
            "category": r["Category"],
            "type": r["Type"],
            "answerable": 1,
            "source_url": r.get("Source", "")
        })

        gold_records.append({
            "id": pid,
            "best_answer": r["Best Answer"],
            "correct_answers": r["Correct Answers"],
            "incorrect_answers": r["Incorrect Answers"],
        })

    pd.DataFrame(records).to_csv(OUT_PATH, index=False)
    pd.DataFrame(gold_records).to_csv(GOLD_PATH, index=False)

    print(f"Wrote {len(records)} rows -> {OUT_PATH} "
          f"(Adversarial={N_ADV}, Non-Adversarial={N_NONADV})")
    print(f"Wrote {len(gold_records)} rows -> {GOLD_PATH}")

if __name__ == "__main__":
    main()