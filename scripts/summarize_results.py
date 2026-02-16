import glob
import os
import csv
from collections import Counter

SCORED_DIR = "results/scored"
OUTPUT_CSV = os.path.join(SCORED_DIR, "results.csv")
EXCLUDED_FILES = {"llama_base_scored.csv", "llama_base_strict_scored.csv"}
TARGET_LABELS = ["hallucinated", "incorrect", "correct", "refused"]


def normalize_label(label: object) -> str:
    return str(label).strip().lower()


def config_name_from_file(path: str) -> str:
    filename = os.path.basename(path)
    if filename.endswith("_scored_clean.csv"):
        return filename.removesuffix("_scored_clean.csv")
    if filename.endswith("_scored.csv"):
        return filename.removesuffix("_scored.csv")
    return filename


def summarize_file(path: str) -> dict[str, int | str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "rubric_label" not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing required column 'rubric_label'")
        counts = Counter(normalize_label(row.get("rubric_label")) for row in reader)

    row: dict[str, int | str] = {"config": config_name_from_file(path)}
    for label in TARGET_LABELS:
        row[label] = counts.get(label, 0)
    return row


def main() -> None:
    scored_paths = sorted(glob.glob(os.path.join(SCORED_DIR, "*_scored.csv")) + glob.glob(os.path.join(SCORED_DIR, "*_scored_clean.csv")))
    filtered_paths = [
        p for p in scored_paths if os.path.basename(p) not in EXCLUDED_FILES
    ]

    rows = [summarize_file(path) for path in filtered_paths]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", *TARGET_LABELS])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
