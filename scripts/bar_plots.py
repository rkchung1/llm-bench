import csv
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_CSV = "results/scored/results.csv"
OUT_DIR = "results/plots"
LABELS = ["hallucinated", "incorrect", "correct", "refused"]
COLORS = {
    "hallucinated": "#d62728",
    "incorrect": "#ff7f0e",
    "correct": "#2ca02c",
    "refused": "#1f77b4",
}

CONFIG_RENAMES = {
    "llama_base_scored_clean.csv": "llama_base",
    "llama_base_strict_scored_clean.csv": "llama_base_strict",
}

PER_CONFIG_PAIRS = [
    ("llama_base", "llama_base_strict"),
    ("llama_rag3", "llama_rag3_strict"),
    ("llama_rag6", "llama_rag6_strict"),
]


def normalize_config_name(name: str) -> str:
    return CONFIG_RENAMES.get(name, name)


def load_results(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"config", *LABELS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = list(reader)
        for row in rows:
            row["config"] = normalize_config_name(row["config"])
        return rows


def plot_grouped(rows: list[dict[str, str]], out_path: str) -> None:
    configs = [row["config"] for row in rows]
    x = np.arange(len(configs))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, label in enumerate(LABELS):
        y = [int(row[label]) for row in rows]
        ax.bar(
            x + (i - 1.5) * width,
            y,
            width=width,
            label=label.capitalize(),
            color=COLORS[label],
        )

    ax.set_title("Answer Outcome Counts by Config")
    ax.set_xlabel("Config")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_config(rows: list[dict[str, str]], out_path: str) -> None:
    row_by_config = {row["config"]: row for row in rows}
    ordered_rows: list[dict[str, str]] = []
    for left, right in PER_CONFIG_PAIRS:
        if left in row_by_config:
            ordered_rows.append(row_by_config[left])
        if right in row_by_config:
            ordered_rows.append(row_by_config[right])

    # Append any configs not listed in PER_CONFIG_PAIRS to avoid dropping data.
    paired_configs = {cfg for pair in PER_CONFIG_PAIRS for cfg in pair}
    for row in rows:
        if row["config"] not in paired_configs:
            ordered_rows.append(row)

    n = len(ordered_rows)
    cols = 2
    rows_n = max(3, (n + cols - 1) // cols)
    if n > rows_n * cols:
        raise ValueError(f"plot_per_config supports up to {rows_n * cols} configs, got {n}")
    fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 4 * rows_n), squeeze=False)

    for idx, row in enumerate(ordered_rows):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        values = [int(row[label]) for label in LABELS]
        bars = ax.bar(
            [label.capitalize() for label in LABELS],
            values,
            color=[COLORS[label] for label in LABELS],
        )
        ax.set_title(row["config"])
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.bar_label(bars, padding=2, fontsize=8)
        ax.tick_params(axis="x", rotation=20)

    for idx in range(n, rows_n * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_results(RESULTS_CSV)
    if not rows:
        raise ValueError(f"No data rows found in {RESULTS_CSV}")

    grouped_path = os.path.join(OUT_DIR, "outcome_counts_grouped.png")
    per_config_path = os.path.join(OUT_DIR, "outcome_counts_per_config.png")
    plot_grouped(rows, grouped_path)
    plot_per_config(rows, per_config_path)
    print(f"Saved: {grouped_path}")
    print(f"Saved: {per_config_path}")


if __name__ == "__main__":
    main()
