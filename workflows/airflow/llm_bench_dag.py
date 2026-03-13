"""
Airflow DAG for the llm-bench pipeline.

Pipeline stages:
1) Run inference for each config in parallel.
2) Score each config output via scripts/score_results.py (LLM-as-judge).
3) Clean base/base_strict scored files to align with RAG prompt IDs.
4) Summarize scored outputs.
5) Generate plots.

To use this DAG, set:
- LLM_BENCH_ROOT: absolute path to this repo (default shown below)
- OPENAI_API_KEY (required by scripts/score_results.py)
- OPENAI_MODEL (optional, defaults to gpt-4.1-mini)
- OPENAI_BASE_URL (optional, defaults to https://api.openai.com/v1)
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

from airflow.decorators import dag, task


DEFAULT_REPO_ROOT = "/Users/ryanchung/Desktop/school/projects/llm-bench"
DEFAULT_CONFIGS = [
    "base",
    "base_strict",
    "rag3",
    "rag3_strict",
    "rag6",
    "rag6_strict",
]


def _repo_root() -> str:
    return os.environ.get("LLM_BENCH_ROOT", DEFAULT_REPO_ROOT)


def _task_env() -> Dict[str, str]:
    env = os.environ.copy()
    keys = [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "LLM_JUDGE_MODEL",
        "OLLAMA_URL",
    ]
    try:
        from airflow.models import Variable

        for key in keys:
            if not env.get(key):
                value = Variable.get(key, default_var=None)
                if value:
                    env[key] = value
    except Exception:
        # Variable backend may be unavailable in some local setups; keep env-only behavior.
        pass
    return env


def _run_cmd(cmd: List[str], cwd: str, env: Optional[Dict[str, str]] = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n\n"
            f"stdout:\n{proc.stdout}\n\n"
            f"stderr:\n{proc.stderr}"
        )


def _parse_configs() -> List[str]:
    raw = os.environ.get("LLM_BENCH_CONFIGS", "")
    if not raw.strip():
        return DEFAULT_CONFIGS
    return [c.strip() for c in raw.split(",") if c.strip()]


@dag(
    dag_id="llm_bench_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["llm-bench", "evaluation"],
)
def llm_bench_pipeline():
    @task
    def selected_configs() -> List[str]:
        return _parse_configs()

    @task
    def run_single_config(config: str) -> str:
        env = _task_env()
        if config.startswith("rag") and not env.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "RAG config requires OPENAI_API_KEY (used by rag/retrieve.py embeddings). "
                "Set it in scheduler env or Airflow Variable OPENAI_API_KEY."
            )
        cmd = [
        "python", "-m", "bench.run_llama_configs",
        "--config", config,
        "--limit", "5",
        "--overwrite",
        ]
        _run_cmd(cmd, cwd=_repo_root(), env=env)
        return config

    @task
    def score_outputs(configs: List[str]) -> None:
        root = _repo_root()
        env = _task_env()
        if not env.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "score_outputs requires OPENAI_API_KEY for scripts/score_results.py. "
                "Set it in scheduler env or Airflow Variable OPENAI_API_KEY."
            )
        for cfg in configs:
            input_jsonl = f"results/llama_{cfg}.jsonl"
            output_csv = f"results/scored/llama_{cfg}_scored.csv"
            cmd = [
                "python",
                "scripts/score_results.py",
                "--input-jsonl",
                input_jsonl,
                "--output-csv",
                output_csv,
                "--prompts-gold",
                "bench/prompts_gold.csv",
                "--rubric",
                "results/rubric.md",
            ]
            _run_cmd(cmd, cwd=root, env=env)

    @task
    def clean_base_scored() -> None:
        _run_cmd(["python", "scripts/clean_results.py"], cwd=_repo_root())

    @task
    def summarize_scored() -> None:
        _run_cmd(["python", "scripts/summarize_results.py"], cwd=_repo_root())

    @task
    def generate_plots() -> None:
        _run_cmd(["python", "scripts/bar_plots.py"], cwd=_repo_root())

    configs = selected_configs()
    run_tasks = run_single_config.expand(config=configs)
    score = score_outputs(configs)
    clean = clean_base_scored()
    summary = summarize_scored()
    plots = generate_plots()

    run_tasks >> score >> clean >> summary >> plots


dag = llm_bench_pipeline()
