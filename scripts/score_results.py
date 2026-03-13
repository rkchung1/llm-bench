import argparse
import csv
import json
import os
import re
from typing import Any

import requests

DEFAULT_OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))

VALID_LABELS = {
    "correct": "Correct",
    "incorrect": "Incorrect",
    "hallucinated": "Hallucinated",
    "refused": "Refused",
}


def load_rubric(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_gold(path: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "best_answer", "correct_answers", "incorrect_answers"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            pid = (row.get("id") or "").strip()
            if pid:
                out[pid] = row
    return out


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def call_judge(
    openai_base_url: str,
    openai_api_key: str,
    judge_model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    endpoint = openai_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    r = requests.post(endpoint, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"OpenAI judge returned no choices: {data}")
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


def normalize_label(value: str) -> str:
    key = (value or "").strip().lower()
    return VALID_LABELS.get(key, "")


def parse_judge_response(raw_text: str) -> tuple[str, str]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("empty judge response")

    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"judge did not return JSON object: {text[:200]}")

    label = normalize_label(str(parsed.get("label", "")))
    reasoning = str(parsed.get("reasoning", "")).strip()
    if not label:
        raise ValueError(f"invalid label in judge response: {parsed}")
    if not reasoning:
        reasoning = "No reasoning provided."
    return label, reasoning


def build_prompts(rubric_text: str, question: str, model_answer: str, gold_row: dict[str, str]) -> tuple[str, str]:
    system_prompt = (
        "You are an expert evaluator for an LLM reliability benchmark.\n"
        "Use the rubric and best/correct answers to assign exactly one label.\n"
        "Return ONLY valid JSON: "
        '{"label":"Correct|Incorrect|Hallucinated|Refused","reasoning":"..."}'
    )
    user_prompt = (
        "RUBRIC:\n"
        f"{rubric_text}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "MODEL ANSWER:\n"
        f"{model_answer}\n\n"
        "GOLD BEST ANSWER:\n"
        f"{gold_row.get('best_answer', '')}\n\n"
        "GOLD CORRECT ANSWERS (semicolon-separated):\n"
        f"{gold_row.get('correct_answers', '')}\n\n"
        "GOLD INCORRECT ANSWERS (semicolon-separated):\n"
        f"{gold_row.get('incorrect_answers', '')}\n\n"
        "Respond with JSON only."
    )
    return system_prompt, user_prompt


def score_file(
    input_jsonl: str,
    output_csv: str,
    prompts_gold_csv: str,
    rubric_md: str,
    openai_base_url: str,
    openai_api_key: str,
    judge_model: str,
) -> None:
    rubric_text = load_rubric(rubric_md)
    gold_by_id = load_gold(prompts_gold_csv)
    rows = load_jsonl(input_jsonl)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "rubric_label", "reasoning"])
        writer.writeheader()

        total = len(rows)
        for idx, row in enumerate(rows, start=1):
            pid = str(row.get("prompt_id", "")).strip()
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            gen_error = str(row.get("error", "")).strip()

            if not pid:
                continue

            gold_row = gold_by_id.get(pid)
            if not gold_row:
                label = "Refused"
                reasoning = "Prompt id not found in prompts_gold.csv."
            elif gen_error and not answer:
                label = "Refused"
                reasoning = f"Generation error: {gen_error}"
            else:
                system_prompt, user_prompt = build_prompts(
                    rubric_text=rubric_text,
                    question=question,
                    model_answer=answer,
                    gold_row=gold_row,
                )
                raw_judge = call_judge(
                    openai_base_url=openai_base_url,
                    openai_api_key=openai_api_key,
                    judge_model=judge_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                label, reasoning = parse_judge_response(raw_judge)

            writer.writerow(
                {
                    "prompt_id": pid,
                    "rubric_label": label,
                    "reasoning": reasoning,
                }
            )

            if idx % 10 == 0 or idx == total:
                print(f"[{idx}/{total}] scored")

    print(f"Wrote scored output: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, help="Input config results JSONL")
    parser.add_argument("--output-csv", required=True, help="Output scored CSV path")
    parser.add_argument(
        "--prompts-gold",
        default="bench/prompts_gold.csv",
        help="Path to prompts_gold.csv",
    )
    parser.add_argument(
        "--rubric",
        default="results/rubric.md",
        help="Path to rubric markdown file",
    )
    parser.add_argument(
        "--openai-base-url",
        default=DEFAULT_OPENAI_BASE_URL,
        help="OpenAI-compatible base URL (default: https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (default from OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Judge model name (default from OPENAI_MODEL or LLM_JUDGE_MODEL)",
    )
    args = parser.parse_args()
    if not args.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required (or pass --openai-api-key).")

    score_file(
        input_jsonl=args.input_jsonl,
        output_csv=args.output_csv,
        prompts_gold_csv=args.prompts_gold,
        rubric_md=args.rubric,
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
        judge_model=args.judge_model,
    )


if __name__ == "__main__":
    main()
