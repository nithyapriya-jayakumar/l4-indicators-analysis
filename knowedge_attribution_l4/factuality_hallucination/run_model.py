#!/usr/bin/env python3
"""
evaluate_models.py
Runs L4 evaluation (TruthfulQA + HaluEval QA) for:
- LLaMA-3.3-70B-Versatile (Groq)
- Mistral Large 3 (OpenRouter)

Outputs ONLY RAW RESPONSES:
  responses/truthfulqa_<model>.csv
  responses/halueval_<model>.csv
  responses/logs/*.jsonl

No BLEURT scoring is done in this file.
"""

import os
import json
from dataclasses import dataclass

import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset

# ============================================================
# INITIALIZATION
# ============================================================

load_dotenv()
SEED = 42

OUTPUT_DIR = "responses"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODELS = [
    ("llama-3.3-70b", "groq", "llama-3.3-70b-versatile"),
    ("mistral-large-3", "openrouter", "mistralai/mistral-large-2407"),
]

# ============================================================
# HELPERS
# ============================================================

def save_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# MODEL CLIENTS
# ============================================================

@dataclass
class GroqClient:
    model_id: str
    api_key: str = GROQ_API_KEY

    def generate(self, system_prompt, user_prompt, max_tokens=256, retries=1):
        from groq import Groq
        client = Groq(api_key=self.api_key)

        for attempt in range(retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                pred = resp.choices[0].message.content.strip()
                if pred:
                    return pred
            except Exception:
                pass

        return "NO_ANSWER_GENERATED"


@dataclass
class OpenRouterClient:
    model_id: str
    api_key: str = OPENROUTER_API_KEY

    def generate(self, system_prompt, user_prompt, max_tokens=256, retries=1):

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }

        for attempt in range(retries + 1):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
                out = resp.json()
                pred = out["choices"][0]["message"]["content"].strip()
                if pred:
                    return pred
            except Exception:
                pass

        return "NO_ANSWER_GENERATED"


# ============================================================
# LOAD DATASETS (unchanged)
# ============================================================

def load_truthfulqa(n=None):
    ds = load_dataset("truthful_qa", "multiple_choice")["validation"].shuffle(SEED)
    items = []
    for ex in ds:
        true_refs = [c for c, l in zip(ex["mc2_targets"]["choices"], ex["mc2_targets"]["labels"]) if l == 1]
        false_refs = [c for c, l in zip(ex["mc2_targets"]["choices"], ex["mc2_targets"]["labels"]) if l == 0]

        if true_refs:
            items.append({
                "question": ex["question"],
                "true_refs": true_refs,
                "false_refs": false_refs,
            })

        if n and len(items) >= n:
            break

    return items


def load_halueval(n=None):
    url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
    lines = requests.get(url, timeout=30).text.splitlines()

    items = []
    for line in lines:
        ex = json.loads(line)
        items.append({
            "knowledge": ex["knowledge"],
            "question": ex["question"],
            "right_answer": ex["right_answer"],
            "hallucinated_answer": ex["hallucinated_answer"],
        })

        if n and len(items) >= n:
            break

    return items


# ============================================================
# EVALUATION CHANNELS — RAW OUTPUT ONLY
# ============================================================

def run_truthfulqa(model_name, client, items):
    out = []
    log_path = f"{LOG_DIR}/truthfulqa_{model_name}.jsonl"

    for ex in tqdm(items, desc=f"TQA ({model_name})"):
        pred = client.generate("Answer truthfully.", ex["question"], retries=1)

        rec = {
            "question": ex["question"],
            "true_refs": ex["true_refs"],
            "false_refs": ex["false_refs"],
            "model_answer": pred,
        }

        out.append(rec)
        save_jsonl(log_path, rec)

    pd.DataFrame(out).to_csv(
        f"{OUTPUT_DIR}/truthfulqa_{model_name}.csv",
        index=False
    )


def run_halueval(model_name, client, items):
    out = []
    log_path = f"{LOG_DIR}/halueval_{model_name}.jsonl"

    for ex in tqdm(items, desc=f"HaluEval ({model_name})"):
        prompt = f"KNOWLEDGE:\n{ex['knowledge']}\n\nQUESTION:\n{ex['question']}"

        pred = client.generate(
            "Answer ONLY using the knowledge provided.",
            prompt,
            retries=1,
        )

        rec = {
            "knowledge": ex["knowledge"],
            "question": ex["question"],
            "right_answer": ex["right_answer"],
            "hallucinated_answer": ex["hallucinated_answer"],
            "model_answer": pred,
        }

        out.append(rec)
        save_jsonl(log_path, rec)

    pd.DataFrame(out).to_csv(
        f"{OUTPUT_DIR}/halueval_{model_name}.csv",
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main(n_tqa=50, n_halu=50):

    tqa_items = load_truthfulqa(n_tqa)
    halu_items = load_halueval(n_halu)

    for name, provider, model_id in MODELS:

        print(f"\n=== Running model: {name} ===")

        if provider == "groq":
            client = GroqClient(model_id)
        else:
            client = OpenRouterClient(model_id)

        run_truthfulqa(name, client, tqa_items)
        run_halueval(name, client, halu_items)

    print("\n✓ Evaluation Complete!")


if __name__ == "__main__":
    main()
