import json
import pandas as pd
import random

# -----------------------------
# Paths
# -----------------------------
HOTPOTQA_PATH = "hotpot_dev_distractor_v1.json"
OUTPUT_CSV = "citation_prompts.csv"

# -----------------------------
# Settings
# -----------------------------
N_PROMPTS = 100
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# -----------------------------
# Load HotpotQA
# -----------------------------
with open(HOTPOTQA_PATH, "r", encoding="utf8") as f:
    hotpot_data = json.load(f)

questions = [item["question"] for item in hotpot_data]
print(f"Loaded {len(questions)} HotpotQA questions.")

if len(questions) < N_PROMPTS:
    raise ValueError("Not enough HotpotQA questions!")

# -----------------------------
# Sample 100 questions
# -----------------------------
sampled_questions = random.sample(questions, N_PROMPTS)

# -----------------------------
# Build dataset WITHOUT instructions
# -----------------------------
prompts = []
for i, q in enumerate(sampled_questions, start=1):

    prompts.append({
        "id": i,
        "source": "HotpotQA",
        "category": "Multi-hop",
        "prompt_text": q   # <-- ONLY the question
    })

df = pd.DataFrame(prompts)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"Saved 100 prompts to: {OUTPUT_CSV}")
