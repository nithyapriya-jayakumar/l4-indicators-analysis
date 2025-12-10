from datasets import load_dataset
import json
import re
import random

# ==========================================================
# 1. Load GSM8K (works in 2025)
# ==========================================================
print("Downloading GSM8K dataset...")
dataset = load_dataset("gsm8k", "main")

data = dataset["test"]   # Use test split

print("Total GSM8K problems:", len(data))

# ==========================================================
# 2. Extract numeric final answers from step-by-step reasoning
# ==========================================================

def extract_final_answer(text):
    # GSM8K final answer pattern: #### 42
    match = re.search(r"####\s*(-?\d+(\.\d+)?)", text)
    if match:
        return match.group(1)
    return None

filtered = []

for item in data:
    q = item["question"]
    raw = item["answer"]
    final = extract_final_answer(raw)

    if final is not None:
        filtered.append({
            "question": q,
            "gold_answer": final
        })

print("Numeric problems with extractable answers:", len(filtered))

# ==========================================================
# 3. Sample 40 items
# ==========================================================
random.seed(42)
sampled = random.sample(filtered, 40)

# ==========================================================
# 4. Write JSONL
# ==========================================================
output_path = "math_40.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for i, item in enumerate(sampled, 1):
        obj = {
            "id": f"M{i:03}",
            "type": "math",
            "question": item["question"],
            "gold_answer": item["gold_answer"]
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ Saved 40-item math dataset to {output_path}")

# Preview
print("\nPreview:")
for i in range(3):
    print(sampled[i])
