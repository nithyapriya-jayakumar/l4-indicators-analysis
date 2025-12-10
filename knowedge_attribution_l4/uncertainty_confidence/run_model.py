import os
import json
import re
import time
import requests
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm

# ============================================================
# 1. Load API keys
# ============================================================
load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")

if not GROQ_KEY:
    raise ValueError("Missing GROQ_API_KEY")
if not DEEPSEEK_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY")

# Groq client for LLaMA
llama_client = Groq(api_key=GROQ_KEY)

# DeepSeek endpoint
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


# ============================================================
# 2. JSON REPAIR — works for BOTH models
# ============================================================
def extract_json(text):
    """Extract valid JSON even if the model adds ```json fencing or extra text."""
    if not isinstance(text, str):
        return {"error": "Output not string", "raw": str(text)}

    # Remove markdown fences
    text = text.replace("```json", "").replace("```", "").strip()

    # Extract the JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group()
        try:
            return json.loads(candidate)
        except:
            return {"error": "JSON parse failed", "raw": candidate}

    return {"error": "No JSON found", "raw": text}


# ============================================================
# 3. SYSTEM PROMPT (same for both models)
# ============================================================
SYSTEM_PROMPT = """
Respond ONLY in JSON with:

{
  "answer": "<string or null>",
  "confidence": <float 0–1>,
  "confidence_label": "<low|medium|high>",
  "rationale": "<short explanation>"
}

Rules:
- unanswerable → answer=null, low confidence
- ambiguous → hedge with medium/low confidence
- factual → answer with appropriate confidence

confidence_label:
  low <0.33
  medium 0.33–0.66
  high >0.66

Return ONLY JSON.
"""


# ============================================================
# 4. Query LLaMA (Groq)
# ============================================================
def ask_llama(prompt):
    try:
        resp = llama_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        raw = resp.choices[0].message.content
        return extract_json(raw)
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 5. Query DeepSeek
# ============================================================
def ask_deepseek(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_KEY}"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    try:
        r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=40)
        result = r.json()
        raw = result["choices"][0]["message"]["content"]
        return extract_json(raw)
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 6. Build prompt from dataset entry
# ============================================================
def build_prompt(item):
    """Convert dataset entry → model input prompt."""

    cat = item["category"]
    q = item["question"]

    if item["choices"]:
        # Multiple-choice formatting
        choice_str = "\n".join([f"{i}. {c}" for i, c in enumerate(item["choices"])])
    else:
        choice_str = "None"

    prompt = f"""
Category: {cat}
Question: {q}
Choices:
{choice_str}

Return ONLY JSON.
"""
    return prompt.strip()


# ============================================================
# 7. MAIN EVALUATION LOOP
# ============================================================
def run_full_eval():
    # Load your 100-prompt dataset
    with open("uncertainty_dataset_100.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    output_file = open("results_combined.jsonl", "w", encoding="utf-8")

    print("\n🚀 Starting evaluation on 100 prompts...")

    for item in tqdm(dataset, desc="Evaluating prompts"):
        prompt = build_prompt(item)

        llama_out = ask_llama(prompt)
        deep_out = ask_deepseek(prompt)

        record = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "choices": item["choices"],
            "gold_answer_index": item["gold_answer_index"],
            "gold_answer_text": item["gold_answer_text"],
            "is_answerable": item["is_answerable"],
            "model_llama": llama_out,
            "model_deepseek": deep_out
        }

        output_file.write(json.dumps(record) + "\n")
        time.sleep(0.3)  # avoid rate-limit issues

    output_file.close()
    print("\n✅ DONE! Results saved to: results_combined.jsonl\n")


# ============================================================
# Run it!
# ============================================================
if __name__ == "__main__":
    run_full_eval()
