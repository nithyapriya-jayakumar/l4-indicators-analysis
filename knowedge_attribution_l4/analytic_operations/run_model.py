import os
import json
import time
import requests
from groq import Groq
from dotenv import load_dotenv   # ← ADD THIS



# ======================================================
# 1. Load API Keys
# ======================================================
load_dotenv()  # ← AND THIS
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment variables.")
if not DEEPSEEK_API_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY in environment variables.")

groq_client = Groq(api_key=GROQ_API_KEY)

# ======================================================
# 2. Model Names
# ======================================================
QWEN_MODEL = "qwen/qwen3-32b"
DEEPSEEK_MODEL = "deepseek-reasoner"



# ======================================================
# 3. Load datasets
# ======================================================
def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items

#math_items = load_jsonl("math_40.jsonl")
translation_items = load_jsonl("translation_40.jsonl")
#summarization_items = load_jsonl("summarization_40.jsonl")

# ======================================================
# 4. DeepSeek Official API Call
# ======================================================
def call_deepseek(messages):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# ======================================================
# 5. Qwen (Groq) API Call
# ======================================================
def call_qwen(messages):
    resp = groq_client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ======================================================
# 6. Generic evaluation loop
# ======================================================
def evaluate_model(model_name, model_fn, items, build_prompt_fn, out_path):
    print(f"\n=== Running {model_name} on {len(items)} items ===")
    outputs = []

    for item in items:
        prompt = build_prompt_fn(item)
        messages = [{"role": "user", "content": prompt}]

        try:
            reply = model_fn(messages)
        except Exception as e:
            reply = f"[ERROR] {str(e)}"

        outputs.append({
            "id": item["id"],
            "model_output": reply,
            "gold": item,
        })

        time.sleep(0.3)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved -> {out_path}")


# ======================================================
# 7. Task-specific prompts
# ======================================================
# def math_prompt(item):
#     return (
#         "Solve the math problem. Give ONLY the numeric final answer.\n\n"
#         f"Problem: {item['question']}\n"
#     )

def translation_prompt(item):
    return (
        f"Translate the following text from {item['source_lang']} to {item['target_lang']}.\n\n"
        f"Text: {item['source_text']}"
    )

# def summarization_prompt(item):
#     return (
#         "Summarize the following article in 3–5 sentences. "
#         "Do NOT add extra facts.\n\n"
#         f"Article:\n{item['article']}"
#     )


# ======================================================
# 8. Run Qwen
# ======================================================
# evaluate_model(
#     "Qwen3-32B Math", call_qwen, math_items, math_prompt,
#     "responses_math_qwen.jsonl"
# )

evaluate_model(
    "Qwen3-32B Translation", call_qwen, translation_items, translation_prompt,
    "responses_translation_qwen.jsonl"
)

# evaluate_model(
#     "Qwen3-32B Summarization", call_qwen, summarization_items, summarization_prompt,
#     "responses_summarization_qwen.jsonl"
# )

# ======================================================
# 9. Run DeepSeek
# ======================================================
# evaluate_model(
#     "DeepSeek Math", call_deepseek, math_items, math_prompt,
#     "responses_math_deepseek.jsonl"
# )

evaluate_model(
    "DeepSeek Translation", call_deepseek, translation_items, translation_prompt,
    "responses_translation_deepseek.jsonl"
)

# evaluate_model(
#     "DeepSeek Summarization", call_deepseek, summarization_items, summarization_prompt,
#     "responses_summarization_deepseek.jsonl"
# )


print("\n=== ALL DONE ===")
