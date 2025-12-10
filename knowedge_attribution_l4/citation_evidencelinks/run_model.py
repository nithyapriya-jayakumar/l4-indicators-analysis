import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import time

# =====================================================
# Load API keys
# =====================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ Missing GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

# =====================================================
# System prompt — strict citation requirement
# =====================================================
SYSTEM_MESSAGE = (
    "Answer the question clearly and factually. "
    "You MUST include at least one real, verifiable citation such as a URL, DOI, PubMed ID, or arXiv link. "
    "The citation must directly support a specific fact in your answer. "
    "Do NOT guess or fabricate citations."
)

# =====================================================
# Models to evaluate (Groq hosted)
# =====================================================
MODEL_A = "llama-3.3-70b-versatile"
MODEL_B = "qwen/qwen3-32b"


# =====================================================
# Robust model call with timeout + retries + fallback
# =====================================================
def run_model(model_name, prompt, max_retries=3):
    """
    Runs a Groq-hosted model with:
    - API timeout
    - Retry on errors or empty responses
    - Fallback non-empty message
    """

    for attempt in range(1, max_retries + 1):

        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,   # Deterministic output
                max_tokens=700,
                timeout=20         # Explicit timeout (seconds)
            )

            output = response.choices[0].message.content

            # If model returned something meaningful
            if output and output.strip():
                return output.strip()

            # If output is empty → retry
            print(f"⚠️ Empty response from {model_name}, retrying ({attempt}/{max_retries})...")
        
        except Exception as e:
            print(f"⚠️ Exception in {model_name}, retrying ({attempt}/{max_retries}): {e}")

        # Short backoff
        time.sleep(1)

    # Final fallback (never return empty string)
    return f"❌ ERROR: {model_name} returned no usable output after {max_retries} attempts."


# =====================================================
# Load dataset of HotpotQA questions
# =====================================================
CSV_FILE = "citation_prompts.csv"
df = pd.read_csv(CSV_FILE)

print(f"\nLoaded {len(df)} prompts.\n")

results = []

# =====================================================
# Run both models on each prompt
# =====================================================
for idx, row in df.iterrows():

    prompt_id = row["id"]
    question = row["prompt_text"]

    print("\n============================================")
    print(f"Prompt {prompt_id}   ({idx+1}/{len(df)})")
    print("QUESTION:", question)
    print("============================================\n")

    # Get responses from both models
    output_a = run_model(MODEL_A, question)
    output_b = run_model(MODEL_B, question)

    # Display outputs
    print(f"🟩 Model A ({MODEL_A}) Output:\n{output_a}\n")
    print("--------------------------------------------\n")
    print(f"🟦 Model B ({MODEL_B}) Output:\n{output_b}\n")

    # Save for CSV
    results.append([
        prompt_id,
        question,
        output_a,
        output_b
    ])


# =====================================================
# Save results to CSV for scoring
# =====================================================
OUTPUT_FILE = "citation_outputs.csv"

df_out = pd.DataFrame(
    results,
    columns=[
        "id",
        "question",
        "llama_3_3_70b_answer",
        "qwen3_32b_answer"
    ]
)

df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\n============================================")
print("✓ DONE — All Prompts Completed Successfully!")
print(f"Saved to {OUTPUT_FILE}")
print("============================================\n")
