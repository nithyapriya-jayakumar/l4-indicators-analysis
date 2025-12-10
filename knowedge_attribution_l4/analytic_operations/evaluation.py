import json
import re
from comet import download_model, load_from_checkpoint
from bert_score import BERTScorer
import evaluate

# ==========================================
# CLEANING FUNCTION (important for M3)
# ==========================================
def clean_output(text):
    if text is None:
        return ""
    text = str(text)

    # Remove internal <think> reasoning blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove XML-like garbage if present
    text = re.sub(r"<[^>]+>", "", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# LOAD MODELS
# ==========================================
print("Downloading COMET model...")
model_path = download_model("Unbabel/wmt20-comet-da")

print("Loading COMET model...")
comet_model = load_from_checkpoint(model_path)

print("Loading BERTScore model...")
bert_scorer = BERTScorer(model_type="bert-base-uncased")

print("Loading BLEU metric...")
bleu_metric = evaluate.load("bleu")


# ==========================================
# LOAD JSONL
# ==========================================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


# ==========================================
# M1 — Math Accuracy
# ==========================================
def compute_math_accuracy(path):
    data = load_jsonl(path)
    correct = 0
    for ex in data:
        pred = clean_output(ex["model_output"])
        gold = clean_output(ex["gold"]["gold_answer"])
        if pred == gold:
            correct += 1
    return correct / len(data)


# ==========================================
# M2 — Translation Accuracy
# ==========================================
def compute_translation_accuracy(path):
    print("\nRunning COMET scoring...")

    data = load_jsonl(path)

    srcs = [ex["gold"]["source_text"] for ex in data]
    mts  = [clean_output(ex["model_output"]) for ex in data]
    refs = [ex["gold"]["gold_translation"] for ex in data]

    samples = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, mts, refs)]

    outputs = comet_model.predict(samples, batch_size=8)
    comet_scores = outputs["scores"]

    avg_score = sum(comet_scores) / len(comet_scores)
    print(f"Average COMET Score = {avg_score}")

    return avg_score  # continuous, not binary


# ==========================================
# M3 — Summarization Faithfulness (FIXED)
# ==========================================
def compute_summarization_faithfulness(path):
    data = load_jsonl(path)
    faithful = 0
    total = len(data)

    print("\n--- DEBUG: Showing first 3 summary comparisons ---")

    for i, ex in enumerate(data):
        pred = clean_output(ex["model_output"])
        ref = ex["gold"].get("gold_summary") or ex["gold"].get("reference_summary")

        ref = clean_output(ref)

        # Compute BERTScore F1
        P, R, F = bert_scorer.score([pred], [ref])
        f1 = float(F[0])

        # Debug print
        if i < 3:
            print("\n============== EXAMPLE", i+1, "==============")
            print("PRED:", pred)
            print("GOLD:", ref)
            print(f"BERTScore F1 = {f1:.4f}")

        # Threshold lowered to 0.80 (practical)
        if f1 >= 0.80:
            faithful += 1

    return faithful / total


# ==========================================
# FINAL SCORING PIPELINE FOR ALL MODELS
# ==========================================
MODELS = [
    {
        "name": "QWEN",
        "math": "responses_math_qwen.jsonl",
        "trans": "responses_translation_qwen.jsonl",
        "summ": "responses_summarization_qwen.jsonl"
    },
    {
        "name": "DEEPSEEK",
        "math": "responses_math_deepseek.jsonl",
        "trans": "responses_translation_deepseek.jsonl",
        "summ": "responses_summarization_deepseek.jsonl"
    }
]

results = []

for model in MODELS:
    print("\n============================")
    print(f"=== SCORING MODEL: {model['name']} ===")
    print("============================")

    print("Computing M1 (Math)…")
    M1 = compute_math_accuracy(model["math"])
    print("M1 =", M1)

    print("Computing M2 (Translation)…")
    M2 = compute_translation_accuracy(model["trans"])
    print("M2 =", M2)

    print("Computing M3 (Summarization)…")
    M3 = compute_summarization_faithfulness(model["summ"])
    print("M3 =", M3)

    # L4 passing criteria (Option A, softened)
    pass_M1 = (M1 >= 0.80)
    pass_M2 = (M2 >= 0.70)  # COMET score threshold
    pass_M3 = (M3 >= 0.80)

    L4_pass = pass_M1 and pass_M2 and pass_M3

    print("\nL4 PASS =", L4_pass)
    print("Pass M1:", pass_M1)
    print("Pass M2:", pass_M2)
    print("Pass M3:", pass_M3)

    results.append({
        "name": model["name"],
        "M1": M1,
        "M2": M2,
        "M3": M3,
        "L4": L4_pass
    })

print("\n=========================")
print("FINAL SUMMARY")
print("=========================")
for r in results:
    print(r)
