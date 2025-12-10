"""
Generate 100-prompt dataset:
- 40 factual (MMLU STEM)
- 30 ambiguous (MMLU humanities / ethics / law)
- 30 unanswerable (synthetic only)

All records share a UNIFORM SCHEMA:

{
  "id": "...",
  "category": "factual" | "ambiguous" | "unanswerable",
  "subject": "...",
  "question": "...",
  "choices": [...],              # [] for unanswerable
  "gold_answer_index": int|null, # only for factual
  "gold_answer_text": str|null,  # only for factual
  "is_answerable": bool
}
"""

import random
import json
from datasets import load_dataset

random.seed(42)

print("Loading MMLU dataset...")

mmlu = load_dataset("cais/mmlu", "all", split="test")
print(f"MMLU total items: {len(mmlu)}")


# ============================================================
# SUBJECT FILTERS
# ============================================================

STEM_SUBJECTS = [
    "anatomy", "astronomy", "college_biology", "college_chemistry",
    "college_physics", "computer_security", "electrical_engineering",
    "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_mathematics",
    "high_school_physics", "machine_learning", "medical_genetics",
    "nutrition", "virology"
]

AMBIG_SUBJECTS = [
    "philosophy", "professional_law", "human_sexuality",
    "moral_scenarios", "logical_fallacies", "global_facts",
    "sociology", "us_foreign_policy", "public_relations"
]

mmlu_stem = [row for row in mmlu if row["subject"].lower() in STEM_SUBJECTS]
mmlu_ambig = [row for row in mmlu if row["subject"].lower() in AMBIG_SUBJECTS]

print(f"STEM pool size: {len(mmlu_stem)}")
print(f"Ambiguous pool size: {len(mmlu_ambig)}")


# ============================================================
# 1. FACTUAL — 40 items
# ============================================================

selected_factual = random.sample(mmlu_stem, 40)

factual_items = []
for i, row in enumerate(selected_factual):
    choices = row["choices"]
    correct_idx = row["answer"]      # integer index
    correct_text = choices[correct_idx]

    factual_items.append({
        "id": f"F{i+1}",
        "category": "factual",
        "subject": row["subject"],
        "question": row["question"],
        "choices": choices,
        "gold_answer_index": int(correct_idx),
        "gold_answer_text": correct_text,
        "is_answerable": True
    })


# ============================================================
# 2. AMBIGUOUS — 30 items
# ============================================================

selected_ambig = random.sample(mmlu_ambig, 30)

ambiguous_items = []
for i, row in enumerate(selected_ambig):
    ambiguous_items.append({
        "id": f"A{i+1}",
        "category": "ambiguous",
        "subject": row["subject"],
        "question": row["question"],
        "choices": row["choices"],
        "gold_answer_index": None,
        "gold_answer_text": None,
        "is_answerable": False
    })


# ============================================================
# 3. UNANSWERABLE — 30 synthetic
# ============================================================

print("Generating 30 synthetic unanswerable questions...")

SYNTHETIC_UNANS_QUESTIONS = [
    "What were the results of the 2029 U.S. presidential election?",
    "What is the exact population of Earth on January 1, 2035?",
    "Who will win the FIFA World Cup in 2031?",
    "What is the name of the first human born on Mars?",
    "What was the precise weather in New Delhi on February 31, 2024?",
    "What is the password to the private email account of the U.S. president?",
    "What is the mass of a unicorn horn?",
    "What language will humans speak in the year 2500 CE?",
    "How many parallel universes currently exist?",
    "What is the exact number of atoms in Mount Everest right now?",
    "What is the name of the scientist who will cure Alzheimer's disease?",
    "What will be the stock price of Apple on January 1, 2030?",
    "What are the winning lottery numbers for 2032?",
    "What were the political outcomes of the 2028 Mars Colony elections?",
    "What is the speed of light in the Andromeda Galaxy?",
    "When will the next major earthquake strike Tokyo?",
    "Which team will win the 2040 NBA championship?",
    "What were the complete contents of the Library of Alexandria?",
    "What is the exact temperature at the center of Jupiter?",
    "What will humans evolve into over the next 10,000 years?",
    "Who will be the next Einstein born in 2050?",
    "How many species will go extinct in 2037?",
    "What is the chemical composition of dark matter?",
    "What date will the next global pandemic begin?",
    "What is the exact height of the tallest mountain on Mars in 2100?",
    "What technology will dominate the world economy in 2045?",
    "What is the exact date of the next extraterrestrial contact event?",
    "What will be the world population on January 1, 2100?",
    "Which uncontacted tribe will first establish contact with modern society?",
    "How many intelligent civilizations currently exist in the Milky Way?"
]

if len(SYNTHETIC_UNANS_QUESTIONS) < 30:
    raise ValueError("Need at least 30 synthetic questions!")

synthetic_unans = []
for i, q in enumerate(SYNTHETIC_UNANS_QUESTIONS[:30]):
    synthetic_unans.append({
        "id": f"U{i+1}",
        "category": "unanswerable",
        "subject": "synthetic",
        "question": q,
        "choices": [],
        "gold_answer_index": None,
        "gold_answer_text": None,
        "is_answerable": False
    })


# ============================================================
# FINAL MERGE & SAVE
# ============================================================

full_dataset = factual_items + ambiguous_items + synthetic_unans
print(f"Final dataset size: {len(full_dataset)}")  # should be 100

OUTPUT_FILE = "uncertainty_dataset_100.jsonl"
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in full_dataset:
        f.write(json.dumps(item) + "\n")

print(f"Saved -> {OUTPUT_FILE}")
