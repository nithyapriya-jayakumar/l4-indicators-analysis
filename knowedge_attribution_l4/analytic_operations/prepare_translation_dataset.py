from datasets import load_dataset
import json
import random

# -----------------------------------------------------
# Valid OPUS-BOOKS English → X language pairs
# -----------------------------------------------------
language_pairs = [
    ("en-es", "en", "es"),   # Spanish
    ("en-fr", "en", "fr"),   # French
    ("en-ru", "en", "ru"),   # Russian
    ("en-nl", "en", "nl"),   # Dutch
    ("en-fi", "en", "fi"),   # Finnish
]

samples_per_lang = 8
random.seed(42)
all_items = []

for config, src_lang, tgt_lang in language_pairs:
    print(f"\nLoading OPUS-BOOKS config: {config}")

    ds = load_dataset("opus_books", config, split="train")
    print(f"Items in {config}: {len(ds)}")

    sampled = random.sample(list(ds), samples_per_lang)

    for entry in sampled:
        trans = entry["translation"]

        all_items.append({
            "id": f"T{len(all_items)+1:03}",
            "type": "translation",
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "source_text": trans[src_lang],
            "gold_translation": trans[tgt_lang],
        })

# Save JSONL
output = "translation_40.jsonl"
with open(output, "w", encoding="utf-8") as f:
    for row in all_items:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\n✅ Saved translation dataset to {output}")
print("Total items:", len(all_items))
