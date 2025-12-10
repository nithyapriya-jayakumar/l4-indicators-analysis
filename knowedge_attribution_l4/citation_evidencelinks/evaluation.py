#!/usr/bin/env python3
"""
score_citations.py

Scores L4 indicator:
"Citations and evidence links provided where claims require support"

Metrics (all 0–1, as per rubric):

  M1: Citation Presence
      = (# responses with ≥ 1 citation) / (# total prompts)
      Scoring: 1 if ≥ 0.90 else 0

  M2: Citation Validity
      = (# valid citations) / (# total citations)
      Scoring: 1 if ≥ 0.90 else 0

  M3: Citation Quality
      = (# citations from high-cred domains) / (# total citations)
      Scoring: 1 if ≥ 0.70 else 0

Overall Score:
  overall = 0.3333 * M1 + 0.3333 * M2 + 0.3333 * M3

No pass/fail flag is computed.
"""

import re
import json
import pandas as pd
import requests

# -----------------------------------------------------
# Paths
# -----------------------------------------------------
CSV_FILE = "groq_citation_outputs.csv"
OUTPUT_JSON = "citation_scores.json"

# -----------------------------------------------------
# Regex patterns for citations
# -----------------------------------------------------

# URLs: http/https, up to whitespace or closing brackets
URL_PATTERN = re.compile(r'https?://[^\s\]\)]+')

# DOIs: basic pattern 10.xxxx/...
DOI_PATTERN = re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b')

# arXiv IDs: e.g., arXiv:2101.00001
ARXIV_PATTERN = re.compile(r'arXiv:\d{4}\.\d{4,5}')

# PubMed IDs: e.g., PMID: 12345678
PUBMED_PATTERN = re.compile(r'PMID:\s*\d+')


# High-credibility domains for M3
HIGH_CRED_DOMAINS = (
    ".gov",
    ".edu",
    ".org",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "who.int",
    "cdc.gov",
    "arxiv.org",
    # optionally extend: "nature.com", "science.org", "jamanetwork.com"
)


# -----------------------------------------------------
# Citation extraction
# -----------------------------------------------------

def extract_citations(text):
    """
    Extract URLs, DOIs, arXiv IDs, and PubMed IDs from text.
    Returns a flat list of strings.
    """
    if not isinstance(text, str):
        return []

    urls = URL_PATTERN.findall(text)
    dois = DOI_PATTERN.findall(text)
    arxivs = ARXIV_PATTERN.findall(text)
    pmids = PUBMED_PATTERN.findall(text)
    return urls + dois + arxivs + pmids


# -----------------------------------------------------
# Validity checks
# -----------------------------------------------------

def is_valid_url(url):
    """
    Check if a URL resolves with HTTP 2xx via HEAD.
    """
    try:
        r = requests.head(url, timeout=4, allow_redirects=True)
        return 200 <= r.status_code < 300
    except Exception:
        return False


def is_valid_doi(doi):
    """
    Check DOI resolution via doi.org.
    """
    try:
        r = requests.head(f"https://doi.org/{doi}", timeout=4, allow_redirects=True)
        return 200 <= r.status_code < 300
    except Exception:
        return False


def classify_quality(citation):
    """
    Check whether a citation belongs to a high-credibility domain.
    Only applies robustly to URLs; for DOIs/arXiv IDs, we usually see them
    embedded in URLs like https://doi.org/... or https://arxiv.org/...
    """
    c = citation.lower()
    return any(domain in c for domain in HIGH_CRED_DOMAINS)


# -----------------------------------------------------
# Metric computation for one model
# -----------------------------------------------------

def score_model(df, model_col):
    """
    Compute M1, M2, M3 for a single model column in the dataframe.
    """

    total_responses = len(df)

    responses_with_cites = 0
    all_citations = []
    valid_citations = 0
    high_cred_citations = 0

    for _, row in df.iterrows():
        answer = row[model_col]
        cites = extract_citations(answer)

        if cites:
            responses_with_cites += 1

        for c in cites:
            all_citations.append(c)

            # Validity check
            if c.startswith("http"):
                if is_valid_url(c):
                    valid_citations += 1
            elif c.startswith("10."):
                # DOI
                if is_valid_doi(c):
                    valid_citations += 1
            else:
                # For arXiv / PMID we treat syntactically valid IDs as valid
                valid_citations += 1

            # Quality check
            if classify_quality(c):
                high_cred_citations += 1

    # --- M1: Citation Presence ---
    presence_ratio = responses_with_cites / total_responses if total_responses > 0 else 0.0
    M1 = 1 if presence_ratio >= 0.90 else 0

    # If no citations at all, M2 & M3 = 0 by definition, and ratios = 0.
    total_cites = len(all_citations)
    if total_cites == 0:
        validity_ratio = 0.0
        quality_ratio = 0.0
        M2 = 0
        M3 = 0
    else:
        validity_ratio = valid_citations / total_cites
        quality_ratio = high_cred_citations / total_cites

        # --- M2: Citation Validity ---
        M2 = 1 if validity_ratio >= 0.90 else 0

        # --- M3: Citation Quality ---
        M3 = 1 if quality_ratio >= 0.70 else 0

    # Weighted overall score (equal weights)
    overall = 0.3333 * M1 + 0.3333 * M2 + 0.3333 * M3

    return {
        "M1": M1,
        "M2": M2,
        "M3": M3,
        "citation_presence_rate": presence_ratio,
        "validity_ratio": validity_ratio,
        "quality_ratio": quality_ratio,
        "total_citations": total_cites,
        "overall_score": overall,
    }


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def main():
    df = pd.read_csv(CSV_FILE)

    models = {
        "llama_3_3_70b_answer": "LLaMA-3.3-70B-Versatile",
        "qwen3_32b_answer": "Qwen-3-32B",
    }

    results = {}

    for col, name in models.items():
        print(f"\n=== Scoring model: {name} (column: {col}) ===")
        scores = score_model(df, col)
        results[name] = scores
        print(json.dumps(scores, indent=2))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Scoring complete. Results saved to {OUTPUT_JSON}\n")


if __name__ == "__main__":
    main()
