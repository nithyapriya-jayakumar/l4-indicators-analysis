# L4 Indicators Analysis  
This repository contains my implementation and evaluation of four L4 ethical AI indicators as part of the AI Ethics Assignment.  
Each indicator is implemented in a separate folder with its own dataset, model execution script, and evaluation pipeline.

---

## 📂 Project Structure

l4-indicators-analysis/
│
├── factuality_hallucination/
├── uncertainty_confidence/
├── knowledge_attribution_l4/
└── citation_evidencelinks/


Each folder includes:
- `run_model.py` – main script for running model inference  
- `evaluation.py` – scoring and evaluation script  
- dataset files (JSON/CSV/JSONL)  
- response/output files  

---

#  L4 Indicators Overview

## 1️ Factuality / Hallucination  
Measures how accurately the model generates information and identifies hallucinated content.  
**Folder:** `factuality_hallucination/`

---

## 2️ Uncertainty & Confidence  
Evaluates whether the model expresses confidence appropriately relative to its output.  
**Folder:** `uncertainty_confidence/`

---

## 3️ Knowledge Attribution (L4)  
Assesses whether the model properly attributes its answers to the correct sources.  
(Example dataset: HotpotQA — not included due to size limits.)  
**Folder:** `knowledge_attribution_l4/`

---

## 4️ Citation & Evidence Links  
Checks whether the model provides evidence-backed citations and links statements to supporting information.  
**Folder:** `citation_evidencelinks/`

---

##  Datasets

Large benchmark datasets such as `hotpot_dev_distractor_v1.json` are **not uploaded** due to GitHub file size limits.  
Download from the official source if needed:

🔗 HotpotQA Dataset: https://github.com/hotpotqa/hotpot

Place datasets into their corresponding indicator folders before running code.

---

##  API Key Setup (.env)

Create a `.env` file in the project root and add the API keys used in this project:

GROQ_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here


The `.env` file is ignored by GitHub for security and must not be uploaded.  
Scripts automatically load these values using `python-dotenv`.

---
## 🧪 Evaluation Metrics (BLEURT & COMET)

Some indicators use advanced text evaluation metrics:

- **BLEURT** – A learned evaluation metric for assessing factuality and semantic similarity.  
- **COMET** – A neural evaluation metric commonly used for translation and reliability scoring.

These are installed automatically through `requirements.txt`.  
Evaluation scripts run BLEURT or COMET only for indicators that require them.

---

##  Running the Code

### Installation
pip install -r requirements.txt


### Run model inference


python run_model.py


### Evaluate results


python evaluation.py


Each indicator folder runs independently and produces its own output files.

---

##  Notes
- `.env` is excluded using `.gitignore` to protect API keys.  
- Large datasets are not included in the repository.  
- Each indicator is structured independently for clarity.

---

## License
This repository is for academic and research use only.
 
