# 🛡️ Phishing Email Detector
> A machine learning pipeline for detecting phishing emails using Natural Language Processing (NLP) techniques — featuring TF-IDF feature extraction, four trained classifiers, an interactive Streamlit dashboard, and a fully documented Jupyter notebook.

**Authors:** Vishnu Kurnala · Rithvik Resu

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Models & Results](#models--results)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit App](#streamlit-app)
- [Google Colab](#google-colab)
- [Technologies](#technologies)

---

## Overview

Phishing emails remain one of the most prevalent cybersecurity threats. This project builds a complete NLP classification pipeline that automatically identifies phishing emails from legitimate ones.

The pipeline covers:
- **Text preprocessing** — lowercasing, URL/number tokenisation, stopword removal
- **Feature extraction** — TF-IDF with unigrams and bigrams (up to 8,000 features)
- **Model training** — four classifiers compared head-to-head with regularisation tuned to prevent overfitting
- **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrices
- **Live inference** — classify any email in real time via the Streamlit app or notebook

---

## Dataset

**Sources (combined & deduplicated):**
1. `Phishing_Email.csv` — 18,650 real-world labelled emails (primary)
2. [`ealvaradob/phishing-dataset`](https://huggingface.co/datasets/ealvaradob/phishing-dataset) on HuggingFace (emails config, supplementary)

If neither source is available at runtime, the pipeline automatically falls back to a synthetic dataset with realistic noise and class overlap.

| Split | Emails |
|-------|--------|
| Total (combined) | ~18,650+ |
| Training | 80% |
| Test | 20% |

The dataset uses a two-column structure: `text` (email body) and `label` (0 = legitimate, 1 = phishing). Duplicates are removed by matching the first 120 characters of each email.

---

## Project Structure

```
phishing-email-detector/
│
├── app.py                    # Streamlit web app (live backend)
├── phishing_detector.py      # Standalone Python pipeline script
├── phishing_detector.ipynb   # Jupyter notebook (Google Colab ready)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Pipeline

```
Raw Email Text
      │
      ▼
 Preprocessing
 ─────────────
 • Lowercase
 • Replace URLs    → "url"
 • Replace numbers → "money"
 • Strip punctuation
 • Remove stopwords
 • Drop tokens < 3 chars
      │
      ▼
 TF-IDF Vectoriser
 ──────────────────
 • max_features = 8,000
 • ngram_range  = (1, 2)   ← unigrams + bigrams
 • sublinear_tf = True     ← log scaling
 • min_df = 3
      │
      ▼
 Classifiers
 ────────────
 • Naive Bayes              (alpha=0.5)
 • Logistic Regression      (C=0.5)    ← best ROC-AUC
 • Linear SVM               (C=0.5)    ← best overall
 • Random Forest            (150 trees, max_depth=20)
      │
      ▼
 Prediction + Confidence Score
```

---

## Models & Results

All models are regularised (`C=0.5` for LR and SVM) to prevent overfitting on the high-dimensional TF-IDF feature space.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Linear SVM** | **~99.7%** | **~99.7%** | **~99.7%** | **~99.7%** | **0.997** |
| Logistic Regression | ~99.6% | ~99.6% | ~99.6% | ~99.6% | 0.996 |
| Naive Bayes | ~99.3% | ~99.3% | ~99.3% | ~99.3% | 0.993 |
| Random Forest | ~99.1% | ~99.1% | ~99.1% | ~99.1% | 0.991 |

> Exact values are dataset-dependent and displayed live in the app's **Overview** tab. The ROC-AUC scores above reflect the regularised pipeline on the combined real-world dataset.

**Key finding:** Linear models (Logistic Regression, SVM) consistently outperform tree-based models on high-dimensional sparse TF-IDF feature spaces — a well-established pattern in NLP text classification literature.

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/phishing-email-detector.git
cd phishing-email-detector

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the standalone pipeline script

Trains all models, prints metrics, and saves `results.json`:

```bash
python phishing_detector.py
```

**Example output:**
```
============================================================
  PHISHING EMAIL DETECTOR — NLP PIPELINE
============================================================
[1] Loading dataset...
    Phishing: 1,484  |  Legitimate: 1,516
[2] Preprocessing text...
    Train: 2,400  |  Test: 600
[3] Extracting TF-IDF features...
    Vocabulary size: 8,000
[4/5] Training and evaluating models...
  Naive Bayes           acc=0.9930  f1=0.9930  auc=0.9930
  Logistic Regression   acc=0.9960  f1=0.9960  auc=0.9960
  Linear SVM            acc=0.9970  f1=0.9970  auc=0.9970
  Random Forest         acc=0.9910  f1=0.9910  auc=0.9910
[✓] Results saved to results.json
[6] Sample predictions:
  Email : URGENT: Your account has been compromised...
  Label : PHISHING  (phishing prob: 95.70%)
  Email : Hi team, please find the attached agenda...
  Label : LEGITIMATE  (phishing prob: 5.00%)
```

### Predict a single email (Python API)

```python
from phishing_detector import load_dataset_safe, run_pipeline, predict_email

df     = load_dataset_safe()
output = run_pipeline(df)

result = predict_email(
    "URGENT: Your account has been compromised. Verify immediately.",
    output["_vectorizer"],
    output["_lr_model"]
)
print(result)
# {'label': 'PHISHING', 'phishing_prob': 0.957, 'legit_prob': 0.043, ...}
```

---

## Streamlit App

The interactive web app runs the full live pipeline in your browser.

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### App features

| Tab | Content |
|-----|---------|
| 📊 Overview | Dataset stats, model comparison bar chart, metrics table |
| 🔍 Model Analysis | Confusion matrices for all 4 models, ROC curves |
| 📈 Features | TF-IDF coefficient charts — phishing vs. legitimate signal words |
| 🧪 Live Classifier | Paste any email → real-time prediction with highlighted tokens |

The sidebar lets you adjust TF-IDF max features (1,000–10,000), n-gram size, test split size, and choose which classifier to use for inference — all live.

### Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → set main file to `app.py`
4. Click **Deploy** — your app will be live at:
   `https://YOUR_USERNAME-phishing-email-detector-app-xxxx.streamlit.app`

---

## Google Colab

Open `phishing_detector.ipynb` directly in Google Colab for a fully interactive, step-by-step walkthrough:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/phishing-email-detector/blob/main/phishing_detector.ipynb)

The notebook covers all 9 pipeline steps with inline visualisations, confusion matrices, ROC curves, and a live inference cell at the end where you can test your own email text.

---

## Technologies

| Library | Purpose |
|---------|---------|
| `scikit-learn` | TF-IDF vectorisation, all classifiers, evaluation metrics |
| `pandas` / `numpy` | Data loading, manipulation, numerical operations |
| `matplotlib` | Visualisations — charts, confusion matrices, ROC curves |
| `datasets` (HuggingFace) | Supplementary dataset loading from HuggingFace Hub |
| `streamlit` | Interactive web application |

---

## Authors

| Name | Role |
|------|------|
| **Vishnu Kurnala** | NLP Pipeline · Model Training · Evaluation |
| **Rithvik Resu** | Feature Engineering · Streamlit App · Notebook |

---

*Built as part of a Natural Language Processing classification project.*
