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
- **Feature extraction** — TF-IDF with unigrams and bigrams
- **Model training** — four classifiers compared head-to-head
- **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrices
- **Live inference** — classify any email in real time via the Streamlit app or notebook

---

## Dataset

**Source:** [`ealvaradob/phishing-dataset`](https://huggingface.co/datasets/ealvaradob/phishing-dataset) on HuggingFace (emails configuration)

| Split | Emails |
|-------|--------|
| Total | 3,000 |
| Phishing (label = 1) | 1,484 (49.5%) |
| Legitimate (label = 0) | 1,516 (50.5%) |
| Training | 2,400 (80%) |
| Test | 600 (20%) |

The dataset uses a simple two-column structure: `text` (email body) and `label` (0 = legitimate, 1 = phishing). If HuggingFace is unavailable at runtime, the pipeline automatically falls back to a synthetic dataset with realistic noise and class overlap.

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
 • max_features = 5,000
 • ngram_range  = (1, 2)   ← unigrams + bigrams
 • sublinear_tf = True     ← log scaling
 • min_df = 2
      │
      ▼
 Classifiers
 ────────────
 • Naive Bayes
 • Logistic Regression  ← best performer
 • Linear SVM
 • Random Forest
      │
      ▼
 Prediction + Confidence Score
```

---

## Models & Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** |
| Naive Bayes | 99.83% | 100.00% | 99.66% | 99.83% | 1.0000 |
| Linear SVM | 99.83% | 100.00% | 99.66% | 99.83% | 1.0000 |
| Random Forest | 98.67% | 98.98% | 98.32% | 98.65% | 0.9990 |

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
    Vocabulary size: 4,821

[4/5] Training and evaluating models...
  Naive Bayes           acc=0.9983  f1=0.9983  auc=1.0000
  Logistic Regression   acc=1.0000  f1=1.0000  auc=1.0000
  Linear SVM            acc=0.9983  f1=0.9983  auc=1.0000
  Random Forest         acc=0.9867  f1=0.9865  auc=0.9990

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

The sidebar lets you adjust TF-IDF settings, test split size, and choose which classifier to use for inference — all live.

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
| `matplotlib` / `seaborn` | Visualisations — charts, confusion matrices, ROC curves |
| `datasets` (HuggingFace) | Dataset loading from HuggingFace Hub |
| `streamlit` | Interactive web application |

---

## Authors

| Name | Role |
|------|------|
| **Vishnu Kurnala** | NLP Pipeline · Model Training · Evaluation |
| **Rithvik Resu** | Feature Engineering · Streamlit App · Notebook |

---

*Built as part of a Natural Language Processing classification project.*
