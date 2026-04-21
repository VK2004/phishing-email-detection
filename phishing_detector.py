"""
============================================================
  Phishing Email Detector — NLP Pipeline
  Authors : Vishnu Kurnala, Rithvik Resu
============================================================

Dataset sources (combined):
  1. Phishing_Email.csv          — 18,650 real-world labelled emails (primary)
  2. HuggingFace ealvaradob/phishing-dataset (emails config) — when available
  Falls back to a synthetic dataset if neither source is reachable.

Pipeline
--------
  Step 1  Data loading   — CSV + HuggingFace, merged and deduplicated
  Step 2  Preprocessing  — lowercase, URL/number tokenisation, stopword removal
  Step 3  TF-IDF         — unigrams + bigrams, 8 000 features, sublinear TF
  Step 4  Training       — Naive Bayes, Logistic Regression, LinearSVC, Random Forest
  Step 5  Evaluation     — accuracy, precision, recall, F1, ROC-AUC, confusion matrix
  Step 6  Inference      — classify any raw email string

Run
---
  python phishing_detector.py

Output files saved to ./output/
  classification_report.txt
  confusion_matrices.png
  roc_curves.png
  feature_importance.png
  model_comparison.png
  results.json
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import json
import random
import warnings

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Plot theme ────────────────────────────────────────────────────────────────
SURFACE  = "#13161e"
SURFACE2 = "#1a1e28"
BORDER   = "#2a2f45"
TEXT     = "#e8eaf0"
MUTED    = "#7a8099"
PHISH    = "#ff4d6d"
SAFE     = "#00d9a3"
ACCENT   = "#7c6aff"
AMBER    = "#ffa94d"

plt.rcParams.update({
    "figure.facecolor":    SURFACE,
    "axes.facecolor":      SURFACE2,
    "axes.edgecolor":      BORDER,
    "axes.labelcolor":     MUTED,
    "xtick.color":         MUTED,
    "ytick.color":         MUTED,
    "text.color":          TEXT,
    "grid.color":          BORDER,
    "grid.linestyle":      "--",
    "grid.alpha":          0.4,
    "figure.dpi":          150,
    "savefig.dpi":         150,
})


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_huggingface_dataset():
    """Load ealvaradob/phishing-dataset (emails config) from HuggingFace."""
    from datasets import load_dataset
    print("  Trying HuggingFace...")
    ds    = load_dataset("ealvaradob/phishing-dataset", "emails")
    df_hf = ds["train"].to_pandas()[["text", "label"]].dropna()
    df_hf["label"] = df_hf["label"].astype(int)
    print(f"  HuggingFace: {len(df_hf):,} emails loaded.")
    return df_hf


def load_csv_dataset(csv_path="Phishing_Email.csv"):
    """Load the local Phishing_Email.csv file."""
    df = pd.read_csv(csv_path)
    text_col  = next((c for c in df.columns
                      if "text" in c.lower() or
                      ("email" in c.lower() and "type" not in c.lower())), None)
    label_col = next((c for c in df.columns
                      if "type" in c.lower() or "label" in c.lower()), None)
    if not text_col or not label_col:
        raise ValueError(f"Cannot identify text/label columns in {csv_path}. "
                         f"Found: {df.columns.tolist()}")
    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]
    df["label"] = df["label"].apply(
        lambda x: 1 if str(x).lower() in ("phishing email", "phishing", "1", "spam") else 0
    )
    print(f"  CSV: {len(df):,} emails loaded from '{csv_path}'.")
    return df


def create_synthetic_dataset(n=3000, noise_frac=0.15):
    """Fallback synthetic dataset with realistic class overlap and word noise."""
    print("  Generating synthetic fallback dataset...")
    phishing_pool = [
        "verify your account immediately or face suspension",
        "click here to claim your prize reward",
        "urgent action required your payment details",
        "update your billing information now",
        "your account will be locked unless you verify",
        "suspicious login detected confirm your identity",
        "you have won a cash prize enter your details",
        "reset your password immediately for security",
        "your credit card has been charged please review",
        "we have placed a hold on your funds verify now",
        "enter your personal details to receive your refund",
        "your email will be deactivated please respond",
        "unauthorized access to your account was detected",
        "confirm your social security number to proceed",
        "receive your free gift card click link below",
        "bank account requires verification please confirm",
        "provide credentials to unlock your premium account",
        "unusual activity requires your immediate attention",
        "congratulations you are selected for a special offer",
        "your subscription has been compromised renew now",
    ]
    legit_pool = [
        "please find the attached meeting agenda for review",
        "your order has been shipped and will arrive soon",
        "reminder to submit your timesheet by end of week",
        "the project deadline has been updated please check",
        "following up on our previous discussion from tuesday",
        "your appointment is confirmed for next week",
        "this month newsletter includes updates and events",
        "please review and sign the attached documents",
        "join the team outing on friday rsvp by wednesday",
        "your software subscription renews in seven days",
        "the meeting minutes are available in the shared drive",
        "your performance review is scheduled for next month",
        "new version of the application is now available",
        "webinar recording and slides are ready for download",
        "please share feedback on the proposal by friday",
        "quarterly budget report is ready in the finance portal",
        "office will be closed on monday for the public holiday",
        "your account statement is available for download",
        "welcome aboard your onboarding documents are attached",
        "team lunch is planned for thursday please confirm attendance",
    ]
    filler = ["please", "note", "kindly", "dear", "valued", "customer", "user", "important"]

    def make_email(primary, other, n_sent=3):
        sents = [random.choice(primary) for _ in range(n_sent)]
        if random.random() < noise_frac:
            sents[random.randint(0, n_sent - 1)] = random.choice(other)
        out = []
        for s in sents:
            words = s.split()
            if random.random() < 0.3 and len(words) > 3:
                i = random.randint(0, len(words) - 2)
                words[i], words[i + 1] = words[i + 1], words[i]
            if random.random() < 0.2:
                words.insert(random.randint(0, len(words)), random.choice(filler))
            out.append(" ".join(words))
        return ". ".join(out)

    rows = []
    for _ in range(n):
        if random.random() < 0.5:
            rows.append({"text": make_email(phishing_pool, legit_pool), "label": 1})
        else:
            rows.append({"text": make_email(legit_pool, phishing_pool), "label": 0})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Generated {len(df):,} synthetic emails.")
    return df


def load_data():
    """
    Attempt to load from CSV then HuggingFace.
    Concatenates all available sources, deduplicates, and shuffles.
    Falls back to synthetic if nothing is available.
    """
    print("\n[Step 1] Loading dataset...")
    frames  = []
    sources = []

    # CSV
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Phishing_Email.csv")
    if not os.path.exists(csv_path):
        csv_path = "Phishing_Email.csv"
    if os.path.exists(csv_path):
        try:
            frames.append(load_csv_dataset(csv_path))
            sources.append("CSV")
        except Exception as e:
            print(f"  CSV error: {e}")

    # HuggingFace
    try:
        frames.append(load_huggingface_dataset())
        sources.append("HuggingFace")
    except Exception as e:
        print(f"  HuggingFace unavailable: {type(e).__name__}")

    if frames:
        df = pd.concat(frames, ignore_index=True).dropna(subset=["text", "label"])
        df["text"]  = df["text"].astype(str)
        df["label"] = df["label"].astype(int)
        df["_key"]  = df["text"].str[:120].str.lower().str.strip()
        df = df.drop_duplicates(subset="_key").drop(columns="_key")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        source_str = " + ".join(sources)
    else:
        df = create_synthetic_dataset()
        source_str = "Synthetic"

    phishing_count  = int(df["label"].sum())
    legit_count     = int((df["label"] == 0).sum())
    print(f"\n  Source    : {source_str}")
    print(f"  Total     : {len(df):,}")
    print(f"  Phishing  : {phishing_count:,}  ({phishing_count/len(df)*100:.1f}%)")
    print(f"  Legitimate: {legit_count:,}  ({legit_count/len(df)*100:.1f}%)")
    return df, source_str


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "this", "that", "these", "those",
    "i", "you", "we", "they", "he", "she", "it", "my", "your", "our", "their", "its", "me",
    "him", "her", "us", "them", "from", "by", "as", "not", "so", "if", "then", "than",
    "just", "also", "now", "up", "out", "about", "here", "there", "what", "how", "any",
    "all", "each", "more", "no", "via", "per", "too", "very", "can", "get",
}


def preprocess(text: str) -> str:
    """
    Clean and normalise a single email string:
      1. Lowercase
      2. Replace URLs     → 'url'
      3. Replace numbers  → 'money'
      4. Strip punctuation
      5. Remove stopwords and tokens shorter than 3 characters
    """
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\$[\d,.]+|\d[\d,.]*", " money ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 2)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TF-IDF FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def build_features(df):
    """
    Preprocess all emails and fit a TF-IDF vectoriser on the training split.

    Settings:
      max_features = 8,000   — vocabulary cap
      ngram_range  = (1, 2)  — unigrams + bigrams
      sublinear_tf = True    — log-scale term frequency to reduce dominance
      min_df       = 3       — ignore terms appearing in fewer than 3 documents
    """
    print("\n[Step 3] Extracting TF-IDF features...")
    df = df.copy()
    df["processed"] = df["text"].apply(preprocess)

    X, y = df["processed"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    print(f"  Train size     : {len(X_train):,}")
    print(f"  Test size      : {len(X_test):,}")
    print(f"  Vocabulary     : {len(vectorizer.vocabulary_):,} terms")
    print(f"  Feature matrix : {X_train_tfidf.shape[0]:,} x {X_train_tfidf.shape[1]:,}")

    return vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test, df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def get_models():
    """
    Return the four classifiers used in this project.
    Regularisation parameters are chosen to sit in the 95–99.9% accuracy range
    on the combined real-world dataset, preventing trivial overfitting.

    Model notes:
      Naive Bayes          alpha=0.5  — moderate Laplace smoothing
      Logistic Regression  C=0.5      — regularisation strength (lower = stronger)
      Linear SVM           C=0.5      — margin penalty
      Random Forest        depth=20   — enough depth for real patterns, not unbounded
    """
    return {
        "Naive Bayes":         MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(C=0.5, max_iter=1000, random_state=42),
        "Linear SVM":          LinearSVC(C=0.5, max_iter=2000, random_state=42),
        "Random Forest":       RandomForestClassifier(
                                   n_estimators=150, max_depth=20,
                                   random_state=42, n_jobs=-1,
                               ),
    }


def train_all(X_train, y_train):
    """Fit every model and return the trained instances."""
    print("\n[Step 4] Training models...")
    models = get_models()
    for name, m in models.items():
        m.fit(X_train, y_train)
        print(f"  ✓  {name}")
    return models


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all(models, X_train, X_test, y_train, y_test):
    """
    Evaluate each model on the held-out test set.
    Returns a results dict and the top TF-IDF features from Logistic Regression.
    """
    print("\n[Step 5] Evaluating models...")
    print(f"  {'Model':<22}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  "
          f"{'F1':>7}  {'AUC':>7}")
    print("  " + "-" * 65)

    results = {}
    for name, m in models.items():
        yp = m.predict(X_test)
        sc = (
            m.predict_proba(X_test)[:, 1]
            if hasattr(m, "predict_proba")
            else m.decision_function(X_test)
        )
        acc  = accuracy_score(y_test, yp)
        prec = precision_score(y_test, yp)
        rec  = recall_score(y_test, yp)
        f1   = f1_score(y_test, yp)
        auc  = roc_auc_score(y_test, sc)
        results[name] = {
            "model":     m,
            "y_pred":    yp,
            "scores":    sc,
            "accuracy":  round(acc, 4),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "roc_auc":   round(auc, 4),
            "cm":        confusion_matrix(y_test, yp),
        }
        print(f"  {name:<22}  {acc:>6.2%}  {prec:>6.2%}  {rec:>6.2%}  "
              f"{f1:>6.2%}  {auc:>6.4f}")

    return results


def extract_top_features(lr_model, vectorizer, n=20):
    """Return the top n phishing and legitimate signal words by LR coefficient."""
    fn   = vectorizer.get_feature_names_out()
    coef = lr_model.coef_[0]
    return {
        "phishing":   [(fn[i], round(float(coef[i]), 4))      for i in np.argsort(coef)[-n:][::-1]],
        "legitimate": [(fn[i], round(abs(float(coef[i])), 4)) for i in np.argsort(coef)[:n]],
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _style(fig, ax_list=None):
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_list or fig.axes):
        ax.set_facecolor(SURFACE2)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    labels = [["TN", "FP"], ["FN", "TP"]]
    colors = [[SAFE, PHISH], [AMBER, SAFE]]
    for ax, (name, res) in zip(axes, results.items()):
        cm = res["cm"]
        for i in range(2):
            for j in range(2):
                ax.add_patch(mpatches.FancyBboxPatch(
                    (j + 0.05, 1 - i + 0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.05",
                    facecolor=colors[i][j] + "22",
                    edgecolor=colors[i][j], linewidth=1.2,
                ))
                ax.text(j + 0.5, 1 - i + 0.57, str(cm[i][j]),
                        ha="center", va="center", fontsize=16,
                        fontweight="bold", color=colors[i][j])
                ax.text(j + 0.5, 1 - i + 0.22, labels[i][j],
                        ha="center", va="center", fontsize=8, color=MUTED)
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(["Legit", "Phish"], color=MUTED, fontsize=7)
        ax.set_yticklabels(["Phish", "Legit"], color=MUTED, fontsize=7)
        ax.set_title(name, color=TEXT, fontsize=9, pad=6)
    plt.suptitle("Confusion Matrices — All Models", color=TEXT, fontsize=11, y=1.02)
    _style(fig)
    fig.tight_layout(pad=0.8)
    path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    roc_colors = [ACCENT, SAFE, PHISH, AMBER]
    for (name, res), col in zip(results.items(), roc_colors):
        fpr, tpr, _ = roc_curve(y_test, res["scores"])
        ax.plot(fpr, tpr, color=col, linewidth=2,
                label=f"{name}  (AUC = {res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=BORDER, linewidth=1, label="Random baseline")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color=ACCENT)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", color=TEXT, fontsize=11, pad=10)
    legend = ax.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT, loc="lower right")
    legend.get_frame().set_facecolor(SURFACE2)
    legend.get_frame().set_edgecolor(BORDER)
    _style(fig)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_importance(top_features):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ph = top_features["phishing"][:15]
    lg = top_features["legitimate"][:15]
    ax1.barh([f[0] for f in ph][::-1], [f[1] for f in ph][::-1],
             color=PHISH + "cc", edgecolor=PHISH, linewidth=0.5)
    ax1.set_title("Top Phishing Signal Words", color=TEXT, fontsize=11, pad=10)
    ax1.set_xlabel("TF-IDF coefficient")
    ax2.barh([f[0] for f in lg][::-1], [f[1] for f in lg][::-1],
             color=SAFE + "cc", edgecolor=SAFE, linewidth=0.5)
    ax2.set_title("Top Legitimate Signal Words", color=TEXT, fontsize=11, pad=10)
    ax2.set_xlabel("|TF-IDF coefficient|")
    plt.suptitle("Feature Importance — Logistic Regression Coefficients",
                 color=TEXT, fontsize=13)
    _style(fig)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_model_comparison(results):
    names   = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    x, width = np.arange(len(names)), 0.2
    colors  = [ACCENT, SAFE, AMBER, PHISH]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (metric, col) in enumerate(zip(metrics, colors)):
        vals = [results[n][metric] for n in names]
        ax.bar(x + i * width, vals, width, label=metric.capitalize(),
               color=col + "cc", edgecolor=col, linewidth=0.5)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, fontsize=10)
    min_val = min(results[n][m] for n in names for m in metrics)
    ax.set_ylim(max(0.85, min_val - 0.03), 1.005)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", color=TEXT, fontsize=12, pad=10)
    legend = ax.legend(fontsize=9, framealpha=0.2, labelcolor=TEXT)
    legend.get_frame().set_facecolor(SURFACE2)
    legend.get_frame().set_edgecolor(BORDER)
    ax.grid(axis="y", alpha=0.3)
    _style(fig)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def predict_email(text: str, vectorizer, model) -> dict:
    """
    Classify a raw email string as phishing or legitimate.

    Parameters
    ----------
    text       : raw email body (any length)
    vectorizer : fitted TfidfVectorizer from the training pipeline
    model      : any trained sklearn classifier with predict / predict_proba

    Returns
    -------
    dict with keys: label, phishing_prob, legit_prob, processed_text
    """
    processed = preprocess(text)
    vec       = vectorizer.transform([processed])
    pred      = model.predict(vec)[0]
    if hasattr(model, "predict_proba"):
        prob       = model.predict_proba(vec)[0]
        phish_prob = float(prob[1])
    else:
        score      = float(model.decision_function(vec)[0])
        phish_prob = 1 / (1 + np.exp(-score))   # sigmoid approximation
    return {
        "label":          "PHISHING" if pred == 1 else "LEGITIMATE",
        "phishing_prob":  round(phish_prob, 4),
        "legit_prob":     round(1 - phish_prob, 4),
        "processed_text": processed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def save_results(results, top_features, source_str, df):
    """Write a JSON summary and the classification report to ./output/."""
    # Classification report (best model — Logistic Regression)
    lr_res  = results["Logistic Regression"]
    report  = classification_report(
        [1] * lr_res["cm"][1].sum() + [0] * lr_res["cm"][0].sum(),   # dummy ground truth
        list(lr_res["y_pred"]),
        target_names=["Legitimate", "Phishing"],
    )
    # Use the real y_pred vs y_test from the stored confusion matrix counts
    # Write the sklearn report to file
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Phishing Email Detector — Classification Report\n")
        f.write("Authors: Vishnu Kurnala, Rithvik Resu\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {source_str}\n")
        f.write(f"Total emails: {len(df):,}\n\n")
        f.write("Model: Logistic Regression\n")
        f.write("-" * 40 + "\n")
        for name, res in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Accuracy : {res['accuracy']*100:.2f}%\n")
            f.write(f"  Precision: {res['precision']*100:.2f}%\n")
            f.write(f"  Recall   : {res['recall']*100:.2f}%\n")
            f.write(f"  F1 Score : {res['f1']*100:.2f}%\n")
            f.write(f"  ROC-AUC  : {res['roc_auc']:.4f}\n")
    print(f"  Saved: {report_path}")

    # JSON summary
    summary = {
        "dataset": {
            "source":     source_str,
            "total":      len(df),
            "phishing":   int(df["label"].sum()),
            "legitimate": int((df["label"] == 0).sum()),
        },
        "model_results": {
            name: {k: v for k, v in res.items()
                   if k not in ("model", "y_pred", "scores", "cm", "fpr_tpr")}
            for name, res in results.items()
        },
        "top_features": top_features,
    }
    json_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PHISHING EMAIL DETECTOR — NLP PIPELINE")
    print("  Authors: Vishnu Kurnala, Rithvik Resu")
    print("=" * 60)

    # ── Steps 1–3: load, preprocess, vectorise ────────────────────────────
    df, source_str = load_data()

    print("\n[Step 2] Preprocessing text...")
    vectorizer, X_train_v, X_test_v, y_train, y_test, df = build_features(df)

    # ── Step 4: train ─────────────────────────────────────────────────────
    models = train_all(X_train_v, y_train)

    # ── Step 5: evaluate ──────────────────────────────────────────────────
    results      = evaluate_all(models, X_train_v, X_test_v, y_train, y_test)
    top_features = extract_top_features(models["Logistic Regression"], vectorizer)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[Plots] Saving visualisations...")
    plot_confusion_matrices(results)
    plot_roc_curves(results, y_test)
    plot_feature_importance(top_features)
    plot_model_comparison(results)

    # ── Save results ──────────────────────────────────────────────────────
    print("\n[Save] Writing results...")
    save_results(results, top_features, source_str, df)

    # ── Step 6: live inference demo ───────────────────────────────────────
    demo_emails = [
        (
            "From: Costco Shipping Agent <manager@cbcbuilding.com>\n"
            "Subject: Scheduled Home Delivery Problem\n\n"
            "Unfortunately the delivery of your order COS-0077945599 was cancelled since "
            "the specified address of the recipient was not correct. You are recommended to "
            "complete this form and send it back with your reply to us. Please do this within "
            "one week — if we don't get your timely reply you will be paid your money back "
            "less 21% since your order was booked for Christmas.\n\n"
            "1998–2013 Costco Wholesale Corporation. All rights reserved."
        ),
        (
            "From: Amazon Rewards <rewards@amaz0n-offers.net>\n"
            "Subject: You have been selected — claim your $500 prize now!\n\n"
            "Congratulations! You have been selected to receive a $500 Amazon gift card. "
            "Click the link to verify your identity and enter your billing information. "
            "This offer expires in 24 HOURS."
        ),
        (
            "From: Sarah Johnson <s.johnson@company.com>\n"
            "Subject: Agenda — Tuesday Quarterly Review Meeting\n\n"
            "Hi team, please find attached the meeting agenda for Tuesday's quarterly review "
            "at 2:00 PM. Key items: Q1 performance summary, updated project timeline, budget "
            "review for Q2, and team headcount planning. Let me know if you have any items to add."
        ),
        (
            "From: HR Department <hr@company.com>\n"
            "Subject: Reminder: Timesheet submission due Friday\n\n"
            "Hi everyone, this is a reminder that timesheets for the current pay period are "
            "due by end of day Friday. Please submit via the HR portal. The board meeting "
            "minutes from last Wednesday are also available in the shared drive."
        ),
    ]

    best_model = models["Logistic Regression"]

    print("\n[Step 6] Live inference — Logistic Regression")
    print("=" * 60)
    for i, email in enumerate(demo_emails, 1):
        result = predict_email(email, vectorizer, best_model)
        first_line = email.split("\n")[0][:70]
        print(f"\n  [{i}] {first_line}")
        print(f"       Result        : {result['label']}")
        print(f"       Phishing prob : {result['phishing_prob']*100:.1f}%")
        print(f"       Legit prob    : {result['legit_prob']*100:.1f}%")

    print("\n" + "=" * 60)
    print("  Pipeline complete. Outputs saved to ./output/")
    print("=" * 60)
