"""
Phishing Email Detector — NLP Pipeline
=======================================
Dataset : HuggingFace ealvaradob/phishing-dataset (emails config)
          Falls back to synthetic dataset when network is unavailable.

Pipeline:
  1. Data loading  (HuggingFace datasets library)
  2. Text preprocessing  (regex, stopword removal)
  3. Feature extraction  (TF-IDF, unigrams + bigrams)
  4. Model training      (Naive Bayes, Logistic Regression, LinearSVC, Random Forest)
  5. Evaluation          (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
  6. Inference           (predict any email text)
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import re
import json
import random
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
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
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_huggingface_dataset():
    """
    Load the ealvaradob/phishing-dataset from HuggingFace.
    Returns a pandas DataFrame with columns: text, label (0=legit, 1=phishing).
    """
    from datasets import load_dataset
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("ealvaradob/phishing-dataset", "emails")
    df = ds["train"].to_pandas()
    # Ensure column names are standard
    df = df.rename(columns={"text": "text", "label": "label"})
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    print(f"  Loaded {len(df):,} emails from HuggingFace.")
    return df


def create_synthetic_dataset(n=3000, noise_frac=0.15):
    """
    Fallback: generate a realistic synthetic email dataset with intentional
    class overlap and word-level noise to produce meaningful metrics.
    """
    print("HuggingFace unavailable — generating synthetic dataset...")

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
    filler = ["please","note","kindly","dear","valued","customer","user","important"]

    def make_email(primary_pool, other_pool, n_sent=3):
        sents = [random.choice(primary_pool) for _ in range(n_sent)]
        if random.random() < noise_frac:
            sents[random.randint(0, n_sent - 1)] = random.choice(other_pool)
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


def load_dataset_safe():
    """Try HuggingFace, fall back to synthetic."""
    try:
        return load_huggingface_dataset()
    except Exception as e:
        print(f"  HuggingFace error: {e}")
        return create_synthetic_dataset()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","have","has","had","do","does","did",
    "will","would","could","should","may","might","this","that","these","those",
    "i","you","we","they","he","she","it","my","your","our","their","its","me",
    "him","her","us","them","from","by","as","not","so","if","then","than",
    "just","also","now","up","out","about","here","there","what","how","any",
    "all","each","more","no","its","via","per","too","very","can","get",
}

def preprocess_text(text: str) -> str:
    """
    Clean and normalise a single email string.
    Steps:
      - Lowercase
      - Replace URLs with token <url>
      - Replace monetary amounts with <money>
      - Strip punctuation and digits
      - Remove stopwords and short tokens (< 3 chars)
    """
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\$[\d,.]+ | \d[\d,.]*", " money ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE EXTRACTION  (TF-IDF)
# ══════════════════════════════════════════════════════════════════════════════

def build_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2)):
    """
    Return a configured TfidfVectorizer.
    - sublinear_tf=True  → apply log(1+tf) to dampen high-frequency terms
    - ngram_range=(1,2)  → capture unigrams AND bigrams
    - min_df=2           → ignore terms that appear only once
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_models():
    """Return dict of model_name → sklearn estimator."""
    return {
        "Naive Bayes":          MultinomialNB(alpha=1.5),
        "Logistic Regression":  LogisticRegression(C=0.3, max_iter=1000, random_state=42),
        "Linear SVM":           LinearSVC(C=0.3, max_iter=2000, random_state=42),
        "Random Forest":        RandomForestClassifier(
                                    n_estimators=200, max_depth=8, random_state=42
                                ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Fit model, return dict of metrics + predictions."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probability scores (for ROC-AUC)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)

    return {
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "precision":        round(precision_score(y_test, y_pred), 4),
        "recall":           round(recall_score(y_test, y_pred), 4),
        "f1":               round(f1_score(y_test, y_pred), 4),
        "roc_auc":          round(roc_auc_score(y_test, scores), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def run_pipeline(df):
    """End-to-end pipeline. Returns full results dict."""

    # -- Preprocess
    print("\n[2] Preprocessing text...")
    df["processed"] = df["text"].apply(preprocess_text)

    # -- Split
    X, y = df["processed"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # -- TF-IDF
    print("\n[3] Extracting TF-IDF features...")
    vectorizer = build_tfidf_vectorizer()
    X_train_v  = vectorizer.fit_transform(X_train)
    X_test_v   = vectorizer.transform(X_test)
    print(f"    Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # -- Train & evaluate all models
    print("\n[4/5] Training and evaluating models...")
    models  = get_models()
    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_train_v, X_test_v, y_train, y_test)
        results[name] = metrics
        print(f"  {name:22s}  acc={metrics['accuracy']:.4f}  "
              f"f1={metrics['f1']:.4f}  auc={metrics['roc_auc']:.4f}")

    # -- Top TF-IDF features (from Logistic Regression coefficients)
    lr   = models["Logistic Regression"]
    fn   = vectorizer.get_feature_names_out()
    coef = lr.coef_[0]
    top_features = {
        "phishing":   [(fn[i], round(float(coef[i]), 3)) for i in np.argsort(coef)[-20:][::-1]],
        "legitimate": [(fn[i], round(float(coef[i]), 3)) for i in np.argsort(coef)[:20]],
    }

    # -- ROC curve (Logistic Regression)
    lr_scores    = lr.predict_proba(X_test_v)[:, 1]
    fpr, tpr, _  = roc_curve(y_test, lr_scores)
    roc_data     = {
        "fpr": [round(x, 4) for x in fpr.tolist()],
        "tpr": [round(x, 4) for x in tpr.tolist()],
    }

    # -- Sample predictions
    samples = []
    for idx in df.sample(10, random_state=7).index:
        vec  = vectorizer.transform([df.loc[idx, "processed"]])
        prob = lr.predict_proba(vec)[0]
        t    = df.loc[idx, "text"]
        samples.append({
            "text":          t[:140] + ("..." if len(t) > 140 else ""),
            "true_label":    int(df.loc[idx, "label"]),
            "pred_label":    int(lr.predict(vec)[0]),
            "phishing_prob": round(float(prob[1]), 3),
        })

    return {
        "dataset_info": {
            "total":      len(df),
            "phishing":   int(df["label"].sum()),
            "legitimate": int((df["label"] == 0).sum()),
            "train_size": len(X_train),
            "test_size":  len(X_test),
        },
        "model_results":     results,
        "top_features":      top_features,
        "roc_data":          roc_data,
        "sample_predictions": samples,
        "_vectorizer":       vectorizer,   # kept in memory for inference
        "_lr_model":         lr,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — LIVE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def predict_email(text: str, vectorizer, model) -> dict:
    """
    Predict whether a raw email string is phishing or legitimate.
    Returns probability and label.
    """
    processed = preprocess_text(text)
    vec       = vectorizer.transform([processed])
    prob      = model.predict_proba(vec)[0]
    label     = model.predict(vec)[0]
    return {
        "label":          "PHISHING" if label == 1 else "LEGITIMATE",
        "phishing_prob":  round(float(prob[1]), 4),
        "legit_prob":     round(float(prob[0]), 4),
        "processed_text": processed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PHISHING EMAIL DETECTOR — NLP PIPELINE")
    print("=" * 60)

    # 1. Load
    print("\n[1] Loading dataset...")
    df = load_dataset_safe()
    print(f"    Phishing: {df['label'].sum():,}  |  Legitimate: {(df['label']==0).sum():,}")

    # 2-5. Full pipeline
    output = run_pipeline(df)

    # Save results (without sklearn objects)
    save_output = {k: v for k, v in output.items() if not k.startswith("_")}
    with open("results.json", "w") as f:
        json.dump(save_output, f, indent=2)
    print("\n[✓] Results saved to results.json")

    # 6. Demo inference
    print("\n[6] Sample predictions:")
    print("-" * 60)
    demo_emails = [
        "URGENT: Your account has been compromised. Click here to verify your identity immediately.",
        "Hi team, please find the attached agenda for Tuesday's quarterly review meeting.",
    ]
    for email in demo_emails:
        result = predict_email(email, output["_vectorizer"], output["_lr_model"])
        print(f"  Email : {email[:80]}...")
        print(f"  Label : {result['label']}  (phishing prob: {result['phishing_prob']:.2%})")
        print()

    print("Pipeline complete.")
