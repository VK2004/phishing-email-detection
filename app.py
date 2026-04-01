"""
Phishing Email Detector — Streamlit App
========================================
Run with:  streamlit run app.py

Pipeline:
  1. Data loading  (HuggingFace datasets OR synthetic fallback)
  2. Text preprocessing
  3. TF-IDF feature extraction
  4. Model training  (NB, LR, SVM, RF)
  5. Evaluation & visualisation
  6. Live inference on any email text
"""

import re
import random
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* Metric cards */
  [data-testid="metric-container"] {
      background: #1e2130;
      border: 1px solid #2e3250;
      border-radius: 10px;
      padding: 14px 18px;
  }
  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
      background: #1e2130;
      border-radius: 8px;
      border: 1px solid #2e3250;
      padding: 6px 18px;
  }
  .stTabs [aria-selected="true"] {
      background: #2d2f6b !important;
      border-color: #7c6aff !important;
  }
  /* Header accent */
  .block-container { padding-top: 2rem; }
  h1 { letter-spacing: -0.5px; }
  /* Result box */
  .phish-box {
      background: rgba(255,77,109,0.12);
      border: 1px solid #ff4d6d;
      border-radius: 10px;
      padding: 18px 22px;
      margin-top: 12px;
  }
  .safe-box {
      background: rgba(0,217,163,0.1);
      border: 1px solid #00d9a3;
      border-radius: 10px;
      padding: 18px 22px;
      margin-top: 12px;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STOPWORDS
# ─────────────────────────────────────────────
STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","have","has","had","do","does","did",
    "will","would","could","should","may","might","this","that","these","those",
    "i","you","we","they","he","she","it","my","your","our","their","its","me",
    "him","her","us","them","from","by","as","not","so","if","then","than",
    "just","also","now","up","out","about","here","there","what","how","any",
    "all","each","more","no","via","per","too","very","can","get",
}

# ─────────────────────────────────────────────
# DATA & PIPELINE  (cached so it runs once)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    """Try HuggingFace; fall back to synthetic dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("ealvaradob/phishing-dataset", "emails")
        df = ds["train"].to_pandas()[["text","label"]].dropna()
        df["label"] = df["label"].astype(int)
        return df, "HuggingFace — ealvaradob/phishing-dataset"
    except Exception:
        pass

    # ── Synthetic fallback ──
    random.seed(42); np.random.seed(42)
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

    def make_email(primary, other, n=3):
        sents = [random.choice(primary) for _ in range(n)]
        if random.random() < 0.15:
            sents[random.randint(0, n-1)] = random.choice(other)
        out = []
        for s in sents:
            words = s.split()
            if random.random() < 0.3 and len(words) > 3:
                i = random.randint(0, len(words)-2)
                words[i], words[i+1] = words[i+1], words[i]
            if random.random() < 0.2:
                words.insert(random.randint(0, len(words)), random.choice(filler))
            out.append(" ".join(words))
        return ". ".join(out)

    rows = []
    for _ in range(3000):
        if random.random() < 0.5:
            rows.append({"text": make_email(phishing_pool, legit_pool), "label": 1})
        else:
            rows.append({"text": make_email(legit_pool, phishing_pool), "label": 0})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df, "Synthetic dataset (HuggingFace unavailable)"


def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\$[\d,.]+|\d[\d,.]*", " money ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 2)


@st.cache_resource(show_spinner=False)
def train_models(_df):
    """Train all four classifiers. Cached so they persist across reruns."""
    df = _df.copy()
    df["processed"] = df["text"].apply(preprocess)

    X, y = df["processed"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    models = {
        "Naive Bayes":          MultinomialNB(alpha=1.5),
        "Logistic Regression":  LogisticRegression(C=0.3, max_iter=1000, random_state=42),
        "Linear SVM":           LinearSVC(C=0.3, max_iter=2000, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    }

    results = {}
    for name, m in models.items():
        m.fit(Xtr, y_train)
        yp = m.predict(Xte)
        sc = m.predict_proba(Xte)[:,1] if hasattr(m, "predict_proba") else m.decision_function(Xte)
        results[name] = {
            "model":    m,
            "accuracy":  round(accuracy_score(y_test, yp), 4),
            "precision": round(precision_score(y_test, yp), 4),
            "recall":    round(recall_score(y_test, yp), 4),
            "f1":        round(f1_score(y_test, yp), 4),
            "roc_auc":   round(roc_auc_score(y_test, sc), 4),
            "cm":        confusion_matrix(y_test, yp),
            "fpr_tpr":   roc_curve(y_test, sc)[:2],
        }

    # Top features from Logistic Regression
    lr   = models["Logistic Regression"]
    fn   = vec.get_feature_names_out()
    coef = lr.coef_[0]
    top_features = {
        "phishing":   [(fn[i], float(coef[i])) for i in np.argsort(coef)[-20:][::-1]],
        "legitimate": [(fn[i], abs(float(coef[i]))) for i in np.argsort(coef)[:20]],
    }

    return vec, results, top_features, y_test


# ─────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────
DARK = "#0d0f14"
SURFACE = "#13161e"
SURFACE2 = "#1a1e28"
BORDER = "#2a2f45"
TEXT = "#e8eaf0"
MUTED = "#7a8099"
PHISH = "#ff4d6d"
SAFE = "#00d9a3"
ACCENT = "#7c6aff"
AMBER = "#ffa94d"

def style_fig(fig, ax_list=None):
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_list or fig.axes):
        ax.set_facecolor(SURFACE2)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)


def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    colors = np.array([[SAFE, PHISH], [AMBER, SAFE]])
    for i in range(2):
        for j in range(2):
            ax.add_patch(mpatches.FancyBboxPatch(
                (j + 0.05, 1 - i + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=colors[i][j] + "22",
                edgecolor=colors[i][j], linewidth=1.5,
            ))
            ax.text(j + 0.5, 1 - i + 0.55, str(cm[i][j]),
                    ha="center", va="center", fontsize=22,
                    fontweight="bold", color=colors[i][j])
            ax.text(j + 0.5, 1 - i + 0.2, labels[i][j],
                    ha="center", va="center", fontsize=9, color=MUTED)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Pred: Legit", "Pred: Phish"], color=MUTED, fontsize=9)
    ax.set_yticklabels(["Actual: Phish", "Actual: Legit"], color=MUTED, fontsize=9)
    ax.set_title(f"Confusion Matrix — {model_name}", color=TEXT, fontsize=11, pad=10)
    style_fig(fig)
    return fig


def plot_roc(results):
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = [ACCENT, SAFE, PHISH, AMBER]
    for (name, res), col in zip(results.items(), colors):
        fpr, tpr = res["fpr_tpr"]
        ax.plot(fpr, tpr, color=col, linewidth=2,
                label=f"{name} (AUC={res['roc_auc']:.3f})")
    ax.plot([0,1],[0,1], "--", color=BORDER, linewidth=1)
    ax.fill_between([0,1],[0,1], alpha=0.03, color=ACCENT)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", color=TEXT, fontsize=11, pad=10)
    legend = ax.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT)
    legend.get_frame().set_facecolor(SURFACE2)
    legend.get_frame().set_edgecolor(BORDER)
    style_fig(fig)
    return fig


def plot_features(top_features, category):
    feats = top_features[category][:14]
    words  = [f[0] for f in feats]
    scores = [f[1] for f in feats]
    color  = PHISH if category == "phishing" else SAFE
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.barh(words[::-1], scores[::-1], color=color + "cc", edgecolor=color, linewidth=0.5)
    ax.set_xlabel("TF-IDF coefficient weight")
    ax.set_title(f"Top {'Phishing' if category == 'phishing' else 'Legitimate'} Signal Words",
                 color=TEXT, fontsize=11, pad=10)
    style_fig(fig)
    return fig


def plot_model_comparison(results):
    names   = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    x       = np.arange(len(names))
    width   = 0.2
    colors  = [ACCENT, SAFE, AMBER, PHISH]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (metric, col) in enumerate(zip(metrics, colors)):
        vals = [results[n][metric] for n in names]
        ax.bar(x + i * width, vals, width, label=metric.capitalize(),
               color=col + "cc", edgecolor=col, linewidth=0.5)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, color=MUTED, fontsize=9)
    ax.set_ylim(0.94, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", color=TEXT, fontsize=11, pad=10)
    legend = ax.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT)
    legend.get_frame().set_facecolor(SURFACE2)
    legend.get_frame().set_edgecolor(BORDER)
    style_fig(fig)
    return fig


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def predict_email(text, vec, model):
    processed = preprocess(text)
    v    = vec.transform([processed])
    prob = model.predict_proba(v)[0] if hasattr(model, "predict_proba") else None
    pred = model.predict(v)[0]
    phish_prob = float(prob[1]) if prob is not None else (1.0 if pred == 1 else 0.0)
    return pred, phish_prob, processed


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Phishing Detector")
    st.markdown("---")
    st.markdown("**Pipeline settings**")

    max_features = st.slider("TF-IDF max features", 1000, 10000, 5000, 500)
    ngram_max    = st.selectbox("Max n-gram size", [1, 2, 3], index=1)
    test_size    = st.slider("Test split %", 10, 40, 20, 5)

    st.markdown("---")
    st.markdown("**Select classifier for inference**")
    selected_model = st.selectbox(
        "Model",
        ["Naive Bayes", "Logistic Regression", "Linear SVM", "Random Forest"],
        index=1,
    )
    st.markdown("---")
    st.caption("Built with Streamlit · sklearn · TF-IDF")


# ─────────────────────────────────────────────
# LOAD + TRAIN
# ─────────────────────────────────────────────
with st.spinner("Loading dataset and training models…"):
    df, source = load_data()
    vec, results, top_features, y_test = train_models(df)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🛡️ Phishing Email Detector")
st.caption(f"Dataset: {source}  ·  {len(df):,} emails  ·  TF-IDF + 4 classifiers")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Model Analysis", "📈 Features", "🧪 Live Classifier"])


# ══════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════
with tab1:
    st.subheader("Dataset summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total emails",  f"{len(df):,}")
    c2.metric("Phishing",      f"{df['label'].sum():,}",  f"{df['label'].mean()*100:.1f}%")
    c3.metric("Legitimate",    f"{(df['label']==0).sum():,}", f"{(1-df['label'].mean())*100:.1f}%")
    c4.metric("Train / Test",  f"{int(len(df)*0.8):,} / {int(len(df)*0.2):,}")

    st.markdown("---")
    st.subheader("Model comparison")
    st.pyplot(plot_model_comparison(results), use_container_width=True)

    st.markdown("---")
    st.subheader("All model metrics")
    rows = []
    for name, res in results.items():
        rows.append({
            "Model":     name,
            "Accuracy":  f"{res['accuracy']*100:.2f}%",
            "Precision": f"{res['precision']*100:.2f}%",
            "Recall":    f"{res['recall']*100:.2f}%",
            "F1 Score":  f"{res['f1']*100:.2f}%",
            "ROC-AUC":   f"{res['roc_auc']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)


# ══════════════════════════════════════
# TAB 2 — MODEL ANALYSIS
# ══════════════════════════════════════
with tab2:
    st.subheader("Confusion matrices")
    cols = st.columns(2)
    for i, (name, res) in enumerate(results.items()):
        with cols[i % 2]:
            st.pyplot(plot_confusion_matrix(res["cm"], name), use_container_width=True)

    st.markdown("---")
    st.subheader("ROC curves")
    st.pyplot(plot_roc(results), use_container_width=True)


# ══════════════════════════════════════
# TAB 3 — FEATURES
# ══════════════════════════════════════
with tab3:
    st.subheader("TF-IDF feature importance (Logistic Regression coefficients)")
    st.caption("Words with the highest positive coefficients are the strongest phishing signals; negative coefficients indicate legitimate email patterns.")

    col_p, col_l = st.columns(2)
    with col_p:
        st.pyplot(plot_features(top_features, "phishing"), use_container_width=True)
    with col_l:
        st.pyplot(plot_features(top_features, "legitimate"), use_container_width=True)

    st.markdown("---")
    st.subheader("Raw feature weights")
    feat_tab = st.radio("Show", ["Phishing signals", "Legitimate signals"], horizontal=True)
    cat  = "phishing" if feat_tab == "Phishing signals" else "legitimate"
    data = [(w, round(s, 4)) for w, s in top_features[cat]]
    st.dataframe(
        pd.DataFrame(data, columns=["Feature", "Weight"]).set_index("Feature"),
        use_container_width=True,
    )


# ══════════════════════════════════════
# TAB 4 — LIVE CLASSIFIER
# ══════════════════════════════════════
with tab4:
    st.subheader("🧪 Classify any email in real time")
    st.caption(f"Using **{selected_model}** — change the model in the sidebar.")

    # Quick-load samples
    st.markdown("**Quick samples:**")
    qc1, qc2, qc3, qc4 = st.columns(4)
    samples = {
        "⚠️ Phishing #1": "URGENT: Your account has been compromised. Verify your identity immediately to avoid suspension. Click here to update your billing information now.",
        "⚠️ Phishing #2": "Congratulations! You have been selected to receive a $500 prize. Enter your personal details and credit card information to claim your reward before it expires.",
        "✅ Legitimate #1": "Hi team, please find the attached meeting agenda for Tuesday's quarterly review. The project deadline has been updated — see the shared document.",
        "✅ Legitimate #2": "Reminder: Please submit your timesheet by end of week so payroll can be processed. The board meeting minutes are now available in the shared drive.",
    }

    if "email_text" not in st.session_state:
        st.session_state.email_text = list(samples.values())[0]

    for (label, text), col in zip(samples.items(), [qc1, qc2, qc3, qc4]):
        if col.button(label, use_container_width=True):
            st.session_state.email_text = text

    email_input = st.text_area(
        "Email text",
        value=st.session_state.email_text,
        height=130,
        placeholder="Paste or type an email here…",
    )

    if st.button("🔍 Classify", type="primary", use_container_width=True):
        if email_input.strip():
            model_obj = results[selected_model]["model"]
            pred, phish_prob, processed = predict_email(email_input, vec, model_obj)

            is_phish  = pred == 1
            label_str = "⚠️ PHISHING" if is_phish else "✅ LEGITIMATE"
            box_class = "phish-box" if is_phish else "safe-box"
            color     = "#ff4d6d" if is_phish else "#00d9a3"

            st.markdown(f"""
            <div class="{box_class}">
              <h3 style="color:{color};margin:0 0 6px">{label_str}</h3>
              <p style="margin:0;color:#aab">
                Phishing probability: <strong style="color:{color}">{phish_prob*100:.1f}%</strong>
                &nbsp;·&nbsp; Model: <strong>{selected_model}</strong>
              </p>
            </div>
            """, unsafe_allow_html=True)

            # Probability bar
            st.progress(phish_prob, text=f"Phishing confidence: {phish_prob*100:.1f}%")

            # Processed tokens with highlights
            st.markdown("**Processed tokens:**")
            PHISH_WORDS = {"verify","account","details","security","prize","immediately",
                           "requires","detected","card","enter","click","receive","information",
                           "update","billing","credit","charged","respond","deactivated","urgent",
                           "compromised","suspicious","locked","unauthorized","funds","credentials"}
            SAFE_WORDS  = {"available","attached","download","week","documents","next","meeting",
                           "team","month","ready","friday","shipped","order","soon","following",
                           "discussion","agenda","reminder","timesheet","quarterly","proposal",
                           "feedback","newsletter","appointment","confirmed","performance","review"}
            tokens_html = ""
            for tok in processed.split():
                if tok in PHISH_WORDS:
                    tokens_html += f'<span style="background:rgba(255,77,109,0.2);border:1px solid #ff4d6d;color:#ff4d6d;border-radius:4px;padding:2px 8px;margin:2px;font-family:monospace;font-size:12px;display:inline-block">{tok}</span>'
                elif tok in SAFE_WORDS:
                    tokens_html += f'<span style="background:rgba(0,217,163,0.15);border:1px solid #00d9a3;color:#00d9a3;border-radius:4px;padding:2px 8px;margin:2px;font-family:monospace;font-size:12px;display:inline-block">{tok}</span>'
                else:
                    tokens_html += f'<span style="background:#1e2130;border:1px solid #2e3250;color:#7a8099;border-radius:4px;padding:2px 8px;margin:2px;font-family:monospace;font-size:12px;display:inline-block">{tok}</span>'
            st.markdown(tokens_html, unsafe_allow_html=True)
            st.caption("🔴 Red = phishing signal word  ·  🟢 Green = legitimate signal word  ·  Gray = neutral")
        else:
            st.warning("Please enter some email text first.")
