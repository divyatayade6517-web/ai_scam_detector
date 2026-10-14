from pathlib import Path
from datetime import datetime
import io

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# -------------------- App setup --------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "spam_model.pkl"
VECTORIZER_PATH = ROOT / "model" / "tfidf_vectorizer.pkl"
DATASET_CANDIDATES = [
    ROOT / "SMSSpamCollection",
    ROOT / "SMSSpamCollection.txt",
    ROOT / "sms+spam+collection" / "SMSSpamCollection",
    ROOT / "data" / "SMSSpamCollection",
]

st.set_page_config(
    page_title="AI Shield | Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(135deg, #07111f 0%, #0b1425 48%, #11152d 100%); color: #edf3ff; }
    [data-testid="stSidebar"] { background: #0a1221; border-right: 1px solid rgba(148,163,184,.16); }
    [data-testid="stHeader"] { background: rgba(7,17,31,.85); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f4f7ff !important; }
    p, label, li { color: #c6d2e5; }
    .hero { padding: 1.35rem 1.5rem; border-radius: 22px; border: 1px solid rgba(125,211,252,.22);
            background: linear-gradient(120deg, rgba(14,165,233,.16), rgba(99,102,241,.15));
            margin-bottom: 1.2rem; }
    .eyebrow { text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; color: #67e8f9; font-weight: 700; }
    .hero-title { font-family: 'Space Grotesk', sans-serif; font-size: clamp(2rem, 4vw, 3rem); font-weight: 700; line-height: 1.1; margin: .35rem 0 .6rem; color: #f8fbff; }
    .hero-copy { color: #bdcbe0; font-size: 1rem; max-width: 760px; }
    .metric-card { background: rgba(17,29,49,.88); border: 1px solid rgba(148,163,184,.18); border-radius: 16px; padding: 1rem 1.1rem; min-height: 105px; }
    .metric-label { color: #9fb1ca; font-size: .82rem; margin-bottom: .35rem; }
    .metric-value { font-family: 'Space Grotesk', sans-serif; color: #f8fbff; font-size: 1.75rem; font-weight: 700; }
    .metric-note { color: #7dd3fc; font-size: .78rem; margin-top: .2rem; }
    .panel { background: rgba(17,29,49,.7); border: 1px solid rgba(148,163,184,.18); border-radius: 18px; padding: 1.15rem 1.25rem; }
    .small-muted { color: #9fb1ca; font-size: .85rem; }
    .pill { display:inline-block; border-radius:999px; padding:.35rem .7rem; font-weight:700; font-size:.8rem; }
    div.stButton > button, div.stDownloadButton > button { border-radius: 11px; border: 1px solid rgba(125,211,252,.3); background: linear-gradient(90deg,#0891b2,#4f46e5); color: white; font-weight: 700; min-height: 2.7rem; }
    div.stButton > button:hover, div.stDownloadButton > button:hover { border-color:#67e8f9; color:white; }
    textarea, input { border-radius: 12px !important; }
    [data-testid="stDataFrame"] { border: 1px solid rgba(148,163,184,.18); border-radius: 12px; }
    hr { border-color: rgba(148,163,184,.18); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Load saved model --------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None, (
            "Model files were not found. Expected both "
            "`model/spam_model.pkl` and `model/tfidf_vectorizer.pkl`."
        )
    try:
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH), None
    except Exception as exc:
        return None, None, f"Could not load model files: {exc}"

model, vectorizer, model_error = load_artifacts()

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- Helpers --------------------
def find_dataset():
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    return None

@st.cache_data
def load_dataset(path_string, modified_time):
    # UCI SMS Spam Collection: each line is label<TAB>message.
    df = pd.read_csv(
        path_string, sep="\t", header=None, names=["label", "message"],
        encoding="utf-8", on_bad_lines="skip"
    )
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str)
    df = df[df["label"].isin(["ham", "spam"])]
    return df

def predict_message(message):
    features = vectorizer.transform([message])
    predicted_label = str(model.predict(features)[0]).lower()
    spam_probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = [str(c).lower() for c in model.classes_]
        if "spam" in classes:
            spam_probability = float(probabilities[classes.index("spam")])
    return predicted_label, spam_probability

def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("## 🛡️ AI Shield")
    st.caption("SMART MESSAGE SECURITY")
    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Message Scanner", "Analytics & History", "Model Insights"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**SYSTEM STATUS**")
    if model_error:
        st.error("Model not ready")
    else:
        st.success("AI model loaded")
    st.caption("Built with Python · scikit-learn · Streamlit")
    st.caption("Predictions are estimates, not guarantees.")

# -------------------- Header --------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Machine learning · Natural language processing</div>
      <div class="hero-title">AI Shield</div>
      <div class="hero-copy">A smart SMS spam detection workspace. Scan a message, review your session activity,
      and explore how your machine-learning model performs.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_error:
    st.error(model_error)
    st.info(
        "Check that your project folder contains a `model` folder with both saved `.pkl` files. "
        "Keep `app.py` in the same project folder as the `model` folder."
    )
    st.stop()

# -------------------- Dashboard --------------------
if page == "Dashboard":
    total_checks = len(st.session_state.history)
    spam_checks = sum(item["Prediction"] == "spam" for item in st.session_state.history)
    ham_checks = total_checks - spam_checks
    spam_rate = (spam_checks / total_checks * 100) if total_checks else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("SESSION SCANS", total_checks, "Messages checked this session")
    with c2: metric_card("SPAM FLAGS", spam_checks, "Predicted as spam")
    with c3: metric_card("NORMAL MESSAGES", ham_checks, "Predicted as ham")
    with c4: metric_card("SPAM FLAG RATE", f"{spam_rate:.1f}%", "Of session scans")

    st.markdown("### Welcome to your message security workspace")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 🔎 Quick scan")
        quick_message = st.text_area(
            "Paste a message to analyze",
            placeholder="Example: Congratulations! You have won a prize. Claim now...",
            height=130,
            key="dashboard_message",
        )
        if st.button("Analyze message", key="dashboard_scan", use_container_width=True):
            if not quick_message.strip():
                st.warning("Please enter a message first.")
            else:
                label, probability = predict_message(quick_message.strip())
                st.session_state.history.insert(0, {
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Message": quick_message.strip(),
                    "Prediction": label,
                    "Spam probability": probability,
                })
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### ✨ What this app can do")
        st.markdown(
            "- Classify SMS messages as **spam** or **normal**\n"
            "- Estimate the model's **spam probability**\n"
            "- Keep a temporary scan history\n"
            "- Export scan results to CSV\n"
            "- Review test-set model metrics"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Recent scans")
    if st.session_state.history:
        recent = pd.DataFrame(st.session_state.history[:5]).copy()
        recent["Message"] = recent["Message"].str.slice(0, 90)
        recent["Spam probability"] = recent["Spam probability"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "Unavailable"
        )
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info("No scans yet. Open Message Scanner or use Quick scan above to analyze your first message.")

# -------------------- Message scanner --------------------
elif page == "Message Scanner":
    st.markdown("### 🔬 Message Scanner")
    st.write("Paste a single SMS message below. The model will analyze it using the same TF-IDF vocabulary used during training.")
    with st.form("scanner_form"):
        message = st.text_area(
            "Message text",
            placeholder="Type or paste the message you want to check...",
            height=180,
        )
        submitted = st.form_submit_button("🛡️ Scan message", use_container_width=True)

    if submitted:
        if not message.strip():
            st.warning("Please enter a message before scanning.")
        else:
            cleaned = message.strip()
            label, probability = predict_message(cleaned)
            st.session_state.history.insert(0, {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Message": cleaned,
                "Prediction": label,
                "Spam probability": probability,
            })
            if label == "spam":
                st.error("🚨 The model classified this message as SPAM.")
            else:
                st.success("✅ The model classified this message as NORMAL (ham).")

            a, b = st.columns(2)
            with a:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("#### Model prediction")
                st.markdown(f"**Category:** `{label.upper()}`")
                st.markdown("</div>", unsafe_allow_html=True)
            with b:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("#### Estimated spam probability")
                if probability is not None:
                    st.markdown(f"### {probability:.1%}")
                    st.progress(min(max(probability, 0.0), 1.0))
                else:
                    st.info("This saved model does not provide probability estimates.")
                st.markdown("</div>", unsafe_allow_html=True)

            suspicious_terms = [
                "free", "winner", "won", "claim", "urgent", "prize", "cash",
                "click", "offer", "selected", "congratulations", "verify", "account"
            ]
            found_terms = sorted({term for term in suspicious_terms if term in cleaned.lower()})
            st.markdown("#### Message signals")
            if found_terms:
                st.write("Potentially suspicious terms found: " + ", ".join(f"`{term}`" for term in found_terms))
                st.caption("These terms are simple indicators only. Their presence does not prove a message is spam.")
            else:
                st.write("No terms from the app's simple indicator list were found.")
                st.caption("A message can still be spam without these terms.")

            st.caption("Probability is the model's estimate, not a guarantee that the message is safe or fraudulent.")

# -------------------- Analytics & History --------------------
elif page == "Analytics & History":
    st.markdown("### 📊 Analytics & History")
    if not st.session_state.history:
        st.info("Your session history is empty. Scan a few messages first.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        total = len(history_df)
        spam_count = int((history_df["Prediction"] == "spam").sum())
        ham_count = total - spam_count
        x1, x2, x3 = st.columns(3)
        with x1: metric_card("TOTAL SCANS", total)
        with x2: metric_card("SPAM", spam_count)
        with x3: metric_card("NORMAL", ham_count)

        chart_data = pd.DataFrame({
            "Category": ["Spam", "Normal"],
            "Count": [spam_count, ham_count],
        }).set_index("Category")
        st.markdown("#### Predictions in this session")
        st.bar_chart(chart_data)

        st.markdown("#### Full scan history")
        display_df = history_df.copy()
        display_df["Message"] = display_df["Message"].str.slice(0, 250)
        display_df["Spam probability"] = display_df["Spam probability"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "Unavailable"
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv_bytes = history_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download history as CSV",
            data=csv_bytes,
            file_name="spam_detector_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("Clear session history", type="secondary"):
            st.session_state.history = []
            st.rerun()
        st.caption("History is kept in this browser session only. It is cleared when the session resets.")

# -------------------- Model insights --------------------
elif page == "Model Insights":
    st.markdown("### 🧠 Model Insights")
    st.write("This page evaluates the saved model on a held-out test split from the original SMS dataset, if the dataset is available.")
    dataset_path = find_dataset()

    if dataset_path is None:
        st.warning("Dataset not found. Place `SMSSpamCollection` next to `app.py` to enable evaluation metrics.")
        st.markdown("Expected file: `SMSSpamCollection`")
    else:
        try:
            df = load_dataset(str(dataset_path), dataset_path.stat().st_mtime)
            if len(df) < 10:
                st.error("The dataset file was found, but it does not contain enough valid rows.")
            else:
                X = df["message"]
                y = df["label"]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                X_test_tfidf = vectorizer.transform(X_test)
                predictions = model.predict(X_test_tfidf)

                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, pos_label="spam", zero_division=0)
                recall = recall_score(y_test, predictions, pos_label="spam", zero_division=0)
                f1 = f1_score(y_test, predictions, pos_label="spam", zero_division=0)

                m1, m2, m3, m4 = st.columns(4)
                with m1: metric_card("ACCURACY", f"{accuracy:.1%}", "All test predictions correct")
                with m2: metric_card("SPAM PRECISION", f"{precision:.1%}", "Predicted spam that was spam")
                with m3: metric_card("SPAM RECALL", f"{recall:.1%}", "Actual spam found")
                with m4: metric_card("SPAM F1-SCORE", f"{f1:.1%}", "Precision/recall balance")

                st.markdown("#### Confusion matrix")
                labels = ["ham", "spam"]
                cm = confusion_matrix(y_test, predictions, labels=labels)
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual normal (ham)", "Actual spam"],
                    columns=["Predicted normal (ham)", "Predicted spam"],
                )
                st.dataframe(cm_df, use_container_width=True)
                st.caption("Rows are actual labels; columns are model predictions.")

                report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose().round(3)
                st.markdown("#### Full classification report")
                st.dataframe(report_df, use_container_width=True)

                st.caption(
                    f"Dataset: {len(df):,} valid messages. Test split: {len(y_test):,} messages. "
                    "These results are meaningful only if this split matches the model's training setup. "
                    "If you changed the split or preprocessing when training, retrain and evaluate consistently."
                )
        except Exception as exc:
            st.error(f"Could not evaluate the model using the dataset: {exc}")
            st.info("Check that SMSSpamCollection is the original tab-separated UCI dataset and that your saved vectorizer matches the saved model.")

st.markdown("---")
st.markdown(
    '<div class="small-muted">AI Shield · Educational project · Machine-learning predictions can be wrong. '
    'Do not use this demo as the only basis for security decisions.</div>',
    unsafe_allow_html=True,
)
