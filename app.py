"""Streamlit demo: Twitter Sentiment Analysis.

Run the notebook first (it saves the model into models/), then:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from sentiment_utils import load_artifacts, predict_sentiment

MODELS_DIR = Path(__file__).parent / "models"
MAX_BATCH_ROWS = 5000

EXAMPLES = [
    "(type my own)",
    "I absolutely love this, best day ever!",
    "This is not good at all, I'm really disappointed.",
    "Not bad, actually pretty happy with it.",
    "worst service ever, never coming back",
    "It arrived on Tuesday.",
]

LABEL_ICON = {"Positive": "🟢", "Negative": "🔴", "Uncertain": "🟡", "No usable text": "⚪"}

st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="💬", layout="centered")


@st.cache_resource
def get_artifacts():
    return load_artifacts(MODELS_DIR)


st.title("💬 Twitter Sentiment Analysis")
st.caption("TF-IDF + a classic machine-learning classifier, trained on the Sentiment140 tweet dataset.")

# ---- load model ----------------------------------------------------------
try:
    vectorizer, model, metadata = get_artifacts()
except FileNotFoundError:
    st.error("No trained model found in the `models/` folder.")
    st.info("Open `notebooks/sentiment_analysis.ipynb`, run all cells (it saves "
            "`vectorizer.joblib` and `model.joblib`), then reload this page.")
    st.stop()

if "MOCK" in str(metadata.get("data_source", "")):
    st.warning("This model was trained on the tiny synthetic MOCK sample, so its "
               "predictions are not meaningful. Train on the full Sentiment140 file.")

# ---- sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("About the model")
    st.write(f"**Algorithm:** {metadata.get('model_name', type(model).__name__)}")
    st.write(f"**Trained on:** {metadata.get('data_source', 'unknown')}")
    if metadata.get("n_train"):
        st.write(f"**Training tweets:** {metadata['n_train']:,}")
    metrics = metadata.get("metrics", {})
    if metrics:
        st.subheader("Test-set metrics")
        for name, value in metrics.items():
            st.metric(name, f"{value:.3f}")
    st.caption("Scores between 0.40 and 0.60 are shown as *Uncertain*. The model only "
               "knows positive vs negative, and it learned from tweets, so other "
               "writing styles may be less reliable.")

tab_single, tab_batch = st.tabs(["Single text", "Batch (CSV)"])

# ---- single text ---------------------------------------------------------
with tab_single:
    example = st.selectbox("Try an example", EXAMPLES)
    default_text = "" if example == EXAMPLES[0] else example
    text = st.text_area("Enter a tweet or short review", value=default_text, height=120)

    if st.button("Analyze sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            result = predict_sentiment([text], vectorizer, model)[0]
            icon = LABEL_ICON[result["label"]]
            if result["label"] == "No usable text":
                st.info(f"{icon} Nothing left to analyse after cleaning (only links, "
                        "numbers, symbols or very common words).")
            else:
                score = result["positive_score"]
                st.subheader(f"{icon} {result['label']}")
                st.progress(score, text=f"Positive score: {score:.1%}")
                st.caption(f"Text the model actually saw: `{result['clean_text']}`")

# ---- batch ---------------------------------------------------------------
with tab_batch:
    st.write("Upload a CSV with a column of texts; every row gets a sentiment label.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded, encoding="latin-1")
        except Exception as exc:  # unreadable or empty file
            st.error(f"Could not read that CSV: {exc}")
            st.stop()

        if data.empty:
            st.warning("The file has no rows.")
        else:
            text_cols = [c for c in data.columns
                         if pd.api.types.is_string_dtype(data[c]) or pd.api.types.is_object_dtype(data[c])]
            if not text_cols:
                st.error("No text column found in this CSV.")
            else:
                column = st.selectbox("Column that contains the text", text_cols)
                if len(data) > MAX_BATCH_ROWS:
                    st.info(f"Only the first {MAX_BATCH_ROWS:,} rows are analysed.")
                    data = data.head(MAX_BATCH_ROWS)

                if st.button("Analyze all rows"):
                    preds = predict_sentiment(data[column].astype(str).tolist(), vectorizer, model)
                    out = data.copy()
                    out["sentiment"] = [p["label"] for p in preds]
                    out["positive_score"] = [round(p["positive_score"], 3) for p in preds]

                    st.bar_chart(out["sentiment"].value_counts())
                    st.dataframe(out)
                    st.download_button("Download results as CSV",
                                       out.to_csv(index=False).encode("utf-8"),
                                       file_name="sentiment_results.csv", mime="text/csv")
