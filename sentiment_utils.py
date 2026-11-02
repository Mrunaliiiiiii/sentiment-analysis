"""Shared helpers for the notebook and the Streamlit app.

The text-cleaning code lives in ONE place on purpose: the model must see text
that was cleaned in exactly the same way at training time and at prediction
time. (In the old project the cleaning was applied to one dataframe, but the
model was trained on a different, uncleaned one.)
"""

import html
import json
import re
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --------------------------------------------------------------------------
# Stop words
# --------------------------------------------------------------------------
# sklearn's list contains words that flip or strengthen sentiment
# ("not good" is not the same as "good"), so we keep those.
KEEP_WORDS = {
    "not", "no", "nor", "never", "nothing", "nobody", "none", "neither",
    "cannot", "without", "very", "too", "much", "enough", "well", "cry",
    "against", "least",
}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) - KEEP_WORDS

# Tweets often drop the apostrophe: "dont", "cant", "didnt" ... -> "not"
NEGATION_NO_APOSTROPHE = {
    "dont", "cant", "wont", "didnt", "doesnt", "isnt", "wasnt", "arent",
    "werent", "couldnt", "shouldnt", "wouldnt", "hasnt", "havent", "hadnt",
    "aint", "cannot",
}

# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_NON_LETTERS = re.compile(r"[^a-z\s]")      # drops digits, punctuation, emojis
_REPEATS = re.compile(r"(.)\1{2,}")          # "sooooo" -> "soo"


def clean_text(text):
    """Turn a raw tweet into a clean string of lowercase words.

    Steps: decode HTML entities, lowercase, remove URLs and @mentions, keep
    hashtag words, expand "n't" to "not", keep letters only, shorten repeated
    letters, map apostrophe-less negations ("dont") to "not", remove stop
    words (but keep negations) and drop one-letter tokens.
    """
    if not isinstance(text, str):
        return ""

    t = html.unescape(text).lower().replace("\u2019", "'")
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _HASHTAG.sub(r"\1", t)

    # contractions must be handled before punctuation is stripped
    t = re.sub(r"\bwon't\b", "will not", t)
    t = re.sub(r"\bcan't\b", "can not", t)
    t = re.sub(r"n't\b", " not", t)

    t = _NON_LETTERS.sub(" ", t)
    t = _REPEATS.sub(r"\1\1", t)

    tokens = []
    for tok in t.split():
        if tok in NEGATION_NO_APOSTROPHE:
            tok = "not"
        if len(tok) > 1 and tok not in STOPWORDS:
            tokens.append(tok)
    return " ".join(tokens)


# --------------------------------------------------------------------------
# Loading and predicting
# --------------------------------------------------------------------------
def load_artifacts(models_dir):
    """Load the saved vectorizer, model and (optional) metadata."""
    models_dir = Path(models_dir)
    vectorizer = joblib.load(models_dir / "vectorizer.joblib")
    model = joblib.load(models_dir / "model.joblib")
    meta_path = models_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return vectorizer, model, metadata


def positive_scores(model, X):
    """Score for the positive class, between 0 and 1.

    Logistic Regression / Naive Bayes give real probabilities. LinearSVC has no
    predict_proba, so its decision value is squashed with a sigmoid; that is a
    useful confidence score but not a calibrated probability. The ranking
    (and therefore ROC-AUC) is the same either way.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return expit(model.decision_function(X))


def predict_sentiment(texts, vectorizer, model, uncertain_band=(0.40, 0.60)):
    """Predict sentiment for a list of raw texts.

    Returns one dict per text with keys: text, clean_text, label, positive_score.
    label is "Positive", "Negative", "Uncertain" (score inside uncertain_band)
    or "No usable text" (nothing left after cleaning).
    """
    texts = list(texts)
    cleaned = [clean_text(t) for t in texts]

    usable = [i for i, c in enumerate(cleaned) if c]
    scores = {}
    if usable:
        X = vectorizer.transform([cleaned[i] for i in usable])
        scores = dict(zip(usable, positive_scores(model, X)))

    low, high = uncertain_band
    results = []
    for i, (raw, clean) in enumerate(zip(texts, cleaned)):
        if i not in scores:
            results.append({"text": raw, "clean_text": clean,
                            "label": "No usable text", "positive_score": np.nan})
            continue
        score = float(scores[i])
        if score > high:
            label = "Positive"
        elif score < low:
            label = "Negative"
        else:
            label = "Uncertain"
        results.append({"text": raw, "clean_text": clean,
                        "label": label, "positive_score": score})
    return results
