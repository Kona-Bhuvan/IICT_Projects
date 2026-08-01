import os
import re

import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from flask import Flask, render_template, request

# ==========================================
# SETUP — mirrors the training notebook exactly
# ==========================================
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

NLP = spacy.load("en_core_web_sm")
VECTOR_SIZE = NLP("placeholder").vector.shape[0]

URGENCY_KEYWORDS = ["urgent", "verify", "account", "bank", "suspend", "immediate", "security", "login"]
URGENCY_PATTERN = "|".join(URGENCY_KEYWORDS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "model_artifacts")

best_model = joblib.load(os.path.join(ARTIFACT_DIR, "best_model.joblib"))
scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib"))
tfidf = joblib.load(os.path.join(ARTIFACT_DIR, "tfidf.joblib"))

MODEL_NAME = type(best_model).__name__

app = Flask(__name__)


# ==========================================
# FEATURE EXTRACTION — must match the notebook's functions exactly,
# or predictions will be silently wrong (feature order/count mismatch).
# ==========================================

def clean_text(text):
    text = re.sub(r"<[^>]+>", "", str(text))
    text = re.sub(r"https?://[^\s]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
    return " ".join(cleaned)


def extract_metadata_features(subject, body):
    url_count = len(re.findall(r"https?://[^\s]+", body))
    urgency_flag = int(
        bool(re.search(URGENCY_PATTERN, subject, re.IGNORECASE))
        or bool(re.search(URGENCY_PATTERN, body, re.IGNORECASE))
    )
    body_length = len(body)
    subject_length = len(subject)
    uppercase_ratio = sum(1 for c in body if c.isupper()) / (len(body) + 1)
    exclamation_count = body.count("!")

    return np.array(
        [[url_count, urgency_flag, body_length, subject_length, uppercase_ratio, exclamation_count]],
        dtype="float64",
    )


def extract_word_embedding(text):
    text = str(text)[:1000]
    if not text.strip():
        return np.zeros((1, VECTOR_SIZE), dtype="float32")
    doc = NLP(text)
    vec = doc.vector
    if vec.shape[0] != VECTOR_SIZE:
        vec = np.zeros(VECTOR_SIZE, dtype="float32")
    return vec.reshape(1, -1)


def predict_email(subject, body):
    subject = subject or ""
    body = body or ""

    meta_feats = extract_metadata_features(subject, body)
    cleaned_body = clean_text(body)
    tfidf_feats = tfidf.transform([cleaned_body]).toarray()
    embedding_feats = extract_word_embedding(cleaned_body)

    final_feats = np.hstack([meta_feats, tfidf_feats, embedding_feats])
    scaled_feats = scaler.transform(final_feats)

    prediction = int(best_model.predict(scaled_feats)[0])
    probabilities = best_model.predict_proba(scaled_feats)[0] if hasattr(best_model, "predict_proba") else None

    return {
        "prediction": prediction,
        "label": "PHISHING" if prediction == 1 else "LEGITIMATE",
        "confidence": round(float(probabilities[prediction]) * 100, 1) if probabilities is not None else None,
        "phishing_prob": round(float(probabilities[1]) * 100, 1) if probabilities is not None else None,
        "legit_prob": round(float(probabilities[0]) * 100, 1) if probabilities is not None else None,
        "url_count": int(meta_feats[0][0]),
        "urgency_flag": bool(meta_feats[0][1]),
        "exclamation_count": int(meta_feats[0][5]),
        "uppercase_ratio": round(float(meta_feats[0][4]) * 100, 1),
    }


# ==========================================
# ROUTES
# ==========================================

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    subject = ""
    body = ""

    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        body = request.form.get("body", "").strip()
        if subject or body:
            result = predict_email(subject, body)

    return render_template(
        "index.html",
        result=result,
        subject=subject,
        body=body,
        model_name=MODEL_NAME,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
