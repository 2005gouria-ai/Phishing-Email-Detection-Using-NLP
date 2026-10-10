"""
Phishing Email Detection - Flask web application.

Serves a simple UI plus a JSON API that classifies an email as
"Phishing" or "Legitimate" using the trained TF-IDF + Random Forest model.

Run:
    python app.py
Then open http://127.0.0.1:5000

If the model file is missing, the app trains it automatically on first start.
"""

import json
import os

import joblib
from flask import Flask, jsonify, render_template, request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "phishing_model.pkl")
METRICS_PATH = os.path.join(HERE, "model", "metrics.json")

app = Flask(__name__)

_model = None
_metrics = {}


def get_model():
    """Load the trained model, training it on the fly if it does not exist."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            print("Model not found - training now (this runs once)...")
            from train_model import train
            train()
        _model = joblib.load(MODEL_PATH)
    return _model


def load_metrics():
    global _metrics
    if not _metrics and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as f:
            _metrics = json.load(f)
    return _metrics


@app.route("/")
def index():
    return render_template("index.html", metrics=load_metrics())


@app.route("/predict", methods=["POST"])
def predict_form():
    """Handle the HTML form submission."""
    text = (request.form.get("email_text") or "").strip()
    if not text:
        return render_template(
            "index.html",
            metrics=load_metrics(),
            error="Please paste some email text to analyse.",
        )
    result = classify(text)
    return render_template("index.html", metrics=load_metrics(), result=result, text=text)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    """JSON API: POST {"text": "..."} -> {"label", "confidence", ...}."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400
    return jsonify(classify(text))


def classify(text):
    model = get_model()
    proba = model.predict_proba([text])[0]
    # classes_ are [0, 1]; index 1 == phishing
    classes = list(model.classes_)
    phish_idx = classes.index(1) if 1 in classes else len(classes) - 1
    phishing_prob = float(proba[phish_idx])
    label = "Phishing" if phishing_prob >= 0.5 else "Legitimate"
    return {
        "label": label,
        "is_phishing": bool(phishing_prob >= 0.5),
        "confidence": round(max(phishing_prob, 1 - phishing_prob) * 100, 2),
        "phishing_probability": round(phishing_prob * 100, 2),
    }


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    get_model()  # warm up / train if needed before serving
    app.run(host="127.0.0.1", port=5000, debug=True)
