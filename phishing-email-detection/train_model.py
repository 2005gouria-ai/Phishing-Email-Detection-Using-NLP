"""
Train the phishing-email classifier.

Pipeline:
    TF-IDF (word + character n-grams)  ->  Random Forest

Outputs (saved to model/):
    phishing_model.pkl   trained scikit-learn Pipeline
    metrics.json         evaluation metrics on the held-out test set

Run:
    python train_model.py
"""

import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "emails.csv")
MODEL_DIR = os.path.join(HERE, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run 'python scripts/generate_sample_data.py' first."
        )
    df = pd.read_csv(path)
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("Dataset must contain 'label' and 'text' columns.")
    df = df.dropna(subset=["label", "text"])
    return df["text"].astype(str), df["label"].astype(int)


def build_pipeline():
    """TF-IDF word n-grams combined with char n-grams, feeding a Random Forest.

    Word n-grams capture phrases like "verify your account"; character
    n-grams catch obfuscated tokens, odd URLs and misspellings that phishing
    emails rely on, and make the model robust to words it has never seen.
    """
    features = FeatureUnion(
        [
            ("word_tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                stop_words="english",
            )),
            ("char_tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
            )),
        ]
    )
    return Pipeline(
        steps=[
            ("features", features),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]
    )


def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision_phishing": round(float(precision), 4),
        "recall_phishing": round(float(recall), 4),
        "f1_phishing": round(float(f1), 4),
        "confusion_matrix": cm,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model": "TF-IDF + RandomForestClassifier",
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 60)
    print(f"Trained on {len(X_train)} emails, tested on {len(X_test)}")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision (phish): {precision:.4f}")
    print(f"Recall (phish)   : {recall:.4f}")
    print(f"F1 (phish)       : {f1:.4f}")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=["legit", "phishing"]))
    print(f"Saved model   -> {os.path.relpath(MODEL_PATH, HERE)}")
    print(f"Saved metrics -> {os.path.relpath(METRICS_PATH, HERE)}")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    train()
