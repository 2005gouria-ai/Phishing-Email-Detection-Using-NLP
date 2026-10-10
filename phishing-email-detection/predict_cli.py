"""
Command-line prediction tool.

Examples:
    python predict_cli.py "Your account has been suspended, verify now at http://x.xyz"
    echo "Some email text" | python predict_cli.py
"""

import sys

import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "model", "phishing_model.pkl")


def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()

    text = text.strip()
    if not text:
        print("Usage: python predict_cli.py \"<email text>\"  (or pipe text via stdin)")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run 'python train_model.py' first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    proba = float(model.predict_proba([text])[0][1])
    label = "PHISHING" if proba >= 0.5 else "LEGITIMATE"
    print(f"Verdict    : {label}")
    print(f"Phishing p : {proba * 100:.2f}%")


if __name__ == "__main__":
    main()
