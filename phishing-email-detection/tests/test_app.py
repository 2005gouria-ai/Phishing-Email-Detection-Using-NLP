"""
Smoke tests for the Flask app. Run with:

    pip install -r requirements-dev.txt
    pytest -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402


def client():
    app.config.update(TESTING=True)
    return app.test_client()


def test_home_page_loads():
    r = client().get("/")
    assert r.status_code == 200
    assert b"Phishing Email Detector" in r.data


def test_health():
    r = client().get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


def test_api_flags_phishing():
    r = client().post("/api/predict", json={
        "text": "Urgent: verify your account at http://secure-login.xyz "
                "and confirm your password now."
    })
    assert r.status_code == 200
    assert r.get_json()["label"] == "Phishing"


def test_api_accepts_legitimate():
    r = client().post("/api/predict", json={
        "text": "Hi, just confirming our lunch meeting tomorrow at 1pm."
    })
    assert r.status_code == 200
    assert r.get_json()["label"] == "Legitimate"


def test_api_rejects_empty():
    r = client().post("/api/predict", json={"text": ""})
    assert r.status_code == 400
    assert "error" in r.get_json()
