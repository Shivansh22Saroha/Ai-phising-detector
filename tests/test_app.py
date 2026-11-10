"""PhishGuard test suite. Run: pytest -v --tb=short"""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app  # noqa: E402  (import after sys.path tweak)
from features import extract_features  # noqa: E402

client = TestClient(app)

# A clean, valid feature payload: the features of "example.com".
VALID = extract_features("https://example.com")


@pytest.fixture(autouse=True)
def temp_log(tmp_path, monkeypatch):
    """Send prediction logs to a temp file so tests never touch logs/predictions.jsonl."""
    log = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("LOG_PATH", str(log))
    return log


# 1. Health check ------------------------------------------------------------
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# 2. UI loads -------------------------------------------------------------------
def test_ui_loads():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "PhishGuard" in r.text


# 3. Domain logic: an obviously phishy host must score higher than a clean one --
def test_phishy_host_scores_higher_than_clean_host():
    clean = client.post("/predict", json=extract_features("https://google.com")).json()
    phishy = client.post(
        "/predict", json=extract_features("http://secure-login-verify-account8842.firebaseapp.com")
    ).json()
    assert phishy["phishing_probability"] > clean["phishing_probability"]
    assert phishy["verdict"] == "phishing"
    assert clean["verdict"] == "legitimate"


def test_path_does_not_change_score():
    """The model sees only the hostname, so adding a path must not change the score
    (guards against the PhiUSIIL 'has a path = phishing' shortcut)."""
    a = client.post("/predict-url", json={"url": "https://github.com"}).json()
    b = client.post("/predict-url", json={"url": "https://github.com/anthropics/docs/x?y=1"}).json()
    assert a["phishing_probability"] == b["phishing_probability"]


# 4. Boundary / contract tests: every violation must be 422, never 500 --------
@pytest.mark.parametrize(
    "field, value",
    [
        ("digit_ratio", 1.5),            # above max 1.0
        ("host_length", 0),              # below min 1
        ("host_length", 254),            # above DNS max 253
        ("is_ip", 2),                    # binary flag must be 0/1
        ("num_subdomains", -1),          # negative count
        ("host_length", "12"),           # string instead of int (strict mode)
        ("host_length", 12.5),           # float instead of int
        ("is_ip", True),                 # bool instead of int (strict mode)
    ],
)
def test_out_of_range_or_wrong_type_returns_422(field, value):
    payload = {**VALID, field: value}
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_and_unknown_fields_return_422():
    missing = {k: v for k, v in VALID.items() if k != "host_entropy"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**VALID, "colour": 1}).status_code == 422


def test_nan_returns_422():
    """'What happens if a feature arrives as NaN?' -> rejected, not passed to the model.
    Python's json accepts the bare NaN token, so send it as raw text."""
    body = json.dumps({**VALID, "vowel_ratio": float("nan")})  # json.dumps writes NaN
    r = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_inconsistent_features_return_422():
    # 10 digits in a 5-character hostname is impossible.
    payload = {**VALID, "host_length": 5, "num_digits": 10, "longest_label_length": 5}
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("url", ["notaurl", "ftp://files.example.com", "https://exa mple.com", "ab", ""])
def test_bad_url_returns_422(url):
    assert client.post("/predict-url", json={"url": url}).status_code == 422


# 5. URL endpoint returns 2-3 tips, and predictions are logged ---------------
def test_predict_url_returns_tips_and_logs(temp_log):
    r = client.post("/predict-url", json={"url": "http://paypal.com.account-verify.security-check.ru/login"})
    assert r.status_code == 200
    body = r.json()
    assert body["host"] == "paypal.com.account-verify.security-check.ru"
    assert 2 <= len(body["tips"]) <= 3
    record = json.loads(temp_log.read_text().strip().splitlines()[-1])
    assert record["host"] == body["host"]
    assert set(record["features"]) == set(VALID)
    assert "/login" not in json.dumps(record)  # privacy: the URL path is never logged
