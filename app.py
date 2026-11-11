"""PhishGuard API: FastAPI service + Pydantic data contract.

Routes
    GET  /             simple UI (static/index.html)
    GET  /health       liveness + model info
    POST /predict      engineered features in -> prediction (the data contract)
    POST /predict-url  raw URL in -> features -> same contract -> prediction + tips
    GET  /docs         auto-generated Swagger UI (FastAPI built-in)
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from features import FEATURE_NAMES, SUSPICIOUS_WORDS, extract_features, normalize_host, to_row

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "model" / "model.joblib"))
META_PATH = MODEL_PATH.with_name("model_meta.json")
THRESHOLD = 0.5

# Stage 3: load ONCE when the process starts, never inside a request handler.
MODEL = joblib.load(MODEL_PATH)
META = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}


# --------------------------------------------------------------------------- #
# Data contract. Every row of the README table is enforced here.
#   strict=True       -> "5" (string) or true (bool) is rejected for an int field
#   extra="forbid"    -> unknown / misspelled fields are rejected
#   allow_inf_nan=False -> NaN and Infinity are rejected for float fields
# Any violation becomes HTTP 422 automatically (FastAPI + Pydantic).
# --------------------------------------------------------------------------- #
class PhishFeatures(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    host_length: int = Field(ge=1, le=253, description="Characters in the hostname (DNS max 253)")
    num_subdomains: int = Field(ge=0, le=125, description="Labels beyond domain + TLD")
    num_hyphens: int = Field(ge=0, le=252, description="'-' characters in the hostname")
    num_digits: int = Field(ge=0, le=253, description="Digit characters in the hostname")
    digit_ratio: float = Field(ge=0.0, le=1.0, description="num_digits / host_length")
    is_ip: int = Field(ge=0, le=1, description="1 if the host is a raw IP address")
    host_entropy: float = Field(ge=0.0, le=6.0, description="Shannon entropy of the hostname (bits/char)")
    suspicious_keyword_count: int = Field(ge=0, le=len(SUSPICIOUS_WORDS), description="Bait words in the host")
    tld_length: int = Field(ge=0, le=63, description="Length of the top-level domain (0 for IPs)")
    has_punycode: int = Field(ge=0, le=1, description="1 if the host contains 'xn--' (IDN / homograph)")
    longest_label_length: int = Field(ge=1, le=253, description="Longest dot-separated label")
    is_free_hosting: int = Field(ge=0, le=1, description="1 if hosted on a free site builder")
    is_risky_tld: int = Field(ge=0, le=1, description="1 if the TLD is on the high-abuse list")
    vowel_ratio: float = Field(ge=0.0, le=1.0, description="Vowels / letters in the host")
    max_consonant_run: int = Field(ge=0, le=253, description="Longest run of consecutive consonants")
    digit_letter_switches: int = Field(ge=0, le=252, description="Letter<->digit transitions")

    @model_validator(mode="after")
    def check_consistency(self) -> "PhishFeatures":
        # Cross-field rules: values that can't occur together for any real hostname.
        if max(self.num_digits, self.num_hyphens, self.longest_label_length) > self.host_length:
            raise ValueError("num_digits, num_hyphens and longest_label_length must be <= host_length")
        if self.max_consonant_run > self.longest_label_length:
            raise ValueError("max_consonant_run must be <= longest_label_length")
        if abs(self.digit_ratio - self.num_digits / self.host_length) > 0.01:
            raise ValueError("digit_ratio must equal num_digits / host_length")
        if self.is_ip and (self.tld_length != 0 or self.num_subdomains != 0):
            raise ValueError("an IP host has tld_length 0 and num_subdomains 0")
        return self


class UrlRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    url: str = Field(min_length=4, max_length=2048, examples=["https://secure-login-verify.example.xyz/account"])

    @field_validator("url")
    @classmethod
    def check_url(cls, v: str) -> str:
        v = v.strip()
        if any(ch.isspace() or ord(ch) < 32 for ch in v):
            raise ValueError("URL must not contain spaces or control characters")
        if "://" in v and urlsplit(v).scheme.lower() not in ("http", "https"):
            raise ValueError("only http and https URLs are supported")
        try:
            host = normalize_host(v)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if "." not in host and ":" not in host:
            raise ValueError("hostname must be a domain (e.g. example.com) or an IP address")
        return v


class Prediction(BaseModel):
    verdict: str
    is_phishing: int
    phishing_probability: float
    threshold: float


class UrlPrediction(Prediction):
    host: str
    features: dict[str, float]
    tips: list[str]


# --------------------------------------------------------------------------- #
app = FastAPI(title="PhishGuard", version="1.0.0",
              description="Phishing domain detector (CSL236 end-term). Model scores the hostname; "
                          "rule-based tips cover the rest of the URL.")


def _json_safe(obj):
    """Replace NaN/Infinity with strings so an error response can always be encoded."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default 422 echoes the bad input back. If that input is NaN, JSON
    # encoding fails and the client gets a 500 instead of a 422. Sanitize first.
    return JSONResponse(status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))})


def _score(feats: PhishFeatures) -> Prediction:
    row = [to_row(feats.model_dump())]
    proba = float(MODEL.predict_proba(row)[0][1])
    is_phish = int(proba >= THRESHOLD)
    return Prediction(verdict="phishing" if is_phish else "legitimate", is_phishing=is_phish,
                      phishing_probability=round(proba, 4), threshold=THRESHOLD)


def _log(endpoint: str, feats: PhishFeatures, pred: Prediction, host: str | None = None) -> None:
    """Stage 8 monitoring: one JSON line per prediction, read later by drift_check.py.
    Only the hostname is logged, never the full URL (paths/queries can hold tokens)."""
    path = Path(os.getenv("LOG_PATH", ROOT / "logs" / "predictions.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "endpoint": endpoint,
              "host": host, "features": feats.model_dump(),
              "phishing_probability": pred.phishing_probability, "verdict": pred.verdict}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def build_tips(url: str, feats: dict[str, float], pred: Prediction) -> list[str]:
    """2-3 rule-based tips. These check things the model deliberately doesn't see."""
    parts = urlsplit(url if "://" in url else "http://" + url)
    tips = []
    if parts.scheme.lower() == "http":
        tips.append("This link uses http, not https, so anything you type is sent unencrypted.")
    if "@" in parts.netloc:
        tips.append("The link contains '@'. Browsers ignore everything before it, so the real "
                    "destination is the part after the '@'.")
    if feats["is_ip"]:
        tips.append("The address is a raw IP number instead of a name. Real companies don't send "
                    "you to bare IP addresses.")
    if feats["has_punycode"]:
        tips.append("The domain uses look-alike (non-English) characters that can imitate a real brand.")
    if feats["is_free_hosting"]:
        tips.append("The site is on a free hosting platform. Anyone can create these in minutes; "
                    "banks and big brands don't use them for logins.")
    if feats["suspicious_keyword_count"]:
        tips.append("The domain itself contains words like 'login', 'verify' or 'secure'. Real sites "
                    "rarely need them in the domain name.")
    if feats["num_subdomains"] >= 3:
        real = ".".join(normalize_host(url).split(".")[-2:])
        tips.append(f"Many subdomains. The part that matters is the end: this site belongs to '{real}'.")
    if feats["is_risky_tld"]:
        tips.append("The domain ending is a cheap extension that is often abused for phishing.")
    if len(url) > 100:
        tips.append("Very long links are often used to hide the real domain. Read it from the right.")

    if pred.is_phishing:
        tips.append("Don't enter passwords or OTPs here. Open the official site by typing its address yourself.")
    else:
        tips.append("Looks normal, but still check the spelling of the domain before signing in.")
    tips.append("If a message pressures you to act urgently, verify through the official app or helpline.")
    return tips[:3]


# --------------------------------------------------------------------------- #
# HEAD too: Render and uptime monitors probe with HEAD requests.
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL is not None,
            "algorithm": META.get("algorithm"), "f1": META.get("f1")}


@app.post("/predict", response_model=Prediction)
def predict(features: PhishFeatures) -> Prediction:
    pred = _score(features)
    _log("/predict", features, pred)
    return pred


@app.post("/predict-url", response_model=UrlPrediction)
def predict_url(req: UrlRequest) -> UrlPrediction:
    host = normalize_host(req.url)
    raw = extract_features(req.url)
    try:
        # Same contract as /predict: extracted values must pass it too.
        feats = PhishFeatures(**raw)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=jsonable_encoder(exc.errors(include_url=False))) from exc
    pred = _score(feats)
    _log("/predict-url", feats, pred, host=host)
    return UrlPrediction(**pred.model_dump(), host=host, features=feats.model_dump(),
                         tips=build_tips(req.url, raw, pred))


assert list(PhishFeatures.model_fields) == FEATURE_NAMES, "schema and features.py out of sync"
