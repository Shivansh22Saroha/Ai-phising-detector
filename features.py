"""URL -> numeric feature extraction (our own feature engineering).

Used by BOTH train.py and app.py so training and serving can never drift apart.
Pure standard library on purpose: the serving image stays lean (no pandas).

Why hostname-only?
EDA on PhiUSIIL showed that the full URL leaks the label: every URL with a path,
a trailing slash or an http:// scheme is phishing, and legitimate rows are almost
all bare "https://www.<domain>" homepages. A model trained on the full URL learns
"has a path = phishing" and would flag https://github.com/anthropics. So the ML
model only sees the hostname (lower-cased, leading "www." removed). Signals that
live in the rest of the URL (@, http, long paths) are handled by rule-based tips
in app.py instead.
"""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

SUSPICIOUS_WORDS = (
    "login", "signin", "verify", "secure", "account", "update", "bank",
    "confirm", "password", "paypal", "wallet", "billing", "support",
)

# Free website builders / hosting that phishing kits abuse because a subdomain
# costs nothing and inherits the platform's reputation and HTTPS certificate.
FREE_HOSTING = (
    "firebaseapp.com", "web.app", "weebly.com", "wixsite.com", "godaddysites.com",
    "000webhostapp.com", "blogspot.com", "github.io", "netlify.app", "vercel.app",
    "pages.dev", "glitch.me", "herokuapp.com", "ipfs", "repl.co", "webflow.io",
    "square.site", "sites.google.com", "r2.dev", "workers.dev",
)

# Cheap TLDs that threat reports repeatedly list as phishing-heavy.
RISKY_TLDS = {
    "xyz", "top", "club", "online", "site", "shop", "info", "icu", "live", "buzz",
    "cfd", "sbs", "app", "dev", "link", "click", "vip", "work", "support", "rest",
    "fun", "monster", "cyou", "ru", "tk", "ml", "ga", "cf", "gq", "cn",
}
CONSONANT_RUN = re.compile(r"[bcdfghjklmnpqrstvwxz]+")
DIGIT_LETTER_SWITCH = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")

# Order matters: the model is trained and queried with exactly this column order.
FEATURE_NAMES = [
    "host_length",
    "num_subdomains",
    "num_hyphens",
    "num_digits",
    "digit_ratio",
    "is_ip",
    "host_entropy",
    "suspicious_keyword_count",
    "tld_length",
    "has_punycode",
    "longest_label_length",
    "is_free_hosting",
    "is_risky_tld",
    "vowel_ratio",
    "max_consonant_run",
    "digit_letter_switches",
]


def normalize_host(url: str) -> str:
    """'HTTPS://WWW.PayPal.com/login' -> 'paypal.com'. Raises ValueError if no host."""
    url = url.strip()
    if "://" not in url:  # let users type 'example.com' without a scheme
        url = "http://" + url
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    if not host:
        raise ValueError("URL has no hostname")
    return host


def _entropy(text: str) -> float:
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def host_features(host: str) -> dict[str, float]:
    """Feature dict for an already-normalized hostname."""
    labels = host.split(".")
    is_ip = _is_ip(host)
    digits = sum(ch.isdigit() for ch in host)
    letters = [ch for ch in host if "a" <= ch <= "z"]
    vowels = sum(ch in "aeiou" for ch in letters)
    return {
        "host_length": len(host),
        # Labels beyond domain + TLD. Naive: 'bbc.co.uk' counts as 1 subdomain.
        "num_subdomains": 0 if is_ip else max(len(labels) - 2, 0),
        "num_hyphens": host.count("-"),
        "num_digits": digits,
        "digit_ratio": round(digits / len(host), 4),
        "is_ip": int(is_ip),
        "host_entropy": round(_entropy(host), 4),
        "suspicious_keyword_count": sum(w in host for w in SUSPICIOUS_WORDS),
        "tld_length": 0 if is_ip else len(labels[-1]),
        "has_punycode": int("xn--" in host),
        "longest_label_length": max(len(lbl) for lbl in labels),
        "is_free_hosting": int(any(f in host for f in FREE_HOSTING)),
        "is_risky_tld": int(not is_ip and labels[-1] in RISKY_TLDS),
        # Random machine-made names ('xkqzvbt') have unusually few vowels...
        "vowel_ratio": round(vowels / len(letters), 4) if letters else 0.0,
        # ...and long consonant runs.
        "max_consonant_run": max((len(m) for m in CONSONANT_RUN.findall(host)), default=0),
        # 'caseid1008934573' style ids flip between letters and digits.
        "digit_letter_switches": len(DIGIT_LETTER_SWITCH.findall(host)),
    }


def extract_features(url: str) -> dict[str, float]:
    """Full pipeline for one URL: normalize the host, then compute features."""
    return host_features(normalize_host(url))


def to_row(feats: dict[str, float]) -> list[float]:
    """Dict -> list in FEATURE_NAMES order, ready for model.predict."""
    return [float(feats[name]) for name in FEATURE_NAMES]
