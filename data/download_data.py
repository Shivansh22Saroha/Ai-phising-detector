"""Download the PhiUSIIL Phishing URL dataset and write a slim two-column CSV.

Source: Prasad, A. & Chandra, S. (2024). PhiUSIIL Phishing URL (Website).
UCI Machine Learning Repository, id 967. https://doi.org/10.1016/j.cose.2023.103545
License: CC BY 4.0.

The raw file is ~54 MB (over the 50 MB commit limit), so it is fetched here instead.
UCI labels: 1 = legitimate, 0 = phishing. We recode to is_phishing (1 = phishing).

Usage:
    python data/download_data.py                 # full dataset
    python data/download_data.py --max-rows 20000   # stratified sample (fast CI runs)
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
OUT = Path(__file__).resolve().parent / "phiusil_urls.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=0, help="stratified sample size (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zip", type=Path, help="use a local zip/CSV instead of downloading")
    args = ap.parse_args()

    if args.zip:
        raw = args.zip.read_bytes()
    else:
        print(f"Downloading {URL} ...")
        raw = urllib.request.urlopen(URL, timeout=120).read()

    if args.zip and args.zip.suffix.lower() == ".csv":
        text = raw.decode("utf-8-sig")
    else:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            text = zf.read(name).decode("utf-8-sig")

    rows = {0: [], 1: []}
    for rec in csv.DictReader(io.StringIO(text)):
        url = (rec.get("URL") or "").strip()
        if not url:
            continue
        is_phishing = 1 - int(rec["label"])  # UCI: 1 = legitimate -> flip
        rows[is_phishing].append(url)

    if args.max_rows:
        rng = random.Random(args.seed)
        total = len(rows[0]) + len(rows[1])
        for k in rows:
            take = round(args.max_rows * len(rows[k]) / total)
            rows[k] = rng.sample(rows[k], min(take, len(rows[k])))

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "is_phishing"])
        for label, urls in rows.items():
            w.writerows((u, label) for u in urls)
    print(f"Wrote {OUT} (legit={len(rows[0])}, phishing={len(rows[1])})")


if __name__ == "__main__":
    main()
