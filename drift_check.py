"""Bonus: simple drift check.

Compares the mean of each feature in the prediction log (logs/predictions.jsonl)
against the training data's mean/std saved by train.py (model/feature_stats.json).
A feature is flagged when the live mean is more than --z training standard
deviations away from the training mean. Many flags = traffic no longer looks like
the training data, which is the signal to collect fresh data and retrain.

Usage: python drift_check.py [--log logs/predictions.jsonl] [--z 1.0] [--min_rows 30]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=ROOT / "logs" / "predictions.jsonl")
    ap.add_argument("--stats", type=Path, default=ROOT / "model" / "feature_stats.json")
    ap.add_argument("--z", type=float, default=1.0, help="flag threshold in training std units")
    ap.add_argument("--min_rows", type=int, default=30)
    args = ap.parse_args()

    train = json.loads(args.stats.read_text())
    rows = [json.loads(line)["features"] for line in args.log.read_text().splitlines() if line.strip()]
    if len(rows) < args.min_rows:
        print(f"Only {len(rows)} logged predictions (need {args.min_rows}); not enough to judge drift.")
        return 0

    print(f"{len(rows)} logged predictions vs training data\n")
    print(f"{'feature':26} {'train mean':>11} {'live mean':>10} {'live std':>9} {'shift (z)':>9}")
    flagged = []
    for name, ref in train.items():
        values = [r[name] for r in rows]
        live_mean = statistics.fmean(values)
        live_std = statistics.pstdev(values)
        z = abs(live_mean - ref["mean"]) / ref["std"] if ref["std"] > 0 else 0.0
        mark = "  <-- DRIFT" if z > args.z else ""
        if mark:
            flagged.append(name)
        print(f"{name:26} {ref['mean']:11.3f} {live_mean:10.3f} {live_std:9.3f} {z:9.2f}{mark}")

    print(f"\n{len(flagged)} of {len(train)} features drifted (threshold z > {args.z}).")
    if len(flagged) >= 3:
        print("Recommendation: collect recent labelled URLs and retrain.")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
