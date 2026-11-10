"""Train a phishing-host classifier and track every run in MLflow (SQLite backend).

Examples:
    python train.py --n_estimators 100 --max_depth 6                  # Random Forest (default)
    python train.py --model logreg --C 1.0
    python train.py --model hgb --learning_rate 0.1 --max_iter 300
    mlflow ui --backend-store-uri sqlite:///mlflow.db

The model file is only replaced when the new run's test F1 beats the saved
model's F1 (stored in model/model_meta.json), so model/model.joblib is always
the best run so far. Use --force_save to overwrite regardless (CI baseline).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, host_features, normalize_host

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "phiusil_urls.csv"
MODEL_DIR = ROOT / "model"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train PhishGuard model")
    ap.add_argument("--model", choices=["rf", "logreg", "hgb"], default="rf")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--max_rows", type=int, default=0, help="subsample hosts (0 = all)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    # Random Forest
    ap.add_argument("--n_estimators", type=int, default=100)
    ap.add_argument("--max_depth", type=int, default=12)
    ap.add_argument("--min_samples_leaf", type=int, default=1)
    # Logistic Regression
    ap.add_argument("--C", type=float, default=1.0)
    # Histogram Gradient Boosting
    ap.add_argument("--learning_rate", type=float, default=0.1)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--force_save", action="store_true", help="always overwrite model.joblib")
    ap.add_argument("--tracking_uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    ap.add_argument("--experiment", default="phishguard")
    return ap.parse_args()


def load_dataset(path: Path, max_rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists() and path == DEFAULT_DATA:
        # Fresh clone: the CSV is not committed (>50 MB), so fetch it first.
        print(f"{path.name} not found, running data/download_data.py ...")
        subprocess.run([sys.executable, str(ROOT / "data" / "download_data.py")], check=True)
    df = pd.read_csv(path)
    df["host"] = df["url"].map(normalize_host)
    # One row per host: many URLs share a host, and leaving duplicates in would let
    # the same host land in both train and test (leakage, inflated scores).
    labels = df.groupby("host")["is_phishing"].agg(["nunique", "first"])
    labels = labels[labels["nunique"] == 1]  # drop hosts labelled both ways
    if max_rows and max_rows < len(labels):
        labels = labels.sample(n=max_rows, random_state=seed)
    X = pd.DataFrame([host_features(h) for h in labels.index], columns=FEATURE_NAMES)
    y = labels["first"].reset_index(drop=True).astype(int)
    return X, y


def build_model(args: argparse.Namespace):
    if args.model == "rf":
        params = {"n_estimators": args.n_estimators, "max_depth": args.max_depth,
                  "min_samples_leaf": args.min_samples_leaf}
        model = RandomForestClassifier(**params, class_weight="balanced",
                                       n_jobs=-1, random_state=args.seed)
    elif args.model == "logreg":
        params = {"C": args.C}
        # Scaling matters for LR (features range 0-1 vs 0-110); trees don't need it.
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=args.C, class_weight="balanced", max_iter=2000))
    else:
        params = {"learning_rate": args.learning_rate, "max_iter": args.max_iter}
        model = HistGradientBoostingClassifier(**params, class_weight="balanced",
                                               random_state=args.seed)
    return model, params


def main() -> None:
    args = parse_args()
    X, y = load_dataset(args.data, args.max_rows, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    model, params = build_model(args)

    with mlflow.start_run(run_name=f"{args.model}-{int(time.time())}") as run:
        mlflow.log_param("algorithm", args.model)
        mlflow.log_params(params)
        mlflow.log_params({"n_train": len(X_train), "n_test": len(X_test),
                           "n_features": len(FEATURE_NAMES), "seed": args.seed})

        start = time.perf_counter()
        # Plain numpy arrays: the server has no pandas, so the model must not
        # depend on column names. Column ORDER is fixed by FEATURE_NAMES instead.
        model.fit(X_train.to_numpy(), y_train.to_numpy())
        train_seconds = time.perf_counter() - start

        pred = model.predict(X_test.to_numpy())
        proba = model.predict_proba(X_test.to_numpy())[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "train_seconds": train_seconds,
        }
        mlflow.log_metrics(metrics)
        print(f"[{args.model}] {params} -> " +
              ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        MODEL_DIR.mkdir(exist_ok=True)
        meta_path = MODEL_DIR / "model_meta.json"
        best_f1 = json.loads(meta_path.read_text())["f1"] if meta_path.exists() else -1.0
        if args.force_save or metrics["f1"] > best_f1:
            joblib.dump(model, MODEL_DIR / "model.joblib", compress=3)
            meta = {"algorithm": args.model, "params": params, "run_id": run.info.run_id,
                    "features": FEATURE_NAMES, **{k: round(v, 4) for k, v in metrics.items()}}
            meta_path.write_text(json.dumps(meta, indent=2))
            # Training-data statistics, used later by drift_check.py.
            stats = {c: {"mean": float(X_train[c].mean()), "std": float(X_train[c].std())}
                     for c in FEATURE_NAMES}
            (MODEL_DIR / "feature_stats.json").write_text(json.dumps(stats, indent=2))
            mlflow.set_tag("saved_as_best", "true")
            reason = "forced" if args.force_save else f"> previous {best_f1:.4f}"
            print(f"Saved model/model.joblib (f1 {metrics['f1']:.4f}, {reason})")
        else:
            mlflow.set_tag("saved_as_best", "false")
            print(f"Kept existing model (f1 {metrics['f1']:.4f} <= best {best_f1:.4f})")


if __name__ == "__main__":
    main()
