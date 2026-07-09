"""Classification metrics for the flood benchmark, with recall reported prominently
(false negatives -- missed floods -- are the safety-critical failure mode) alongside
false-positive rate (alarm-fatigue proxy)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
)

LABELS = ["no_flood", "flood_watch", "active_flood"]


def load_predictions(path: str) -> pd.DataFrame:
    rows = [json.loads(line) for line in open(path)]
    # agent output nests the example; baseline output flattens it
    normalized = []
    for r in rows:
        if "example" in r:
            ex = r["example"]
            normalized.append({**ex, "classification": r["classification"], "confidence": r.get("confidence")})
        else:
            normalized.append(r)
    return pd.DataFrame(normalized)


def classification_metrics(df: pd.DataFrame) -> dict:
    y_true = df["label"]
    y_pred = df["classification"]
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    # False-negative rate specifically for missing an active_flood (predicting no_flood
    # or flood_watch when the true label was active_flood) -- the safety-critical error.
    active_flood_rows = df[df["label"] == "active_flood"]
    missed_active_flood = (active_flood_rows["classification"] != "active_flood").mean() if len(active_flood_rows) else None

    # False-positive rate: predicted active_flood when truth was no_flood.
    no_flood_rows = df[df["label"] == "no_flood"]
    false_alarm_rate = (no_flood_rows["classification"] == "active_flood").mean() if len(no_flood_rows) else None

    return {
        "n": len(df),
        "accuracy": report["accuracy"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {label: report[label] for label in LABELS if label in report},
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "missed_active_flood_rate": missed_active_flood,
        "false_alarm_rate": false_alarm_rate,
    }


def calibration_metrics(df: pd.DataFrame) -> dict | None:
    if "confidence" not in df.columns or df["confidence"].isna().all():
        return None
    df = df.dropna(subset=["confidence"])
    # One-vs-rest Brier score for the "active_flood" class (the highest-stakes class).
    y_true_binary = (df["label"] == "active_flood").astype(int)
    pred_is_active = (df["classification"] == "active_flood").astype(int)
    # confidence is reported for whatever class was predicted; convert to P(active_flood)
    prob_active = np.where(pred_is_active == 1, df["confidence"], 1 - df["confidence"])
    return {"brier_score_active_flood": float(brier_score_loss(y_true_binary, prob_active))}
