"""Classification metrics for the Mineirinho Creek water-level benchmark.

Given the severe class imbalance (low=98.84%, medium=0.78%, high=0.27%,
flood=0.11% of the labeled dataset), accuracy is close to meaningless -- a
trivial always-predict-low classifier scores ~98.8%. Metrics here center on
per-class recall (especially `flood`, the rarest and most safety-critical
class) and an ordinal-distance error that accounts for the low<medium<high<flood
severity ordering, since predicting `low` when the truth is `flood` is a much
worse mistake than predicting `high`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
)

LABELS = ["low", "medium", "high", "flood"]
ORDINAL_INDEX = {label: i for i, label in enumerate(LABELS)}


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

    # False-negative rate specifically for missing flood (predicting anything
    # other than flood when the true label was flood) -- the safety-critical
    # error: underestimating that the creek has actually overflowed onto the bank.
    flood_rows = df[df["label"] == "flood"]
    missed_flood_rate = (flood_rows["classification"] != "flood").mean() if len(flood_rows) else None

    # False-positive rate: predicted flood when truth was low (the base/dry-weather rate).
    low_rows = df[df["label"] == "low"]
    false_alarm_rate = (low_rows["classification"] == "flood").mean() if len(low_rows) else None

    # Mean absolute ordinal error: |predicted_index - true_index| on the
    # low(0) < medium(1) < high(2) < flood(3) scale. Rows with an unrecognized
    # predicted label (shouldn't happen given the tool schema enum, but be safe)
    # are excluded rather than crashing.
    valid = df[df["classification"].isin(ORDINAL_INDEX)]
    ordinal_errors = (
        valid["label"].map(ORDINAL_INDEX) - valid["classification"].map(ORDINAL_INDEX)
    ).abs()
    mean_ordinal_error = float(ordinal_errors.mean()) if len(ordinal_errors) else None

    return {
        "n": len(df),
        "accuracy": report["accuracy"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {label: report[label] for label in LABELS if label in report},
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "missed_flood_rate": missed_flood_rate,
        "false_alarm_rate": false_alarm_rate,
        "mean_ordinal_error": mean_ordinal_error,
    }


def calibration_metrics(df: pd.DataFrame) -> dict | None:
    if "confidence" not in df.columns or df["confidence"].isna().all():
        return None
    df = df.dropna(subset=["confidence"])
    # One-vs-rest Brier score for the "flood" class (the highest-stakes class).
    y_true_binary = (df["label"] == "flood").astype(int)
    pred_is_flood = (df["classification"] == "flood").astype(int)
    # confidence is reported for whatever class was predicted; convert to P(flood)
    prob_flood = np.where(pred_is_flood == 1, df["confidence"], 1 - df["confidence"])
    return {"brier_score_flood": float(brier_score_loss(y_true_binary, prob_flood))}
