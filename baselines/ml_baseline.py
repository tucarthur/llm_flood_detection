"""Structured-feature ML baseline (logistic regression + gradient-boosted trees).

Uses leave-one-event-out cross-validation (train on 2 of {sandy, harvey, irma}, test on
the third) rather than a random split -- with only 3 joint-track events, a random split
would let rows from the same event/baseline leak between train and test, inflating
scores. This is also a more honest test of generalization to an unseen storm.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["discharge_value", "pct_above_baseline", "day_index_in_window"]


def build_features(examples_path: str, usgs_csv: str) -> pd.DataFrame:
    examples = [json.loads(line) for line in open(examples_path)]
    usgs = pd.read_csv(usgs_csv, dtype={"site_id": str})
    usgs["site_id"] = usgs["site_id"].str.zfill(8)
    discharge = usgs[usgs["parameter_name"] == "discharge_cfs"]

    rows = []
    for ex in examples:
        sub = discharge[discharge["site_id"] == ex["site_id"]].sort_values("date")
        baseline = sub.head(3)["value"].mean()
        target_row = sub[sub["date"] == ex["target_date"]]
        if target_row.empty or pd.isna(baseline):
            continue
        value = target_row["value"].iloc[0]
        dates_sorted = sorted(sub["date"].unique())
        day_index = dates_sorted.index(ex["target_date"])
        rows.append(
            {
                **ex,
                "discharge_value": value,
                "pct_above_baseline": 100.0 * (value - baseline) / baseline,
                "day_index_in_window": day_index,
            }
        )
    return pd.DataFrame(rows)


def leave_one_event_out_eval(df: pd.DataFrame, model_name: str = "gbt") -> list[dict]:
    events = df["event"].unique()
    predictions = []
    for held_out in events:
        train = df[df["event"] != held_out]
        test = df[df["event"] == held_out]
        if train.empty or test.empty:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[FEATURE_COLS])
        X_test = scaler.transform(test[FEATURE_COLS])

        model = (
            GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=0)
            if model_name == "gbt"
            else LogisticRegression(max_iter=1000)
        )
        model.fit(X_train, train["label"])
        preds = model.predict(X_test)

        for (_, row), pred in zip(test.iterrows(), preds):
            predictions.append({**row.to_dict(), "classification": pred})
    return predictions


def run(examples_path: str, usgs_csv: str, out_dir: str) -> None:
    df = build_features(examples_path, usgs_csv)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for model_name in ("gbt", "logreg"):
        preds = leave_one_event_out_eval(df, model_name)
        out_path = Path(out_dir) / f"{model_name}_baseline.jsonl"
        with open(out_path, "w") as f:
            for p in preds:
                f.write(json.dumps(p, default=str) + "\n")
        print(f"Wrote {len(preds)} {model_name} predictions to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", default="data/processed/examples.jsonl")
    parser.add_argument("--usgs", default="data/raw/usgs_daily_values.csv")
    parser.add_argument("--out-dir", default="experiments/results")
    args = parser.parse_args()
    run(args.examples, args.usgs, args.out_dir)
