"""Rule-based classical flood detector: classifies from % deviation of discharge above
the site's pre-event baseline. This is the actual incumbent approach this benchmark
must beat to justify an LLM-agent architecture.

Thresholds are fixed a priori (not fit on this dataset) to avoid circularity with the
labels, which are themselves anchored to independent event timelines, not to gauge
deviation -- see data/align.py docstring.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

WATCH_THRESHOLD_PCT = 25.0  # discharge >25% above baseline -> flood_watch
FLOOD_THRESHOLD_PCT = 60.0  # discharge >60% above baseline -> active_flood


def classify_from_deviation(pct_above_baseline: float) -> str:
    if pct_above_baseline >= FLOOD_THRESHOLD_PCT:
        return "active_flood"
    if pct_above_baseline >= WATCH_THRESHOLD_PCT:
        return "flood_watch"
    return "no_flood"


def run(examples_path: str, usgs_csv: str, out_path: str) -> None:
    examples = [json.loads(line) for line in open(examples_path)]
    usgs = pd.read_csv(usgs_csv, dtype={"site_id": str})
    usgs["site_id"] = usgs["site_id"].str.zfill(8)
    discharge = usgs[usgs["parameter_name"] == "discharge_cfs"]

    results = []
    for ex in examples:
        site_id = ex["site_id"]
        sub = discharge[discharge["site_id"] == site_id].sort_values("date")
        baseline = sub.head(3)["value"].mean()
        target_row = sub[sub["date"] == ex["target_date"]]
        if target_row.empty or pd.isna(baseline):
            pred = "no_flood"
            pct = None
        else:
            value = target_row["value"].iloc[0]
            pct = 100.0 * (value - baseline) / baseline
            pred = classify_from_deviation(pct)
        results.append({**ex, "classification": pred, "pct_above_baseline": pct})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} threshold-baseline predictions to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", default="data/processed/examples.jsonl")
    parser.add_argument("--usgs", default="data/raw/usgs_daily_values.csv")
    parser.add_argument("--out", default="experiments/results/threshold_baseline.jsonl")
    args = parser.parse_args()
    run(args.examples, args.usgs, args.out)
