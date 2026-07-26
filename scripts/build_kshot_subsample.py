"""Build the stratified subsample used for the K-shot sweep.

Why a subsample at all: each K-shot call carries K exemplar images per class on top of the
query image (K=4 on 3 classes is 13 images per request), and provider throughput is
input-token-bound. The full 1,592-example set at 3 models x 3 K levels x 3 episodes is not
affordable; the K=0 cells were already run at full n, so the sweep only needs enough
examples to estimate the *shape* of the curve.

What is preserved exactly:
  * EVERY flood frame (all 75, all 10 events). Flood recall and per-event detection are
    the headline metrics and must not lose a single event.
  * Day/night proportion within each of the other classes, since night frames carry
    streetlight glare that materially changes difficulty.

What changes: the class prior moves further from the true distribution, so accuracy and
precision on this subsample are not comparable to full-set numbers. Per-class recall and
the conditional rates remain unbiased; report the subsample's own prior alongside.

Usage:
    python -m scripts.build_kshot_subsample                    # defaults: 150 risky, 150 no_risk
    python -m scripts.build_kshot_subsample --n-risky 200 --n-no-risk 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data.taxonomy import event_id, map_label

TEST_EXAMPLES = Path("data/processed/test_examples.jsonl")
DEFAULT_OUT = Path("data/processed/kshot_subsample.jsonl")


def stratified_by_night(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows preserving the group's day/night proportion."""
    if n >= len(df):
        return df
    night, day = df[df["is_night"]], df[~df["is_night"]]
    n_night = min(round(n * len(night) / len(df)), len(night))
    n_day = min(n - n_night, len(day))
    return pd.concat([night.sample(n_night, random_state=seed), day.sample(n_day, random_state=seed)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-risky", type=int, default=150)
    parser.add_argument("--n-no-risk", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(TEST_EXAMPLES)]
    df = pd.DataFrame(rows)
    df["tax_label"] = df["label"].map(lambda l: map_label(l, "3class"))

    parts = [
        df[df["tax_label"] == "flood"],  # all of it, untouched
        stratified_by_night(df[df["tax_label"] == "risky"], args.n_risky, args.seed),
        stratified_by_night(df[df["tax_label"] == "no_risk"], args.n_no_risk, args.seed),
    ]
    sub = pd.concat(parts).sort_values("datetime")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = ["image_path", "kaggle_path", "datetime", "season", "is_night", "place", "label"]
    with open(out, "w") as f:
        for _, row in sub.iterrows():
            f.write(json.dumps({k: row[k] for k in keep}) + "\n")

    n_events = sub.apply(lambda r: event_id(r.to_dict()), axis=1).nunique()
    flood = sub[sub["tax_label"] == "flood"]
    flood_events = flood.apply(lambda r: event_id(r.to_dict()), axis=1).nunique()
    print(f"wrote {out}: n={len(sub)} across {n_events} events")
    print(sub["tax_label"].value_counts().to_string())
    print(f"\nflood: {len(flood)} frames / {flood_events} events (must be 75 / 10)")
    print("gold 4-class breakdown:")
    print(sub["label"].value_counts().to_string())
    print(f"night fraction: {sub['is_night'].mean():.3f}")


if __name__ == "__main__":
    main()
