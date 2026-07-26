"""Supervised learning curve as a function of annotation budget.

This extends baselines/kshot_probe.py past the region where a strict one-frame-per-event
support set is possible, in order to locate where supervision overtakes a zero-shot VLM.
Two axes, because they answer different questions:

**balanced** -- K frames per class, drawn spreading across available events round-robin
  before taking a second frame from any event. Comparable to the K-shot arm's x-axis, but
  above K=6 it necessarily reuses events, since only 6 flood events exist per fold. Hard
  ceiling: the flood class has 46-73 frames per fold, so K cannot exceed ~46.

**natural** -- a total budget of N labeled frames drawn at the bank's own class
  proportions. This is the realistic annotation-effort axis ("I can afford to label N
  images"), it reaches the full ~900, and it is the axis to quote when answering "at what
  point should I stop prompting and train a model?"

Both hold the leave-one-season-out protocol: training frames come only from the three
seasons that are not the evaluation fold.

Usage:
    python -m baselines.supervised_budget_curve
    python -m baselines.supervised_budget_curve --balanced 12 24 --natural 100 400
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.kshot_probe import BANK_DIR, TEST_EMB, TEST_EXAMPLES, fit_predict, write_cell
from data.enoe_images import SEASONS
from data.taxonomy import labels_for, map_label

RESULTS_DIR = Path("experiments/results")
TAXONOMY = "3class"


def sample_balanced(pool: pd.DataFrame, k: int, rng: np.random.Generator) -> np.ndarray:
    """K frames per class, spread across events before doubling up on any one event.

    Concentrating a class's budget in one event would mean K near-duplicate frames, which
    inflates the apparent budget without adding information.
    """
    rows = []
    for label in labels_for(TAXONOMY):
        label_pool = pool[pool["tax_label"] == label]
        by_event = defaultdict(list)
        for idx, event in zip(label_pool.index, label_pool["event"]):
            by_event[event].append(idx)
        for indices in by_event.values():
            rng.shuffle(indices)
        events = sorted(by_event)
        rng.shuffle(events)
        picked, round_no = [], 0
        while len(picked) < k:
            added = False
            for event in events:
                if round_no < len(by_event[event]):
                    picked.append(by_event[event][round_no])
                    added = True
                    if len(picked) == k:
                        break
            if not added:  # exhausted every frame of this class
                break
            round_no += 1
        rows.extend(picked)
    return np.array(rows)


def sample_natural(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray:
    """N frames total at the pool's own class proportions -- the annotation-budget axis."""
    if n >= len(pool):
        return pool.index.to_numpy()
    proportions = pool["tax_label"].value_counts(normalize=True)
    rows = []
    for label, share in proportions.items():
        label_pool = pool[pool["tax_label"] == label]
        take = min(int(round(n * share)), len(label_pool))
        rows.extend(rng.choice(label_pool.index.to_numpy(), size=take, replace=False))
    return np.array(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced", type=int, nargs="*", default=[12, 24, 40])
    parser.add_argument("--natural", type=int, nargs="*", default=[50, 100, 200, 400])
    parser.add_argument("--episodes", type=int, default=10, help="Independent draws per budget point")
    parser.add_argument("--probe", nargs="*", default=["logreg", "prototype"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = labels_for(TAXONOMY)
    bank_emb = np.load(BANK_DIR / "embeddings.npy")
    bank = pd.read_parquet(BANK_DIR / "metadata.parquet").reset_index(drop=True)
    bank["tax_label"] = bank["label"].map(lambda l: map_label(l, TAXONOMY))
    bank["event"] = bank["path"].str.slice(0, 10).str.replace("/", "-") + "|" + bank["place"]

    examples = [json.loads(line) for line in open(TEST_EXAMPLES)]
    test_emb = np.load(TEST_EMB)
    test_season = np.array([ex["season"] for ex in examples])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Report the per-fold ceiling for the flood class, which is what bounds `balanced`.
    flood_per_fold = {
        s: int((bank[(bank["season"] != s) & (bank["tax_label"] == "flood")]).shape[0]) for s in SEASONS
    }
    print(f"flood frames available per fold: {flood_per_fold} -> balanced K ceiling {min(flood_per_fold.values())}\n")

    jobs = [("balanced", k) for k in args.balanced] + [("natural", n) for n in args.natural]
    for kind in args.probe:
        for mode, budget in jobs:
            for episode in range(args.episodes):
                rng = np.random.default_rng(args.seed + 1000 * episode + budget)
                preds = np.empty(len(examples), dtype=object)
                proba = np.zeros((len(examples), len(labels)))
                actual_sizes = []
                for season in SEASONS:
                    pool = bank[bank["season"] != season]
                    rows = sample_balanced(pool, budget, rng) if mode == "balanced" else sample_natural(pool, budget, rng)
                    actual_sizes.append(len(rows))
                    X, y = bank_emb[rows], bank["tax_label"].to_numpy()[rows]
                    if len(np.unique(y)) < len(labels):
                        # A budget too small to cover every class cannot be scored fairly.
                        raise SystemExit(
                            f"{mode} budget {budget} left a fold without all classes "
                            f"({sorted(set(y))}) -- raise the budget"
                        )
                    mask = test_season == season
                    fold_preds, fold_proba = fit_predict(kind, X, y, test_emb[mask], labels, args.seed)
                    preds[mask] = fold_preds
                    proba[mask] = fold_proba
                out = RESULTS_DIR / f"probe_{kind}_3class_{mode}{budget}_e{episode}.jsonl"
                write_cell(out, examples, preds, proba, labels, f"dinov2-vitb14+{kind}",
                           {"budget_mode": mode, "budget": budget, "episode": episode,
                            "train_size_per_fold": actual_sizes})
            print(f"{kind} {mode}={budget}: {args.episodes} cells, train size/fold {actual_sizes}")


if __name__ == "__main__":
    main()
