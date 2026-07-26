"""Episode (support-set) sampler for the K-shot arm.

Protocol: N-way K-shot episodic evaluation. Rather than picking one "good" support set,
we draw E independent episodes per K and report mean +/- CI across them -- the few-shot
literature's standard, and the only defensible answer to support-set draw variance when
K is small.

Two constraints are specific to this dataset and both are enforced here:

**One frame per event.** The camera fires every few minutes, so K frames drawn frame-wise
can easily be K near-duplicates of one rising-water event -- visually K=1. Support sets
are therefore sampled over *events* (date, camera), taking a single frame from each.

**K is capped by flood events, not by budget.** Distinct flood events per season are
1/2/3/4, ten in total. Holding out one season leaves at most six, so K_max = 6 for the
flood class under leave-one-season-out. The builder refuses larger K rather than silently
sampling the same event twice.

A manifest holds, per episode, one support set per held-out season (i.e. per LOSO fold).
At classification time the fold whose held-out season matches the query's season is the
one used, which is what keeps support and query event-disjoint.

Usage:
    python -m data.episodes --k 2 --n-episodes 3 --out experiments/episodes_k2.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data.enoe_images import SEASONS
from data.taxonomy import labels_for, map_label, ordinal_index

BANK_METADATA = Path("knowledge_base/image_bank_backup/metadata.parquet")
BANK_IMAGES_DIR = Path("knowledge_base/image_bank_images")


def load_bank(metadata_path: Path = BANK_METADATA) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found -- the exemplar bank must be present "
            "(built by knowledge_base/build_image_bank.ipynb)"
        )
    bank = pd.read_parquet(metadata_path)
    # event = (calendar date, camera); `path` is 'YYYY/MM/DD/<file>-<place>.jpg'
    bank["event"] = bank["path"].str.slice(0, 10).str.replace("/", "-") + "|" + bank["place"]
    return bank


def local_path(bank_path: str, images_dir: Path = BANK_IMAGES_DIR) -> Path:
    """Bank paths are CSV-style; local files use the flattened '/'->'_' convention."""
    return images_dir / bank_path.replace("/", "_")


def events_available(bank: pd.DataFrame, taxonomy: str) -> pd.DataFrame:
    """Distinct events per (taxonomy label, season) -- the table that bounds K."""
    bank = bank.copy()
    bank["tax_label"] = bank["label"].map(lambda l: map_label(l, taxonomy))
    return bank.groupby(["tax_label", "season"])["event"].nunique().unstack(fill_value=0)


def max_k(bank: pd.DataFrame, taxonomy: str) -> int:
    """Largest K supported for every class in every fold (one frame per event)."""
    pivot = events_available(bank, taxonomy)
    return int(min(pivot.sum(axis=1) - pivot.max(axis=1)))


def build_episodes(
    k: int,
    n_episodes: int,
    seed: int = 42,
    taxonomy: str = "3class",
    images_dir: Path = BANK_IMAGES_DIR,
) -> dict:
    bank = load_bank()
    bank["tax_label"] = bank["label"].map(lambda l: map_label(l, taxonomy))
    limit = max_k(bank, taxonomy)
    if k > limit:
        pivot = events_available(bank, taxonomy)
        raise ValueError(
            f"K={k} exceeds what the data supports (max {limit}). Distinct events per "
            f"class/season:\n{pivot.to_string()}\n"
            "One frame per event is not negotiable -- sampling the same event twice would "
            "make K meaningless."
        )

    labels = labels_for(taxonomy)
    ordinals = ordinal_index(taxonomy)
    rng = np.random.default_rng(seed)
    episodes = []

    for episode_index in range(n_episodes):
        supports: dict[str, list[dict]] = {}
        for held_out in SEASONS:
            pool = bank[bank["season"] != held_out]
            chosen: list[dict] = []
            for label in labels:
                label_pool = pool[pool["tax_label"] == label]
                event_ids = np.sort(label_pool["event"].unique())
                picked_events = rng.choice(event_ids, size=k, replace=False)
                for event in picked_events:
                    frames = label_pool[label_pool["event"] == event].sort_values("path")
                    # one frame per event, chosen at random among that event's frames
                    row = frames.iloc[int(rng.integers(len(frames)))]
                    chosen.append({
                        "label": row["label"],          # gold 4-class label
                        "tax_label": label,             # label in the run's taxonomy
                        "path": row["path"],
                        "season": row["season"],
                        "place": row["place"],
                        "is_night": bool(row["is_night"]),
                        "event": event,
                        "image_available_locally": local_path(row["path"], images_dir).exists(),
                    })
            # In-context learning is order-sensitive; fix ascending severity by construction.
            chosen.sort(key=lambda ex: ordinals[ex["tax_label"]])
            supports[held_out] = chosen
        episodes.append({"episode": episode_index, "supports": supports})

    missing = sum(
        1
        for ep in episodes
        for support in ep["supports"].values()
        for ex in support
        if not ex["image_available_locally"]
    )
    return {
        "taxonomy": taxonomy,
        "k": k,
        "n_episodes": n_episodes,
        "seed": seed,
        "max_k_supported": limit,
        "images_dir": str(images_dir),
        "missing_images": missing,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": episodes,
    }


def load_manifest(path: str | Path) -> dict:
    return json.load(open(path))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, help="Exemplars per class (one frame per event)")
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--taxonomy", default="3class")
    parser.add_argument("--out", default="")
    parser.add_argument("--show-limits", action="store_true", help="Print the event table and K ceiling, then exit")
    args = parser.parse_args()

    if args.show_limits:
        bank = load_bank()
        print(events_available(bank, args.taxonomy).to_string())
        print(f"\nmax K (one frame per event, leave-one-season-out): {max_k(bank, args.taxonomy)}")
        return

    if args.k is None:
        parser.error("--k is required (or pass --show-limits)")
    manifest = build_episodes(args.k, args.n_episodes, args.seed, args.taxonomy)
    out = args.out or f"experiments/episodes_{args.taxonomy}_k{args.k}_e{args.n_episodes}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)

    n_support = len(manifest["episodes"][0]["supports"]["2018-2019"])
    print(
        f"wrote {out}: k={args.k}, {args.n_episodes} episodes x 4 folds, "
        f"{n_support} exemplars per call, missing_images={manifest['missing_images']}"
    )
    if manifest["missing_images"]:
        print("WARNING: some support images are absent locally and will be silently skipped")


if __name__ == "__main__":
    main()
