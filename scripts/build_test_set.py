"""Build the final evaluation set: sample rows from the annotation CSV, write the
examples JSONL, and download the selected images from Kaggle.

Sampling strategy (mirrors the exemplar-bank builder's shape, but random instead of
farthest-point -- FPS over-selects outliers, which is right for RAG references and
wrong for an unbiased eval sample):
  - keep EVERY medium/high/flood row (they're too rare to subsample: 535/186/75);
  - per season, randomly sample as many 'low' rows as that season has rare rows,
    stratified by is_night so the day/night mix of the season's lows is preserved.

Note the class prior this produces (~50% low vs the true ~99%): per-class recall and
macro-F1 remain unbiased estimates, but precision/accuracy are not comparable to
full-distribution numbers.

Images land in --images-dir using the flattened-name convention from
data.enoe_images.local_image_path, so the agent and eval code work unchanged.
Re-running skips already-downloaded files (safe to resume after interruption).

Usage:
    python -m scripts.build_test_set                  # sample + download
    python -m scripts.build_test_set --no-download    # just write manifest/JSONL
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from data.enoe_images import KAGGLE_DATASET, RARE_LABELS, load_annotations, local_image_path


def sample_lows_for_season(low_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Randomly sample n lows, keeping the season's day/night proportion."""
    night = low_df[low_df["is_night"]]
    day = low_df[~low_df["is_night"]]
    n_night = min(round(n * len(night) / len(low_df)), len(night))
    n_day = min(n - n_night, len(day))
    return pd.concat(
        [night.sample(n_night, random_state=seed), day.sample(n_day, random_state=seed)]
    )


def build_test_df(seed: int) -> pd.DataFrame:
    df = load_annotations()
    parts = []
    for season, season_df in df.groupby("season"):
        rare = season_df[season_df["level"].isin(RARE_LABELS)]
        lows = sample_lows_for_season(season_df[season_df["level"] == "low"], len(rare), seed)
        print(
            f"{season}: rare={len(rare)} {rare['level'].value_counts().to_dict()}, "
            f"lows sampled={len(lows)} (night={int(lows['is_night'].sum())})"
        )
        parts.append(pd.concat([rare, lows]))
    test_df = pd.concat(parts).sort_values("datetime")
    print(f"\ntotal: {len(test_df)}  {test_df['level'].value_counts().to_dict()}")
    return test_df


def download_one(row: pd.Series, images_dir: Path, retries: int = 2) -> tuple[str, bool]:
    target = local_image_path(row, images_dir)
    if target.exists():
        return str(target), True
    # Kaggle's single-file download saves the file's basename into -p (zipping only
    # non-image formats, so .jpg arrives as-is); rename to the flattened convention.
    downloaded = images_dir / Path(row["path"]).name
    cmd = [
        "kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
        "-f", row["kaggle_path"], "-p", str(images_dir), "--force",
    ]
    for _ in range(retries + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and downloaded.exists():
            downloaded.rename(target)
            return str(target), True
    return f"{row['kaggle_path']}: {result.stderr.strip().splitlines()[-1] if result.stderr else 'download failed'}", False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images-dir", default="data/raw/test_images")
    parser.add_argument("--out", default="data/processed/test_examples.jsonl")
    parser.add_argument("--manifest", default="data/processed/test_manifest.csv")
    parser.add_argument("--no-download", action="store_true", help="Only sample + write manifest/JSONL")
    parser.add_argument("--workers", type=int, default=8, help="Parallel Kaggle downloads")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    test_df = build_test_df(args.seed)

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(args.manifest)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for _, row in test_df.iterrows():
            f.write(json.dumps({
                "image_path": str(local_image_path(row, images_dir)),
                "kaggle_path": row["kaggle_path"],
                "datetime": row["datetime"].isoformat(),
                "season": row["season"],
                "is_night": bool(row["is_night"]),
                "place": row["place"],
                "label": row["level"],
            }) + "\n")
    print(f"wrote {args.manifest} and {args.out}")

    if args.no_download:
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for _, row in test_df.iterrows()]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, row, images_dir) for row in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            msg, ok = fut.result()
            if not ok:
                failures.append(msg)
            if i % 100 == 0 or i == len(rows):
                print(f"downloaded {i}/{len(rows)} ({len(failures)} failures)", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} downloads FAILED (rerun to retry):", file=sys.stderr)
        for msg in failures[:20]:
            print(f"  {msg}", file=sys.stderr)
        sys.exit(1)
    print("all downloads complete")


if __name__ == "__main__":
    main()
