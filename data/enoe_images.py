"""Loader for the E-Noe river-camera dataset (Mineirinho Creek, Sao Carlos, Brazil).

Source: Ranieri, Souza, Nishijima, Krishnamachari & Ueyama, "A deep learning workflow
enhanced with optical flow fields for flood risk estimation," Applied Intelligence, 2024.
Kaggle: caetanoranieri/river-images-at-sao-carlos (CC-BY-NC-SA-4.0).

Annotation CSV (`flood_images_annot_2.csv`) is the labeled subset actually used in the
paper: SHOP/SHOP2 camera only (the other location, SESC/SESC2, was not carefully
labeled per the dataset's own documentation), restricted to the Nov-Feb rainy season
across 4 years (2018-19 through 2021-22) -- this is the paper's own definition of
"season," used here directly for leave-one-season-out CV rather than calendar years.

Label taxonomy (ordinal, physically defined via a fixed reference marker at the canal):
low < medium < high < flood, where `flood` specifically means water has risen above the
top of the concrete canal wall and is touching the grass bank -- not just "more severe,"
a distinct physical threshold.

Severe class imbalance (see data/README.md): low=67803 (98.84%), medium=535 (0.78%),
high=186 (0.27%), flood=75 (0.11%) across the full 68599-row labeled set.

Camera relocation: `place=SHOP` covers only 2018-11-01 to 2019-01-24 (~3 months, the
start of the 2018-2019 season); `place=SHOP2` covers everything after, through
2022-02-28. This is a one-time early relocation, not a per-season alternation -- the
2018-2019 season is therefore a mix of both angles, and has by far the fewest positive
examples of any season (2 flood, 13 high) -- treat leave-one-season-out results for that
fold as low-confidence given the tiny positive-class sample size.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).parent / "raw" / "flood_images_annot_2.csv"
KAGGLE_DATASET = "caetanoranieri/river-images-at-sao-carlos"
KAGGLE_PREFIX = "enoe/enoe2/"  # CSV `path` values are relative to this prefix in the dataset

LABELS = ["low", "medium", "high", "flood"]


def season_for(dt: pd.Timestamp) -> str:
    """Nov-Feb rainy season, matching the source paper's own definition -- a
    November date belongs to the season starting that year; a Jan/Feb date
    belongs to the season that started the previous November."""
    year = dt.year if dt.month >= 11 else dt.year - 1
    return f"{year}-{year + 1}"


def load_annotations(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- copy flood_images_annot_2.csv into data/raw/ first."
        )
    df = pd.read_csv(csv_path, index_col=0, parse_dates=["datetime"])
    df["season"] = df["datetime"].apply(season_for)
    df["hour"] = df["datetime"].dt.hour
    df["is_night"] = (df["hour"] < 6) | (df["hour"] >= 18)
    df["kaggle_path"] = KAGGLE_PREFIX + df["path"]
    return df


def local_image_path(row: pd.Series, images_dir: Path) -> Path:
    """Local filename convention used by the dev-sample downloader: CSV `path`
    with '/' replaced by '_' (e.g. '2018/11/01/x-SHOP.jpg' -> '2018_11_01_x-SHOP.jpg')."""
    return images_dir / row["path"].replace("/", "_")


def build_examples(csv_path: Path = CSV_PATH, images_dir: Path = None, only_existing: bool = True) -> list[dict]:
    """Build classification examples from the annotation CSV. If images_dir is
    given and only_existing=True, restricts to rows whose image file is
    actually present locally (e.g. the dev sample) rather than the full 68599."""
    df = load_annotations(csv_path)
    if images_dir is not None and only_existing:
        df = df[df.apply(lambda r: local_image_path(r, images_dir).exists(), axis=1)]

    examples = []
    for _, row in df.iterrows():
        examples.append(
            {
                "image_path": str(local_image_path(row, images_dir)) if images_dir else None,
                "kaggle_path": row["kaggle_path"],
                "datetime": row["datetime"].isoformat(),
                "season": row["season"],
                "is_night": bool(row["is_night"]),
                "place": row["place"],
                "label": row["level"],
            }
        )
    return examples


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="data/raw/sample_images")
    parser.add_argument("--out", default="data/processed/examples.jsonl")
    parser.add_argument("--all-rows", action="store_true", help="Include rows with no local image file (image_path will be null)")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    examples = build_examples(images_dir=images_dir, only_existing=not args.all_rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} examples to {args.out}")
    by_label = pd.Series([e["label"] for e in examples]).value_counts()
    print(by_label)
    by_season = pd.Series([e["season"] for e in examples]).value_counts().sort_index()
    print(by_season)
