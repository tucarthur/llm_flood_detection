"""One representative frame per three-level class, as a single figure.

The three frames are deliberately drawn from the SAME camera position (SHOP2) and all in
daylight. The label is defined by a geometric relation -- how much bare concrete wall remains
between the water surface and the grass line -- so a panel that changed viewpoint or
illumination at the same time as water level would not show the reader what distinguishes the
classes. Holding both fixed makes water level the only thing that varies across the panels.

Frames were chosen by inspection from the daytime SHOP2 candidates in the evaluation set:

  no_risk  2019-01-31 11:51  shallow flow, canal bed visible, wall margin at its widest
  risky    2020-01-23 15:24  turbid and much higher, margin clearly reduced, still below the
                             wall top -- this one is a `high` frame, the upper half of `risky`
  flood    2021-02-02 15:12  wall entirely submerged, water in contact with the grass on both
                             banks, which is the physical threshold that defines the class

Usage:
    python -m scripts.make_class_examples
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

DEFAULT_OUT = Path("paper/figures/class_examples.pdf")

# (three-level class, original label, datetime shown, image path)
PANELS = [
    ("no_risk", "low", "2019-01-31 11:51",
     "data/raw/test_images/2019_01_31_20190131_115126-SHOP2.jpg"),
    ("risky", "high", "2020-01-23 15:24",
     "data/raw/test_images/2020_01_23_20200123_152412-SHOP2.jpg"),
    ("flood", "flood", "2021-02-02 15:12",
     "data/raw/test_images/2021_02_02_20210202_151222-SHOP2.jpg"),
]

TEXT_SECONDARY = "#52514e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    missing = [p for *_, p in PANELS if not Path(p).exists()]
    if missing:
        raise SystemExit(f"missing source frames: {missing}")

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.15))
    fig.patch.set_facecolor("white")

    for ax, (klass, original, when, path) in zip(axes, PANELS):
        ax.imshow(mpimg.imread(path))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#d8d8d4")
        # Class first, since that is what the panel illustrates; the source label is given
        # too because the mapping merges medium and high and the reader needs to see which
        # side of that merge the example came from.
        label = klass if original == klass else f"{klass}  ({original})"
        ax.set_title(label, fontsize=9, pad=4, fontfamily="monospace")
        ax.set_xlabel(when, fontsize=7, color=TEXT_SECONDARY, labelpad=3)

    fig.tight_layout(pad=0.4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
