"""Confusion matrices for gemma-4-26b across four prompting conditions.

The figure exists to substantiate a claim that scalar metrics can only imply. Section 5.10
reports that this model's flood recall rises from 0.613 to 0.924 between zero-shot and K=4
while its macro-F1 collapses from 0.682 to 0.201, and concludes it is "not a better classifier"
but one "progressively converting into a flood detector that calls almost everything flood".
Two numbers cannot show that; the matrices can. At K=4, 80% of dry frames and 91% of risky
frames are predicted `flood` -- the prediction mass has moved into one column irrespective of
the true class.

Design notes, in the order the choices were made:

* Form: a confusion matrix encodes magnitude, so the colour job is SEQUENTIAL -- a single hue,
  light to dark. Not a diverging or categorical scale, and never a rainbow: the cell value has
  no meaningful midpoint and no identity to carry.
* Row-normalised, so the diagonal reads directly as per-class recall and rows are comparable
  across panels even though the conditions differ in evaluation-set size (the zero-shot cells
  are restricted to the same 375 rows as the K-shot cells before scoring).
* Counts are printed alongside the proportion, because a proportion over 27 flood frames and
  one over 150 dry frames are not equally informative and the reader should see which is which.
* In-cell text switches to white above 0.55 so it stays legible against the dark end of the
  ramp; this is contrast, not decoration.

Usage:
    python -m scripts.plot_confusion_progression
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.bootstrap_supervised import Cell, R

DEFAULT_OUT = Path("paper/figures/confusion_progression.pdf")
SUBSAMPLE = "data/processed/kshot_subsample.jsonl"
LABELS = ["no_risk", "risky", "flood"]
MODEL = "gemma4_26b_a4b"

# Sequential, single hue. "Blues" is perceptually ordered light->dark, keeps its ordering under
# every common CVD simulation, and survives a greyscale print, which a multi-hue scale does not.
CMAP = "Blues"
TEXT_SECONDARY = "#52514e"
GRID = "#d8d8d4"

PANELS = [
    ("Zero-shot, no criteria", [f"{R}/zsnaive_3class_{MODEL}.jsonl"], True),
    ("Zero-shot, with criteria", [f"{R}/zs_3class_{MODEL}.jsonl"], True),
    ("$K$=1", [f"{R}/k1_e{i}_{MODEL}.jsonl" for i in range(3)], False),
    ("$K$=4", [f"{R}/k4_e{i}_{MODEL}.jsonl" for i in range(3)], False),
]


def subsample_keys() -> set:
    keys = set()
    for line in open(SUBSAMPLE):
        row = json.loads(line)
        ex = row.get("example", row)
        keys.add((ex["datetime"], ex["place"]))
    return keys


def restricted(path: str, keys: set, workdir: Path) -> str:
    """Zero-shot cells are full-n; restrict them to the K-shot rows so all panels share rows."""
    out = workdir / (Path(path).stem + ".sub.jsonl")
    if not out.exists():
        with open(out, "w") as f:
            for line in open(path):
                ex = json.loads(line)["example"]
                if (ex["datetime"], ex["place"]) in keys:
                    f.write(line)
    return str(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--workdir", default="/tmp/confusion_progression")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    keys = subsample_keys()

    mats = []
    for title, paths, needs_restrict in PANELS:
        use = [restricted(p, keys, workdir) if needs_restrict else p for p in paths]
        total = np.zeros((3, 3))
        for p in use:
            total += Cell(p).conf.sum(axis=0)
        # Mean over episodes, so a K-shot panel is one episode's worth of counts.
        mats.append((title, total / len(use)))

    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.0))
    fig.patch.set_facecolor("white")

    for ax, (title, counts) in zip(axes, mats):
        norm = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(title, fontsize=9.5, pad=7)
        ax.set_xticks(range(3), LABELS, fontsize=7.5, rotation=45, ha="right", family="monospace")
        if ax is axes[0]:
            ax.set_yticks(range(3), LABELS, fontsize=7.5, family="monospace")
            ax.set_ylabel("true class", fontsize=8.5, color=TEXT_SECONDARY)
        else:
            ax.set_yticks(range(3), [""] * 3)
        ax.set_xlabel("predicted", fontsize=8.5, color=TEXT_SECONDARY)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        # A 2px surface gap between cells, so adjacent fills read as separate marks.
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)

        for i in range(3):
            for j in range(3):
                v = norm[i, j]
                ax.text(j, i - 0.13, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.55 else "#1a1a1a")
                ax.text(j, i + 0.20, f"n={counts[i, j]:.0f}", ha="center", va="center",
                        fontsize=6.2,
                        color="#e8eef5" if v > 0.55 else TEXT_SECONDARY)

    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cax = fig.add_axes((0.945, 0.22, 0.011, 0.56))
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("share of true class", fontsize=7.5, color=TEXT_SECONDARY)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_edgecolor(GRID)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
