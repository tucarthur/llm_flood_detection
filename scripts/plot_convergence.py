"""Plot ResNet-50 training convergence across the four leave-one-season-out folds.

One panel, training loss against epoch, one colour per fold. Folds are an identity
dimension, so the palette is categorical and assigned in fixed slot order; it was
validated for colour-vision deficiency separation under an all-pairs pairlist, since
convergence curves interleave and any pair may need distinguishing. The aqua slot sits
below 3:1 contrast on a white surface, which obliges direct labels rather than
legend-only identification -- so each curve is labelled at its right-hand end in addition
to the legend.

Best-epoch markers show where early stopping selected the reported weights. The y axis is
logarithmic because the loss spans two orders of magnitude and a linear axis compresses
everything after the first epoch into a flat line.

Usage:
    python -m scripts.plot_convergence
    python -m scripts.plot_convergence --history <csv> --out <pdf>
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

DEFAULT_HISTORY = Path("experiments/results/resnet50_3class_history.csv")
DEFAULT_OUT = Path("paper/figures/resnet50_convergence.pdf")

# Categorical slots 1, 2, 3, 7 of the reference palette. Slot 4 (yellow) is skipped
# deliberately: paired with slot 2 (orange) it fails the all-pairs separation floors.
FOLD_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d8d8d4"


def load_history(path: Path) -> dict[str, list[dict]]:
    by_fold: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_fold[row["held_out_season"]].append({
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_acc": float(row["val_acc"]),
                "is_best": row["is_best_epoch"] == "1",
                "n_train": int(row["n_train"]),
            })
    for rows in by_fold.values():
        rows.sort(key=lambda r: r["epoch"])
    return dict(by_fold)


def plot(history: dict[str, list[dict]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    max_epoch = max(r["epoch"] for rows in history.values() for r in rows)

    for slot, season in enumerate(sorted(history)):
        rows = history[season]
        colour = FOLD_COLOURS[slot % len(FOLD_COLOURS)]
        epochs = [r["epoch"] for r in rows]
        losses = [r["train_loss"] for r in rows]
        ax.plot(epochs, losses, color=colour, linewidth=2.0, solid_capstyle="round",
                label=f"{season} (n={rows[0]['n_train']})", zorder=3)

        best = next((r for r in rows if r["is_best"]), None)
        if best is not None:
            # 2px surface ring so a marker landing on another curve stays readable
            ax.plot(best["epoch"], best["train_loss"], marker="o", markersize=8,
                    color=colour, markeredgecolor="white", markeredgewidth=2, zorder=4)

        # Direct label at the curve end: the palette's contrast WARN obliges visible
        # labels rather than legend-only identity. Curves end at different epochs, so a
        # label can land on top of another fold's line -- a surface-coloured halo keeps it
        # legible without moving it away from the curve it identifies.
        ax.annotate(season, xy=(epochs[-1], losses[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    color=colour, fontsize=7.5, va="center", ha="left", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Training loss (log scale)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_xlim(-0.3, max_epoch + 2.6)
    ax.set_xticks(range(0, max_epoch + 1, 2))

    ax.grid(True, which="major", axis="both", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=3)

    legend = ax.legend(title="Held-out season", frameon=False, fontsize=8,
                       title_fontsize=8, loc="upper right")
    legend.get_title().set_color(TEXT_PRIMARY)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Vector PDF only -- the figure goes into a LaTeX document, where a raster would
    # resample badly at print resolution.
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default=str(DEFAULT_HISTORY))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    history = load_history(Path(args.history))
    print(f"{len(history)} folds, "
          f"{sum(len(v) for v in history.values())} epoch records")
    plot(history, Path(args.out))


if __name__ == "__main__":
    main()
