"""Compare the two supervised learners across annotation budgets.

Two panels, because the budget axes are not interchangeable: the left panel holds
class-balanced budgets (K-shot and balanced sampling), the right holds budgets drawn at the
pool's natural class prior. Plotting them on one axis would imply that 100 balanced images
and 100 natural-prior images are the same quantity, which is the opposite of one of the
findings.

Two series per panel, one per learner, coloured categorically (identity, not magnitude) and
direct-labelled as well as legended. Incomplete budget points -- those whose draws have not
all finished -- are omitted rather than plotted with fewer draws, so nothing here depends on
a run that has not landed.

Reads experiments/results/supervised_curve_summary.json, written by the scoring step.

Usage:
    python -m scripts.plot_supervised_curve
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

DEFAULT_IN = Path("experiments/results/supervised_curve_summary.json")
DEFAULT_OUT = Path("paper/figures/supervised_curve.pdf")

# Categorical slots 1 and 2; validated all-pairs, both above 3:1 on a light surface.
COLOURS = {"resnet": "#eb6834", "logreg": "#2a78d6"}
LABELS = {"resnet": "ResNet-50 (fine-tuned)", "logreg": "DINOv2 + logreg"}
REQUIRED_DRAWS = 3  # a ResNet budget point is only plotted once all its draws exist

TEXT_SECONDARY = "#52514e"
GRID = "#d8d8d4"


def load(path: Path):
    raw = json.load(open(path))
    panels: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for key, entry in raw.items():
        axis, imgs = key.split("|")
        imgs = int(imgs)
        # Skip points where the slower learner has not finished all its draws. The `all`
        # point is a single deterministic run, so it is exempt.
        if entry["resnet"]["n"] < REQUIRED_DRAWS and imgs != 900:
            print(f"  skipping {axis} {imgs} imgs -- only {entry['resnet']['n']}/{REQUIRED_DRAWS} draws")
            continue
        panel = panels.setdefault(axis, {"resnet": [], "logreg": []})
        for learner in ("resnet", "logreg"):
            panel[learner].append((imgs, entry[learner]["f1"]))
    for panel in panels.values():
        for series in panel.values():
            series.sort()
    return panels


def plot(panels, out: Path):
    order = ["balanced-family", "natural"]
    titles = {
        "balanced-family": "Class-balanced budgets",
        "natural": "Natural-prior budgets",
    }
    present = [a for a in order if a in panels]
    fig, axes = plt.subplots(1, len(present), figsize=(9.2, 3.6), sharey=True)
    if len(present) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, axis in zip(axes, present):
        ax.set_facecolor("white")
        # A panel with only two measured budgets spans a gap where a point is still
        # pending; a solid connector there would assert a crossover location we have not
        # measured, so those segments are dashed.
        sparse = len(panels[axis]["resnet"]) < 3
        for learner in ("logreg", "resnet"):
            pts = panels[axis][learner]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=COLOURS[learner], linewidth=2.0, marker="o",
                    markersize=6, markeredgecolor="white", markeredgewidth=1.5,
                    linestyle="--" if sparse else "-",
                    solid_capstyle="round", label=LABELS[learner], zorder=3)
            ax.annotate(LABELS[learner].split(" ")[0], xy=(xs[-1], ys[-1]),
                        xytext=(5, -2), textcoords="offset points",
                        color=COLOURS[learner], fontsize=7.5, va="center", zorder=5,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
        ax.set_xscale("log")
        ax.set_title(titles[axis], color=TEXT_SECONDARY, fontsize=9.5, pad=8)
        ax.set_xlabel("Labelled training images per fold (log)", color=TEXT_SECONDARY, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=3)

    axes[0].set_ylabel("macro-F$_1$", color=TEXT_SECONDARY, fontsize=9)
    legend = axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(DEFAULT_IN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    panels = load(Path(args.summary))
    for axis, series in panels.items():
        print(f"  {axis}: {len(series['resnet'])} budget points")
    plot(panels, Path(args.out))


if __name__ == "__main__":
    main()
