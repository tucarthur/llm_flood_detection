"""Cluster-bootstrap confidence intervals for every supervised row in the paper.

`eval.metrics.bootstrap_ci` is the reference implementation and stays the definition of the
estimator. It is also too slow to use here: it rebuilds a DataFrame and calls
`classification_report` once per resample, so 2000 resamples cost ~3.3 minutes per results
file. The supervised curve tables aggregate 220 probe draw-files and 24 ResNet draw-files,
which that path would take roughly twelve hours to cover.

This module computes the identical quantity by resampling precomputed per-event confusion
matrices instead of rows. Summing 3x3 integer matrices with the resample multiplicities is
algebraically the same as concatenating the rows and re-scoring them, so the speedup costs
nothing in fidelity -- `--verify` checks this against `eval.metrics.bootstrap_ci` on a real
file and requires exact agreement to 1e-12.

Two estimator details are inherited deliberately rather than reinvented:

* Frame-level metrics count a twice-sampled event twice. Event-level metrics do not: the
  reference concatenates the sampled events and groups by `event_id`, which collapses
  duplicates, so event detection is a rate over the DISTINCT events in a resample. That is
  reproduced here (`--event-multiplicity` computes the alternative for comparison). It
  matters: the distinct-event variant effectively bootstraps ~63% of the events per draw.
* Multi-draw budget points report a mean over independent draws. The interval for such a row
  resamples events ONCE per bootstrap iteration and applies that same resample to every
  draw, so the interval describes sampling uncertainty of the reported mean. Draw-to-draw
  variance is a separate quantity and stays in the table's existing +/- column.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.taxonomy import labels_for, map_label, top_label
from eval.metrics import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, load_predictions

R = "experiments/results"


class Cell:
    """One scored predictions file, reduced to per-event sufficient statistics.

    Everything the bootstrap needs is additive over events, so a resample never has to touch
    the underlying rows again.
    """

    def __init__(self, path: str, taxonomy: str = "3class"):
        df = load_predictions(path)
        labels = labels_for(taxonomy)
        flood = top_label(taxonomy)
        index = {label: i for i, label in enumerate(labels)}
        self.flood_i = index[flood]

        gold = df["label"].map(lambda v: index[map_label(v, taxonomy)]).to_numpy()
        # A prediction outside the taxonomy cannot occur for probes but can for a prompted
        # cell; -1 keeps it out of the confusion matrix, matching zero_division=0 scoring.
        pred = df["classification"].map(lambda v: index.get(v, -1)).to_numpy()
        event_key = df["event_id"].to_numpy()
        self.event_ids = np.array(sorted(set(event_key)))
        event_pos = {eid: i for i, eid in enumerate(self.event_ids)}
        event = np.array([event_pos[e] for e in event_key])

        n_events, n_labels = len(self.event_ids), len(labels)
        self.conf = np.zeros((n_events, n_labels, n_labels), dtype=np.int64)
        valid = pred >= 0
        np.add.at(self.conf, (event[valid], gold[valid], pred[valid]), 1)

        # Dry-weather false alarms are defined against the ORIGINAL `low` label, not against
        # the taxonomy's least-severe class, so they stay comparable across taxonomies.
        is_low = (df["label"] == "low").to_numpy()
        self.low_total = np.bincount(event[is_low], minlength=n_events)
        self.low_flood = np.bincount(event[is_low & (pred == self.flood_i)], minlength=n_events)

        gold_flood, pred_flood = gold == self.flood_i, pred == self.flood_i
        self.has_flood = np.bincount(event[gold_flood], minlength=n_events) > 0
        self.detected = np.bincount(event[gold_flood & pred_flood], minlength=n_events) > 0
        self.has_nonflood = np.bincount(event[~gold_flood], minlength=n_events) > 0
        self.alarmed = np.bincount(event[~gold_flood & pred_flood], minlength=n_events) > 0

    def score(self, mult: np.ndarray, event_multiplicity: bool = False) -> dict:
        """Metrics for a resample given per-event multiplicities."""
        conf = np.tensordot(mult, self.conf, axes=(0, 0))
        tp = np.diag(conf).astype(float)
        support, predicted = conf.sum(axis=1), conf.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            recall = np.where(support > 0, tp / support, 0.0)
            precision = np.where(predicted > 0, tp / predicted, 0.0)
            denom = precision + recall
            f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

        # Event-level counts: distinct events unless multiplicity is requested (see module
        # docstring -- the reference collapses duplicates here and not at frame level).
        w = mult.astype(float) if event_multiplicity else (mult > 0).astype(float)
        n_flood_ev = float(w[self.has_flood].sum())
        n_det_ev = float(w[self.has_flood & self.detected].sum())
        n_nonflood_ev = float(w[self.has_nonflood].sum())
        n_alarm_ev = float(w[self.has_nonflood & self.alarmed].sum())

        low_total = float(mult @ self.low_total)
        return {
            "macro_f1": float(f1.mean()),
            "flood_recall": float(recall[self.flood_i]),
            "dry_false_alarm_rate": float(mult @ self.low_flood) / low_total if low_total else None,
            "event_detection_rate": n_det_ev / n_flood_ev if n_flood_ev else None,
            "n_flood_events_detected": n_det_ev,
            "event_false_alarm_rate": n_alarm_ev / n_nonflood_ev if n_nonflood_ev else None,
        }


    def restrict(self, event_ids: np.ndarray) -> "Cell":
        """A view of this cell over `event_ids`, in that order.

        Paired tests need both sides indexed identically so that one multiplicity vector
        applies to each, which is what makes the comparison paired rather than two
        independent resamples.
        """
        position = {eid: i for i, eid in enumerate(self.event_ids)}
        take = np.array([position[e] for e in event_ids])
        view = object.__new__(Cell)
        view.flood_i = self.flood_i
        view.event_ids = np.asarray(event_ids)
        for attr in ("conf", "low_total", "low_flood", "has_flood", "detected", "has_nonflood", "alarmed"):
            setattr(view, attr, getattr(self, attr)[take])
        return view


METRICS = (
    "macro_f1",
    "flood_recall",
    "dry_false_alarm_rate",
    "event_detection_rate",
    "n_flood_events_detected",
    "event_false_alarm_rate",
)


def bootstrap(
    cells: list[Cell],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci: float = 95.0,
    event_multiplicity: bool = False,
) -> dict:
    """Interval for the mean over `cells` of each metric.

    One cell is an ordinary single-run arm. Several cells are the independent draws of one
    budget point, which share a single event resample per iteration so that the interval
    refers to the mean the table actually prints.
    """
    reference = cells[0]
    n_events = len(reference.event_ids)
    for cell in cells[1:]:
        if not np.array_equal(cell.event_ids, reference.event_ids):
            raise SystemExit("draws of one budget point must be scored on the same events")

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in METRICS}
    for _ in range(n_resamples):
        # Sampling positions (not ids) reproduces the reference's rng.choice over the
        # sorted event_id array exactly, so the two implementations agree draw for draw.
        sampled = rng.choice(n_events, size=n_events, replace=True)
        mult = np.bincount(sampled, minlength=n_events)
        per_cell = [cell.score(mult, event_multiplicity) for cell in cells]
        for m in METRICS:
            values = [c[m] for c in per_cell if c[m] is not None]
            if values:
                draws[m].append(float(np.mean(values)))

    point_cells = [cell.score(np.ones(n_events, dtype=np.int64), event_multiplicity) for cell in cells]
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    out = {"n_events": n_events, "n_cells": len(cells), "n_resamples": n_resamples, "ci": ci, "metrics": {}}
    for m in METRICS:
        values = [c[m] for c in point_cells if c[m] is not None]
        out["metrics"][m] = {
            "point": float(np.mean(values)) if values else None,
            "lo": float(np.percentile(draws[m], lo_q)) if draws[m] else None,
            "hi": float(np.percentile(draws[m], hi_q)) if draws[m] else None,
        }
    return out


def paired(
    cells_a: list[Cell],
    cells_b: list[Cell],
    metric: str = "macro_f1",
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci: float = 95.0,
    event_multiplicity: bool = False,
) -> dict:
    """Interval on (A - B) with events resampled jointly for both arms.

    Every arm is scored on the identical evaluation set, so the paired interval removes the
    shared event-sampling variation that dominates the two marginal intervals. It is the
    test Section 2.9 of the paper requires before any claim that one arm beats another.

    Either side may be several draws of one budget point, in which case that side's value is
    the mean over its draws under the same resample -- matching how the tables report it.
    """
    shared = np.array(sorted(set(cells_a[0].event_ids) & set(cells_b[0].event_ids)))
    a = [c.restrict(shared) for c in cells_a]
    b = [c.restrict(shared) for c in cells_b]

    def value(cells: list[Cell], mult: np.ndarray) -> float | None:
        scores = [c.score(mult, event_multiplicity)[metric] for c in cells]
        scores = [s for s in scores if s is not None]
        return float(np.mean(scores)) if scores else None

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_resamples):
        sampled = rng.choice(len(shared), size=len(shared), replace=True)
        mult = np.bincount(sampled, minlength=len(shared))
        va, vb = value(a, mult), value(b, mult)
        if va is not None and vb is not None:
            deltas.append(va - vb)

    ones = np.ones(len(shared), dtype=np.int64)
    point_a, point_b = value(a, ones), value(b, ones)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    lo = float(np.percentile(deltas, lo_q)) if deltas else None
    hi = float(np.percentile(deltas, hi_q)) if deltas else None
    return {
        "metric": metric,
        "n_shared_events": len(shared),
        "n_draws_a": len(cells_a),
        "n_draws_b": len(cells_b),
        "a": point_a,
        "b": point_b,
        "delta": point_a - point_b,
        "lo": lo,
        "hi": hi,
        # A CI straddling zero means the arms are not distinguishable at this event count,
        # however far apart the point estimates look.
        "significant": lo is not None and (lo > 0 or hi < 0),
    }


# Every comparison the paper's text asserts, and therefore has to substantiate. The paper's
# summary count of resolved vs unresolved differences is taken from this dict, so a claim
# added to the text belongs here too.
def _probe(stem: str) -> list[str]:
    return [f"{R}/probe_logreg_3class_{stem}_e{i}.jsonl" for i in range(10)]


def _resnet(stem: str) -> list[str]:
    return [f"{R}/resnet50_3class_{stem}_e{i}.jsonl" for i in range(3)]


_ZS = [f"{R}/zsnaive_3class_gemma4_26b_a4b.jsonl"]

PAIRED_CLAIMS = {
    "resnet_vs_logreg_full": {
        "a": [f"{R}/resnet50_3class.jsonl"],
        "b": [f"{R}/probe_logreg_3class_kall.jsonl"],
        "claim": "Sec 3.1: the fine-tuned CNN is the strongest arm on aggregate correctness",
    },
    # Sec 3.3: where supervision overtakes zero-shot prompting. Run across the whole budget
    # range because the point at which the crossover becomes RESOLVABLE is the claim.
    "balanced24_vs_zeroshot": {"a": _probe("balanced24"), "b": _ZS, "claim": "Sec 3.3: 72 balanced frames"},
    "balanced40_vs_zeroshot": {"a": _probe("balanced40"), "b": _ZS, "claim": "Sec 3.3: 120 balanced frames"},
    "natural100_vs_zeroshot": {"a": _probe("natural100"), "b": _ZS, "claim": "Sec 3.3: 100 natural frames"},
    "natural200_vs_zeroshot": {"a": _probe("natural200"), "b": _ZS, "claim": "Sec 3.3: 200 natural frames"},
    "natural400_vs_zeroshot": {"a": _probe("natural400"), "b": _ZS, "claim": "Sec 3.3: 400 natural frames"},
    "all_vs_zeroshot": {"a": [f"{R}/probe_logreg_3class_kall.jsonl"], "b": _ZS, "claim": "Sec 3.3: full pool"},
    # Sec 3.6: probe against CNN on matched training sets, at every shared budget point.
    "probe_vs_resnet_k1": {"a": _probe("k1"), "b": _resnet("kshot1"), "claim": "Sec 3.6: 3 images"},
    "probe_vs_resnet_k2": {"a": _probe("k2"), "b": _resnet("kshot2"), "claim": "Sec 3.6: 6 images"},
    "probe_vs_resnet_k4": {"a": _probe("k4"), "b": _resnet("kshot4"), "claim": "Sec 3.6: 12 images"},
    "probe_vs_resnet_k6": {"a": _probe("k6"), "b": _resnet("kshot6"), "claim": "Sec 3.6: 18 images"},
    "probe_vs_resnet_bal12": {"a": _probe("balanced12"), "b": _resnet("balanced12"), "claim": "Sec 3.6: 36 images"},
    "probe_vs_resnet_bal40": {"a": _probe("balanced40"), "b": _resnet("balanced40"), "claim": "Sec 3.6: 120 images"},
    "probe_vs_resnet_nat100": {"a": _probe("natural100"), "b": _resnet("natural100"), "claim": "Sec 3.6: 100 images"},
    "probe_vs_resnet_nat400": {"a": _probe("natural400"), "b": _resnet("natural400"), "claim": "Sec 3.6: 400 images"},
    # Sec 5.3: the criteria-prompted zero-shot model against the CNN at each learning rate.
    # These are the comparisons that decide whether "zero-shot detects events supervision
    # cannot" survives, and at lr 3e-5 it does not.
    "zeroshot_vs_resnet_lr3e-5": {
        "a": [f"{R}/zs_3class_gemma4_31b.jsonl"],
        "b": [f"{R}/resnet50_3class_lr3e-05.jsonl"],
        "claim": "Sec 5.3: zero-shot vs CNN at lr 3e-5",
    },
    "zeroshot_vs_resnet_lr1e-4": {
        "a": [f"{R}/zs_3class_gemma4_31b.jsonl"],
        "b": [f"{R}/resnet50_3class_lr1e-04.jsonl"],
        "claim": "Sec 5.3: zero-shot vs CNN at lr 1e-4 (headline configuration)",
    },
    "zeroshot_vs_resnet_lr3e-4": {
        "a": [f"{R}/zs_3class_gemma4_31b.jsonl"],
        "b": [f"{R}/resnet50_3class_lr3e-04.jsonl"],
        "claim": "Sec 5.3: zero-shot vs CNN at lr 3e-4",
    },
    "probe_vs_resnet_all": {
        "a": [f"{R}/probe_logreg_3class_kall.jsonl"],
        "b": [f"{R}/resnet50_3class.jsonl"],
        "claim": "Sec 3.6: full pool",
    },
}


def verify_paired(path_a: str, path_b: str, n_resamples: int = 200) -> None:
    """Assert the paired implementation reproduces eval.metrics.paired_bootstrap_delta."""
    from eval.metrics import paired_bootstrap_delta

    reference = paired_bootstrap_delta(
        load_predictions(path_a), load_predictions(path_b), metric="macro_f1", n_resamples=n_resamples
    )
    fast = paired([Cell(path_a)], [Cell(path_b)], metric="macro_f1", n_resamples=n_resamples)
    worst = 0.0
    for key in ("a", "b", "delta", "lo", "hi"):
        worst = max(worst, abs(reference[key] - fast[key]))
        print(f"  {key:6s} reference={reference[key]:.12f} fast={fast[key]:.12f}")
    if worst > 1e-12:
        raise SystemExit(f"FAIL: diverges from eval.metrics.paired_bootstrap_delta by {worst:.2e}")
    print(f"OK: matches eval.metrics.paired_bootstrap_delta to {worst:.2e}")


def verify(path: str, n_resamples: int = 200) -> None:
    """Assert this implementation reproduces eval.metrics.bootstrap_ci exactly."""
    from eval.metrics import bootstrap_ci

    reference = bootstrap_ci(
        load_predictions(path),
        metrics=("flood_recall", "macro_f1", "event_detection_rate"),
        n_resamples=n_resamples,
    )
    fast = bootstrap([Cell(path)], n_resamples=n_resamples)
    worst = 0.0
    for metric in ("flood_recall", "macro_f1", "event_detection_rate"):
        for bound in ("point", "lo", "hi"):
            a, b = reference["metrics"][metric][bound], fast["metrics"][metric][bound]
            worst = max(worst, abs(a - b))
            print(f"  {metric:22s} {bound:5s} reference={a:.12f} fast={b:.12f} delta={abs(a - b):.2e}")
    if worst > 1e-12:
        raise SystemExit(f"FAIL: diverges from eval.metrics.bootstrap_ci by {worst:.2e}")
    print(f"OK: matches eval.metrics.bootstrap_ci to {worst:.2e} over {n_resamples} resamples")


# --------------------------------------------------------------------------------------
# The rows of each supervised table in the paper
# --------------------------------------------------------------------------------------
def table_rows() -> dict[str, dict[str, list[str]]]:
    """table -> row label -> the draw files averaged for that row."""
    kshot = {"1": "k1", "2": "k2", "4": "k4", "6": "k6"}
    balanced = {"12": "balanced12", "24": "balanced24", "40": "balanced40"}
    natural = {"50": "natural50", "100": "natural100", "200": "natural200", "400": "natural400"}

    def probe_draws(kind: str, stem: str) -> list[str]:
        return [f"{R}/probe_{kind}_3class_{stem}_e{i}.jsonl" for i in range(10)]

    def resnet_draws(stem: str) -> list[str]:
        return [f"{R}/resnet50_3class_{stem}_e{i}.jsonl" for i in range(3)]

    curve: dict[str, list[str]] = {}
    for learner in ("logreg", "prototype"):
        for axis, points in (("K", kshot), ("balanced", balanced), ("natural", natural)):
            for budget, stem in points.items():
                curve[f"{learner}|{axis}|{budget}"] = probe_draws(learner, stem)
        curve[f"{learner}|all|-"] = [f"{R}/probe_{learner}_3class_kall.jsonl"]

    two: dict[str, list[str]] = {}
    for budget, stem in (("K=1", "kshot1"), ("K=2", "kshot2"), ("K=4", "kshot4"), ("K=6", "kshot6"),
                         ("12/class", "balanced12"), ("40/class", "balanced40"),
                         ("natural100", "natural100"), ("natural400", "natural400")):
        two[f"resnet|{budget}"] = resnet_draws(stem)
    two["resnet|all"] = [f"{R}/resnet50_3class.jsonl"]

    return {
        "fullbudget": {
            "ResNet-50 (fine-tuned)": [f"{R}/resnet50_3class.jsonl"],
            "DINOv2 + logreg": [f"{R}/probe_logreg_3class_kall.jsonl"],
            "DINOv2 + prototype": [f"{R}/probe_prototype_3class_kall.jsonl"],
            "gemma-4-26b, no criteria": [f"{R}/zsnaive_3class_gemma4_26b_a4b.jsonl"],
            "gemma-4-31b, with criteria": [f"{R}/zs_3class_gemma4_31b.jsonl"],
        },
        "lrband": {
            "3e-05": [f"{R}/resnet50_3class_lr3e-05.jsonl"],
            "1e-04": [f"{R}/resnet50_3class_lr1e-04.jsonl"],
            "3e-04": [f"{R}/resnet50_3class_lr3e-04.jsonl"],
        },
        "curve": curve,
        "twolearners": two,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=f"{R}/supervised_ci.json")
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--tables", nargs="*", default=None, help="Subset of tables to compute")
    parser.add_argument("--verify", metavar="RESULTS_JSONL", help="Check against eval.metrics and exit")
    parser.add_argument(
        "--verify-paired", nargs=2, metavar=("A", "B"), help="Check the paired test against eval.metrics and exit"
    )
    parser.add_argument("--paired", action="store_true", help="Run the paired tests for PAIRED_CLAIMS")
    parser.add_argument(
        "--event-multiplicity",
        action="store_true",
        help="Count duplicated events with multiplicity in event-level metrics (the reference "
        "implementation does not; see module docstring)",
    )
    args = parser.parse_args()

    if args.verify:
        verify(args.verify)
        return
    if args.verify_paired:
        verify_paired(*args.verify_paired)
        return
    if args.paired:
        out = {}
        for name, spec in PAIRED_CLAIMS.items():
            missing = [p for p in spec["a"] + spec["b"] if not Path(p).exists()]
            if missing:
                print(f"  SKIP {name}: {len(missing)} files missing")
                continue
            for metric in ("macro_f1", "flood_recall", "event_detection_rate"):
                result = paired(
                    [Cell(p) for p in spec["a"]],
                    [Cell(p) for p in spec["b"]],
                    metric=metric,
                    n_resamples=args.resamples,
                )
                result["claim"] = spec["claim"]
                out[f"{name}|{metric}"] = result
                verdict = "DISTINGUISHABLE" if result["significant"] else "not distinguishable"
                print(
                    f"  {name} [{metric}]: {result['a']:.3f} - {result['b']:.3f} = "
                    f"{result['delta']:+.3f} [{result['lo']:+.3f}, {result['hi']:+.3f}]  {verdict}"
                )
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.out}")
        return

    tables = table_rows()
    wanted = args.tables or list(tables)
    out: dict[str, dict] = {}
    for table in wanted:
        out[table] = {}
        for row, paths in tables[table].items():
            missing = [p for p in paths if not Path(p).exists()]
            if missing:
                print(f"  SKIP {table}/{row}: {len(missing)}/{len(paths)} draw files missing")
                continue
            cells = [Cell(p) for p in paths]
            result = bootstrap(cells, n_resamples=args.resamples, event_multiplicity=args.event_multiplicity)
            out[table][row] = result
            f1, fr = result["metrics"]["macro_f1"], result["metrics"]["flood_recall"]
            print(
                f"  {table}/{row}: macro-F1 {f1['point']:.3f} [{f1['lo']:.3f}, {f1['hi']:.3f}]  "
                f"flood rec. {fr['point']:.3f} [{fr['lo']:.3f}, {fr['hi']:.3f}]  ({len(cells)} draws)"
            )

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
