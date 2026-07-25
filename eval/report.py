"""Aggregate every results file into one comparison report, grouped by prompted task.

Runs are only comparable within a taxonomy, so the table is grouped by it and never
averaged across. Two columns exist to keep the reader honest:

* `parse_fail` -- the fraction of rows that fell back to the least-severe label. Fallbacks
  concentrate on ambiguous frames, so a cell with a high rate has had its flood recall
  deflated by a formatting failure rather than a vision failure. Anything above a couple
  of percent invalidates cross-arm comparison until it is fixed or excluded.
* `event_det` -- per-event detection rate, with a cluster-bootstrap CI over events. This
  is the number to quote for an alarm system; per-frame `flood_rec` is computed over
  correlated near-duplicate frames and reads higher than the event evidence supports.

Published prior-work numbers (Ranieri et al. 2024) are deliberately NOT included: they
were produced on a different evaluation set, and putting them in this table would imply a
matched comparison that does not exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.faithfulness import evaluate_file
from eval.metrics import (
    bootstrap_ci,
    calibration_metrics,
    event_scores,
    frame_scores,
    load_predictions,
    prepare,
)


def summarize(name: str, path: str, resamples: int, with_ci: bool = True) -> dict:
    df = load_predictions(path)
    prepared, taxonomy = prepare(df)
    frame = frame_scores(prepared, taxonomy)
    event = event_scores(prepared, taxonomy)

    # Repeat the headline metric with fallback rows removed: the gap between the two is
    # how much of the arm's apparent behaviour is a parsing artifact.
    strict, _ = prepare(df, taxonomy, drop_parse_failures=True)
    strict_frame = frame_scores(strict, taxonomy) if len(strict) else {}

    summary = {
        "name": name,
        "taxonomy": taxonomy,
        "n": frame["n"],
        "parse_failure_rate": frame["parse_failure_rate"],
        "accuracy": frame["accuracy"],
        "macro_f1": frame["macro_f1"],
        "flood_recall": frame["flood_recall"],
        "flood_recall_excl_parse_failures": strict_frame.get("flood_recall"),
        "false_alarm_rate": frame["false_alarm_rate"],
        "mean_ordinal_error": frame["mean_ordinal_error"],
        **{k: v for k, v in event.items() if k != "note"},
    }
    if with_ci:
        summary["ci"] = bootstrap_ci(
            df,
            taxonomy,
            metrics=("flood_recall", "event_detection_rate", "macro_f1"),
            n_resamples=resamples,
        )
    calibration = calibration_metrics(df, taxonomy)
    if calibration:
        summary["brier_score_flood"] = round(calibration["brier_score_flood"], 3)

    # Faithfulness only means anything when exemplar/reference context was actually
    # injected; gate on that rather than on a filename convention.
    if any(json.loads(line).get("exemplar_context") for line in open(path)):
        faithfulness = evaluate_file(path)
        rate = faithfulness["mean_faithfulness_rate"]
        summary["mean_faithfulness_rate"] = round(rate, 3) if rate is not None else None
    return summary


def _fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci_str(summary: dict, metric: str) -> str:
    ci = summary.get("ci", {}).get("metrics", {}).get(metric)
    if not ci or ci["lo"] is None:
        # --no-ci: still show the point estimate rather than hiding the metric entirely.
        return _fmt(summary.get(metric), 2)
    return f"{_fmt(ci['point'], 2)} [{_fmt(ci['lo'], 2)},{_fmt(ci['hi'], 2)}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--out", default="experiments/results/comparison_report.json")
    parser.add_argument(
        "--resamples",
        type=int,
        default=500,
        help="Cluster-bootstrap resamples. 500 is a fast pass; use 2000 for anything reported",
    )
    parser.add_argument("--no-ci", action="store_true", help="Skip bootstrap CIs (much faster)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_files = {p.stem: p for p in sorted(results_dir.glob("*.jsonl"))} if results_dir.exists() else {}

    summaries = []
    for name, path in run_files.items():
        if name == "comparison_report":
            continue
        try:
            summaries.append(summarize(name, str(path), args.resamples, with_ci=not args.no_ci))
        except (ValueError, KeyError) as exc:
            print(f"  skipped {name}: {type(exc).__name__}: {exc}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summaries, f, indent=2)

    for taxonomy in sorted({s["taxonomy"] for s in summaries}):
        rows = [s for s in summaries if s["taxonomy"] == taxonomy]
        print(f"\n=== taxonomy: {taxonomy} ===")
        print(
            f"{'name':<34}{'n':>6}{'parse_fail':>12}{'macro_f1':>10}{'flood_rec':>11}"
            f"{'(excl fail)':>12}{'false_al':>10}{'event_det [95% CI]':>24}{'faithful':>10}"
        )
        for s in sorted(rows, key=lambda r: -(r["macro_f1"] or 0)):
            print(
                f"{s['name']:<34}{s['n']:>6}{_fmt(s['parse_failure_rate']):>12}"
                f"{_fmt(s['macro_f1']):>10}{_fmt(s['flood_recall']):>11}"
                f"{_fmt(s.get('flood_recall_excl_parse_failures')):>12}"
                f"{_fmt(s['false_alarm_rate']):>10}{_ci_str(s, 'event_detection_rate'):>24}"
                f"{_fmt(s.get('mean_faithfulness_rate')):>10}"
            )
        n_events = {s.get("ci", {}).get("n_events") for s in rows if s.get("ci")}
        if n_events - {None}:
            print(f"  (CIs: cluster bootstrap over {max(n_events - {None})} events, n={args.resamples})")

    print(f"\nWrote full report to {args.out}")


if __name__ == "__main__":
    main()
