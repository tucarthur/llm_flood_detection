"""Aggregate metrics across all baselines + agent runs into one comparison table."""
from __future__ import annotations

import json
from pathlib import Path

from eval.faithfulness import evaluate_file
from eval.metrics import calibration_metrics, classification_metrics, load_predictions


def summarize(name: str, path: str) -> dict:
    df = load_predictions(path)
    metrics = classification_metrics(df)
    calibration = calibration_metrics(df)
    summary = {
        "name": name,
        "n": metrics["n"],
        "accuracy": round(metrics["accuracy"], 3),
        "macro_recall": round(metrics["macro_recall"], 3),
        "macro_f1": round(metrics["macro_f1"], 3),
        "missed_active_flood_rate": metrics["missed_active_flood_rate"],
        "false_alarm_rate": metrics["false_alarm_rate"],
    }
    if calibration:
        summary["brier_score_active_flood"] = round(calibration["brier_score_active_flood"], 3)
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--out", default="experiments/results/comparison_report.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_files = {
        "threshold_baseline": results_dir / "threshold_baseline.jsonl",
        "gbt_baseline": results_dir / "gbt_baseline.jsonl",
        "logreg_baseline": results_dir / "logreg_baseline.jsonl",
        "agent_full": results_dir / "sample_agent.jsonl",
        "agent_no_tools": results_dir / "agent_no_tools.jsonl",
        "agent_no_rag": results_dir / "agent_no_rag.jsonl",
        "agent_zero_shot": results_dir / "agent_zero_shot.jsonl",
    }

    summaries = []
    for name, path in run_files.items():
        if not path.exists():
            continue
        summaries.append(summarize(name, str(path)))
        if "agent" in name:
            faithfulness = evaluate_file(str(path))
            summaries[-1]["mean_faithfulness_rate"] = (
                round(faithfulness["mean_faithfulness_rate"], 3)
                if faithfulness["mean_faithfulness_rate"] is not None
                else None
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summaries, f, indent=2)

    header = f"{'name':<20}{'n':>5}{'acc':>8}{'macro_recall':>14}{'macro_f1':>10}{'missed_flood':>14}{'false_alarm':>13}{'faithful':>10}"
    print(header)
    for s in summaries:
        missed = round(s["missed_active_flood_rate"], 3) if s["missed_active_flood_rate"] is not None else "-"
        false_alarm = round(s["false_alarm_rate"], 3) if s["false_alarm_rate"] is not None else "-"
        faithful = s.get("mean_faithfulness_rate", "-")
        print(
            f"{s['name']:<20}{s['n']:>5}{s['accuracy']:>8}{s['macro_recall']:>14}{s['macro_f1']:>10}"
            f"{missed:>14}{false_alarm:>13}{faithful:>10}"
        )
    print(f"\nWrote full report to {args.out}")


if __name__ == "__main__":
    main()
