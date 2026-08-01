"""Prompt-paraphrase spread across the five prompt variants, per model.

The five variants share byte-identical operative content -- judging rule, level definitions,
rarity note, cost-asymmetry instruction -- and differ only in incidental framing. The spread
across them is therefore a measure of how much an arbitrary wording choice is worth, and the
paper uses it as a yardstick: an effect is interpretable only if it exceeds the paraphrase
spread for the same model and metric.

Two scopes, because coverage is not uniform:

* `sub`  -- all four models, restricted to the 375-row subsample. v4 and v5 were only ever run
            at that size for the Gemma models and nemotron, so this is the only row set on
            which all five variants exist for every model. It is the scope for CROSS-MODEL
            comparison.
* `full` -- the 1,592-row evaluation set. Only minimax has all five variants at this size.

The distinction matters beyond coverage: macro-F1 moves with the class prior, which differs
between the two sets (flood is 4.7% of the full set, 20% of the subsample), so spreads from the
two scopes are not interchangeable.

The criteria effect (v1 with criteria vs the same prompt without) is computed on the SAME rows
as the spread it is compared against, since comparing an effect measured at one prior to a
spread measured at another would not be meaningful.

Usage:
    python -m scripts.sensitivity_spread --scope sub
    python -m scripts.sensitivity_spread --scope full --models minimax_m3
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from scripts.bootstrap_supervised import Cell, R, paired

MODELS = ["gemma4_31b", "gemma4_26b_a4b", "nemotron_nano_vl_8b", "minimax_m3"]
SHORT = {"gemma4_31b": "gemma-31b", "gemma4_26b_a4b": "gemma-26b",
         "nemotron_nano_vl_8b": "nemotron", "minimax_m3": "minimax"}
METRICS = ["macro_f1", "flood_recall", "event_detection_rate"]
SUBSAMPLE = "data/processed/kshot_subsample.jsonl"


def variant_path(model: str, v: str) -> str | None:
    """v1 is the zero-shot criteria cell; v2-v5 are the sensitivity cells.

    v4/v5 exist at full n only for minimax; every other model has them on the subsample.
    """
    if v == "v1":
        p = f"{R}/zs_3class_{model}.jsonl"
    else:
        full = f"{R}/sens_{v}_3class_{model}.jsonl"
        sub = f"{R}/sens_{v}_sub_{model}.jsonl"
        p = full if Path(full).exists() else sub
    return p if Path(p).exists() else None


def subsample_keys() -> set:
    keys = set()
    for line in open(SUBSAMPLE):
        row = json.loads(line)
        ex = row.get("example", row)
        keys.add((ex["datetime"], ex["place"]))
    return keys


def restricted_cell(path: str, keys: set | None, workdir: Path) -> Cell:
    """A Cell over `path`, optionally restricted to `keys`.

    Restriction is done by writing a filtered copy rather than by filtering inside Cell, so the
    scored file remains an ordinary results file and the same code path is used throughout.
    """
    if keys is None:
        return Cell(path)
    out = workdir / (Path(path).stem + ".sub.jsonl")
    if not out.exists():
        with open(out, "w") as f:
            for line in open(path):
                ex = json.loads(line)["example"]
                if (ex["datetime"], ex["place"]) in keys:
                    f.write(line)
    return Cell(str(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["sub", "full"], default="sub")
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--workdir", default="/tmp/sensitivity_spread")
    parser.add_argument("--out", default=f"{R}/sensitivity_spread.json")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    keys = subsample_keys() if args.scope == "sub" else None
    n_rows = 375 if args.scope == "sub" else 1592

    print(f"scope={args.scope} ({n_rows} rows per cell)\n")
    out: dict = {"scope": args.scope, "n_rows": n_rows, "models": {}}

    for model in args.models:
        paths = {v: variant_path(model, v) for v in ("v1", "v2", "v3", "v4", "v5")}
        missing = [v for v, p in paths.items() if p is None]
        # In `full` scope a subsample-only cell would silently contribute 375 rows, so drop it.
        if args.scope == "full":
            for v, p in list(paths.items()):
                if p and "_sub_" in p:
                    paths[v] = None
                    missing.append(v)
        present = {v: p for v, p in paths.items() if p}
        if len(present) < 2:
            print(f"{SHORT[model]}: only {len(present)} variant(s) at this scope -- skipped\n")
            continue

        scores = {}
        for v, p in present.items():
            cell = restricted_cell(p, keys, workdir)
            scores[v] = cell.score(np.ones(len(cell.event_ids), dtype=np.int64))

        print(f"--- {SHORT[model]} ({len(present)} variants{', missing ' + ','.join(missing) if missing else ''}) ---")
        header = "        " + "".join(f"{v:>10s}" for v in present) + "     range        sd"
        print(header)
        model_out = {"variants": list(present), "metrics": {}}
        for m in METRICS:
            vals = [scores[v][m] for v in present]
            rng = max(vals) - min(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"{m:8s}" + "".join(f"{x:10.3f}" for x in vals) + f"{rng:10.3f}{sd:10.3f}")
            model_out["metrics"][m] = {"values": dict(zip(present, vals)), "range": rng, "sd": sd}

        # The substantive effect the spread is a yardstick for, on the same rows.
        naive = f"{R}/zsnaive_3class_{model}.jsonl"
        if Path(naive).exists():
            a = restricted_cell(paths["v1"], keys, workdir)
            b = restricted_cell(naive, keys, workdir)
            print("criteria effect (v1 with criteria minus without), paired:")
            model_out["criteria_effect"] = {}
            for m in METRICS:
                r = paired([a], [b], m)
                verdict = "resolved" if r["significant"] else "ns"
                bigger = abs(r["delta"]) > model_out["metrics"][m]["range"]
                print(f"  {m:20s} {r['delta']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] {verdict:9s}"
                      f" {'EXCEEDS spread' if bigger else 'within paraphrase spread'}")
                model_out["criteria_effect"][m] = {
                    "delta": r["delta"], "lo": r["lo"], "hi": r["hi"],
                    "significant": r["significant"], "exceeds_spread": bool(bigger),
                }
        print()
        out["models"][model] = model_out

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
