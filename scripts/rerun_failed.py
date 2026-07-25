"""Re-classify the failed rows of a completed cell, in place.

Two kinds of row need repairing after a long unattended run:

* `api_error` -- a provider-side 500/503/429 that survived the SDK's own retries. The row
  holds a placeholder prediction (the least-severe label) purely because the request never
  landed. Observed at ~3-4% on the Gemini path during high-demand periods.
* `parse_failed` without `api_error` -- the completion came back but could not be parsed.
  Rare under `--json-mode schema`; expected under `off`.

Both leave a conservative placeholder that deflates flood recall, and neither is
missing-at-random, so scoring a cell before repairing it understates that cell.

The run configuration is read back from each row's own `config` block (taxonomy, prompt
variant, criteria, json mode, exemplar settings) and the provider from the cell's
`.meta.json` sidecar, so a repair cannot silently apply different settings than the
original run. Rows that fail again keep their placeholder and are reported.

Replaces the earlier scripts/rerun_unparseable.py, which had rotted: it called a
`classify_baseline` method that no longer exists and matched a fallback marker string the
classifier never writes.

Usage:
    python -m scripts.rerun_failed --results experiments/results/zs_3class_gemma4_31b.jsonl
    python -m scripts.rerun_failed --results ... --only api_error --rpm 8
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from agent.classifier import ClassifierAgent


def load_meta(results_path: Path) -> dict:
    meta_path = results_path.with_suffix(results_path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}
    return json.load(open(meta_path))


def build_agent(config: dict, provider: str, model: str, rpm: float) -> ClassifierAgent:
    """Rebuild the exact agent the cell was run with."""
    image_rag = config.get("image_rag")
    image_retriever = None
    if image_rag == "fewshot":
        from agent.image_retrieval import FixedFewShotExemplarProvider

        image_retriever = FixedFewShotExemplarProvider()
    elif image_rag == "similarity":
        from agent.image_retrieval import ImageBankRetriever

        image_retriever = ImageBankRetriever()

    return ClassifierAgent(
        image_retriever=image_retriever,
        use_image_rag=image_rag is not None,
        image_rag_mode=image_rag or "similarity",
        n_exemplars=config.get("n_exemplars", 3) or 3,
        taxonomy=config["taxonomy"],
        prompt_variant=config.get("prompt_variant", "v1"),
        include_criteria=config.get("include_criteria", True),
        json_mode=config.get("json_mode", "schema"),
        provider=provider,
        model=model,
        requests_per_minute=rpm,
    )


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Results JSONL to patch in place")
    parser.add_argument(
        "--only",
        choices=["all", "api_error", "parse_failed"],
        default="all",
        help="Which failures to retry (default: both kinds)",
    )
    parser.add_argument("--provider", default="", help="Defaults to the provider in the .meta.json sidecar")
    parser.add_argument(
        "--rpm",
        type=float,
        default=8.0,
        help="Deliberately below the run's own rate: these rows failed once already, often "
        "because the endpoint was saturated",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Report what would be retried, change nothing")
    args = parser.parse_args()

    path = Path(args.results)
    rows = [json.loads(line) for line in open(path)]

    def needs_retry(row: dict) -> bool:
        if args.only == "api_error":
            return bool(row.get("api_error"))
        if args.only == "parse_failed":
            return bool(row.get("parse_failed")) and not row.get("api_error")
        return bool(row.get("parse_failed")) or bool(row.get("api_error"))

    targets = [i for i, r in enumerate(rows) if needs_retry(r)]
    n_api = sum(1 for i in targets if rows[i].get("api_error"))
    print(
        f"{len(targets)}/{len(rows)} rows to retry "
        f"({n_api} api_error, {len(targets) - n_api} parse-only)"
    )
    if not targets:
        return
    if args.dry_run:
        for i in targets[:10]:
            reason = rows[i].get("api_error") or (rows[i].get("rationale", "")[:120])
            print(f"  line {i + 1}: {reason}")
        return

    meta = load_meta(path)
    provider = args.provider or meta.get("provider")
    if not provider:
        raise SystemExit(
            "provider unknown: no .meta.json sidecar next to the results file -- pass --provider"
        )
    config = rows[targets[0]].get("config")
    if not config:
        raise SystemExit(
            "rows carry no `config` block (pre-rework results file) -- rerun the cell instead"
        )
    model = rows[targets[0]].get("model") or meta.get("model")
    print(f"provider={provider} model={model} config={config}")

    agent = build_agent(config, provider, model, args.rpm)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        repaired = list(
            tqdm(
                pool.map(agent.classify, [rows[i]["example"] for i in targets]),
                total=len(targets),
                desc="re-classifying",
            )
        )

    fixed = 0
    for i, result in zip(targets, repaired):
        if not result["parse_failed"] and not result.get("api_error"):
            rows[i] = result
            fixed += 1

    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)

    still_failing = len(targets) - fixed
    print(f"repaired {fixed}/{len(targets)} (still failing: {still_failing}); backup at {path}.bak")
    if still_failing:
        print("Rerun to retry the remainder, or report the residual rate with the cell's metrics.")


if __name__ == "__main__":
    main()
