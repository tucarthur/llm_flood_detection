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
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from agent.classifier import ClassifierAgent

# Repairs are flushed to disk every this many successes.
CHECKPOINT_EVERY = 25


def load_meta(results_path: Path) -> dict:
    meta_path = results_path.with_suffix(results_path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}
    return json.load(open(meta_path))


def build_agent(config: dict, provider: str, model: str, rpm: float, meta: dict | None = None) -> ClassifierAgent:
    """Rebuild the exact agent the cell was run with.

    An episode-manifest cell records `image_rag: "fewshot"` in its per-row config, which is
    indistinguishable from a hand-curated calibration cell by config alone. The manifest path
    and episode index live only in the cell's .meta.json sidecar, so they must be consulted
    here -- rebuilding from the config alone would silently substitute a completely different
    support set and corrupt the repaired rows.
    """
    meta = meta or {}
    image_rag = config.get("image_rag")
    image_retriever = None
    if meta.get("episodes_manifest"):
        from agent.image_retrieval import EpisodeExemplarProvider

        image_retriever = EpisodeExemplarProvider(meta["episodes_manifest"], meta.get("episode", 0))
        print(f"episode support set: {meta['episodes_manifest']} episode {meta.get('episode', 0)}")
    elif image_rag == "fewshot":
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
    parser.add_argument(
        "--error-pause",
        type=float,
        default=30.0,
        help="Seconds to wait after a raised provider error, which arrive in bursts",
    )
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

    agent = build_agent(config, provider, model, args.rpm, meta)
    from concurrent.futures import ThreadPoolExecutor

    # The backup is taken BEFORE any writing, so it holds the cell as it was on entry
    # regardless of how far the run gets.
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))

    def write_back() -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        tmp.rename(path)

    errors: list[str] = []

    def attempt(index: int):
        """Classify one row, turning a raised provider error into a kept placeholder.

        `agent.classify` handles the failure modes it knows about, but an error the SDK
        re-raises after exhausting its own retries -- a sustained 429 on a throttled
        endpoint, in practice -- propagates out of the worker. Letting it escape used to
        kill the whole cell, and since the write-back happened only after every row had
        finished, it discarded every repair made up to that point: three cells crashed at
        30-50% and saved nothing. A row that fails here keeps its placeholder and is
        retried by the next invocation, which is what this script's contract has always
        said it does.
        """
        try:
            return index, agent.classify(rows[index]["example"])
        except Exception as exc:  # noqa: BLE001 -- any provider error must not end the run
            errors.append(f"{type(exc).__name__}: {exc}")
            # 429s arrive in bursts, so pausing after one keeps the next few rows from
            # being spent against a window that has not reopened yet.
            time.sleep(args.error_pause)
            return index, None

    fixed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, result in tqdm(
            pool.map(attempt, targets), total=len(targets), desc="re-classifying"
        ):
            if result is None or result["parse_failed"] or result.get("api_error"):
                continue
            rows[index] = result
            fixed += 1
            # Checkpoint, so an interruption costs at most CHECKPOINT_EVERY repairs
            # rather than the entire run.
            if fixed % CHECKPOINT_EVERY == 0:
                write_back()

    write_back()

    still_failing = len(targets) - fixed
    print(f"repaired {fixed}/{len(targets)} (still failing: {still_failing}); backup at {path}.bak")
    if errors:
        counts: dict[str, int] = {}
        for e in errors:
            counts[e.split(":")[0]] = counts.get(e.split(":")[0], 0) + 1
        print(f"raised errors (row kept its placeholder): {counts}")
    if still_failing:
        print("Rerun to retry the remainder, or report the residual rate with the cell's metrics.")


if __name__ == "__main__":
    main()
