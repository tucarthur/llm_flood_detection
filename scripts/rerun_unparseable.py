"""Re-classify results rows whose baseline fell back to the unparseable-response
placeholder (classification 'low', confidence 0.0), replacing them in place.

Only rows whose rationale starts with the fallback marker are touched; everything
else is preserved byte-for-byte in order. The original file is backed up to
<results>.bak before rewriting. Rows that fail again keep their fallback form.

Usage:
    python -m scripts.rerun_unparseable --results experiments/results/baseline_gemma4_31b.jsonl \
        --provider gemini --rpm 15
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from agent.classifier import ClassifierAgent

FALLBACK_MARKER = "unparseable baseline response"


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Results JSONL to patch in place")
    parser.add_argument("--provider", required=True, choices=["vllm", "gemini", "nvidia", "groq"])
    parser.add_argument("--model", default="", help="Defaults per provider env, must match the original run")
    parser.add_argument("--rpm", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    path = Path(args.results)
    rows = [json.loads(line) for line in open(path)]
    failed_idx = [i for i, r in enumerate(rows) if r["rationale"].startswith(FALLBACK_MARKER)]
    print(f"{len(failed_idx)} unparseable rows out of {len(rows)}")
    if not failed_idx:
        return

    agent = ClassifierAgent(use_rag=False, provider=args.provider, model=args.model, requests_per_minute=args.rpm)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        new_results = list(
            tqdm(
                pool.map(agent.classify_baseline, [rows[i]["example"] for i in failed_idx]),
                total=len(failed_idx),
                desc="re-classifying",
            )
        )

    fixed = 0
    for i, result in zip(failed_idx, new_results):
        if not result["rationale"].startswith(FALLBACK_MARKER):
            rows[i] = result
            fixed += 1

    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)
    print(f"fixed {fixed}/{len(failed_idx)} (still failing: {len(failed_idx) - fixed}); backup at {path}.bak")


if __name__ == "__main__":
    main()
