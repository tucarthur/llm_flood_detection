"""CLI entrypoint: run the classifier agent over a JSONL examples file, write JSONL results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from agent.classifier import ClassifierAgent
from agent.retrieval import KnowledgeBaseRetriever
from agent.tools import StructuredDataTools

USGS_CSV = Path(__file__).parent.parent / "data" / "raw" / "usgs_daily_values.csv"


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to examples JSONL")
    parser.add_argument("--out", required=True, help="Path to write results JSONL")
    parser.add_argument("--no-tools", action="store_true", help="Ablation: disable structured-data tool calls")
    parser.add_argument("--no-rag", action="store_true", help="Ablation: disable RAG retrieval")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N examples")
    args = parser.parse_args()

    examples = [json.loads(line) for line in open(args.input)]
    if args.limit:
        examples = examples[: args.limit]

    structured_tools = StructuredDataTools(pd.read_csv(USGS_CSV, dtype={"site_id": str}))
    retriever = KnowledgeBaseRetriever()
    agent = ClassifierAgent(
        structured_tools,
        retriever,
        use_tools=not args.no_tools,
        use_rag=not args.no_rag,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for ex in tqdm(examples, desc="classifying"):
            result = agent.classify(ex)
            f.write(json.dumps(result) + "\n")
            f.flush()

    print(f"Wrote {len(examples)} results to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
