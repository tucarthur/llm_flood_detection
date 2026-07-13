"""CLI entrypoint: run the classifier agent over a JSONL examples file, write JSONL results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from agent.classifier import ClassifierAgent
from agent.retrieval import KnowledgeBaseRetriever


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to examples JSONL")
    parser.add_argument("--out", required=True, help="Path to write results JSONL")
    parser.add_argument("--no-rag", action="store_true", help="Ablation: disable RAG retrieval")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N examples")
    args = parser.parse_args()

    examples = [json.loads(line) for line in open(args.input)]
    if args.limit:
        examples = examples[: args.limit]

    retriever = KnowledgeBaseRetriever()
    agent = ClassifierAgent(
        retriever,
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
