"""CLI entrypoint: run the classifier agent over a JSONL examples file, write JSONL results."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
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
    parser.add_argument("--no-rag", action="store_true", help="Ablation: disable text RAG retrieval")
    parser.add_argument("--image-rag", action="store_true", help="Ablation: enable image-exemplar RAG retrieval")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Ablation floor: one image+prompt completion per example, no tools/RAG at all",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N examples")
    parser.add_argument(
        "--provider",
        choices=["vllm", "gemini"],
        default="vllm",
        help="Inference backend: local vLLM server or the Gemini API (needs GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--model", default="", help="Model name (defaults: VLLM_MODEL for vllm, GEMINI_MODEL or gemini-3.5-flash for gemini)"
    )
    parser.add_argument(
        "--base-url", default="", help="OpenAI-compatible endpoint override (defaults per provider)"
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=None,
        help="Throttle: max API requests/minute (default: 5 for gemini free tier, unthrottled for vllm)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent classification requests; --rpm is still enforced globally across workers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to --out, skipping as many input examples as it already contains lines",
    )
    args = parser.parse_args()
    if args.baseline and args.image_rag:
        parser.error("--baseline is the no-tools ablation floor; it can't be combined with --image-rag")
    rpm = args.rpm if args.rpm is not None else (5.0 if args.provider == "gemini" else 0.0)

    examples = [json.loads(line) for line in open(args.input)]
    if args.limit:
        examples = examples[: args.limit]

    already_done = 0
    if args.resume and Path(args.out).exists():
        already_done = sum(1 for _ in open(args.out))
        examples = examples[already_done:]
        print(f"Resuming: {already_done} results already in {args.out}, {len(examples)} to go", file=sys.stderr)

    retriever = None if args.baseline else KnowledgeBaseRetriever()
    image_retriever = None
    if args.image_rag:
        from agent.image_retrieval import ImageBankRetriever  # lazy: avoids requiring torch unless used

        image_retriever = ImageBankRetriever()

    agent = ClassifierAgent(
        retriever,
        use_rag=not (args.no_rag or args.baseline),
        image_retriever=image_retriever,
        use_image_rag=args.image_rag,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        requests_per_minute=rpm,
    )

    classify = agent.classify_baseline if args.baseline else agent.classify
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # pool.map keeps results in input order regardless of which worker finishes first,
    # so the output file stays line-aligned with the input (which --resume relies on).
    with open(args.out, "a" if args.resume else "w") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in tqdm(pool.map(classify, examples), total=len(examples), desc="classifying"):
            f.write(json.dumps(result) + "\n")
            f.flush()

    print(f"Wrote {len(examples)} results to {args.out}" + (f" (after {already_done} resumed)" if already_done else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
