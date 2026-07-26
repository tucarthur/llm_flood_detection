"""CLI entrypoint: run the classifier agent over a JSONL examples file, write JSONL
results plus a `<out>.meta.json` sidecar recording exactly how the cell was produced.

One invocation = one cell of the experiment matrix. A cell is defined by
(taxonomy, prompt variant, criteria on/off, exemplar condition, K, model, json mode);
all of it is recorded in the sidecar and in every result row, because provider
endpoints drift behind stable model names over a multi-week matrix.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from agent.classifier import JSON_MODES, ClassifierAgent
from agent.prompts import PROMPT_VARIANTS
from data.taxonomy import TAXONOMIES


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to examples JSONL")
    parser.add_argument("--out", required=True, help="Path to write results JSONL")
    parser.add_argument(
        "--taxonomy",
        choices=TAXONOMIES,
        default="3class",
        help="Prompted task. '3class' (no_risk/risky/flood) is the primary; '4class' is the "
        "published taxonomy kept for comparability; 'binary' is the diagnostic arm",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=sorted(PROMPT_VARIANTS),
        default="v1",
        help="Framing paraphrase, for prompt-sensitivity reporting. Physical criteria are "
        "identical across variants; only incidental wording changes",
    )
    parser.add_argument(
        "--no-criteria",
        action="store_true",
        help="Naive zero-shot arm: withhold the verbal level definitions entirely",
    )
    parser.add_argument(
        "--json-mode",
        choices=JSON_MODES,
        default="schema",
        help="Structured-output enforcement. Hold this FIXED across the whole matrix -- "
        "verify per-provider support with scripts/check_json_mode.py first",
    )
    parser.add_argument("--image-rag", action="store_true", help="Attach labeled image exemplars")
    parser.add_argument(
        "--image-rag-mode",
        choices=["similarity", "fewshot"],
        default="similarity",
        help=(
            "Only used with --image-rag. 'similarity': DINOv2 nearest-neighbour retrieval "
            "from the exemplar bank. 'fewshot': a fixed curated calibration set covering "
            "each level, meant to teach the class boundaries rather than match the scene"
        ),
    )
    parser.add_argument(
        "--n-exemplars",
        type=int,
        default=3,
        help="Exemplars attached per call when --image-rag is set",
    )
    parser.add_argument(
        "--episodes",
        default="",
        help="K-shot arm: path to an episode manifest from data/episodes.py. Implies "
        "--image-rag with a fixed, event-disjoint support set instead of similarity retrieval",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Which episode within --episodes to run (one cell per episode; average across them)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N examples")
    parser.add_argument(
        "--provider",
        choices=["vllm", "gemini", "nvidia", "groq"],
        default="vllm",
        help="Inference backend: local vLLM server, Gemini API, NVIDIA NIM, or Groq (each needs its <PROVIDER>_API_KEY)",
    )
    parser.add_argument(
        "--model", default="", help="Model name (defaults: VLLM_MODEL / GEMINI_MODEL / NVIDIA_MODEL / GROQ_MODEL per provider)"
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

    # provider-default throttles: gemini free tier is 5 req/min; NVIDIA NIM's is ~40,
    # kept under with headroom. groq allows 30 req/min but its 30k tokens/min cap binds
    # first (~2.4k tokens per vision request -> ~12/min), and it also caps at 1,000
    # requests/DAY -- long groq runs must split across days (--resume). vllm is
    # self-hosted, no throttle.
    rpm = args.rpm if args.rpm is not None else {"gemini": 5.0, "nvidia": 30.0, "groq": 12.0}.get(args.provider, 0.0)

    examples = [json.loads(line) for line in open(args.input)]
    if args.limit:
        examples = examples[: args.limit]

    already_done = 0
    if args.resume and Path(args.out).exists():
        already_done = sum(1 for _ in open(args.out))
        examples = examples[already_done:]
        print(f"Resuming: {already_done} results already in {args.out}, {len(examples)} to go", file=sys.stderr)

    image_retriever = None
    episode_meta = {}
    if args.episodes:
        from agent.image_retrieval import EpisodeExemplarProvider

        image_retriever = EpisodeExemplarProvider(args.episodes, args.episode)
        # The support set is a fixed calibration set, not similarity-retrieved, so the
        # prompt must frame it as such (see agent/prompts.py:format_exemplars).
        args.image_rag = True
        args.image_rag_mode = "fewshot"
        args.n_exemplars = image_retriever.k
        episode_meta = {
            "episodes_manifest": args.episodes,
            "episode": args.episode,
            "k": image_retriever.k,
            # exact images per fold, so a cell is traceable to what it actually saw
            "support_paths": {s: image_retriever.support_paths(s) for s in image_retriever.supports},
        }
    elif args.image_rag:
        if args.image_rag_mode == "fewshot":
            from agent.image_retrieval import FixedFewShotExemplarProvider

            image_retriever = FixedFewShotExemplarProvider()
        else:
            from agent.image_retrieval import ImageBankRetriever  # lazy: avoids requiring torch unless used

            image_retriever = ImageBankRetriever()

    agent = ClassifierAgent(
        image_retriever=image_retriever,
        use_image_rag=args.image_rag,
        image_rag_mode=args.image_rag_mode,
        n_exemplars=args.n_exemplars,
        taxonomy=args.taxonomy,
        prompt_variant=args.prompt_variant,
        include_criteria=not args.no_criteria,
        json_mode=args.json_mode,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        requests_per_minute=rpm,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()

    n_failed = 0
    n_errored = 0

    def classify_resiliently(example: dict) -> dict:
        """A provider-side error must not kill a multi-hour unattended cell.

        Transient 429/500s do happen (a spontaneous 500 was observed during provider
        pilots), and the SDK's own retries can be exhausted. Without this, one bad
        request propagates out of pool.map and takes down the whole run. Errored rows are
        written in the normal schema with `api_error` set, keeping the file line-aligned
        with the input so --resume still works; rerun the cell to fill them in.
        """
        try:
            return agent.classify(example)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad; see docstring
            return {
                "observations": "",
                "rationale": f"api error: {type(exc).__name__}: {exc}"[:500],
                "classification": agent.fallback_label,
                "confidence": 0.0,
                "cited_evidence": [],
                "example": example,
                "parse_failed": True,
                "api_error": f"{type(exc).__name__}: {exc}"[:500],
                "resolved_on_attempt": None,
                "attempts": [],
                "latency_s": None,
                "exemplar_context": "",
                "config": agent.config(),
                "model": agent.model,
                "usage": {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

    # pool.map keeps results in input order regardless of which worker finishes first,
    # so the output file stays line-aligned with the input (which --resume relies on).
    with open(out_path, "a" if args.resume else "w") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in tqdm(pool.map(classify_resiliently, examples), total=len(examples), desc="classifying"):
            n_failed += bool(result["parse_failed"])
            n_errored += bool(result.get("api_error"))
            f.write(json.dumps(result) + "\n")
            f.flush()

    meta = {
        "input": args.input,
        "out": args.out,
        "provider": args.provider,
        "n_examples": len(examples),
        "n_resumed": already_done,
        "parse_failures": n_failed,
        "parse_failure_rate": round(n_failed / len(examples), 4) if examples else None,
        "api_errors": n_errored,
        "temperature": 0,
        "workers": args.workers,
        "rpm": rpm,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_s": round(time.monotonic() - started_monotonic, 1),
        "git_commit": _git_commit(),
        **agent.config(),
        **episode_meta,
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Wrote {len(examples)} results to {args.out}"
        + (f" (after {already_done} resumed)" if already_done else "")
        + f"; {n_failed} parse failures; meta -> {meta_path}",
        file=sys.stderr,
    )
    if n_errored:
        print(
            f"WARNING: {n_errored} rows failed with a provider-side error and hold placeholder "
            "predictions. Filter on `api_error` and rerun that cell before scoring it.",
            file=sys.stderr,
        )
    if n_failed:
        print(
            f"WARNING: {n_failed} rows ({n_failed / len(examples):.1%}) fell back to the "
            "least-severe label. Fallbacks are not random -- they concentrate on ambiguous "
            "frames -- so report this rate and re-check metrics with them excluded.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
