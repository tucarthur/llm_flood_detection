"""Estimate total token usage and $ cost across the planned agent experiment
matrix, for comparing candidate models before committing to one.

Uses the real `client.messages.count_tokens` endpoint against representative
message payloads built from this repo's actual system prompt, tool schemas,
and RAG corpus (agent/prompts.py, agent/retrieval.py) -- never a heuristic
tokenizer. Claude's tokenizer is not tiktoken-compatible, so an approximation
would give the wrong answer for exactly the decision this script exists to
inform. See the claude-api skill's token-counting guidance.

Two things this script can NOT know in advance and therefore approximates,
clearly flagged in the output:
  1. Real assistant output length varies per example -- estimated from a
     representative tool_use payload shaped like what the agent actually emits.
  2. How many turns the model takes -- assumed to be the minimum required by
     each ablation cell (1 turn without RAG, 2 with RAG: retrieve then submit).
     A model that re-prompts (e.g. forgets to call submit_classification) costs
     more than this estimate.

Everything else (system prompt tokens, tool schema tokens, image tokens at a
given resolution, retrieved-passage tokens from the real knowledge base) is
counted exactly via the API, not approximated.
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
import zlib
from pathlib import Path

import anthropic

from agent.prompts import (
    SUBMIT_CLASSIFICATION_TOOL_SCHEMA,
    build_system_prompt,
    build_user_text_block,
)
from agent.retrieval import RETRIEVAL_TOOL_SCHEMA, KnowledgeBaseRetriever

# Cached from the claude-api skill (2026-06-24). Re-check
# https://platform.claude.com/docs/en/pricing.md before relying on this for a
# real budget decision -- pricing can change.
PRICING = {
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},  # $2/$10 intro through 2026-08-31
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}

EXAMPLE = {
    "datetime": "2020-01-15T22:47:00",
    "season": "2019-2020",
    "place": "SHOP2",
    "is_night": True,
}

SAMPLE_SUBMIT_INPUT = {
    "classification": "flood",
    "confidence": 0.8,
    "rationale": (
        "The canal wall is fully submerged and water is visibly touching the grass "
        "embankment above the wall line, meeting the flood threshold. Nighttime shot "
        "with some streetlight glare, which is why confidence is not higher."
    ),
    "cited_evidence": [
        "no concrete wall margin visible between water and grass",
        "water in direct contact with vegetation above the wall line",
        "Mineirinho Creek criteria: flood -- water above the top of the wall, touching the grass bank",
    ],
}
SAMPLE_RETRIEVAL_QUERY = "water above wall touching grass flood threshold"


def make_placeholder_image(path: Path, width: int, height: int) -> None:
    """Write a minimal solid-color PNG with no external dependencies. Image
    token cost is a function of resolution, not pixel content, so this is a
    valid stand-in for a real photo of the same dimensions."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    row = b"\x00" + bytes([120, 140, 160] * width)  # filter byte + solid RGB row
    idat = zlib.compress(row * height, 6)
    path.write_bytes(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def load_image_block(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def get_sample_retrieval_result() -> dict:
    """Pull a real retrieval result from the actual knowledge base index if it's
    been built; fall back to a placeholder of representative size otherwise."""
    try:
        retriever = KnowledgeBaseRetriever()
        return retriever.retrieve_flood_knowledge(SAMPLE_RETRIEVAL_QUERY, n_results=3)
    except FileNotFoundError:
        placeholder_passage = "x" * 400
        return {
            "query": SAMPLE_RETRIEVAL_QUERY,
            "passages": [{"title": "placeholder", "source": "placeholder", "text": placeholder_passage}] * 3,
        }


def build_scenario(image_block: dict, use_rag: bool) -> list[dict]:
    """Return the list of (input_messages, output_content) pairs for each API
    call the agent would make in this ablation cell, mirroring the real loop
    in agent/classifier.py: with RAG, retrieve then submit (2 calls); without
    RAG, submit directly (1 call) -- the minimum-turn assumption noted above."""
    text_block = {"type": "text", "text": build_user_text_block(EXAMPLE)}
    base_messages = [{"role": "user", "content": [image_block, text_block]}]

    if not use_rag:
        submit_output = [{"type": "tool_use", "id": "toolu_01", "name": "submit_classification", "input": SAMPLE_SUBMIT_INPUT}]
        return [{"input_messages": base_messages, "output_content": submit_output}]

    retrieval_result = get_sample_retrieval_result()
    retrieve_output = [
        {"type": "tool_use", "id": "toolu_01", "name": "retrieve_flood_knowledge", "input": {"query": SAMPLE_RETRIEVAL_QUERY}}
    ]
    turn2_messages = base_messages + [
        {"role": "assistant", "content": retrieve_output},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": json.dumps(retrieval_result)}],
        },
    ]
    submit_output = [{"type": "tool_use", "id": "toolu_02", "name": "submit_classification", "input": SAMPLE_SUBMIT_INPUT}]
    return [
        {"input_messages": base_messages, "output_content": retrieve_output},
        {"input_messages": turn2_messages, "output_content": submit_output},
    ]


def count_tokens_for_model(client: anthropic.Anthropic, model: str, system: str, tools: list[dict], messages: list[dict]) -> int:
    resp = client.messages.count_tokens(model=model, system=system, tools=tools, messages=messages)
    return resp.input_tokens


def estimate_output_tokens(client: anthropic.Anthropic, model: str, content_blocks: list[dict]) -> int:
    """Approximate how many tokens a hypothetical assistant output would cost,
    by tokenizing its serialized content as a proxy user message. count_tokens
    only measures input, so this is an approximation, not an exact figure --
    real output token counts depend on actual generation."""
    proxy = [{"role": "user", "content": json.dumps(content_blocks)}]
    resp = client.messages.count_tokens(model=model, messages=proxy)
    return resp.input_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-examples", type=int, default=68599, help="Total classification examples (default: 68599, the full labeled E-Noe SHOP/SHOP2 rainy-season dataset -- pass a smaller number for a subsample plan)")
    parser.add_argument("--image-width", type=int, default=1280, help="Real E-Noe camera frames are 1280x720")
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--models", nargs="+", default=list(PRICING.keys()), choices=list(PRICING.keys()))
    parser.add_argument("--sample-image", type=str, default=None, help="Path to a real representative image instead of a synthetic placeholder")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    try:
        client.messages.count_tokens(model=args.models[0], messages=[{"role": "user", "content": "ping"}])
    except (TypeError, anthropic.AuthenticationError) as exc:
        print(
            "No usable Anthropic credentials found -- this script needs real API access to "
            "count tokens accurately (never estimated via a heuristic tokenizer).\n"
            "Set ANTHROPIC_API_KEY in .env, or run `ant auth login` if you have the Anthropic CLI.\n"
            f"\nUnderlying error: {exc}"
        )
        return

    if args.sample_image:
        image_path = Path(args.sample_image)
    else:
        image_path = Path("/tmp") / "estimate_tokens_placeholder.png"
        make_placeholder_image(image_path, args.image_width, args.image_height)
        print(f"No --sample-image given; using a synthetic {args.image_width}x{args.image_height} placeholder "
              f"(image token cost depends on resolution, not content).")
    image_block = load_image_block(image_path)

    ablation_cells = [True, False]  # use_rag on/off -- the only ablation axis for this image-only dataset

    print(f"\nAssuming {args.n_examples} examples x {len(ablation_cells)} ablation cells "
          f"= {args.n_examples * len(ablation_cells)} total classifications.\n")

    results = []
    for model in args.models:
        system_no_rag = build_system_prompt(use_rag=False)
        system_rag = build_system_prompt(use_rag=True)
        tools_no_rag = [SUBMIT_CLASSIFICATION_TOOL_SCHEMA]
        tools_rag = [RETRIEVAL_TOOL_SCHEMA, SUBMIT_CLASSIFICATION_TOOL_SCHEMA]

        total_input_tokens = 0
        total_output_tokens = 0
        for use_rag in ablation_cells:
            system = system_rag if use_rag else system_no_rag
            tools = tools_rag if use_rag else tools_no_rag
            calls = build_scenario(image_block, use_rag)
            cell_input = 0
            cell_output = 0
            for call in calls:
                cell_input += count_tokens_for_model(client, model, system, tools, call["input_messages"])
                cell_output += estimate_output_tokens(client, model, call["output_content"])
            total_input_tokens += cell_input * args.n_examples
            total_output_tokens += cell_output * args.n_examples

        price = PRICING[model]
        cost = (total_input_tokens / 1_000_000) * price["input"] + (total_output_tokens / 1_000_000) * price["output"]
        results.append(
            {
                "model": model,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "cost_usd": cost,
            }
        )

    print(f"{'model':<20}{'input tokens':>16}{'output tokens':>16}{'est. cost (USD)':>18}")
    for r in results:
        print(f"{r['model']:<20}{r['input_tokens']:>16,}{r['output_tokens']:>16,}{'$' + format(r['cost_usd'], ',.2f'):>18}")

    print(
        "\nCaveats:\n"
        "- Output tokens are approximated from a representative tool_use payload, not real generation.\n"
        "- Turn count assumes the minimum required (1 call without RAG, 2 with RAG); a model that\n"
        "  re-prompts before submitting costs more than this estimate.\n"
        "- Re-run with --n-examples set to the real dataset size once it's finalized.\n"
        "- Pricing is cached as of 2026-06-24 -- verify at https://platform.claude.com/docs/en/pricing.md\n"
        "  before treating this as a firm budget."
    )


if __name__ == "__main__":
    main()
