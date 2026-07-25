"""Pilot one real vision call per JSON mode against a provider, before launching a cell.

Structured-output support is uneven across the four providers this study uses, and the
failure is usually a 400 on an unsupported `response_format` rather than anything subtle.
Enabling JSON mode only where it happens to work would replace a random confound with a
systematic per-model one, so the point of this script is to find ONE mode that works
everywhere and hold it fixed across the whole matrix.

Usage:
    python -m scripts.check_json_mode --provider gemini --model gemma-4-31b-it
    python -m scripts.check_json_mode --provider nvidia --model <model>   # repeat per model

Reports, per mode: whether the call succeeded, whether the response parsed, whether the
reasoning fields came back non-empty, and the finish reason. A mode that parses but
returns an empty `observations` field is a warning sign -- it means constrained decoding
is suppressing the reasoning the schema is meant to preserve.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from agent.classifier import JSON_MODES, ClassifierAgent
from data.taxonomy import TAXONOMIES

DEFAULT_IMAGE_DIR = Path("data/raw/sample_images")


def pick_sample_image(explicit: str | None) -> str:
    if explicit:
        return explicit
    images = sorted(DEFAULT_IMAGE_DIR.glob("*.jpg"))
    if not images:
        raise SystemExit(
            f"no sample images in {DEFAULT_IMAGE_DIR} -- pass --image with a path to one"
        )
    return str(images[0])


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["vllm", "gemini", "nvidia", "groq"])
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--taxonomy", choices=TAXONOMIES, default="3class")
    parser.add_argument("--image", default=None, help="Sample image (defaults to the first dev-sample frame)")
    args = parser.parse_args()

    image_path = pick_sample_image(args.image)
    example = {"image_path": image_path, "season": "2018-2019", "is_night": False}
    print(f"provider={args.provider} model={args.model or '<env default>'} image={image_path}\n")

    print(f"{'json_mode':<12}{'call':<10}{'parsed':<9}{'observations':<14}{'label':<10}{'finish_reason':<16}detail")
    for mode in JSON_MODES:
        try:
            agent = ClassifierAgent(
                taxonomy=args.taxonomy,
                json_mode=mode,
                provider=args.provider,
                model=args.model,
                base_url=args.base_url,
            )
            result = agent.classify(example)
        except Exception as exc:  # provider errors are the expected outcome here
            detail = str(exc).replace("\n", " ")[:110]
            print(f"{mode:<12}{'FAILED':<10}{'-':<9}{'-':<14}{'-':<10}{'-':<16}{detail}")
            continue

        attempts = result["attempts"]
        finish = attempts[-1].get("finish_reason") or "-"
        parsed = "no" if result["parse_failed"] else "yes"
        observations = result.get("observations") or ""
        obs = f"{len(observations)} chars" if observations else "EMPTY"
        detail = ""
        if result["parse_failed"]:
            detail = (attempts[-1].get("raw") or "")[:110].replace("\n", " ")
        elif not observations:
            detail = "reasoning field empty -- constrained decoding may be suppressing it"
        elif len(attempts) > 1:
            detail = f"needed {len(attempts)} attempts"
        print(
            f"{mode:<12}{'ok':<10}{parsed:<9}{obs:<14}"
            f"{result['classification']:<10}{str(finish):<16}{detail}"
        )

    print(
        "\nPick the strictest mode that works on EVERY provider/model in the matrix and pass "
        "it as --json-mode to every agent.run invocation. Record it in the cell's meta.json "
        "(agent.run does this automatically)."
    )


if __name__ == "__main__":
    main()
