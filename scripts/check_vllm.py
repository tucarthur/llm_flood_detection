"""Smoke test for the vLLM serving setup (docker-compose.yml) -- run this on the GPU
machine (or against it remotely) after `docker compose up -d`, before merging the
vllm-serving branch. Checks, in order:

1. The server is up and serving the expected model (GET /v1/models).
2. A real end-to-end ClassifierAgent.classify() call against one dev-sample image
   succeeds and returns a valid classification -- this is the part that actually
   matters, since tool-calling reliability varies a lot across open models and a
   healthy /v1/models response doesn't tell you tool calls actually work.

Exits 0 and prints "ALL CHECKS PASSED" only if both succeed; otherwise exits 1 with
a clear indication of which check failed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent


def check_server_up(base_url: str, expected_model: str) -> bool:
    print(f"[1/2] Checking {base_url}/models ...")
    try:
        resp = requests.get(f"{base_url}/models", timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  FAILED: could not reach vLLM server -- {exc}")
        return False

    model_ids = [m["id"] for m in resp.json().get("data", [])]
    if expected_model not in model_ids:
        print(f"  FAILED: expected model '{expected_model}' not in served models: {model_ids}")
        return False

    print(f"  OK -- server up, serving {model_ids}")
    return True


def check_classify_roundtrip() -> bool:
    print("[2/2] Running a real classify() call against a dev-sample image ...")
    examples_path = REPO_ROOT / "data" / "processed" / "examples.jsonl"
    if not examples_path.exists():
        print(
            f"  FAILED: {examples_path} not found -- run `python -m data.enoe_images` first "
            "to build it from data/raw/sample_images."
        )
        return False

    from agent.classifier import ClassifierAgent
    from agent.retrieval import KnowledgeBaseRetriever

    example = json.loads(examples_path.read_text().splitlines()[0])
    retriever = KnowledgeBaseRetriever()
    agent = ClassifierAgent(retriever, use_rag=True)

    try:
        result = agent.classify(example)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, this is a smoke test
        print(f"  FAILED: classify() raised {type(exc).__name__}: {exc}")
        return False

    valid_labels = {"low", "medium", "high", "flood"}
    if result.get("classification") not in valid_labels:
        print(f"  FAILED: classification {result.get('classification')!r} not in {valid_labels}")
        return False

    n_tool_calls = len(result.get("tool_call_log", []))
    if n_tool_calls == 0:
        print(
            "  WARNING: agent never called retrieve_flood_knowledge -- tool-calling may not "
            "be working for this model/tool-call-parser combination even though it produced "
            "a valid final answer. Worth checking before relying on RAG grounding."
        )

    print(f"  OK -- classification={result['classification']!r}, {n_tool_calls} tool call(s) logged")
    return True


def main() -> int:
    load_dotenv()
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

    server_ok = check_server_up(base_url, model)
    classify_ok = check_classify_roundtrip() if server_ok else False

    if server_ok and classify_ok:
        print("\nALL CHECKS PASSED")
        return 0

    print("\nFAILED -- see above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
