"""Grounding-faithfulness check for agent outputs: does each cited_evidence string
actually correspond to something present in the retrieved context injected into the
prompt (text passages and/or image-exemplar descriptions), or is it fabricated/
unverifiable? This is an automatic proxy (word-overlap against retrieved_context) --
flagged low-overlap citations should be manually spot-checked, not trusted outright.

Retrieval is always-on per arm (not model-gated), so this checks grounding quality of
the *rationale*, not whether retrieval happened at all -- see agent/classifier.py.
"""
from __future__ import annotations

import json
import re

STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "at", "for", "is", "was",
    "were", "this", "that", "with", "as", "by", "from", "it", "its", "be", "are",
}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9.]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def citation_overlap(citation: str, tool_log_text: str) -> float:
    cite_words = _content_words(citation)
    if not cite_words:
        return 0.0
    log_words = _content_words(tool_log_text)
    return len(cite_words & log_words) / len(cite_words)


def check_faithfulness(agent_output: dict, overlap_threshold: float = 0.5) -> dict:
    context_text = agent_output.get("retrieved_context", "")
    citations = agent_output.get("cited_evidence", [])

    results = []
    for citation in citations:
        overlap = citation_overlap(citation, context_text)
        results.append({"citation": citation, "overlap": overlap, "grounded": overlap >= overlap_threshold})

    n_grounded = sum(r["grounded"] for r in results)
    return {
        "n_citations": len(citations),
        "n_grounded": n_grounded,
        "faithfulness_rate": n_grounded / len(citations) if citations else None,
        "citation_details": results,
        "had_retrieved_context": len(context_text) > 0,
    }


def evaluate_file(agent_results_path: str) -> dict:
    rows = [json.loads(line) for line in open(agent_results_path)]
    per_example = [check_faithfulness(r) for r in rows]
    rates = [r["faithfulness_rate"] for r in per_example if r["faithfulness_rate"] is not None]
    return {
        "n_examples": len(rows),
        "mean_faithfulness_rate": sum(rates) / len(rates) if rates else None,
        "examples_with_no_citations": sum(1 for r in per_example if r["n_citations"] == 0),
        "examples_with_no_retrieved_context": sum(1 for r in per_example if not r["had_retrieved_context"]),
        "per_example": per_example,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-results", required=True)
    args = parser.parse_args()
    summary = evaluate_file(args.agent_results)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_example"}, indent=2))
