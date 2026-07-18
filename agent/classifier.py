"""The classifier agent: a single-shot classification call over a vision-language model
served through an OpenAI-compatible chat completions API.

Retrieval (text and/or image) is not a tool the model may choose to invoke -- when an
arm enables it, it always runs before the model is called, and its results are injected
directly into the prompt. This means every arm (baseline / text-RAG / image-RAG /
combined) is architecturally identical -- one API call, temperature=0, JSON response --
differing only in what context is present. See resolve_provider for provider config.
"""
from __future__ import annotations

import json
import mimetypes
import os
import threading
import time
from pathlib import Path

from openai import OpenAI

from agent.image_retrieval import ImageBankRetriever
from agent.prompts import (
    IMAGE_RAG_N_RESULTS,
    TEXT_RAG_N_RESULTS,
    TEXT_RAG_QUERY,
    build_single_shot_prompt,
    build_user_text_block,
    format_retrieved_exemplars,
    format_retrieved_passages,
)
from agent.retrieval import KnowledgeBaseRetriever

VALID_LABELS = {"low", "medium", "high", "flood"}

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def resolve_provider(provider: str, model: str = "", base_url: str = "") -> tuple[str, str, str]:
    """-> (model, base_url, api_key) for the given provider, filling unset values from
    env vars. Env is read lazily here (not at import time) so load_dotenv() in the
    entrypoint takes effect."""
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("provider 'gemini' requires GEMINI_API_KEY (set it in .env)")
        return (
            model or os.environ.get("GEMINI_MODEL", "gemma-4-31b-it"),
            base_url or os.environ.get("GEMINI_BASE_URL", GEMINI_OPENAI_BASE_URL),
            api_key,
        )
    if provider == "nvidia":
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            raise ValueError("provider 'nvidia' requires NVIDIA_API_KEY (set it in .env)")
        return (
            # no default model: the study runs several (see .env.example), so pass --model explicitly
            model or os.environ.get("NVIDIA_MODEL", ""),
            base_url or os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key,
        )
    if provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("provider 'groq' requires GROQ_API_KEY (set it in .env)")
        return (
            model or os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
            base_url or os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key,
        )
    if provider == "vllm":
        return (
            model or os.environ.get("VLLM_MODEL", ""),
            base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            os.environ.get("VLLM_API_KEY", "EMPTY"),  # vLLM ignores this unless an auth proxy requires it
        )
    raise ValueError(f"unknown provider: {provider!r} (expected 'vllm', 'gemini', 'nvidia', or 'groq')")


def _image_url_content(image_path: str) -> dict:
    path = Path(image_path)
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    import base64

    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}


def _image_data_to_content(image_data: dict) -> dict:
    """{media_type, data} (as returned by ImageBankRetriever.dispatch) -> an OpenAI
    chat-completions image_url content part."""
    return {"type": "image_url", "image_url": {"url": f"data:{image_data['media_type']};base64,{image_data['data']}"}}


class ClassifierAgent:
    def __init__(
        self,
        retriever: KnowledgeBaseRetriever | None = None,
        use_rag: bool = True,
        image_retriever: ImageBankRetriever | None = None,
        use_image_rag: bool = False,
        client: OpenAI | None = None,
        provider: str = "vllm",
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        requests_per_minute: float = 0.0,
    ):
        self.retriever = retriever
        self.use_rag = use_rag
        self.image_retriever = image_retriever
        self.use_image_rag = use_image_rag
        if use_rag and retriever is None:
            raise ValueError("use_rag=True requires a retriever")
        if use_image_rag and image_retriever is None:
            raise ValueError("use_image_rag=True requires an image_retriever")
        model, base_url, resolved_key = resolve_provider(provider, model, base_url)
        if not model:
            raise ValueError(
                "model is required -- pass the model name vLLM was launched with, or set VLLM_MODEL"
            )
        self.model = model
        # 3-minute cap (vs the client's 600s default) so a hung/queued endpoint fails
        # fast enough to notice instead of silently stalling a whole eval run.
        self.client = client or OpenAI(base_url=base_url, api_key=api_key or resolved_key, timeout=180.0)
        self._min_request_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._next_request_time = 0.0
        self._throttle_lock = threading.Lock()  # one agent may be shared across worker threads

    def _create_completion(self, **kwargs):
        if self._min_request_interval:
            # Space request *starts* min_request_interval apart globally; sleeping while
            # holding the lock is intentional -- it makes queued threads inherit the delay.
            with self._throttle_lock:
                wait = self._next_request_time - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._next_request_time = time.monotonic() + self._min_request_interval
        return self.client.chat.completions.create(**kwargs)

    def classify(self, example: dict) -> dict:
        """Single-shot classification. If use_rag/use_image_rag are set, retrieval
        always runs first (not model-gated) and its results are injected into the
        prompt -- see module docstring."""
        retrieved_text = ""
        retrieved_exemplars_text = ""
        image_content = [_image_url_content(example["image_path"])]

        if self.use_rag:
            retrieval = self.retriever.retrieve_flood_knowledge(TEXT_RAG_QUERY, n_results=TEXT_RAG_N_RESULTS)
            retrieved_text = format_retrieved_passages(retrieval["passages"])

        if self.use_image_rag:
            img_result, image_payloads = self.image_retriever.dispatch(
                "retrieve_similar_examples",
                {"n_results": IMAGE_RAG_N_RESULTS},
                image_path=example["image_path"],
                exclude_season=example["season"],
            )
            retrieved_exemplars_text = format_retrieved_exemplars(img_result["examples"])
            image_content.extend(_image_data_to_content(d) for d in image_payloads)

        system_prompt = build_single_shot_prompt(
            self.use_rag, self.use_image_rag, retrieved_text, retrieved_exemplars_text
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": build_user_text_block(example)}],
            },
        ]

        usage = _empty_usage()
        result = None
        # occasional transient empty/garbled completions happen (observed on NVIDIA NIM);
        # one retry keeps them from becoming silent fallback rows in a long eval run
        for _ in range(2):
            response = self._create_completion(model=self.model, max_tokens=1500, temperature=0, messages=messages)
            _accumulate_usage(usage, response)
            content = response.choices[0].message.content or ""
            result = _parse_response(content)
            if result is not None:
                break
        if result is None:
            result = {
                "classification": "low",
                "confidence": 0.0,
                "rationale": f"unparseable response: {content[:500]}",
                "cited_evidence": [],
            }
        return {
            **result,
            "example": example,
            "retrieved_context": retrieved_text + retrieved_exemplars_text,
            "model": self.model,
            "usage": usage,
        }


def _empty_usage() -> dict:
    return {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _accumulate_usage(total: dict, response) -> None:
    total["requests"] += 1
    u = getattr(response, "usage", None)
    if u is not None:
        total["prompt_tokens"] += u.prompt_tokens or 0
        total["completion_tokens"] += u.completion_tokens or 0
        total["total_tokens"] += u.total_tokens or 0


def _parse_response(content: str) -> dict | None:
    """Extract the JSON object from a completion (tolerating markdown fences or stray
    prose around it). Returns None if there is no valid object to salvage."""
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or parsed.get("classification") not in VALID_LABELS:
        return None
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    cited = parsed.get("cited_evidence")
    return {
        "classification": parsed["classification"],
        "confidence": confidence,
        "rationale": str(parsed.get("rationale", "")),
        "cited_evidence": [str(c) for c in cited] if isinstance(cited, list) else [],
    }
