"""The classifier agent: a tool-calling loop over a vision-language model served
through an OpenAI-compatible chat completions API, grounding its water-level
classification in RAG retrieval before emitting a final structured answer.

Two providers are supported (see resolve_provider):
  - vllm: a self-hosted vLLM server, which must be launched with tool-calling enabled:
        vllm serve <model> --enable-auto-tool-choice --tool-call-parser <parser-for-model>
  - gemini: Google's Gemini API via its OpenAI-compatibility endpoint (needs
    GEMINI_API_KEY; free tier is heavily rate-limited, see --rpm in agent/run.py).

Either way the model must support image input (vision-language), since every
classification call includes the camera frame.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
import time
from pathlib import Path

from openai import OpenAI

from agent.image_retrieval import RETRIEVE_SIMILAR_EXAMPLES_TOOL_SCHEMA, ImageBankRetriever
from agent.prompts import (
    SUBMIT_CLASSIFICATION_TOOL_SCHEMA,
    build_baseline_system_prompt,
    build_system_prompt,
    build_user_text_block,
)
from agent.retrieval import RETRIEVAL_TOOL_SCHEMA, KnowledgeBaseRetriever

VALID_LABELS = set(SUBMIT_CLASSIFICATION_TOOL_SCHEMA["input_schema"]["properties"]["classification"]["enum"])

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_TURNS = 6


def resolve_provider(provider: str, model: str = "", base_url: str = "") -> tuple[str, str, str]:
    """-> (model, base_url, api_key) for the given provider, filling unset values from
    env vars. Env is read lazily here (not at import time) so load_dotenv() in the
    entrypoint takes effect."""
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("provider 'gemini' requires GEMINI_API_KEY (set it in .env)")
        return (
            model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
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
    if provider == "vllm":
        return (
            model or os.environ.get("VLLM_MODEL", ""),
            base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            os.environ.get("VLLM_API_KEY", "EMPTY"),  # vLLM ignores this unless an auth proxy requires it
        )
    raise ValueError(f"unknown provider: {provider!r} (expected 'vllm', 'gemini', or 'nvidia')")


def _image_url_content(image_path: str) -> dict:
    path = Path(image_path)
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}


def _image_data_to_content(image_data: dict) -> dict:
    """{media_type, data} (as returned by ImageBankRetriever.dispatch) -> an OpenAI
    chat-completions image_url content part."""
    return {"type": "image_url", "image_url": {"url": f"data:{image_data['media_type']};base64,{image_data['data']}"}}


def _to_openai_tool(schema: dict) -> dict:
    """Anthropic-style flat tool schema ({name, description, input_schema}) -> OpenAI /
    vLLM function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["input_schema"],
        },
    }


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

    def _tool_schemas(self) -> list[dict]:
        schemas = [SUBMIT_CLASSIFICATION_TOOL_SCHEMA]
        if self.use_rag:
            schemas = [RETRIEVAL_TOOL_SCHEMA] + schemas
        if self.use_image_rag:
            schemas = [RETRIEVE_SIMILAR_EXAMPLES_TOOL_SCHEMA] + schemas
        return [_to_openai_tool(s) for s in schemas]

    def classify(self, example: dict) -> dict:
        usage = _empty_usage()
        messages = [
            {"role": "system", "content": build_system_prompt(self.use_rag, self.use_image_rag)},
            {
                "role": "user",
                "content": [
                    _image_url_content(example["image_path"]),
                    {"type": "text", "text": build_user_text_block(example)},
                ],
            },
        ]
        tools = self._tool_schemas()
        tool_call_log = []

        for _ in range(MAX_TURNS):
            response = self._create_completion(
                model=self.model,
                max_tokens=1500,
                tools=tools,
                tool_choice="auto",
                messages=messages,
            )
            _accumulate_usage(usage, response)
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            tool_calls = message.tool_calls or []

            submit_call = next((tc for tc in tool_calls if tc.function.name == "submit_classification"), None)
            if submit_call:
                result = json.loads(submit_call.function.arguments)
                return {**result, "example": example, "tool_call_log": tool_call_log, "model": self.model, "usage": usage}

            if not tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "You must call submit_classification with your final answer now.",
                    }
                )
                continue

            # OpenAI/vLLM tool-role messages only accept string content -- any images a
            # tool returns can't go inside the tool result itself, so they're collected
            # here and delivered as an immediate follow-up user turn instead.
            image_followup_content = []
            for tc in tool_calls:
                tool_input = json.loads(tc.function.arguments or "{}")
                if tc.function.name == "retrieve_similar_examples":
                    result, image_payloads = self.image_retriever.dispatch(
                        tc.function.name,
                        tool_input,
                        image_path=example["image_path"],
                        exclude_season=example["season"],
                    )
                    image_followup_content.extend(_image_data_to_content(d) for d in image_payloads)
                else:
                    result = self.retriever.dispatch(tc.function.name, tool_input)
                tool_call_log.append({"tool": tc.function.name, "input": tool_input, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

            if image_followup_content:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Retrieved reference images:"}] + image_followup_content,
                    }
                )

        return {
            "classification": "low",
            "confidence": 0.0,
            "rationale": "agent failed to submit a classification within max turns",
            "cited_evidence": [],
            "example": example,
            "tool_call_log": tool_call_log,
            "model": self.model,
            "usage": usage,
        }

    def classify_baseline(self, example: dict) -> dict:
        """Single-shot ablation baseline: one image + text prompt, one completion,
        no tools and no retrieval. Result schema matches classify() so eval/ code
        works unchanged (tool_call_log is always empty here)."""
        messages = [
            {"role": "system", "content": build_baseline_system_prompt()},
            {
                "role": "user",
                "content": [
                    _image_url_content(example["image_path"]),
                    {"type": "text", "text": build_user_text_block(example)},
                ],
            },
        ]
        response = self._create_completion(model=self.model, max_tokens=1500, messages=messages)
        usage = _empty_usage()
        _accumulate_usage(usage, response)
        content = response.choices[0].message.content or ""
        result = _parse_baseline_response(content)
        if result is None:
            result = {
                "classification": "low",
                "confidence": 0.0,
                "rationale": f"unparseable baseline response: {content[:500]}",
                "cited_evidence": [],
            }
        return {**result, "example": example, "tool_call_log": [], "model": self.model, "usage": usage}


def _empty_usage() -> dict:
    return {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _accumulate_usage(total: dict, response) -> None:
    total["requests"] += 1
    u = getattr(response, "usage", None)
    if u is not None:
        total["prompt_tokens"] += u.prompt_tokens or 0
        total["completion_tokens"] += u.completion_tokens or 0
        total["total_tokens"] += u.total_tokens or 0


def _parse_baseline_response(content: str) -> dict | None:
    """Extract the JSON object from a baseline completion (tolerating markdown fences
    or stray prose around it). Returns None if there is no valid object to salvage."""
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
