"""The classifier agent: a single-shot classification call over a vision-language model
served through an OpenAI-compatible chat completions API.

Every arm is architecturally identical -- one API call, temperature=0, JSON response --
differing only in what context is present. Few-shot exemplar retrieval, when an arm
enables it, is not a tool the model may choose to invoke: it always runs before the
model is called and its results are injected directly into the prompt. See
resolve_provider for provider config.

Two reliability mechanisms exist because the earlier runs lost up to 28.6% of a cell to
silent majority-class imputation:

* **JSON mode** (`json_mode="schema"`) constrains decoding to the response schema, which
  makes the observed failure mode -- a `<thought>` preamble that never reaches JSON --
  structurally impossible. It is only safe because the schema puts the reasoning fields
  first (see agent/prompts.py); with a classification-first schema it would suppress
  reasoning on exactly the ambiguous frames that decide the result. Provider support is
  uneven, so verify with scripts/check_json_mode.py and hold the setting FIXED across
  the whole comparison matrix rather than enabling it per-model.
* **A retry that actually differs.** Retrying an identical request at temperature=0 is
  near-deterministic and reproduces the same failure, which is why the previous 2-attempt
  loop salvaged almost nothing. Attempt 2 shows the model its own malformed output and
  asks for a correction; attempt 3 raises temperature. Every attempt is logged, so rows
  resolved at temperature>0 can be excluded from a strict analysis.

Rows that still fail fall back to the least-severe label, which is a conservative
prediction that deflates flood recall -- so `parse_failed` is recorded per row and the
parse-failure rate must be reported alongside any metric.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
import time
from pathlib import Path

from openai import OpenAI

from agent.prompts import (
    DEFAULT_VARIANT,
    build_system_prompt,
    build_user_text_block,
    format_exemplars,
    response_json_schema,
    severity_sort_key,
)
from data.taxonomy import FALLBACK_LABEL, labels_for

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

JSON_MODES = ("off", "object", "schema")

# Reasoning-trace preambles seen in the wild (Gemma family emits `<thought>`), stripped
# before parsing so an otherwise-valid JSON body after the block is still salvageable.
_THOUGHT_BLOCK = re.compile(r"<(thought|think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)

MAX_ATTEMPTS = 3
RETRY_TEMPERATURE = 0.2
_CORRECTION_REQUEST = (
    "Your previous response was not a single valid JSON object. Respond again with ONLY "
    "the JSON object in the required form -- no thinking block, no prose, no markdown "
    "fence, nothing before the opening brace or after the closing brace."
)


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
    """{media_type, data} (as returned by the exemplar providers) -> an OpenAI
    chat-completions image_url content part."""
    return {"type": "image_url", "image_url": {"url": f"data:{image_data['media_type']};base64,{image_data['data']}"}}


class ClassifierAgent:
    def __init__(
        self,
        image_retriever=None,
        use_image_rag: bool = False,
        image_rag_mode: str = "similarity",
        n_exemplars: int = 3,
        taxonomy: str = "3class",
        prompt_variant: str = DEFAULT_VARIANT,
        include_criteria: bool = True,
        json_mode: str = "schema",
        client: OpenAI | None = None,
        provider: str = "vllm",
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        requests_per_minute: float = 0.0,
    ):
        self.image_retriever = image_retriever
        self.use_image_rag = use_image_rag
        if image_rag_mode not in ("similarity", "fewshot"):
            raise ValueError(f"image_rag_mode must be 'similarity' or 'fewshot', got {image_rag_mode!r}")
        self.image_rag_mode = image_rag_mode
        self.n_exemplars = n_exemplars
        self.taxonomy = taxonomy
        self.valid_labels = set(labels_for(taxonomy))  # also validates the taxonomy name
        self.fallback_label = FALLBACK_LABEL[taxonomy]
        self.prompt_variant = prompt_variant
        self.include_criteria = include_criteria
        if json_mode not in JSON_MODES:
            raise ValueError(f"json_mode must be one of {JSON_MODES}, got {json_mode!r}")
        self.json_mode = json_mode
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

    # -- config record, written into every result row so a results file is self-describing
    def config(self) -> dict:
        return {
            "model": self.model,
            "taxonomy": self.taxonomy,
            "prompt_variant": self.prompt_variant,
            "include_criteria": self.include_criteria,
            "json_mode": self.json_mode,
            "image_rag": self.image_rag_mode if self.use_image_rag else None,
            "n_exemplars": self.n_exemplars if self.use_image_rag else 0,
        }

    def _response_format(self) -> dict | None:
        if self.json_mode == "object":
            return {"type": "json_object"}
        if self.json_mode == "schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "water_level_classification",
                    "strict": True,
                    "schema": response_json_schema(self.taxonomy),
                },
            }
        return None

    def _create_completion(self, messages: list[dict], temperature: float):
        if self._min_request_interval:
            # Space request *starts* min_request_interval apart globally; sleeping while
            # holding the lock is intentional -- it makes queued threads inherit the delay.
            with self._throttle_lock:
                wait = self._next_request_time - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._next_request_time = time.monotonic() + self._min_request_interval
        kwargs = {
            "model": self.model,
            "max_tokens": 1500,
            "temperature": temperature,
            "messages": messages,
        }
        response_format = self._response_format()
        if response_format is not None:
            kwargs["response_format"] = response_format
        started = time.monotonic()
        response = self.client.chat.completions.create(**kwargs)
        return response, time.monotonic() - started

    def _exemplar_context(self, example: dict) -> tuple[str, list[dict]]:
        """-> (rendered prompt section, image content parts), ordered by ascending
        severity so exemplar order is fixed rather than whatever the sampler returned."""
        result, payloads = self.image_retriever.dispatch(
            "retrieve_similar_examples",
            {"n_results": self.n_exemplars},
            image_path=example["image_path"],
            exclude_season=example["season"],
        )
        # Only exemplars whose bytes resolved locally have a payload; pair them up in
        # order, then sort text and images together so descriptions match attachments.
        available = [ex for ex in result["examples"] if ex.get("image_available_locally", True)]
        paired = sorted(
            zip(available, payloads), key=lambda pair: severity_sort_key(pair[0], self.taxonomy)
        )
        ordered_examples = [ex for ex, _ in paired]
        ordered_payloads = [p for _, p in paired]
        kind = "reference" if self.image_rag_mode == "similarity" else "calibration"
        text = format_exemplars(ordered_examples, self.taxonomy, kind=kind)
        return text, [_image_data_to_content(p) for p in ordered_payloads]

    def classify(self, example: dict) -> dict:
        exemplars_text = ""
        query_content = _image_url_content(example["image_path"])
        if self.use_image_rag:
            exemplars_text, exemplar_content = self._exemplar_context(example)
            # Exemplars FIRST, query LAST. With the query first, models resolve "this
            # image" to the most recent attachment and answer about the final exemplar
            # instead: nemotron called 134/150 dry frames 'flood' at K=1, its rationales
            # describing the flood exemplar ("spilling onto the grass embankment") rather
            # than the query frame. Ascending-severity ordering put flood last, so the
            # error was maximally wrong. Query position is load-bearing, not cosmetic.
            image_content = exemplar_content + [query_content]
        else:
            image_content = [query_content]

        system_prompt = build_system_prompt(
            taxonomy=self.taxonomy,
            variant=self.prompt_variant,
            exemplars_text=exemplars_text,
            include_criteria=self.include_criteria,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": image_content
                + [{"type": "text", "text": build_user_text_block(example, self.taxonomy, bool(exemplars_text))}],
            },
        ]

        usage = _empty_usage()
        attempts: list[dict] = []
        result = None
        content = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # Attempt 2 shows the model its own malformed output and asks for a fix;
            # attempt 3 goes back to the clean prompt at a nonzero temperature. Decoding
            # config (json_mode) never changes mid-row -- that would confound the cell.
            temperature = 0.0 if attempt < MAX_ATTEMPTS else RETRY_TEMPERATURE
            if attempt == 2 and content.strip():
                request_messages = messages + [
                    {"role": "assistant", "content": content[:1000]},
                    {"role": "user", "content": _CORRECTION_REQUEST},
                ]
            else:
                request_messages = messages

            response, latency = self._create_completion(request_messages, temperature)
            _accumulate_usage(usage, response)
            choice = response.choices[0]
            content = choice.message.content or ""
            result = _parse_response(content, self.valid_labels)
            attempts.append({
                "attempt": attempt,
                "temperature": temperature,
                "finish_reason": getattr(choice, "finish_reason", None),
                "latency_s": round(latency, 3),
                "completion_chars": len(content),
                "parsed": result is not None,
                # Raw text is kept only for failures: it is the audit trail for the
                # parse-failure rate, and keeping it for every row would bloat the file.
                "raw": None if result is not None else content[:2000],
            })
            if result is not None:
                break

        parse_failed = result is None
        if parse_failed:
            result = {
                "observations": "",
                "rationale": f"unparseable response: {content[:500]}",
                "classification": self.fallback_label,
                "confidence": 0.0,
                "cited_evidence": [],
            }
        return {
            **result,
            "example": example,
            "parse_failed": parse_failed,
            "resolved_on_attempt": None if parse_failed else attempts[-1]["attempt"],
            "attempts": attempts,
            "latency_s": round(sum(a["latency_s"] for a in attempts), 3),
            "exemplar_context": exemplars_text,
            "config": self.config(),
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
        # Reasoning/"thought" tokens are billed separately by some providers and are NOT
        # always included in completion_tokens (observed on the Gemini OpenAI-compat
        # path, where failed rows reported *fewer* completion tokens than parsed ones
        # despite emitting a long <thought> block). Captured separately where exposed so
        # the cost axis isn't silently understated.
        details = getattr(u, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", None) if details is not None else None
        if reasoning:
            total["reasoning_tokens"] = total.get("reasoning_tokens", 0) + reasoning


def _parse_response(content: str, valid_labels: set[str]) -> dict | None:
    """Extract the JSON object from a completion, tolerating markdown fences, a
    reasoning-trace preamble, or stray prose around it. Returns None if there is no
    valid object to salvage."""
    cleaned = _THOUGHT_BLOCK.sub("", content)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or parsed.get("classification") not in valid_labels:
        return None
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    cited = parsed.get("cited_evidence")
    return {
        "observations": str(parsed.get("observations", "")),
        "rationale": str(parsed.get("rationale", "")),
        "classification": parsed["classification"],
        "confidence": confidence,
        "cited_evidence": [str(c) for c in cited] if isinstance(cited, list) else [],
    }
