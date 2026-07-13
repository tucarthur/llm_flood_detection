"""The classifier agent: a tool-calling loop over a vLLM-served vision-language model
(via vLLM's OpenAI-compatible chat completions API) that grounds its water-level
classification in RAG retrieval before emitting a final structured answer.

vLLM must be launched with tool-calling enabled for whatever model it's serving, e.g.:
    vllm serve <model> --enable-auto-tool-choice --tool-call-parser <parser-for-model>
and the model must support image input (vision-language), since every classification
call includes the camera frame.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path

from openai import OpenAI

from agent.image_retrieval import RETRIEVE_SIMILAR_EXAMPLES_TOOL_SCHEMA, ImageBankRetriever
from agent.prompts import SUBMIT_CLASSIFICATION_TOOL_SCHEMA, build_system_prompt, build_user_text_block
from agent.retrieval import RETRIEVAL_TOOL_SCHEMA, KnowledgeBaseRetriever

DEFAULT_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")  # vLLM ignores this unless an auth proxy requires it
DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "")
MAX_TURNS = 6


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
        retriever: KnowledgeBaseRetriever,
        use_rag: bool = True,
        image_retriever: ImageBankRetriever | None = None,
        use_image_rag: bool = False,
        client: OpenAI | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
    ):
        self.retriever = retriever
        self.use_rag = use_rag
        self.image_retriever = image_retriever
        self.use_image_rag = use_image_rag
        if use_image_rag and image_retriever is None:
            raise ValueError("use_image_rag=True requires an image_retriever")
        if not model:
            raise ValueError(
                "model is required -- pass the model name vLLM was launched with, or set VLLM_MODEL"
            )
        self.model = model
        self.client = client or OpenAI(base_url=base_url, api_key=api_key)

    def _tool_schemas(self) -> list[dict]:
        schemas = [SUBMIT_CLASSIFICATION_TOOL_SCHEMA]
        if self.use_rag:
            schemas = [RETRIEVAL_TOOL_SCHEMA] + schemas
        if self.use_image_rag:
            schemas = [RETRIEVE_SIMILAR_EXAMPLES_TOOL_SCHEMA] + schemas
        return [_to_openai_tool(s) for s in schemas]

    def classify(self, example: dict) -> dict:
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
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1500,
                tools=tools,
                tool_choice="auto",
                messages=messages,
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            tool_calls = message.tool_calls or []

            submit_call = next((tc for tc in tool_calls if tc.function.name == "submit_classification"), None)
            if submit_call:
                result = json.loads(submit_call.function.arguments)
                return {**result, "example": example, "tool_call_log": tool_call_log}

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
        }
