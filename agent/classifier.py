"""The classifier agent: a tool-calling loop over Claude's vision API that grounds its
damage-severity classification in RAG retrieval before emitting a final structured answer.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path

import anthropic

from agent.image_retrieval import RETRIEVE_SIMILAR_EXAMPLES_TOOL_SCHEMA, ImageBankRetriever
from agent.prompts import SUBMIT_CLASSIFICATION_TOOL_SCHEMA, build_system_prompt, build_user_text_block
from agent.retrieval import RETRIEVAL_TOOL_SCHEMA, KnowledgeBaseRetriever

MODEL = "claude-sonnet-5"
MAX_TURNS = 6


def _load_image_block(image_path: str) -> dict:
    path = Path(image_path)
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


class ClassifierAgent:
    def __init__(
        self,
        retriever: KnowledgeBaseRetriever,
        use_rag: bool = True,
        image_retriever: ImageBankRetriever | None = None,
        use_image_rag: bool = False,
        client: anthropic.Anthropic | None = None,
    ):
        self.retriever = retriever
        self.use_rag = use_rag
        self.image_retriever = image_retriever
        self.use_image_rag = use_image_rag
        if use_image_rag and image_retriever is None:
            raise ValueError("use_image_rag=True requires an image_retriever")
        self.client = client or anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def _tool_schemas(self) -> list[dict]:
        schemas = [SUBMIT_CLASSIFICATION_TOOL_SCHEMA]
        if self.use_rag:
            schemas = [RETRIEVAL_TOOL_SCHEMA] + schemas
        if self.use_image_rag:
            schemas = [RETRIEVE_SIMILAR_EXAMPLES_TOOL_SCHEMA] + schemas
        return schemas

    def classify(self, example: dict) -> dict:
        image_block = _load_image_block(example["image_path"])
        text_block = {"type": "text", "text": build_user_text_block(example)}
        messages = [{"role": "user", "content": [image_block, text_block]}]
        tools = self._tool_schemas()
        system_prompt = build_system_prompt(self.use_rag, self.use_image_rag)
        tool_call_log = []

        for _ in range(MAX_TURNS):
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=1500,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            submit_block = next(
                (b for b in response.content if b.type == "tool_use" and b.name == "submit_classification"),
                None,
            )
            if submit_block:
                return {**submit_block.input, "example": example, "tool_call_log": tool_call_log}

            if response.stop_reason != "tool_use":
                messages.append(
                    {
                        "role": "user",
                        "content": "You must call submit_classification with your final answer now.",
                    }
                )
                continue

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if block.name == "retrieve_similar_examples":
                    result, image_blocks = self.image_retriever.dispatch(
                        block.name,
                        block.input,
                        image_path=example["image_path"],
                        exclude_season=example["season"],
                    )
                    content = [{"type": "text", "text": json.dumps(result)}] + image_blocks
                else:
                    result = self.retriever.dispatch(block.name, block.input)
                    content = json.dumps(result)
                tool_call_log.append({"tool": block.name, "input": block.input, "result": result})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return {
            "classification": "low",
            "confidence": 0.0,
            "rationale": "agent failed to submit a classification within max turns",
            "cited_evidence": [],
            "example": example,
            "tool_call_log": tool_call_log,
        }
