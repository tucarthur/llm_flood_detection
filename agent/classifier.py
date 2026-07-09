"""The classifier agent: a tool-calling loop over Claude that grounds its classification
in structured USGS gauge tools and RAG retrieval before emitting a final structured answer.
"""
from __future__ import annotations

import json
import os

import anthropic

from agent.prompts import SUBMIT_CLASSIFICATION_TOOL_SCHEMA, build_system_prompt, build_user_prompt
from agent.retrieval import RETRIEVAL_TOOL_SCHEMA, KnowledgeBaseRetriever
from agent.tools import TOOL_SCHEMAS, StructuredDataTools

MODEL = "claude-sonnet-5"
MAX_TURNS = 8


class ClassifierAgent:
    def __init__(
        self,
        structured_tools: StructuredDataTools,
        retriever: KnowledgeBaseRetriever,
        use_tools: bool = True,
        use_rag: bool = True,
        client: anthropic.Anthropic | None = None,
    ):
        self.structured_tools = structured_tools
        self.retriever = retriever
        self.use_tools = use_tools
        self.use_rag = use_rag
        self.client = client or anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def _tool_schemas(self) -> list[dict]:
        schemas = [SUBMIT_CLASSIFICATION_TOOL_SCHEMA]
        if self.use_tools:
            schemas = TOOL_SCHEMAS + schemas
        if self.use_rag:
            schemas = [RETRIEVAL_TOOL_SCHEMA] + schemas
        return schemas

    def _dispatch(self, tool_name: str, tool_input: dict) -> dict:
        if tool_name == "retrieve_flood_knowledge":
            return self.retriever.dispatch(tool_name, tool_input)
        return self.structured_tools.dispatch(tool_name, tool_input)

    def classify(self, example: dict) -> dict:
        structured_dump = None
        if not self.use_tools:
            structured_dump = {
                "baseline": self.structured_tools.get_site_baseline(example["site_id"]),
                "time_series": self.structured_tools.get_gauge_time_series(
                    example["site_id"], example["start_date"], example["end_date"]
                ),
                "target_date_reading": self.structured_tools.get_gauge_reading(
                    example["site_id"], example["target_date"]
                ),
            }
        messages = [{"role": "user", "content": build_user_prompt(example, structured_dump)}]
        tools = self._tool_schemas()
        system_prompt = build_system_prompt(self.use_tools, self.use_rag)
        tool_call_log = []
        if structured_dump is not None:
            tool_call_log.append({"tool": "structured_dump_in_prompt", "input": {}, "result": structured_dump})

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
                # Model stopped without submitting -- force it on the next turn.
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
                result = self._dispatch(block.name, block.input)
                tool_call_log.append({"tool": block.name, "input": block.input, "result": result})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return {
            "classification": "no_flood",
            "confidence": 0.0,
            "rationale": "agent failed to submit a classification within max turns",
            "cited_evidence": [],
            "example": example,
            "tool_call_log": tool_call_log,
        }
