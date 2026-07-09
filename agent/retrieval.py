"""RAG retrieval tool wrapping the Chroma knowledge-base index built in knowledge_base/build_index.py."""
from __future__ import annotations

from pathlib import Path

import chromadb

INDEX_DIR = Path(__file__).parent.parent / "knowledge_base" / "index"
COLLECTION_NAME = "flood_knowledge_base"

RETRIEVAL_TOOL_SCHEMA = {
    "name": "retrieve_flood_knowledge",
    "description": (
        "Search the curated flood-knowledge corpus (NWS/WMO flood definitions, flood-type "
        "descriptions, and historical event summaries) for passages relevant to a query. "
        "Use this to ground your classification in flood-definition criteria and to check "
        "whether the current readings match a documented historical analog."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query."},
            "n_results": {"type": "integer", "description": "Number of passages to return (default 3)."},
        },
        "required": ["query"],
    },
}


class KnowledgeBaseRetriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        if not index_dir.exists():
            raise FileNotFoundError(
                f"No index at {index_dir} -- run `python -m knowledge_base.build_index` first."
            )
        client = chromadb.PersistentClient(path=str(index_dir))
        self.collection = client.get_collection(COLLECTION_NAME)

    def retrieve_flood_knowledge(self, query: str, n_results: int = 3) -> dict:
        res = self.collection.query(query_texts=[query], n_results=n_results)
        passages = [
            {"title": meta["title"], "source": meta["source"], "text": doc}
            for doc, meta in zip(res["documents"][0], res["metadatas"][0])
        ]
        return {"query": query, "passages": passages}

    def dispatch(self, tool_name: str, tool_input: dict) -> dict:
        if tool_name != "retrieve_flood_knowledge":
            return {"error": f"unknown tool {tool_name}"}
        return self.retrieve_flood_knowledge(**tool_input)
