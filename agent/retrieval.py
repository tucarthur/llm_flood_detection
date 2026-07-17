"""RAG retrieval tool wrapping the Chroma knowledge-base index built in knowledge_base/build_index.py."""
from __future__ import annotations

from pathlib import Path

import chromadb

INDEX_DIR = Path(__file__).parent.parent / "knowledge_base" / "index"
COLLECTION_NAME = "flood_knowledge_base"

RETRIEVAL_TOOL_SCHEMA = {
    "name": "retrieve_flood_knowledge",
    "description": (
        "Search the curated flood-knowledge corpus (WMO flood definitions, flood-type "
        "descriptions, this site's classification criteria, official regional flood-risk "
        "context for Sao Carlos, and documented historical flood events near this site) "
        "for passages relevant to a query. Use this to ground your classification in "
        "flood-definition criteria and site context, and to check whether the current "
        "conditions resemble a documented regional event."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query."},
            "n_results": {"type": "integer", "description": "Number of passages to return (default 4)."},
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

    def retrieve_flood_knowledge(self, query: str, n_results: int = 4) -> dict:
        res = self.collection.query(query_texts=[query], n_results=n_results)
        passages = [
            # distance is Chroma's raw query-space distance (smaller = more similar), not
            # a normalized similarity score -- exposed so the model can judge whether a
            # hit is a strong or a weak match rather than treating every result as equally
            # relevant just because it was returned.
            {"title": meta["title"], "source": meta["source"], "text": doc, "distance": dist}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
        return {"query": query, "passages": passages}

    def dispatch(self, tool_name: str, tool_input: dict) -> dict:
        if tool_name != "retrieve_flood_knowledge":
            return {"error": f"unknown tool {tool_name}"}
        return self.retrieve_flood_knowledge(**tool_input)
