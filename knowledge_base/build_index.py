"""Build a persistent Chroma vector index over the curated flood-knowledge corpus.

Uses Chroma's bundled default embedding function (a local ONNX MiniLM model) so no
external embedding API key is required -- only the classifier agent's LLM calls need
ANTHROPIC_API_KEY. Documents are chunked per markdown section (split on '# ' / '## ')
so retrieval can point at a specific definition or event rather than a whole file.
"""
from __future__ import annotations

import re
from pathlib import Path

import chromadb

CORPUS_DIR = Path(__file__).parent / "corpus"
INDEX_DIR = Path(__file__).parent / "index"
COLLECTION_NAME = "flood_knowledge_base"


def chunk_markdown(text: str, source: str) -> list[dict]:
    """Split a markdown file into chunks at top-level ('# ') headings."""
    sections = re.split(r"\n(?=# )", text.strip())
    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        title_match = re.match(r"#+\s*(.+)", section)
        title = title_match.group(1) if title_match else f"{source} chunk {i}"
        chunks.append({"text": section, "title": title, "source": source})
    return chunks


def build_index() -> chromadb.Collection:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, docs, metadatas = [], [], []
    for md_path in sorted(CORPUS_DIR.glob("*.md")):
        text = md_path.read_text()
        for j, chunk in enumerate(chunk_markdown(text, md_path.name)):
            ids.append(f"{md_path.stem}::{j}")
            docs.append(chunk["text"])
            metadatas.append({"title": chunk["title"], "source": chunk["source"]})

    collection.add(ids=ids, documents=docs, metadatas=metadatas)
    return collection


if __name__ == "__main__":
    collection = build_index()
    print(f"Indexed {collection.count()} chunks from {CORPUS_DIR} into {INDEX_DIR}")
