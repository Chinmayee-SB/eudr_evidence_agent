"""Vector search over the chunks, stored locally in Chroma (no account or server).

Chroma's default embedding model (all-MiniLM-L6-v2) runs on this machine. It reads only about the
first 256 tokens of a passage, which covers nearly all of our chunks.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from eudr_agent.config import VECTOR_DIR
from eudr_agent.rag.chunk import Chunk

COLLECTION = "eudr_rules"
BATCH_SIZE = 100


@lru_cache(maxsize=1)
def _client() -> Any:
    return chromadb.PersistentClient(path=str(VECTOR_DIR), settings=ChromaSettings(anonymized_telemetry=False))


def build_vector_index(chunks: list[Chunk]) -> int:
    """Rebuild the collection from scratch, so removed chunks cannot linger. Returns the vector count."""
    client = _client()
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        collection.add(ids=[c.id for c in batch],
                       documents=[f"{c.ref} {c.title} {c.text}" for c in batch],
                       metadatas=[{"kind": c.kind} for c in batch])
    return collection.count()


def search(query: str, k: int, kinds: set[str] | None = None) -> list[str]:
    """Return the ids of the k chunks closest in meaning to the query, best first, optionally of some kinds only."""
    try:
        collection = _client().get_collection(COLLECTION)
    except Exception as e:
        raise FileNotFoundError("No vector index yet. Run: python scripts/build_index.py") from e
    where = {"kind": {"$in": sorted(kinds)}} if kinds else None
    return collection.query(query_texts=[query], n_results=k, where=where, include=[])["ids"][0]
