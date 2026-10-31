"""Search over the chunks: BM25 keywords, vector similarity, or both merged (hybrid).

BM25 is strong on exact legal terms ("geolocation"). Vector search is strong on paraphrases
("legal definition of forest"). Hybrid merges the two rankings with Reciprocal Rank Fusion.
RAG_RETRIEVAL picks the mode, so the three can be compared on the same questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from rank_bm25 import BM25Okapi

from eudr_agent.config import settings
from eudr_agent.rag import vector
from eudr_agent.rag.chunk import Chunk, load_chunks

STOPWORDS = set(
    """a an and are as at be been by can could do does for from has have how if in is it its
    may must not of on or shall should such than that the their them then there these this to
    under was were what when where which who will with would whether any all also into out our
    i we you my me us one""".split()
)

RRF_K = 60        # standard constant: stops the top rank or two from dominating the merged score
CANDIDATES = 20   # how many passages each retriever offers to the merge


def _stem(word: str) -> str:
    """Very light stemming so 'trees'/'tree' and 'applies'/'apply' match."""
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + repl
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("‘", " ").replace("’", " "))
    return [_stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    found_by: list[str] = field(default_factory=list)  # which retrievers returned this passage


def rrf_merge(rankings: dict[str, list[str]], k: int) -> list[tuple[str, float, list[str]]]:
    """Reciprocal Rank Fusion: a passage scores 1 / (RRF_K + rank) in each list it appears in.

    Uses ranks only, so BM25 scores and vector distances never need to be on the same scale.
    Returns (id, merged score, retrievers that found it), best first.
    """
    scores: dict[str, float] = {}
    found_by: dict[str, list[str]] = {}
    for name, ids in rankings.items():
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (RRF_K + rank)
            found_by.setdefault(chunk_id, []).append(name)
    best = sorted(scores, key=lambda i: scores[i], reverse=True)[:k]
    return [(i, scores[i], found_by[i]) for i in best]


class Index:
    def __init__(self, chunks: list[Chunk]):
        self.by_id = {c.id: c for c in chunks}
        self.chunk_ids = [c.id for c in chunks]
        self._bm25 = BM25Okapi([tokenize(c.ref + " " + c.title + " " + c.text) for c in chunks])

    def _bm25_ids(self, query: str, k: int) -> list[str]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunk_ids[i] for i in ranked if scores[i] > 0]

    def search(self, query: str, k: int = 5) -> list[Hit]:
        rankings: dict[str, list[str]] = {}
        if settings.retrieval in ("hybrid", "bm25"):
            rankings["bm25"] = self._bm25_ids(query, CANDIDATES)
        if settings.retrieval in ("hybrid", "vector"):
            rankings["vector"] = vector.search(query, CANDIDATES)
        return [Hit(self.by_id[i], score, names) for i, score, names in rrf_merge(rankings, k)]


@lru_cache(maxsize=1)
def get_index() -> Index:
    return Index(load_chunks())
