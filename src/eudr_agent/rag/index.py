"""Search over the chunks: BM25 keywords, vector similarity, or both merged (hybrid).

BM25 is strong on exact legal terms ("geolocation"). Vector search is strong on paraphrases
("legal definition of forest"). Hybrid merges the two rankings with Reciprocal Rank Fusion.
RAG_RETRIEVAL picks the mode, so the three can be compared on the same questions.

The regulation and the supporting documents (Commission guidance, Indian law) are searched as two shelves.
The supporting documents are far longer, so in one shared ranking they push the binding articles out of the
top results. Each shelf gets its own places instead: k from the regulation, SUPPORT_K from the rest.

With the reranker on (rerank.py, the default), both shelves offer more candidates and a cross-encoder orders
them all together, so a Commission FAQ answer can rank above a loosely related article. The binding law keeps
MIN_REGULATION of the places. If the reranker cannot run, the shelf order above is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from rank_bm25 import BM25Okapi

from eudr_agent.config import settings
from eudr_agent.observability import retrieved, traceable
from eudr_agent.rag import rerank, vector
from eudr_agent.rag.chunk import Chunk, load_chunks

STOPWORDS = set(
    """a an and are as at be been by can could do does for from has have how if in is it its
    may must not of on or shall should such than that the their them then there these this to
    under was were what when where which who will with would whether any all also into out our
    i we you my me us one""".split()
)

RRF_K = 60        # standard constant: stops the top rank or two from dominating the merged score
CANDIDATES = 20   # how many passages each retriever offers to the merge
SUPPORT_K = 3     # places for Commission guidance and Indian law, on top of k from the regulation
TOP_EACH = 2      # each retriever's best passages that always get a place, before the merged ranking fills the rest
SHELVES = {"regulation": {"law", "recital", "note"}, "supporting": {"guidance", "national_law"}}
RERANK = True          # the evaluation switches this off to compare
RERANK_POOL = {"regulation": 12, "supporting": 8}  # candidates each shelf offers the reranker
MIN_REGULATION = 3     # places kept for the regulation shelf after reranking
RERANK_MODE = "keep_winners"  # replace | fuse | keep_winners (see search)


def _stem(word: str) -> str:
    """Light stemming, in steps, so 'clearings'/'clearing'/'clear', 'reserved'/'reserve' and 'applies'/'apply'
    meet: plural first, then -ing/-ed, then a final -e."""
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
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
        self.kinds = [c.kind for c in chunks]
        self.rerank_note = ""  # why the last search was not reranked, if it was not
        self._bm25 = BM25Okapi([tokenize(c.ref + " " + c.title + " " + c.text) for c in chunks])

    def _bm25_ids(self, query: str, k: int, kinds: set[str]) -> list[str]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted((i for i in range(len(scores)) if self.kinds[i] in kinds),
                        key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunk_ids[i] for i in ranked if scores[i] > 0]

    def _winners(self, query: str) -> list[str]:
        """Each retriever's top passages on the regulation shelf, the ones the plain search always keeps."""
        ids = []
        if settings.retrieval in ("hybrid", "bm25"):
            ids += self._bm25_ids(query, TOP_EACH, SHELVES["regulation"])
        if settings.retrieval in ("hybrid", "vector"):
            ids += vector.search(query, TOP_EACH, SHELVES["regulation"])
        return list(dict.fromkeys(ids))

    def _shelf(self, query: str, k: int, kinds: set[str]) -> list[Hit]:
        rankings: dict[str, list[str]] = {}
        if settings.retrieval in ("hybrid", "bm25"):
            rankings["bm25"] = self._bm25_ids(query, CANDIDATES, kinds)
        if settings.retrieval in ("hybrid", "vector"):
            rankings["vector"] = vector.search(query, CANDIDATES, kinds)
        merged = rrf_merge(rankings, len(self.chunk_ids))
        # A passage one retriever ranks first but the other misses scores low in the merge, so it can drop out
        # behind passages both rank middling. Each retriever's top passages keep a place.
        firsts = list(dict.fromkeys(i for ids in rankings.values() for i in ids[:TOP_EACH]))
        rest = [i for i, _, _ in merged if i not in firsts]
        by_id = {i: (score, names) for i, score, names in merged}
        return [Hit(self.by_id[i], *by_id[i]) for i in (firsts + rest)[:k]]

    def _shelves(self, query: str, sizes: dict[str, int]) -> list[Hit]:
        hits = self._shelf(query, sizes["regulation"], SHELVES["regulation"])
        if any(kind in SHELVES["supporting"] for kind in self.kinds):
            hits += self._shelf(query, sizes["supporting"], SHELVES["supporting"])
        return hits

    @traceable(run_type="retriever", name="search", process_inputs=lambda i: {"query": i["query"], "k": i.get("k", 5)},
               process_outputs=retrieved)
    def search(self, query: str, k: int = 5) -> list[Hit]:
        """k + SUPPORT_K passages: reranked across both shelves, or k from the regulation then SUPPORT_K more."""
        self.rerank_note = ""
        if not (RERANK and rerank.enabled()):
            return self._shelves(query, {"regulation": k, "supporting": SUPPORT_K})
        candidates = self._shelves(query, RERANK_POOL)
        try:
            scores = rerank.scores(query, [f"{h.chunk.ref} {h.chunk.title}: {h.chunk.text}" for h in candidates])
        except Exception as error:  # no model (e.g. offline on first run): keep the shelf order
            self.rerank_note = f"not reranked ({type(error).__name__}: {str(error)[:80]})"
            return self._shelves(query, {"regulation": k, "supporting": SUPPORT_K})
        ranked = [Hit(h.chunk, s, [*h.found_by, "rerank"])
                  for s, h in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
        total = k + SUPPORT_K
        if RERANK_MODE == "fuse":  # merge the reranker's order with the first-stage order, by rank
            first = {h.chunk.id: i for i, h in enumerate(candidates, 1)}
            fused = {h.chunk.id: 1 / (RRF_K + i) + 1 / (RRF_K + first[h.chunk.id]) for i, h in enumerate(ranked, 1)}
            ranked = sorted(ranked, key=lambda h: fused[h.chunk.id], reverse=True)
        keep: list[Hit] = []
        if RERANK_MODE == "keep_winners":  # the first stage's clear winners keep their places
            winners = set(self._winners(query))
            keep = [h for h in ranked if h.chunk.id in winners]
        regulation = [h for h in ranked if h.chunk.kind in SHELVES["regulation"] and h not in keep]
        keep += regulation[:max(MIN_REGULATION - len(keep), 0)]
        picked = keep + [h for h in ranked if h not in keep][:max(total - len(keep), 0)]
        return sorted(picked, key=lambda h: ranked.index(h))


@lru_cache(maxsize=1)
def get_index() -> Index:
    return Index(load_chunks())
