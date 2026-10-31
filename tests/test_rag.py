"""Tests for the rules agent. Run `python scripts/build_index.py` once first."""

from __future__ import annotations

import pytest

from eudr_agent.config import CHUNKS_PATH
from eudr_agent.rag.chunk import load_chunks
from eudr_agent.rag.index import get_index, rrf_merge, tokenize

pytestmark = pytest.mark.skipif(not CHUNKS_PATH.exists(), reason="index not built")


def refs() -> dict:
    return {c.ref: c for c in load_chunks()}


def test_definitions_are_split_per_point():
    r = refs()
    assert r["EUDR Art. 2(4)"].text.startswith("‘forest’ means")
    assert "agroforestry" in r["EUDR Art. 2(6)"].text
    assert "31 December, 2020" in r["EUDR Art. 2(13)"].text  # the OJ text has the comma


def test_consolidated_dates_are_current():
    assert "30 December 2026" in refs()["EUDR Art. 38(2)"].text


def test_amendment_markers_removed():
    assert not any("▼" in c.text for c in load_chunks())


def test_chunk_ids_and_citations_are_unique():
    chunks = load_chunks()
    assert len({c.id for c in chunks}) == len(chunks)
    assert len({c.ref for c in chunks}) == len(chunks)


def test_inserted_article_4a_keeps_its_own_citation():
    r = refs()
    assert "Micro or small primary operators" in r["EUDR Art. 4a(2)"].text
    assert "Micro or small primary operators" not in r["EUDR Art. 4(2)"].text


def test_last_recital_stops_before_the_articles():
    r = refs()
    assert len(r["EUDR Recital 86"].text) < 1000
    assert len(r["Reg. 2025/2650 Recital 20"].text) < 2000


def test_rrf_rewards_passages_found_by_both_retrievers():
    merged = rrf_merge({"bm25": ["a", "b"], "vector": ["c", "b"]}, k=3)
    assert merged[0][0] == "b"
    assert merged[0][2] == ["bm25", "vector"]
    assert merged[1][2] == ["bm25"]  # "a": first in BM25 only


def test_tokenize_stems_plurals():
    assert tokenize("trees applies") == tokenize("tree apply")


def test_search_finds_geolocation_definition():
    hits = get_index().search("geolocation plot of land polygon hectares", k=5)
    assert "EUDR Art. 2(28)" in [h.chunk.ref for h in hits]


def test_failed_answer_retries_then_abstains():
    from eudr_agent.config import settings
    from eudr_agent.rag.agent import after_check

    failed = {"status": "no_rule_found", "attempt": 0}
    assert after_check(failed) == "rewrite"
    assert after_check({**failed, "attempt": settings.max_retries}) == "no_rule"


def test_agent_abstains_out_of_scope():
    from eudr_agent.rag.agent import ask

    state = ask("How do I bake sourdough bread at home?")
    assert state["status"] == "no_rule_found"
    assert state["citations"] == []


def test_agent_cites_only_retrieved_passages():
    from eudr_agent.rag.agent import ask

    state = ask("Is coffee one of the relevant commodities?")
    retrieved = {ref for e in state["tracer"].events if e["node"] == "retrieve" for ref, _ in e["hits"]}
    assert state["status"] == "answered"
    assert set(state["citations"]) <= retrieved
