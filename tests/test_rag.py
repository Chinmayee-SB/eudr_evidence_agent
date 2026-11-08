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


# ---------- hand-added PDFs, split along their own headings ----------

from eudr_agent.config import MANUAL_DIR  # noqa: E402
from eudr_agent.rag import pdf_split  # noqa: E402
from eudr_agent.rag.chunk import KINDS  # noqa: E402

has_pdfs = pytest.mark.skipif(not (MANUAL_DIR / "EC_FAQ_v5_2026.pdf").exists(), reason="guidance PDFs not added")


def test_faq_heading_must_come_next_in_sequence():
    faq = pdf_split.FAQ("EC FAQ")
    assert faq.heading("4.9. Is it in scope?") is None  # chapter 4 cannot come before chapter 1
    assert faq.heading("1.1. Is it in scope?") == ("EC FAQ Q1.1", "Is it in scope?")
    assert faq.heading("1.2. Who checks?")[0] == "EC FAQ Q1.2"
    assert faq.heading("1.1. See the earlier answer") is None  # a cross-reference that starts a line
    assert faq.heading("1.2.1 A sub-question?")[0] == "EC FAQ Q1.2.1"
    assert faq.heading("2.1. Next chapter?")[0] == "EC FAQ Q2.1"


def test_act_heading_skips_amendment_notes_and_keeps_editor_markers():
    act = pdf_split.Act("Act")
    assert act.heading("1. Short title - (1) This Act")[0] == "Act s.1"
    assert act.heading("3. Inserted by Act 20 of 2001 Dt. 20-8-2001") is None
    assert act.heading("2. Definitions- In this Act")[0] == "Act s.2" and act.rest == "In this Act"
    assert act.heading("1[3. Power to constitute - The State")[0] == "Act s.3"
    assert act.heading("3A. Inserted power - x")[0] == "Act s.3A"
    assert act.heading("40. Appeals - x") is None  # too far ahead to be the next section


def test_guidance_levels_and_body_lists():
    g = pdf_split.Guidance("G")
    assert g.heading("1. DEFINITIONS OF ‘OPERATOR’")[0] == "G §1"
    assert g.heading("a) for distribution to consumers") is None  # a list inside the text
    assert g.heading("a) Placing on the market")[0] == "G §1(a)"
    assert g.heading("7. This requirement comes on top") is None
    assert g.heading("2. DATE OF EFFECT and TIME-FRAME")[0] == "G §2"
    assert g.heading("1. Introduction")[0] == "G §2.1"


def test_contents_pages_are_recognised():
    toc = [f"1.{i}. Question {i} ........................ {10 + i}" for i in range(5)]
    fees = ["For each elephant ..... Twenty rupees"] * 5
    assert pdf_split._is_contents_page(toc) and not pdf_split._is_contents_page(fees)


@has_pdfs
def test_pdfs_are_cited_by_question_and_section():
    r = refs()
    assert "in situ" in r["EC FAQ Q4.10"].title
    assert "Agroforestry" in r["EC Guidance §11.4(d) (part 1)"].title
    assert r["Karnataka Forest Act s.24 (part 1)"].text.startswith("Any person who")
    assert r["EC FAQ Q4.10"].url.endswith("#page=59")


@has_pdfs
def test_pdf_pieces_are_short_and_not_contents_pages():
    pieces = [c for c in load_chunks() if c.source.endswith(".pdf")]
    assert pieces and all(len(c.text) <= pdf_split.MAX_CHARS for c in pieces)
    assert not any(len(pdf_split.DOTTED_ENTRY.findall(c.text)) >= 3 for c in pieces)


def test_every_chunk_has_a_source_type():
    kinds = {c.kind for c in load_chunks()}
    assert kinds <= set(KINDS) and {"law", "recital", "note"} <= kinds
    assert refs()["EUDR Recital 38"].kind == "recital" and refs()["EUDR Art. 2(4)"].kind == "law"


@has_pdfs
def test_search_reaches_the_guidance():
    hits = [h.chunk.ref for h in get_index().search("How should trees able to reach thresholds in situ be applied?")]
    assert "EC FAQ Q4.10" in hits


# ---------- an answer must rest on a legal source ----------

from eudr_agent.rag import agent as rag  # noqa: E402
from eudr_agent.rag.index import Hit  # noqa: E402
from eudr_agent.tracing import Tracer  # noqa: E402


def _hit(ref: str) -> Hit:
    return Hit(refs()[ref], 1.0)


NOTE = "Note: Hansen Global Forest Change loss is not deforestation"


def test_notes_alone_cannot_carry_an_answer():
    state = {"relevant": [_hit(NOTE)], "citations": [NOTE], "answer": "x", "tracer": Tracer("t")}
    assert rag.check_citations(state)["status"] == "no_rule_found"
    state = {**state, "relevant": [_hit(NOTE), _hit("EUDR Art. 2(3)")], "citations": [NOTE, "EUDR Art. 2(3)"]}
    out = rag.check_citations(state)
    assert out["status"] == "answered"
    assert [s["kind"] for s in out["sources"]] == ["note", "law"]


def test_grading_with_only_notes_searches_again():
    assert rag.after_grade({"relevant": [_hit(NOTE)], "attempt": 0}) == "rewrite"
    assert rag.after_grade({"relevant": [_hit(NOTE), _hit("EUDR Art. 2(3)")], "attempt": 0}) == "answer"


def test_best_legal_passage_is_always_given_to_the_answer():
    notes = [h for h in map(Hit, [c for c in load_chunks() if c.kind == "note"], [1.0] * 5)]
    picked = rag.pick_passages([*notes, _hit("EUDR Art. 2(3)")], n=2)
    assert len(picked) == 2 and picked[0].chunk.ref == "EUDR Art. 2(3)"


def test_dataset_question_is_answered_from_the_law_too():
    state = rag.ask("Does tree loss in the Hansen dataset prove deforestation?")
    assert state["status"] == "answered"
    assert any(s["kind"] in {"law", "recital", "guidance"} for s in state["sources"])
    assert all(s["label"] for s in state["sources"])


@has_pdfs
def test_regulation_keeps_its_places_next_to_the_guidance():
    from eudr_agent.rag.index import SHELVES, SUPPORT_K

    hits = get_index().search("Is shade-grown coffee under trees counted as forest?", k=5)
    kinds = [h.chunk.kind for h in hits]
    assert all(k in SHELVES["regulation"] for k in kinds[:5])
    assert 0 < sum(k in SHELVES["supporting"] for k in kinds) <= SUPPORT_K
    # 2(6) is the decisive one: agroforestry plantations are excluded from 'forest'; 2(4) is the general definition
    assert {"EUDR Art. 2(4)", "EUDR Art. 2(6)"} & {h.chunk.ref for h in hits}


# ---------- verify: quotes and claims ----------

from eudr_agent.config import Settings  # noqa: E402
from eudr_agent.rag import verify as checks  # noqa: E402

PASSAGE = "‘deforestation’ means the conversion of forest to agricultural use, whether human-induced or not;"


def test_quotes_must_appear_in_a_passage():
    assert checks.bad_quotes("It is “the conversion of forest to agricultural use” [EUDR Art. 2(3)].", [PASSAGE]) == []
    assert checks.bad_quotes("It is “the conversion of forest to any other use”.", [PASSAGE])
    # an editor's brackets and the PDF reader's spacing do not count as misquotes
    assert checks.bad_quotes("“clear[] or break[] up any land”", ["shall clear or break up any land for"]) == []
    assert checks.bad_quotes("“maki ng available on the market”", ["making available on the market"]) == []
    assert checks.quotes("defined as “forest”") == []  # a defined term, not a quoted claim


def test_sentences_keep_citations_whole():
    parts = checks.sentences("Coffee is not forest [EUDR Art. 2(6)]. Guidance agrees [EC FAQ Q4.10]. "
                             "Reg. No. 995/2010 is repealed.")
    assert parts == ["Coffee is not forest [EUDR Art. 2(6)].", "Guidance agrees [EC FAQ Q4.10].",
                     "Reg. No. 995/2010 is repealed."]


def test_faithfulness_leaves_out_sentences_without_a_claim():
    verdicts = [{"verdict": "supported"}, {"verdict": "unsupported"}, {"verdict": "no_claim"}]
    assert checks.faithfulness(verdicts) == 0.5 and checks.faithfulness([{"verdict": "no_claim"}]) is None


def _verify_state(**extra):
    return {"answer": "Deforestation is any loss of trees [EUDR Art. 2(3)].", "passages": [["EUDR Art. 2(3)", PASSAGE]],
            "citations": ["EUDR Art. 2(3)"], "relevant": [_hit("EUDR Art. 2(3)")], "question": "What is deforestation?",
            "tracer": Tracer("t"), **extra}


def test_unsupported_claim_is_repaired_once_then_abstains(monkeypatch):
    monkeypatch.setattr(Settings, "llm_enabled", property(lambda self: True))
    monkeypatch.setattr(checks, "judge_claims", lambda question, answer, passages: [
        {"sentence": answer, "verdict": "unsupported", "why": "the definition requires conversion"}])
    first = rag.verify_answer(_verify_state())
    assert first["verification"] == "failed" and rag.after_verify(first) == "answer"
    second = rag.verify_answer(_verify_state(repairs=first["repairs"]))
    assert rag.after_verify(second) == "no_rule" and "do not support" in second["abstain_reason"]
    assert "do not support" in rag.no_rule({**second, "tracer": Tracer("t")})["answer"]


def test_judge_failure_keeps_the_answer_unverified(monkeypatch):
    monkeypatch.setattr(Settings, "llm_enabled", property(lambda self: True))
    monkeypatch.setattr(checks, "judge_claims", lambda *a: (_ for _ in ()).throw(ValueError("bad json")))
    out = rag.verify_answer(_verify_state())
    assert out["verification"] == "not_run" and rag.after_verify(out) == "__end__"


def test_offline_answers_pass_the_quote_check():
    state = rag.ask("What does deforestation mean in this regulation?")
    assert state["status"] == "answered" and state["verification"] == "passed"


def test_retrieval_metrics_arithmetic():
    import importlib.util
    from eudr_agent.config import ROOT

    spec = importlib.util.spec_from_file_location("quality", ROOT / "eval" / "run_rag_quality.py")
    quality = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quality)
    s = quality.retrieval_scores(["x", "A", "y", "B"], {"A", "B"}, k=4)
    assert s["hit"] == 1 and s["recall"] == 1 and s["mrr"] == 0.5
    assert abs(s["ndcg"] - (1 / 1.585 + 1 / 2.322) / (1 + 1 / 1.585)) < 0.01
    assert quality.retrieval_scores(["x"], {"A"}, k=1) == {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}


def test_still_unsupported_sentences_are_removed_and_the_rest_kept(monkeypatch):
    monkeypatch.setattr(Settings, "llm_enabled", property(lambda self: True))
    good, bad = "Deforestation is conversion of forest to agricultural use [EUDR Art. 2(3)].", "Pruning is always fine."
    monkeypatch.setattr(checks, "judge_claims", lambda question, answer, passages: [
        {"sentence": good, "verdict": "supported", "why": ""}, {"sentence": bad, "verdict": "unsupported", "why": "not said"}])
    out = rag.verify_answer(_verify_state(answer=f"{good} {bad}", repairs=1))
    assert out["verification"] == "trimmed" and out["answer"] == good and rag.after_verify(out) == "__end__"
    assert out["citations"] == ["EUDR Art. 2(3)"] and out["sources"][0]["kind"] == "law"


def test_citations_after_the_full_stop_move_inside_the_sentence():
    assert checks.normalise("It is **binding**. [EUDR Art. 3] Next one. [EC FAQ Q1.1][EUDR Art. 9(1)]") == \
        "It is binding [EUDR Art. 3]. Next one [EC FAQ Q1.1][EUDR Art. 9(1)]."
    assert len(checks.sentences("A rule applies. [EUDR Art. 3] The guidance adds detail. [EC FAQ Q1.1]")) == 2


def test_quoting_a_source_label_is_not_a_misquote():
    labels = ["Commission guidance (not legally binding)"]
    assert checks.bad_quotes("The FAQ is “not legally binding”.", ["Polygons describe plots."], labels) == []


def test_sentence_ends_inside_a_closing_quote():
    parts = checks.sentences("It does not define “native tree species.” The recitals distinguish forests [EUDR Recital 32].")
    assert len(parts) == 2


# ---------- grading: one call for all passages ----------

def _grade_state():
    hits = [_hit("EUDR Art. 2(3)"), _hit(NOTE), _hit("EUDR Art. 2(4)")]
    return {"question": "What is deforestation?", "hits": hits, "tracer": Tracer("t")}


def test_grading_uses_one_call_for_all_passages(monkeypatch):
    calls = []
    monkeypatch.setattr(Settings, "llm_enabled", property(lambda self: True))
    monkeypatch.setattr(rag.llm, "chat", lambda *a, **k: calls.append(a) or '{"relevant": [1, 3]}')
    out = rag.grade(_grade_state())
    assert len(calls) == 1 and [h.chunk.ref for h in out["relevant"]] == ["EUDR Art. 2(3)", "EUDR Art. 2(4)"]


def test_unreadable_grading_reply_falls_back_to_one_call_per_passage(monkeypatch):
    replies = iter(["not json", "yes", "no", "no"])
    monkeypatch.setattr(Settings, "llm_enabled", property(lambda self: True))
    monkeypatch.setattr(rag.llm, "chat", lambda *a, **k: next(replies))
    out = rag.grade(_grade_state())
    assert [h.chunk.ref for h in out["relevant"]] == ["EUDR Art. 2(3)"]


# ---------- answer cache ----------

from eudr_agent.rag import cache as answer_cache  # noqa: E402


def test_repeated_question_is_answered_from_the_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(answer_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(answer_cache, "enabled", lambda: True)
    runs = []
    real = rag._GRAPH or rag.build_graph()

    class Counting:
        def invoke(self, *a, **k):
            runs.append(1)
            return {**real.invoke(*a, **k), "verification": "passed"}

    monkeypatch.setattr(rag, "_GRAPH", Counting())
    first = rag.ask("What does deforestation mean in this regulation?")
    again = rag.ask("  what does deforestation   mean in this regulation? ")  # same question, other spacing
    assert len(runs) == 1 and again["answer"] == first["answer"] and again["cached"]
    assert again["tracer"].events[0]["node"] == "cache"
    rag.ask("What does deforestation mean in this regulation?", use_cache=False)
    assert len(runs) == 2  # the evaluation bypasses it


def test_cache_starts_fresh_when_the_setup_changes(monkeypatch):
    before = answer_cache.key("Q?")
    monkeypatch.setattr(answer_cache.settings.__class__, "llm_model", property(lambda self: "another-model"))
    assert answer_cache.key("Q?") != before


def test_abstentions_and_unverified_answers_are_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(answer_cache, "CACHE_DIR", tmp_path)
    answer_cache.put("a", {"status": "no_rule_found", "verification": None})
    answer_cache.put("b", {"status": "answered", "verification": "not_run"})
    assert list(tmp_path.iterdir()) == []



def test_reranker_puts_the_answering_faq_first(monkeypatch):
    from eudr_agent.rag import index as ix
    from eudr_agent.rag import rerank

    try:
        rerank.scores("q", ["p"])
    except Exception as error:  # the model is downloaded on first use
        pytest.skip(f"reranker model not available: {error}")
    monkeypatch.setattr(ix, "RERANK", True)
    idx = get_index()
    hits = idx.search("Can one polygon be drawn around several separate coffee plots?", k=5)
    assert hits[0].chunk.ref == "EC FAQ Q1.14" and "rerank" in hits[0].found_by and not idx.rerank_note
    assert sum(h.chunk.kind in ix.SHELVES["regulation"] for h in hits) >= ix.MIN_REGULATION


def test_search_without_the_reranker_model_keeps_the_shelf_order(monkeypatch):
    from eudr_agent.rag import index as ix
    from eudr_agent.rag import rerank

    monkeypatch.setattr(ix, "RERANK", True)
    monkeypatch.setattr(rerank, "scores", lambda *a: (_ for _ in ()).throw(OSError("offline")))
    idx = get_index()
    hits = idx.search("What is the legal definition of forest?", k=5)
    assert len(hits) == 8 and "not reranked" in idx.rerank_note
