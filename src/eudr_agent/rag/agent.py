"""Rules agent: agentic RAG as a LangGraph state machine.

    retrieve -> grade --(a relevant legal passage)--> answer -> check_citations --(valid)--> verify --(ok)--> END
                  |                                      ^             |                        |
                  |                                      +-------------|--(a claim unsupported, once)
                  +--(weak, retries left)--> rewrite -> retrieve  <----+--(invalid, retries left)
                  +--(weak, no retries left)--> no_rule -> END    <----+--(invalid, no retries left)
                                                         ^------(still unsupported, nothing left after trimming)

verify (see verify.py) checks that quoted phrases appear in the passages and, with an LLM, that each sentence
is supported by them. A failed answer is written again once, told which claims failed. If it still fails, the
unsupported sentences are removed; the agent abstains only if no supported sentence with a legal citation is left.

Two modes, same graph:
  * LLM mode     (LLM_API_KEY set): an LLM grades passages, rewrites queries and writes a cited answer.
  * offline mode (no key):          keyword-coverage grading, glossary query expansion, extractive answer.
Offline mode lets the loop run and be tested with no API key or cost.

An answer must rest on a legal source: EU law, a recital, Commission guidance or Indian law. Our own dataset
notes explain what a dataset can and cannot show, so they may support an answer but never carry it alone.
"""

from __future__ import annotations

import json
import re
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from eudr_agent import llm
from eudr_agent.config import settings
from eudr_agent.rag import cache as answer_cache
from eudr_agent.rag import prompts, verify as checks
from eudr_agent.rag.chunk import KINDS, LEGAL_KINDS
from eudr_agent.rag.index import Hit, get_index, tokenize
from eudr_agent.tracing import Tracer

# Plain-language words a user might type -> the regulation's own terms (offline rewrite).
GLOSSARY = {
    "shade": ["agroforestry", "tree cover", "agricultural plantation"],
    "coffee": ["relevant commodities", "agricultural plantation"],
    "pruning": ["agricultural use", "forest degradation", "tree cover"],
    "thinning": ["agricultural use", "forest degradation", "canopy cover"],
    "farm": ["agricultural use", "agricultural plantation"],
    "cutoff": ["deforestation-free", "31 December 2020"],
    "cut": ["deforestation-free", "31 December 2020"],
    "deadline": ["shall apply from", "date of application"],
    "apply": ["shall apply from", "date of application"],
    "polygon": ["geolocation", "plot of land", "hectares"],
    "gps": ["geolocation", "latitude", "longitude"],
    "boundary": ["geolocation", "plot of land", "polygon"],
    "satellite": ["earth observation", "competent authorities", "checks"],
    "inspector": ["competent authorities", "checks on operators"],
    "penalty": ["penalties", "fines", "confiscation"],
    "fine": ["penalties", "fines"],
    "small": ["micro", "small undertakings", "smallholders"],
    "smallholder": ["smallholders", "micro", "small undertakings"],
    "documents": ["information requirements", "due diligence statement"],
    "evidence": ["information requirements", "verifiable", "due diligence"],
    # dataset words: a question about what a map shows is answered by the definition it is tested against
    "hansen": ["deforestation", "conversion", "agricultural use"],
    "dataset": ["deforestation", "forest degradation"],
    "loss": ["deforestation", "conversion of forest"],
    "tree": ["forest", "canopy cover"],
    "prove": ["deforestation-free", "verifiable"],
    # the EUDR never names reserve forests or protected areas: it defers to the country's own law (Art. 2(40))
    "reserve": ["relevant legislation of the country of production", "legal status", "land use rights"],
    "reserved": ["relevant legislation of the country of production", "legal status", "land use rights"],
    "protected": ["relevant legislation of the country of production", "environmental protection", "legal status"],
    "legal": ["relevant legislation", "legality"],
}

MIN_COVERAGE = 0.6   # offline grading: share of question terms a passage must contain
MIN_RELEVANT = 1     # legal passages needed before answering
ANSWER_PASSAGES = 4  # passages given to the answer step
MAX_REPAIRS = 1      # times an answer with unsupported claims is written again before the agent abstains


class RAGState(TypedDict, total=False):
    question: str
    query: str
    attempt: int
    hits: list[Hit]
    relevant: list[Hit]
    answer: str
    citations: list[str]
    sources: list[dict[str, str]]  # each citation with its source type, e.g. "Commission guidance (not legally binding)"
    status: Literal["answered", "no_rule_found"]
    passages: list[list[str]]      # [ref, text] of the passages the answer was written from
    passage_labels: list[str]      # their source-type labels, as the answer step saw them
    verification: str              # passed, repaired (passed on the second try), trimmed (unsupported
                                   # sentences removed), not_run (the judge failed), or failed
    problems: list[str]            # why the last answer failed verification
    repairs: int
    abstain_reason: str
    tracer: Tracer


def is_legal(hit: Hit) -> bool:
    return hit.chunk.kind in LEGAL_KINDS


def pick_passages(relevant: list[Hit], n: int = ANSWER_PASSAGES) -> list[Hit]:
    """The best passages, making sure the best legal one is among them even if notes rank higher."""
    picked = relevant[:n]
    if not any(is_legal(h) for h in picked):
        legal = next((h for h in relevant if is_legal(h)), None)
        if legal:
            picked = [legal, *picked[:n - 1]]
    return picked


def describe(ref: str, by_ref: dict[str, Hit]) -> dict[str, str]:
    chunk = by_ref[ref].chunk
    return {"ref": ref, "kind": chunk.kind, "label": KINDS.get(chunk.kind, chunk.kind), "url": chunk.url}


# ---------- nodes ----------

def retrieve(state: RAGState) -> RAGState:
    with state["tracer"].span("retrieve") as rec:
        hits = get_index().search(state["query"], k=settings.top_k)
        rec["query"] = state["query"]
        rec["hits"] = [[h.chunk.ref, round(h.score, 4)] for h in hits]
        rec["found_by"] = {h.chunk.ref: h.found_by for h in hits}
        if get_index().rerank_note:
            rec["rerank"] = get_index().rerank_note
        rec["summary"] = f"query='{state['query']}' -> {[h.chunk.ref for h in hits[:3]]}"
    return {"hits": hits}


def _coverage(question: str, text: str) -> float:
    """Share of question words found in the passage (a word also counts if one of its glossary terms is)."""
    q = set(tokenize(question))
    if not q:
        return 0.0
    t = set(tokenize(text))
    found = sum(1 for w in q if w in t or any(x in t for x in tokenize(" ".join(GLOSSARY.get(w, [])))))
    return found / len(q)


def _grade_all(question: str, hits: list[Hit]) -> set[int]:
    """One LLM call for all passages. Returns the indexes judged relevant; raises if the reply is unreadable."""
    passages = "\n\n".join(f"{i}. ({h.chunk.ref}; {h.chunk.title}) {h.chunk.text[:1500]}" for i, h in enumerate(hits, 1))
    text = llm.chat(prompts.GRADE_ALL_SYSTEM, prompts.GRADE_ALL_USER.format(question=question, passages=passages),
                    max_tokens=100)
    numbers = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())["relevant"]
    return {int(n) - 1 for n in numbers if 0 < int(n) <= len(hits)}


def _grade_one(question: str, hit: Hit) -> bool:
    verdict = llm.chat(prompts.GRADE_SYSTEM, prompts.GRADE_USER.format(question=question, ref=hit.chunk.ref,
                                                                       text=hit.chunk.text[:1500]), max_tokens=16)
    return verdict.lower().startswith("y")


def grade(state: RAGState) -> RAGState:
    with state["tracer"].span("grade") as rec:
        scored: list[tuple[float, Hit]] = []
        grades = []
        chosen: set[int] | None = None
        if settings.llm_enabled:
            try:
                chosen = _grade_all(state["question"], state["hits"])
                rec["mode"] = "one call"
            except Exception as error:  # unreadable reply: grade passage by passage, as before
                rec["mode"] = f"per passage ({type(error).__name__})"
        for i, h in enumerate(state["hits"]):
            if settings.llm_enabled:
                ok = i in chosen if chosen is not None else _grade_one(state["question"], h)
                score = h.score
                grades.append([h.chunk.ref, "yes" if ok else "no"])
            else:
                score = _coverage(state["question"], h.chunk.ref + " " + h.chunk.title + " " + h.chunk.text)
                ok = score >= MIN_COVERAGE
                grades.append([h.chunk.ref, round(score, 2)])
            if ok:
                scored.append((score, h))
        # binding law first, then best-covered, retrieval score breaking ties; the answer step quotes the
        # rule before the guidance that explains it
        relevant = [h for _, h in sorted(scored, key=lambda x: (x[1].chunk.kind == "law", x[0], x[1].score),
                                         reverse=True)]
        rec["grades"] = grades
        legal = sum(is_legal(h) for h in relevant)
        rec["summary"] = f"{len(relevant)}/{len(state['hits'])} relevant, {legal} from a legal source"
    return {"relevant": relevant}


def rewrite(state: RAGState) -> RAGState:
    with state["tracer"].span("rewrite") as rec:
        if settings.llm_enabled:
            query = llm.chat(prompts.REWRITE_SYSTEM,
                             prompts.REWRITE_USER.format(question=state["question"], query=state["query"]),
                             max_tokens=60)
        else:
            extra = [t for w in tokenize(state["question"]) for t in GLOSSARY.get(w, [])]
            query = state["question"] + " " + " ".join(dict.fromkeys(extra))
        attempt = state.get("attempt", 0) + 1
        rec["attempt"] = attempt
        rec["summary"] = f"attempt {attempt}: '{query}'"
    return {"query": query, "attempt": attempt}


def _extractive_answer(question: str, hits: list[Hit]) -> tuple[str, list[str]]:
    q = set(tokenize(question))
    lines, cites = [], []
    for h in pick_passages(hits, 2):
        sentences = re.split(r"(?<=[.;])\s+", h.chunk.text)
        best = max(sentences, key=lambda s: len(q & set(tokenize(s))))
        lines.append(f"{h.chunk.ref}: “{best.strip()[:400]}”")
        cites.append(h.chunk.ref)
    return "\n".join(lines), cites


def answer(state: RAGState) -> RAGState:
    with state["tracer"].span("answer") as rec:
        hits = pick_passages(state["relevant"])
        if settings.llm_enabled:
            passages = "\n\n".join(
                f"[P{i}] ({h.chunk.ref}; {KINDS.get(h.chunk.kind, h.chunk.kind)}; {h.chunk.title}) {h.chunk.text[:1800]}"
                for i, h in enumerate(hits, 1))
            user = prompts.ANSWER_USER.format(question=state["question"], passages=passages)
            if state.get("problems"):  # a repair: say what was wrong with the last answer
                user += prompts.REPAIR_USER.format(problems="\n".join(f"- {p}" for p in state["problems"]))
            text = llm.chat(prompts.ANSWER_SYSTEM, user)
            labels = sorted(set(re.findall(r"\[P(\d+)\]", text)), key=int)
            cites = [hits[int(n) - 1].chunk.ref for n in labels if 0 < int(n) <= len(hits)]
            for n in labels:  # show real references instead of P-labels
                if 0 < int(n) <= len(hits):
                    text = text.replace(f"[P{n}]", f"[{hits[int(n) - 1].chunk.ref}]")
            text = checks.normalise(text)
        else:
            text, cites = _extractive_answer(state["question"], hits)
        rec["citations"] = cites
        rec["summary"] = f"cites {cites}" + (" (repair)" if state.get("problems") else "")
    return {"answer": text, "citations": cites, "passages": [[h.chunk.ref, h.chunk.text] for h in hits],
            "passage_labels": list(dict.fromkeys(KINDS.get(h.chunk.kind, "") for h in hits))}


def check_citations(state: RAGState) -> RAGState:
    """Guardrail: an answer must cite passages that were actually retrieved, at least one of them a legal source."""
    with state["tracer"].span("check_citations") as rec:
        by_ref = {h.chunk.ref: h for h in state["relevant"]}
        valid = [c for c in state.get("citations", []) if c in by_ref]
        legal = [c for c in valid if is_legal(by_ref[c])]
        ok = bool(legal) and "NO_RULE_FOUND" not in state.get("answer", "")
        rec["valid"], rec["legal"] = valid, legal
        rec["summary"] = "ok" if ok else ("only dataset notes cited, no legal source" if valid else "no valid citation")
    return {"citations": valid, "sources": [describe(c, by_ref) for c in valid],
            "status": "answered" if ok else "no_rule_found"}


def _trim(state: RAGState, bad: set[str], verdicts: list[dict]) -> RAGState | None:
    """Keep only the sentences that passed. None if nothing supported, or no legal citation, would be left."""
    kept = [x for x in checks.sentences(state["answer"]) if x not in bad]
    supported = {v["sentence"] for v in verdicts if v["verdict"] == "supported"}
    by_ref = {h.chunk.ref: h for h in state["relevant"]}
    cites = [c for c in state["citations"] if any(f"[{c}]" in x for x in kept)]
    if not (set(kept) & supported) or not any(is_legal(by_ref[c]) for c in cites if c in by_ref):
        return None
    return {"answer": " ".join(kept), "citations": cites, "sources": [describe(c, by_ref) for c in cites]}


def verify_answer(state: RAGState) -> RAGState:
    """Quoted phrases must appear in the passages; with an LLM, every claim must be supported by them.
    A failing answer is written again once; if it still fails, the unsupported sentences are removed, and the
    agent abstains only if no supported sentence with a legal citation is left."""
    with state["tracer"].span("verify") as rec:
        answer = state["answer"]
        texts = [text for _, text in state["passages"]]
        misquoted = checks.bad_quotes(answer, texts, state.get("passage_labels", []))
        problems = [f"this quote is not in any passage: “{q}”" for q in misquoted]
        bad = {x for x in checks.sentences(answer) if any(q in x for q in misquoted)}
        verdicts: list[dict] = []
        verification = "passed"
        if settings.llm_enabled:
            try:
                by_ref = {h.chunk.ref: h for h in state["relevant"]}
                verdicts = checks.judge_claims(state["question"], answer, [
                    (r, KINDS.get(by_ref[r].chunk.kind, "") if r in by_ref else "", t) for r, t in state["passages"]])
                rec["verdicts"] = verdicts
                for v in verdicts:
                    if v["verdict"] == "unsupported":
                        problems.append(f"not supported by the passages: “{v['sentence']}” ({v['why']})")
                        bad.add(v["sentence"])
            except Exception as error:  # the judge failed: keep the answer, say it was not verified
                verification = "not_run"
                rec["judge_error"] = f"{type(error).__name__}: {error}"[:200]
        update: RAGState = {"problems": problems}
        if not problems:
            update["verification"] = "repaired" if verification == "passed" and state.get("repairs") else verification
        elif state.get("repairs", 0) < MAX_REPAIRS:
            update |= {"verification": "failed", "repairs": state.get("repairs", 0) + 1}
        else:
            trimmed = _trim(state, bad, verdicts)
            if trimmed:
                update |= trimmed | {"verification": "trimmed"}
            else:
                update |= {"verification": "failed",
                           "abstain_reason": "the answer made claims its sources do not support: "
                                             + "; ".join(problems)[:400]}
        rec["problems"] = problems
        rec["summary"] = update["verification"] + (f": {len(problems)} problem(s)" if problems else "")
    return update


def no_rule(state: RAGState) -> RAGState:
    with state["tracer"].span("no_rule_found") as rec:
        rec["summary"] = "abstain" + (f": {state['abstain_reason']}" if state.get("abstain_reason") else "")
    answer = "No rule found in the indexed EUDR texts for this question."
    if state.get("abstain_reason"):
        answer = f"No answer given: {state['abstain_reason']}"
    return {"status": "no_rule_found", "citations": [], "sources": [], "answer": answer}


# ---------- routing ----------

def after_grade(state: RAGState) -> str:
    if sum(is_legal(h) for h in state["relevant"]) >= MIN_RELEVANT:
        return "answer"
    if state.get("attempt", 0) < settings.max_retries:
        return "rewrite"
    return "no_rule"


def after_check(state: RAGState) -> str:
    """A failed answer is a weak result too: search again while retries remain, then abstain."""
    if state["status"] == "answered":
        return "verify"
    if state.get("attempt", 0) < settings.max_retries:
        return "rewrite"
    return "no_rule"


def after_verify(state: RAGState) -> str:
    if state.get("abstain_reason"):
        return "no_rule"
    return "answer" if state["verification"] == "failed" else END


def build_graph():
    g = StateGraph(RAGState)
    for name, fn in [("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
                     ("answer", answer), ("check_citations", check_citations), ("verify", verify_answer),
                     ("no_rule", no_rule)]:
        g.add_node(name, fn)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"answer": "answer", "rewrite": "rewrite", "no_rule": "no_rule"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("answer", "check_citations")
    g.add_conditional_edges("check_citations", after_check,
                            {"verify": "verify", "rewrite": "rewrite", "no_rule": "no_rule"})
    g.add_conditional_edges("verify", after_verify, {END: END, "answer": "answer", "no_rule": "no_rule"})
    g.add_edge("no_rule", END)
    return g.compile(checkpointer=False)  # run as a worker inside the supervisor, never saved on its own


_GRAPH = None


def ask(question: str, echo: bool = False, tracer: Tracer | None = None, use_cache: bool = True) -> RAGState:
    """Run the rules agent on one question. Returns the final state (answer, citations, status, trace).
    A question already answered with the same setup is answered from the cache (see cache.py)."""
    global _GRAPH
    tracer = tracer or Tracer("rules_agent", echo=echo)
    if use_cache and answer_cache.enabled():
        cached = answer_cache.get(question)
        if cached:
            with tracer.span("cache") as rec:
                rec["created"] = cached["created"]
                rec["summary"] = f"answer reused from {cached['created']} (same question, same texts and model)"
            return {**{k: cached[k] for k in answer_cache.KEPT}, "question": question, "cached": cached["created"],
                    "tracer": tracer}
    if _GRAPH is None:
        _GRAPH = build_graph()
    final = _GRAPH.invoke({"question": question, "query": question, "attempt": 0, "tracer": tracer},
                          {"recursion_limit": 25})
    if use_cache and answer_cache.enabled():
        answer_cache.put(question, final)
    return final
