"""Rules agent: agentic RAG as a LangGraph state machine.

    retrieve -> grade --(enough relevant passages)--> answer -> check_citations --(valid)--> END
                  |                                                    |
                  +--(weak, retries left)--> rewrite -> retrieve  <----+--(invalid, retries left)
                  +--(weak, no retries left)--> no_rule -> END    <----+--(invalid, no retries left)

Two modes, same graph:
  * LLM mode     (LLM_API_KEY set): an LLM grades passages, rewrites queries and writes a cited answer.
  * offline mode (no key):          keyword-coverage grading, glossary query expansion, extractive answer.
Offline mode lets the loop run and be tested with no API key or cost.
"""

from __future__ import annotations

import re
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from eudr_agent import llm
from eudr_agent.config import settings
from eudr_agent.rag import prompts
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
}

MIN_COVERAGE = 0.6   # offline grading: share of question terms a passage must contain
MIN_RELEVANT = 1     # passages needed before answering


class RAGState(TypedDict, total=False):
    question: str
    query: str
    attempt: int
    hits: list[Hit]
    relevant: list[Hit]
    answer: str
    citations: list[str]
    status: Literal["answered", "no_rule_found"]
    tracer: Tracer


# ---------- nodes ----------

def retrieve(state: RAGState) -> RAGState:
    with state["tracer"].span("retrieve") as rec:
        hits = get_index().search(state["query"], k=settings.top_k)
        rec["query"] = state["query"]
        rec["hits"] = [[h.chunk.ref, round(h.score, 4)] for h in hits]
        rec["found_by"] = {h.chunk.ref: h.found_by for h in hits}
        rec["summary"] = f"query='{state['query'][:60]}' -> {[h.chunk.ref for h in hits[:3]]}"
    return {"hits": hits}


def _coverage(question: str, text: str) -> float:
    """Share of question words found in the passage (a word also counts if one of its glossary terms is)."""
    q = set(tokenize(question))
    if not q:
        return 0.0
    t = set(tokenize(text))
    found = sum(1 for w in q if w in t or any(x in t for x in tokenize(" ".join(GLOSSARY.get(w, [])))))
    return found / len(q)


def grade(state: RAGState) -> RAGState:
    with state["tracer"].span("grade") as rec:
        scored: list[tuple[float, Hit]] = []
        grades = []
        for h in state["hits"]:
            if settings.llm_enabled:
                verdict = llm.chat(prompts.GRADE_SYSTEM,
                                   prompts.GRADE_USER.format(question=state["question"], ref=h.chunk.ref,
                                                             text=h.chunk.text[:1500]), max_tokens=16)
                ok, score = verdict.lower().startswith("y"), h.score
                grades.append([h.chunk.ref, verdict])
            else:
                score = _coverage(state["question"], h.chunk.ref + " " + h.chunk.title + " " + h.chunk.text)
                ok = score >= MIN_COVERAGE
                grades.append([h.chunk.ref, round(score, 2)])
            if ok:
                scored.append((score, h))
        # best-covered passages first, retrieval score breaks ties
        relevant = [h for _, h in sorted(scored, key=lambda x: (x[0], x[1].score), reverse=True)]
        rec["grades"] = grades
        rec["summary"] = f"{len(relevant)}/{len(state['hits'])} relevant"
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
        rec["summary"] = f"attempt {attempt}: '{query[:70]}'"
    return {"query": query, "attempt": attempt}


def _extractive_answer(question: str, hits: list[Hit]) -> tuple[str, list[str]]:
    q = set(tokenize(question))
    lines, cites = [], []
    for h in hits[:2]:
        sentences = re.split(r"(?<=[.;])\s+", h.chunk.text)
        best = max(sentences, key=lambda s: len(q & set(tokenize(s))))
        lines.append(f"{h.chunk.ref}: “{best.strip()[:400]}”")
        cites.append(h.chunk.ref)
    return "\n".join(lines), cites


def answer(state: RAGState) -> RAGState:
    with state["tracer"].span("answer") as rec:
        hits = state["relevant"][:4]
        if settings.llm_enabled:
            passages = "\n\n".join(f"[P{i}] ({h.chunk.ref}) {h.chunk.text[:1800]}" for i, h in enumerate(hits, 1))
            text = llm.chat(prompts.ANSWER_SYSTEM,
                            prompts.ANSWER_USER.format(question=state["question"], passages=passages))
            labels = sorted(set(re.findall(r"\[P(\d+)\]", text)), key=int)
            cites = [hits[int(n) - 1].chunk.ref for n in labels if 0 < int(n) <= len(hits)]
            for n in labels:  # show real references instead of P-labels
                if 0 < int(n) <= len(hits):
                    text = text.replace(f"[P{n}]", f"[{hits[int(n) - 1].chunk.ref}]")
        else:
            text, cites = _extractive_answer(state["question"], hits)
        rec["citations"] = cites
        rec["summary"] = f"cites {cites}"
    return {"answer": text, "citations": cites}


def check_citations(state: RAGState) -> RAGState:
    """Guardrail: an answer must cite at least one passage that was actually retrieved."""
    with state["tracer"].span("check_citations") as rec:
        retrieved = {h.chunk.ref for h in state["relevant"]}
        valid = [c for c in state.get("citations", []) if c in retrieved]
        ok = bool(valid) and "NO_RULE_FOUND" not in state.get("answer", "")
        rec["valid"] = valid
        rec["summary"] = "ok" if ok else "no valid citation"
    return {"citations": valid, "status": "answered" if ok else "no_rule_found"}


def no_rule(state: RAGState) -> RAGState:
    with state["tracer"].span("no_rule_found") as rec:
        rec["summary"] = "abstain"
    return {"status": "no_rule_found", "citations": [],
            "answer": "No rule found in the indexed EUDR texts for this question."}


# ---------- routing ----------

def after_grade(state: RAGState) -> str:
    if len(state["relevant"]) >= MIN_RELEVANT:
        return "answer"
    if state.get("attempt", 0) < settings.max_retries:
        return "rewrite"
    return "no_rule"


def after_check(state: RAGState) -> str:
    """A failed answer is a weak result too: search again while retries remain, then abstain."""
    if state["status"] == "answered":
        return END
    if state.get("attempt", 0) < settings.max_retries:
        return "rewrite"
    return "no_rule"


def build_graph():
    g = StateGraph(RAGState)
    for name, fn in [("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
                     ("answer", answer), ("check_citations", check_citations), ("no_rule", no_rule)]:
        g.add_node(name, fn)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"answer": "answer", "rewrite": "rewrite", "no_rule": "no_rule"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("answer", "check_citations")
    g.add_conditional_edges("check_citations", after_check, {END: END, "rewrite": "rewrite", "no_rule": "no_rule"})
    g.add_edge("no_rule", END)
    return g.compile(checkpointer=False)  # run as a worker inside the supervisor, never saved on its own


_GRAPH = None


def ask(question: str, echo: bool = False, tracer: Tracer | None = None) -> RAGState:
    """Run the rules agent on one question. Returns the final state (answer, citations, status, trace)."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    tracer = tracer or Tracer("rules_agent", echo=echo)
    final = _GRAPH.invoke({"question": question, "query": question, "attempt": 0, "tracer": tracer},
                          {"recursion_limit": 25})
    return final
