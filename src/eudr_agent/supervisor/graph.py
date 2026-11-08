"""The supervisor graph: one plot's case from input to the officer's sign-off.

    guardrail --(blocked)--> END
        |
    geo_analyst <--(send_back)-- supervisor --(ask_rules)--> rules_agent
        |                          ^   |                         |
        +--------------------------+   +--(finish)--> verdict    +--> back to supervisor
                                                         |
                                              officer (pauses here, saved)
                                                |                    |
                                  (approve / override)        (documents added, once)
                                                |                    |
                                              pack --> END       guardrail

The supervisor node is the only place that chooses a route (see decide.py). Every other edge is fixed.
State is saved after every node by the checkpointer (see checkpoints.py), under thread id = case id, so a
case paused at the officer can be resumed in another process. One trace file per case: runs/traces/<case_id>.jsonl.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, get_args

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from eudr_agent.geo import agent as geo_agent
from eudr_agent.guardrail import check_input
from eudr_agent.rag import agent as rag_agent
from eudr_agent.state import CaseFile, PlotInput, Verdict
from eudr_agent.supervisor import verdict_rules
from eudr_agent.supervisor.decide import decide
from eudr_agent.supervisor.review import review_evidence
from eudr_agent.tracing import Tracer, new_run_id

MAX_RERUNS = 1   # officer "documents added" re-runs per case
OFFICER_DECISIONS = {"approve", "override", "documents_added"}
VERDICTS = set(get_args(Verdict))

# Tracers hold an open callback (the desk's live view), so they live here, not in the saved state.
_tracers: dict[str, Tracer] = {}


def _tracer(case: CaseFile) -> Tracer:
    case_id = case["case_id"]
    if case_id not in _tracers:  # e.g. resumed in a new process: carry on the same trace file
        _tracers[case_id] = Tracer("supervisor", run_id=case_id)
    return _tracers[case_id]


# ---------- nodes ----------

def guardrail_node(case: CaseFile) -> CaseFile:
    check = check_input(case["plot"], tracer=_tracer(case))
    return {"guardrail": check, "status": "running" if check["passed"] else "blocked"}


def make_geo_node(analyse: Callable[..., Any]) -> Callable[[CaseFile], CaseFile]:
    def geo_node(case: CaseFile) -> CaseFile:
        task = case.get("geo_task")
        report: dict[str, Any] = {"round": len(case.get("geo_reports", [])) + 1, "task": task or "initial screening"}
        new_findings = []
        try:
            # earlier findings go in, so a follow-up builds on them instead of repeating them
            final = analyse(case["guardrail"]["plot"], task=task, tracer=_tracer(case),
                            prior_findings=case.get("findings", []))
            new_findings = final["findings"]
            report |= {"summary": final["summary"], "conflicts": final["conflicts"],
                       "open_questions": final["open_questions"], "hit_limit": final.get("hit_limit", False)}
        except Exception as error:  # a failed round is recorded; the supervisor sees it as gaps
            report["error"] = f"{type(error).__name__}: {error}"[:300]
        return {"findings": [*case.get("findings", []), *new_findings],
                "geo_reports": [*case.get("geo_reports", []), report],
                "geo_task": "", "steps": case.get("steps", 0) + 1}
    return geo_node


def supervisor_node(case: CaseFile) -> CaseFile:
    tracer = _tracer(case)
    with tracer.span("supervisor") as rec:
        review = review_evidence(case["guardrail"]["plot"], case.get("findings", []))
        decision = decide(case, review)
        rec.update(gaps=[g["question"] for g in review["gaps"]], conflicts=[c["text"] for c in review["conflicts"]],
                   not_checkable=review["not_checkable"], decision=decision)
        rec["summary"] = f"{decision['action']} [{decision['source']}]: {decision['reason'][:110]}"
    update: CaseFile = {"review": review, "decisions": [*case.get("decisions", []), decision],
                        "steps": case.get("steps", 0) + 1}
    if decision["action"] == "send_back":
        update |= {"geo_task": decision["task"], "send_backs": case.get("send_backs", 0) + 1}
    elif decision["action"] == "ask_rules":
        update["pending_questions"] = decision["questions"]
    return update


def make_rules_node(ask: Callable[..., Any]) -> Callable[[CaseFile], CaseFile]:
    def rules_node(case: CaseFile) -> CaseFile:
        answers = []
        for question in case.get("pending_questions", []):
            try:
                final = ask(question, tracer=_tracer(case))
                answers.append({"question": question, "answer": final["answer"],
                                "citations": final["citations"], "status": final["status"]})
            except Exception as error:
                answers.append({"question": question, "status": "failed", "citations": [],
                                "error": f"{type(error).__name__}: {error}"[:300]})
        return {"rules": [*case.get("rules", []), *answers], "pending_questions": [],
                "steps": case.get("steps", 0) + 1}
    return rules_node


def _open_items(case: CaseFile) -> list[str]:
    review = case.get("review", {})
    items = [f"Missing: {g['text']}" for g in review.get("gaps", [])]
    items += [f"Not checked: {n}" for n in review.get("not_checkable", [])]
    items += [f"Geo-analyst round {r['round']} failed: {r['error']}" for r in case.get("geo_reports", []) if r.get("error")]
    items += [f"Input: {w}" for w in case["guardrail"]["warnings"]]
    items += [f"Document held back: {d}" for d in case["guardrail"]["dropped"]]
    items += [f"Rule check failed: {r['question']}" for r in case.get("rules", []) if r["status"] == "failed"]
    return items


def verdict_node(case: CaseFile) -> CaseFile:
    with _tracer(case).span("verdict") as rec:
        try:
            verdict, note = verdict_rules.decide(case), "from the fixed verdict rules"
        except NotImplementedError:
            verdict, note = None, "not issued: the verdict rules are not built yet"
        rec.update(verdict=verdict, note=note)
        rec["summary"] = verdict or note
    return {"draft_verdict": verdict, "verdict_note": note, "open_items": _open_items(case),
            "status": "awaiting_officer"}


def validate_officer(value: dict[str, Any], case: CaseFile) -> None:
    """Raise ValueError if the officer's decision cannot be accepted. Checked before the graph resumes."""
    decision = value.get("decision")
    if decision not in OFFICER_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(OFFICER_DECISIONS)}")
    if decision == "override":
        if value.get("verdict") not in VERDICTS:
            raise ValueError(f"an override needs a verdict: one of {sorted(VERDICTS)}")
        if not str(value.get("reason", "")).strip():
            raise ValueError("an override needs a reason, for the record")
    if decision == "approve" and not case.get("draft_verdict") and not str(value.get("reason", "")).strip():
        raise ValueError("no verdict was issued, so approving needs a reason (or override with a verdict)")
    if decision == "documents_added":
        if case.get("reruns", 0) >= MAX_RERUNS:
            raise ValueError("documents were already added once for this case; start a new case")
        if not value.get("documents"):
            raise ValueError("documents_added needs the case's document list")


def officer_node(case: CaseFile) -> CaseFile:
    value = interrupt({"case_id": case["case_id"], "draft_verdict": case.get("draft_verdict"),
                       "verdict_note": case.get("verdict_note"), "open_items": case.get("open_items", [])})
    with _tracer(case).span("officer") as rec:
        rec.update(decision=value)
        rec["summary"] = f"{value['decision']}" + (f": {value.get('reason')}" if value.get("reason") else "")
    officer = {k: value[k] for k in ("decision", "verdict", "reason") if value.get(k)}
    if value["decision"] == "documents_added":
        names = ", ".join(Path(p).name for p in value["documents"])
        return {"officer": officer, "status": "running", "reruns": case.get("reruns", 0) + 1,
                "plot": PlotInput(**{**case["plot"], "documents": list(value["documents"])}),
                # a fresh evidence round with its own budget; rule answers are kept
                "steps": 0, "send_backs": 0,
                "geo_task": f"New documents were added to the case ({names}). Read them and report whether "
                            "they change the evidence."}
    final = value["verdict"] if value["decision"] == "override" else case.get("draft_verdict")
    return {"officer": officer, "final_verdict": final}


def pack_node(case: CaseFile) -> CaseFile:
    with _tracer(case).span("pack") as rec:
        rec["summary"] = f"case closed: {case['officer']['decision']}, final verdict {case.get('final_verdict')}"
    return {"status": "closed"}


# ---------- routing ----------

def after_guardrail(case: CaseFile) -> str:
    return "geo_analyst" if case["guardrail"]["passed"] else END


def after_supervisor(case: CaseFile) -> str:
    return {"send_back": "geo_analyst", "ask_rules": "rules_agent", "finish": "verdict"}[case["decisions"][-1]["action"]]


def after_officer(case: CaseFile) -> str:
    return "guardrail" if case["officer"]["decision"] == "documents_added" else "pack"


def build_graph(checkpointer: Any = None, analyse: Callable[..., Any] = geo_agent.analyse,
                ask: Callable[..., Any] = rag_agent.ask):
    """`analyse` and `ask` are the two workers; tests pass stand-ins so no LLM is needed."""
    g = StateGraph(CaseFile)
    g.add_node("guardrail", guardrail_node)
    g.add_node("geo_analyst", make_geo_node(analyse))
    g.add_node("supervisor", supervisor_node)
    g.add_node("rules_agent", make_rules_node(ask))
    g.add_node("verdict", verdict_node)
    g.add_node("officer", officer_node)
    g.add_node("pack", pack_node)
    g.set_entry_point("guardrail")
    g.add_conditional_edges("guardrail", after_guardrail, {"geo_analyst": "geo_analyst", END: END})
    g.add_edge("geo_analyst", "supervisor")
    g.add_conditional_edges("supervisor", after_supervisor,
                            {"geo_analyst": "geo_analyst", "rules_agent": "rules_agent", "verdict": "verdict"})
    g.add_edge("rules_agent", "supervisor")
    g.add_edge("verdict", "officer")
    g.add_conditional_edges("officer", after_officer, {"guardrail": "guardrail", "pack": "pack"})
    g.add_edge("pack", END)
    return g.compile(checkpointer=checkpointer)


# ---------- running a case ----------

RECURSION_LIMIT = 60  # far above any path the budgets allow; a safety net only


def _config(case_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": case_id}, "recursion_limit": RECURSION_LIMIT}


def start_case(graph: Any, plot: PlotInput, echo: bool = False, on_event: Callable | None = None) -> CaseFile:
    """Run a new case until it pauses for the officer, is blocked, or ends. Returns the saved case file.
    `echo` prints each step; `on_event` lets a screen show each step as it finishes."""
    case_id = f"{plot['plot_id']}-{new_run_id()}"
    _tracers[case_id] = Tracer("supervisor", echo=echo, on_event=on_event, run_id=case_id)
    graph.invoke({"case_id": case_id, "plot": plot, "steps": 0, "send_backs": 0, "reruns": 0,
                  "findings": [], "geo_reports": [], "decisions": [], "rules": [], "status": "running"},
                 _config(case_id))
    return get_case(graph, case_id)


def resume_case(graph: Any, case_id: str, officer: dict[str, Any], echo: bool = False,
                on_event: Callable | None = None) -> CaseFile:
    """Give the officer's decision to a paused case and run it on. Raises ValueError for an invalid decision."""
    case = get_case(graph, case_id)
    if case.get("status") != "awaiting_officer":
        raise ValueError(f"case {case_id} is not waiting for the officer (status: {case.get('status')})")
    validate_officer(officer, case)
    _tracers[case_id] = Tracer("supervisor", echo=echo, on_event=on_event, run_id=case_id)
    graph.invoke(Command(resume=officer), _config(case_id))
    return get_case(graph, case_id)


def get_case(graph: Any, case_id: str) -> CaseFile:
    return graph.get_state(_config(case_id)).values
