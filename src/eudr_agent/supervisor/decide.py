"""The supervisor's decision: the LLM proposes the next step, code enforces the limits.

    review (code) --> propose (LLM) --> enforce (code) --> send_back | ask_rules | finish

Code has the last word on four things, whatever the LLM says:
  1. budgets: at most MAX_SEND_BACKS send-backs, MAX_RULE_QUESTIONS rule questions, MAX_STEPS steps
  2. required evidence: "finish" is refused while a required question is unanswered and a send-back is left
  3. shape: an unknown action, an empty task or no new question is refused
  4. availability: with no LLM, an LLM error or an unreadable reply, the code decision is used instead

The code decision on its own is a complete, simpler supervisor: gaps -> send back, conflicts -> fixed rule
questions, otherwise finish. Every decision records its source ("llm" or "code") and why code overruled.
"""

from __future__ import annotations

import json
from typing import Any

from eudr_agent import llm
from eudr_agent.config import settings
from eudr_agent.state import CaseFile, Decision
from eudr_agent.supervisor import prompts

MAX_STEPS = 10           # agent calls plus supervisor decisions, per case
MAX_SEND_BACKS = 1       # a second round over the same datasets rarely adds evidence; gaps go to the officer
MAX_RULE_QUESTIONS = 3   # per case
MAX_QUESTION_CHARS = 300
ACTIONS = {"send_back", "ask_rules", "finish"}

# The fixed question for each conflict kind, used when the LLM is not. Each is in the rules evaluation set.
RULE_QUESTIONS = {
    "forest_maps_disagree": "Is shade-grown coffee under trees counted as forest?",
    "forest_vs_declared_crop": "Is shade-grown coffee under trees counted as forest?",
    "loss_datasets_disagree": "What does deforestation mean in this regulation?",
}


def budgets(case: CaseFile) -> dict[str, int]:
    asked = len(case.get("rules", []))
    return {"send_backs_left": MAX_SEND_BACKS - case.get("send_backs", 0),
            "rule_questions_left": MAX_RULE_QUESTIONS - asked,
            "steps_left": MAX_STEPS - case.get("steps", 0)}


def _asked(case: CaseFile) -> set[str]:
    return {r["question"].strip().lower() for r in case.get("rules", [])}


def code_decision(case: CaseFile, review: dict[str, Any]) -> Decision:
    left = budgets(case)
    if review["gaps"] and left["send_backs_left"] > 0:
        return Decision(action="send_back", source="code", task=" ".join(g["task"] for g in review["gaps"]),
                        reason="Required evidence is missing: " + "; ".join(g["question"] for g in review["gaps"]))
    questions = list(dict.fromkeys(RULE_QUESTIONS[c["kind"]] for c in review["conflicts"] if c["kind"] in RULE_QUESTIONS))
    questions = [q for q in questions if q.lower() not in _asked(case)][:max(left["rule_questions_left"], 0)]
    if questions:
        return Decision(action="ask_rules", source="code", questions=questions,
                        reason="The datasets disagree, so the relevant rules are looked up: "
                               + "; ".join(c["text"] for c in review["conflicts"]))
    if review["gaps"]:
        reason = "No send-back left; missing evidence goes to the officer: " + "; ".join(g["question"] for g in review["gaps"])
    else:
        reason = "The evidence is as complete as the available tools allow."
    return Decision(action="finish", source="code", reason=reason)


def _brief(case: CaseFile, review: dict[str, Any]) -> str:
    """What the LLM sees: numbers as the tools returned them, document text shortened."""
    findings = []
    for f in case.get("findings", []):
        item = {"tool": f.get("tool"), "dataset": f.get("dataset")}
        if f.get("error"):
            item["error"] = f["error"]
        elif f.get("tool") == "read_document":
            item["document"] = {"file": f["values"]["file"], "text": f["values"]["text"][:600]}
        else:
            item["values"] = f.get("values")
        findings.append(item)
    plot = case["guardrail"]["plot"]
    latest_report = (case.get("geo_reports") or [{}])[-1]
    on_case = [p.rsplit("/", 1)[-1] for p in plot.get("documents", [])]
    read = {f["values"]["file"] for f in case.get("findings", []) if f.get("tool") == "read_document" and not f.get("error")}
    return json.dumps({
        "plot": {k: plot.get(k) for k in ("plot_id", "area_ha", "declared_crop", "crop_since")} |
                {"documents_on_case": on_case, "documents_unread": [d for d in on_case if d not in read]},
        "protected_area_in_force": review["protected_in_force"],
        "input_warnings": case["guardrail"]["warnings"],
        "findings": findings,
        "geo_analyst_report": {k: latest_report.get(k) for k in ("summary", "conflicts", "open_questions", "error")
                               if latest_report.get(k)},
        "code_review": {k: review[k] for k in ("gaps", "conflicts", "not_checkable", "skipped")},
        "rules_answered": [{"question": r["question"], "status": r["status"], "answer": r.get("answer", "")[:400],
                            "citations": r.get("citations", [])} for r in case.get("rules", [])],
        "budgets": budgets(case),
    }, indent=1)


def llm_proposal(case: CaseFile, review: dict[str, Any]) -> dict[str, Any]:
    text = llm.chat(prompts.SUPERVISOR_SYSTEM, _brief(case, review), max_tokens=500)
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    proposal = json.loads(text)
    if not isinstance(proposal, dict):
        raise ValueError("reply is not a JSON object")
    return proposal


def enforce(proposal: dict[str, Any], case: CaseFile, review: dict[str, Any]) -> Decision:
    """Turn an LLM proposal into a decision the case may act on, or replace it with the code decision."""
    left = budgets(case)

    def overrule(why: str) -> Decision:
        return Decision(**code_decision(case, review), overruled=f"LLM proposed '{proposal.get('action')}': {why}")

    action, reason = proposal.get("action"), str(proposal.get("reason", "")).strip()
    if left["steps_left"] <= 0:
        return Decision(action="finish", source="code", reason="Step budget used up; the case goes to the officer as is.",
                        overruled=f"LLM proposed '{action}': no steps left")
    if action not in ACTIONS:
        return overrule("not one of send_back, ask_rules, finish")

    if action == "send_back":
        if left["send_backs_left"] <= 0:
            return overrule("no send-back left")
        task = str(proposal.get("task", "")).strip()
        if not task:
            return overrule("no task given")
        return Decision(action="send_back", source="llm", reason=reason, task=task[:600])

    if action == "ask_rules":
        asked, questions = _asked(case), []
        for q in proposal.get("questions") or []:
            q = str(q).strip()[:MAX_QUESTION_CHARS]
            if q and q.lower() not in asked and q.lower() not in {x.lower() for x in questions}:
                questions.append(q)
        questions = questions[:max(left["rule_questions_left"], 0)]
        if not questions:
            return overrule("no new question within the budget")
        return Decision(action="ask_rules", source="llm", reason=reason, questions=questions)

    # finish
    if review["gaps"] and left["send_backs_left"] > 0:
        return overrule("required evidence is still missing: " + "; ".join(g["question"] for g in review["gaps"]))
    return Decision(action="finish", source="llm", reason=reason)


def decide(case: CaseFile, review: dict[str, Any]) -> Decision:
    if budgets(case)["steps_left"] <= 0:
        return Decision(action="finish", source="code", reason="Step budget used up; the case goes to the officer as is.")
    if not settings.llm_enabled:
        return code_decision(case, review)
    try:
        proposal = llm_proposal(case, review)
    except Exception as error:  # no reply, bad JSON, API error: the case must still move
        return Decision(**code_decision(case, review), overruled=f"LLM unavailable ({type(error).__name__}: {str(error)[:120]})")
    return enforce(proposal, case, review)
