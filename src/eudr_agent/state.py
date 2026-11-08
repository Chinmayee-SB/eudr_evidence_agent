"""The plot's case file: the shared state the supervisor graph passes between agents.

Defined now so all three parts agree on one shape; filled in as each part is built.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Verdict = Literal["clear", "needs_evidence", "high_risk"]


class PlotInput(TypedDict, total=False):
    plot_id: str                 # anonymised, e.g. "K-06"
    geometry: dict[str, Any]     # GeoJSON Polygon or Point
    area_ha: float
    declared_crop: str
    crop_since: int              # year
    documents: list[str]         # paths to mocked grower documents


class Finding(TypedDict, total=False):
    tool: str                    # e.g. "loss_after_2020"
    dataset: str                 # e.g. "Hansen GFC v1.13"
    values: dict[str, Any]       # numbers exactly as the tool returned them
    note: str
    cached: bool                 # True if read from the per-plot cache instead of a live query
    error: str                   # set when the tool failed; the agent must report it, not guess


class Decision(TypedDict, total=False):
    action: Literal["send_back", "ask_rules", "finish"]
    reason: str
    task: str                    # send_back: what the geo-analyst should look at
    questions: list[str]         # ask_rules: what the rules agent should answer
    points: list[str]            # ask_rules: the legal point (A-D, see supervisor/decide.py) of each question
    evidence_seen: int           # ask_rules: findings on the case when the round was asked
    source: Literal["llm", "code"]
    overruled: str               # set when code replaced the LLM's proposal, saying why


class OfficerDecision(TypedDict, total=False):
    decision: Literal["approve", "override", "documents_added"]
    verdict: Verdict             # override only: the officer's verdict
    reason: str                  # required for an override


class CaseFile(TypedDict, total=False):
    """The supervisor graph's state. Everything here is plain JSON, so a checkpoint can save and restore it."""
    case_id: str                 # also the checkpoint thread and the trace run id
    plot: PlotInput              # as submitted
    guardrail: dict[str, Any]    # input check: passed, blocked, dropped, warnings, and the plot agents may see
    findings: list[Finding]      # every geo-analyst finding, across rounds
    geo_reports: list[dict[str, Any]]  # each round's summary, conflicts and open questions
    geo_task: str                # the supervisor's follow-up task for the next geo round
    review: dict[str, Any]       # latest code review: gaps, conflicts, not_checkable
    decisions: list[Decision]    # every supervisor decision, with its reason
    pending_questions: list[str]
    rules: list[dict[str, Any]]  # rules-agent answers: question, point, answer, citations, sources, status
    steps: int                   # agent calls and supervisor decisions so far (budget: MAX_STEPS)
    send_backs: int
    reruns: int                  # officer "documents added" re-runs
    draft_verdict: Verdict | None
    verdict_note: str
    officer: OfficerDecision
    final_verdict: Verdict | None    # the officer's: the draft if approved, theirs if overridden
    open_items: list[str]        # what the evidence file must say is missing or not checked
    status: Literal["running", "blocked", "awaiting_officer", "closed"]
