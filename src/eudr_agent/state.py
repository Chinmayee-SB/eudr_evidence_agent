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


class CaseFile(TypedDict, total=False):
    plot: PlotInput
    findings: list[Finding]      # geo-analyst results
    rules: list[dict[str, Any]]  # rules-agent answers: question, answer, citations
    conflicts: list[str]         # where datasets disagree
    step: int                    # supervisor steps used (max 10)
    draft_verdict: Verdict
    missing_documents: list[str]
    reviewer_gaps: list[str]
    officer_decision: Literal["approved", "overridden", "documents_added"]
