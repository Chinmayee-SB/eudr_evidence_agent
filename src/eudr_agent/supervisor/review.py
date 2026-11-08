"""The supervisor's code review of the geo-analyst's evidence. Fixed rules on the tool numbers, no LLM.

It answers three things:
  gaps       required questions with no answer yet, or whose tool failed -> grounds for a send-back
  conflicts  places where the datasets point different ways              -> grounds for a rules check
  not_checkable  required questions whose tools are not built yet        -> a stated limitation, not a gap

Evidence is defined as questions, not tools. Each question lists the tools that can answer it, and a question
counts only once one of its tools is offered to the geo-analyst. So when a planned tool (e.g. locate_loss) is
added to the geo-analyst, it becomes required here with no change to this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from eudr_agent.state import Finding, PlotInput

MAPS_DISAGREE = 0.30   # forest share gap between GFC2020 and Hansen that counts as disagreement
FOREST_ON_MAP = 0.50   # GFC2020 share above which the plot "looks like forest" on the EU map
CUT_OFF_YEAR = 2020
NO_FOREST = 0.05      # both maps below this share: the plot was not forest at the cut-off


def latest(findings: list[Finding], tool: str) -> Finding | None:
    """The most recent successful result of a tool, so a send-back's retry replaces an earlier failure."""
    ok = [f for f in findings if f.get("tool") == tool and not f.get("error")]
    return ok[-1] if ok else None


def values(findings: list[Finding], tool: str) -> dict[str, Any]:
    found = latest(findings, tool)
    return found["values"] if found else {}


def in_protected_area(findings: list[Finding]) -> bool:
    return bool(values(findings, "protected_area_check").get("inside_protected_area"))


def not_forest_in_2020(findings: list[Finding]) -> bool:
    """Both forest maps agree the plot had almost no forest at the cut-off, so later loss cannot be deforestation."""
    forest = values(findings, "forest_2020")
    return bool(forest) and forest["gfc2020_forest_share"] < NO_FOREST and forest["hansen_tree_cover_share"] < NO_FOREST


def has_loss(findings: list[Finding]) -> bool:
    loss = values(findings, "loss_after_2020")
    return loss.get("hansen_loss_ha", 0) > 0 or loss.get("tmf_deforestation_ha", 0) > 0


# ---------- conflicts ----------

def find_conflicts(plot: PlotInput, findings: list[Finding]) -> list[dict[str, str]]:
    """Disagreements read straight from the numbers. The geo-analyst reports conflicts too, in words;
    these are the ones code can prove, so they hold even if the LLM misses one."""
    conflicts = []
    forest = values(findings, "forest_2020")
    if forest:
        gfc, hansen = forest["gfc2020_forest_share"], forest["hansen_tree_cover_share"]
        if abs(gfc - hansen) >= MAPS_DISAGREE:
            conflicts.append({"kind": "forest_maps_disagree",
                              "text": f"JRC GFC2020 shows {gfc:.0%} of the plot as forest, Hansen tree cover {hansen:.0%}"})
        since = plot.get("crop_since")
        if gfc >= FOREST_ON_MAP and plot.get("declared_crop") == "coffee" and (since is None or since <= CUT_OFF_YEAR):
            declared = f"coffee since {since}" if since else "coffee (year not stated)"
            conflicts.append({"kind": "forest_vs_declared_crop",
                              "text": f"JRC GFC2020 maps {gfc:.0%} of the plot as forest in 2020, "
                                      f"but it is declared as {declared}"})
    loss = values(findings, "loss_after_2020")
    if loss:
        hansen_ha, tmf_ha = loss.get("hansen_loss_ha", 0), loss.get("tmf_deforestation_ha", 0)
        if (hansen_ha > 0) != (tmf_ha > 0):
            conflicts.append({"kind": "loss_datasets_disagree",
                              "text": f"after 2020, Hansen shows {hansen_ha} ha tree-cover loss "
                                      f"and JRC TMF {tmf_ha} ha deforestation"})
    return conflicts


# ---------- required evidence ----------

@dataclass(frozen=True)
class Need:
    key: str
    question: str
    tools: tuple[str, ...]
    when: Callable[[PlotInput, list[Finding], list[dict[str, str]]], bool] = lambda plot, findings, conflicts: True
    task_hint: str = ""


NEEDS = [
    Need("protected_area", "Is the plot inside a protected area?", ("protected_area_check", "run_whisp")),
    Need("forest_2020", "Was the plot forest at the end of 2020?", ("forest_2020", "run_whisp")),
    Need("loss_after_2020", "Was there tree-cover loss after 2020?", ("loss_after_2020", "run_whisp")),
    Need("loss_location", "Where on the plot is the loss: edge or interior, one block or scattered?",
         ("locate_loss",), when=lambda plot, findings, conflicts: has_loss(findings)),
    Need("boundary_shift", "Does the result hold if the boundary moves by 30 m?",
         ("boundary_sensitivity",), when=lambda plot, findings, conflicts: has_loss(findings)),
    Need("land_record", "Does the land record show the land was farmed before 2020?", ("read_document",),
         when=lambda plot, findings, conflicts: bool(conflicts) and bool(plot.get("documents")),
         task_hint="the datasets disagree, and the land record may show whether the land was already farmed"),
]


LOSS_NEEDS = {"loss_after_2020", "loss_location", "boundary_shift"}


def available_tools() -> set[str]:
    """The tools the geo-analyst is actually offered: the single source of what can be checked."""
    from eudr_agent.geo.agent import TOOL_SCHEMAS  # imported here so this module loads without the agent

    return {schema["function"]["name"] for schema in TOOL_SCHEMAS}


def review_evidence(plot: PlotInput, findings: list[Finding], available: set[str] | None = None) -> dict[str, Any]:
    """Review the evidence so far. `plot` is the plot the agents saw (after the input guardrail)."""
    available = available_tools() if available is None else available
    conflicts = find_conflicts(plot, findings)
    protected = in_protected_area(findings)
    gaps, not_checkable, answered, skipped = [], [], [], []

    for need in NEEDS:
        if protected and need.key != "protected_area":
            skipped.append(f"{need.question} (not needed: the plot is inside a protected area in force)")
            continue
        if need.key in LOSS_NEEDS and not_forest_in_2020(findings):
            skipped.append(f"{need.question} (not needed: both maps show no forest in 2020)")
            continue
        if not need.when(plot, findings, conflicts):
            continue
        usable = [t for t in need.tools if t in available]
        if not usable:
            not_checkable.append(f"{need.question} (tool not built yet: {', '.join(need.tools)})")
            continue
        if any(latest(findings, t) for t in usable):
            answered.append(need.key)
            continue
        errors = [f["error"] for f in findings if f.get("tool") in usable and f.get("error")]
        if need.key == "land_record":
            names = ", ".join(Path(p).name for p in plot.get("documents", []))
            task = f"Read the land record ({names}): {need.task_hint}."
        else:
            task = f"Answer: {need.question} Use {' or '.join(usable)}."
        gaps.append({"need": need.key, "question": need.question,
                     "text": f"{need.question} The tool failed: {errors[-1]}" if errors else f"{need.question} Not checked yet.",
                     "task": task + (" It failed last time, so try it again." if errors else "")})

    return {"gaps": gaps, "conflicts": conflicts, "not_checkable": not_checkable, "answered": answered,
            "skipped": skipped, "protected_in_force": protected}
