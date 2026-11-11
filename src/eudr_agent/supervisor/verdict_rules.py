"""The fixed verdict table: the draft verdict the officer approves or overrides. Code only, no LLM.

Rules, checked in order; the first that applies decides:
  1. plot inside a protected area in force                                    -> high_risk
  2. evidence the verdict depends on is missing (the three core checks; where there is loss, also where it
     lies and whether it survives a boundary shift)                         -> needs_evidence
  3. no forest at the end of 2020 on both maps                                -> clear
  4. no deforestation-type loss after 2020 (Hansen loss and TMF deforestation) -> clear
  5. forest on both maps in 2020, both loss datasets show loss, and a block of at least 0.5 ha cleared in
     that forest                                                              -> high_risk
  6. small loss only (every patch under 0.5 ha), not a block, stable when the boundary moves 30 m,
     coffee declared from before 2020                                        -> clear
  7. the result changes when the boundary moves, or the loss datasets disagree  -> needs_evidence
  otherwise                                                                    -> needs_evidence

Why these lines:
  * EUDR Art. 2(13): coffee is deforestation-free if its land was not deforested after 31 Dec 2020. The
    forest-degradation condition is for wood only (2(13)(b)), so TMF "degradation" is not a test for coffee.
  * Art. 2(3): deforestation is conversion of forest to agricultural use, so land that was not forest at the
    cut-off (rule 3), or that has not changed since (rule 4), cannot have been deforested.
  * Art. 2(4) and 2(6): forest needs more than 0.5 ha, and coffee under trees is an agricultural plantation,
    not forest. Patches under 0.5 ha inside a coffee plot match pruning or shade-tree work (rule 6); a block of
    0.5 ha or more inside mapped forest matches clearing (rule 5).
  * Art. 3(b), 2(40): production must be legal under the country's law; in Karnataka a reserved forest bars
    clearing and cultivation (Karnataka Forest Act s.24), so rule 1 decides alone.
  * Rule 5 needs both loss datasets to agree (Hansen loss and TMF deforestation). A block seen by only one of
    them goes to rule 7: TMF can read heavy pruning in shade coffee as clearing (plot K-03), and a wrong
    "high risk" drops a supplier, so the officer checks it instead.
  * Rule 2: a gap the verdict depends on must never turn into "clear". Other gaps (e.g. an unread land record
    on a plot with no loss) cannot change the result, so they are listed as caveats instead.

Each verdict comes with the rule that decided, the reasons in the plot's own numbers, the legal basis, and
caveats (e.g. "rests on the declared crop year; no land record was read"). The officer can override any of it.
"""

from __future__ import annotations

from typing import Any, TypedDict

from eudr_agent.state import CaseFile, Verdict
from eudr_agent.wording import CHECK, boundary_change
from eudr_agent.supervisor.review import (CUT_OFF_YEAR, FOREST_ON_MAP, has_loss, in_protected_area, latest,
                                          not_forest_in_2020, values)

MIN_FOREST_HA = 0.5   # EUDR Art. 2(4): the smallest area that can be forest
CORE_CHECKS = ("protected_area_check", "forest_2020", "loss_after_2020")
LOSS_CHECKS = ("locate_loss", "boundary_sensitivity")  # needed once there is loss


class VerdictResult(TypedDict):
    verdict: Verdict
    rule: int            # 0 = no rule applied (the default)
    title: str
    reasons: list[str]
    basis: list[str]     # the legal passages the rule rests on (all in the rules agent's index)
    caveats: list[str]


def _result(verdict: Verdict, rule: int, title: str, reasons: list[str], basis: list[str],
            caveats: list[str]) -> VerdictResult:
    return VerdictResult(verdict=verdict, rule=rule, title=title, reasons=reasons, basis=basis, caveats=caveats)


def _caveats(case: CaseFile) -> list[str]:
    plot = case["guardrail"]["plot"]
    findings = case.get("findings", [])
    notes = [f"Not checked: {g['question']}" for g in case.get("review", {}).get("gaps", [])
             if g["need"] not in {"protected_area", "forest_2020", "loss_after_2020", "loss_location", "boundary_shift"}]
    read = [f for f in findings if f.get("tool") == "read_document" and not f.get("error")]
    if plot.get("crop_since") and not read:
        notes.append(f"Coffee since {plot['crop_since']} rests on the exporter's declaration; no land record was read.")
    coffee = values(findings, "coffee_probability")
    if coffee:
        notes.append("The global coffee map is not reliable for shade-grown coffee in India, so its score was not used.")
    notes += [w[0].upper() + w[1:] + "." if not w.endswith(".") else w for w in case["guardrail"].get("warnings", [])]
    return notes


def decide(case: CaseFile) -> VerdictResult:
    findings = case.get("findings", [])
    plot = case["guardrail"]["plot"]
    caveats = _caveats(case)

    # 1. protected area
    if in_protected_area(findings):
        areas = values(findings, "protected_area_check").get("in_force", [])
        names = "; ".join(f"{a.get('name', '?')} ({a.get('designation', '?')}, {a.get('overlap_ha', '?')} ha)"
                          for a in areas) or "a protected area"
        return _result("high_risk", 1, "Inside a protected forest area",
                       [f"The plot overlaps {names}, a protected area that is in force."],
                       ["EUDR Art. 3", "EUDR Art. 2(40)", "Karnataka Forest Act s.24"], caveats)

    # 2. missing evidence the verdict depends on
    # loss checks are not needed when both maps show no forest in 2020 (the same skip as review.py)
    core = [c for c in CORE_CHECKS if not (c == "loss_after_2020" and latest(findings, "forest_2020")
                                           and not_forest_in_2020(findings))]
    missing = [c for c in core if not latest(findings, c)]
    if not missing and has_loss(findings) and not not_forest_in_2020(findings):
        missing = [c for c in LOSS_CHECKS if not latest(findings, c)]
    if missing:
        return _result("needs_evidence", 2, "A required check could not be completed",
                       [f"The \"{CHECK.get(c, c)}\" check did not return a result, and the verdict depends on it." for c in missing], [], caveats)

    forest = values(findings, "forest_2020")
    loss = values(findings, "loss_after_2020")
    where = values(findings, "locate_loss")
    shift = values(findings, "boundary_sensitivity")
    gfc, hansen = forest["gfc2020_forest_share"], forest["hansen_tree_cover_share"]
    hansen_ha, tmf_ha = loss.get("hansen_loss_ha", 0), loss.get("tmf_deforestation_ha", 0)

    def unsettled(why_not: str) -> list[str]:
        """A clear verdict is not changed by an unstable boundary check, but the officer should see it."""
        if not shift or shift.get("stable", True):
            return caveats
        what = " ".join(boundary_change(c) for c in shift.get("changes", []))
        return caveats + [f"The boundary check is not stable: {what[:1].lower() + what[1:]} This does not change "
                          f"the verdict, because {why_not}"]

    # 3. not forest at the cut-off
    if not_forest_in_2020(findings):
        return _result("clear", 3, "Not forest in 2020",
                       [f"Both forest maps show almost no forest on the plot in 2020 (EU forest map {gfc:.0%}, Global Forest Watch "
                        f"tree cover {hansen:.0%})."],
                       ["EUDR Art. 2(3)", "EUDR Art. 2(13)"], unsettled("the land was not forest in 2020."))

    # 4. no deforestation-type loss
    if hansen_ha == 0 and tmf_ha == 0:
        reasons = ["Neither satellite source shows forest loss on the plot after 2020 (Global Forest Watch and the "
                   "EU tropical forest monitor)."]
        if loss.get("tmf_degradation_ha"):
            reasons.append(f"The EU tropical forest monitor shows {loss['tmf_degradation_ha']} ha of thinning (degradation). This does not count "
                           "against coffee: the EUDR degradation rule applies to wood only.")
        return _result("clear", 4, "No forest lost since 2020", reasons, ["EUDR Art. 2(13)", "EUDR Art. 2(3)"],
                       unsettled("neither source shows loss inside the plot as drawn, and any loss found with the "
                                 "boundary moved out lies outside the declared plot."))

    # loss exists from here on; the location and boundary checks are required evidence by now (review.py)
    largest = where.get("largest_patch_ha", 0)
    in_forest = where.get("loss_in_gfc2020_forest_ha", 0)
    stable = shift.get("stable", False)
    loss_text = (f"{max(hansen_ha, tmf_ha)} ha of tree loss after 2020 (Global Forest Watch {hansen_ha} ha, "
                 f"EU tropical forest monitor {tmf_ha} ha)")

    # 5. a cleared block inside agreed forest
    both_see_loss = hansen_ha > 0 and tmf_ha > 0
    block = largest >= MIN_FOREST_HA and in_forest >= MIN_FOREST_HA
    if gfc >= FOREST_ON_MAP and hansen >= FOREST_ON_MAP and block and both_see_loss:
        return _result("high_risk", 5, "Forest cleared after 2020",
                       [f"Both forest maps show forest on the plot in 2020 (EU forest map {gfc:.0%}, Global Forest Watch "
                        f"tree cover {hansen:.0%}).",
                        f"{loss_text}; the largest patch is {largest} ha, at or above the 0.5 ha forest minimum, "
                        f"with {in_forest} ha inside 2020 forest."],
                       ["EUDR Art. 2(3)", "EUDR Art. 2(4)", "EUDR Art. 2(13)"], caveats)

    # 6. small, stable loss on land declared as coffee before the cut-off
    since = plot.get("crop_since")
    if largest < MIN_FOREST_HA and where.get("pattern") != "block" and stable and since and since <= CUT_OFF_YEAR:
        return _result("clear", 6, "Only small tree loss on an older coffee farm",
                       [f"{loss_text}, in {where.get('patch_count', 0)} patch(es); the largest is {largest} ha, "
                        "below the 0.5 ha forest minimum.",
                        "The result holds when the boundary moves 30 m in or out.",
                        f"Declared as coffee since {since}: an agricultural plantation, which is excluded from forest."],
                       ["EUDR Art. 2(4)", "EUDR Art. 2(6)", "EUDR Art. 2(3)"], caveats)

    # 7. the result depends on the boundary, or the datasets disagree
    reasons = []
    if not stable:
        reasons += [boundary_change(c) for c in shift.get("changes", [])] \
            or ["The result changes when the boundary moves 30 m."]
    if (hansen_ha > 0) != (tmf_ha > 0):
        reasons.append(f"The two satellite sources disagree: Global Forest Watch shows {hansen_ha} ha of loss, the EU "
                       f"tropical forest monitor {tmf_ha} ha.")
        if block:
            reasons.append(f"Only one source shows the {largest} ha block inside 2020 forest; in shade coffee that "
                           "can be heavy pruning. Check recent imagery or a field visit before calling it clearing.")
    if reasons:
        return _result("needs_evidence", 7, "The evidence is not conclusive", reasons, ["EUDR Art. 2(28)"], caveats)

    return _result("needs_evidence", 0, "No standard check settles it", [f"{loss_text}; none of the standard checks settles it."], [], caveats)
