"""Tests for the fixed verdict rules. Satellite results come from the saved cache; no LLM or Earth Engine."""

from __future__ import annotations

import copy

import pytest

from eudr_agent.geo import tools
from eudr_agent.guardrail import check_input
from eudr_agent.plots import load_plots
from eudr_agent.supervisor.review import review_evidence
from eudr_agent.supervisor.verdict_rules import decide

SATELLITE = ("protected_area_check", "forest_2020", "loss_after_2020", "locate_loss", "boundary_sensitivity",
             "ndvi_series")


def case_for(plot_id, findings=None, read_records=False, **plot_changes):
    plot = {**load_plots()[plot_id], **plot_changes}
    guard = check_input(plot)
    if findings is None:
        findings = [getattr(tools, name)(plot["geometry"]) for name in SATELLITE]
        if read_records:
            findings += [tools.read_document(p) for p in guard["plot"].get("documents", [])]
    return {"guardrail": guard, "findings": findings, "review": review_evidence(guard["plot"], findings)}


def finding(case, tool):
    return next(f for f in case["findings"] if f["tool"] == tool)


@pytest.mark.parametrize("plot_id, verdict, rule", [
    ("N-01", "high_risk", 1),       # Hatgat Reserve Forest
    ("C-01", "clear", 4),           # no loss; 0.08 ha TMF degradation does not count for coffee
    ("C-02", "clear", 6),           # one 0.42 ha patch, stable, coffee since 2010
    ("K-01", "needs_evidence", 7),  # most loss vanishes when the boundary moves in
])
def test_test_plots(plot_id, verdict, rule):
    result = decide(case_for(plot_id, read_records=True))
    assert (result["verdict"], result["rule"]) == (verdict, rule)
    assert result["reasons"]


def test_protected_area_decides_alone_and_names_it():
    result = decide(case_for("N-01", findings=[tools.protected_area_check(load_plots()["N-01"]["geometry"])]))
    assert result["verdict"] == "high_risk" and "Hatgat Reserve Forest" in result["reasons"][0]
    assert "Karnataka Forest Act s.24" in result["basis"]


def test_missing_evidence_is_never_clear():
    case = case_for("C-01")
    case["findings"] = [f for f in case["findings"] if f["tool"] != "loss_after_2020"]
    result = decide(case)
    assert (result["verdict"], result["rule"]) == ("needs_evidence", 2)
    failed = case_for("C-01")
    failed["findings"].append({"tool": "forest_2020", "dataset": "x", "error": "quota"})  # a retry failed: earlier result still holds
    assert decide(failed)["verdict"] == "clear"


def test_degradation_alone_does_not_count_for_coffee():
    result = decide(case_for("C-01"))
    assert result["rule"] == 4 and any("wood only" in r for r in result["reasons"])


def test_no_forest_in_2020_is_clear_even_with_loss():
    case = case_for("K-01")
    finding(case, "forest_2020")["values"] |= {"gfc2020_forest_share": 0.02, "hansen_tree_cover_share": 0.03}
    assert decide(case)["rule"] == 3


def test_cleared_block_in_mapped_forest_is_high_risk():
    case = case_for("C-02")
    finding(case, "locate_loss")["values"] |= {"largest_patch_ha": 1.2, "loss_in_gfc2020_forest_ha": 1.1,
                                               "pattern": "block"}
    result = decide(case)
    assert (result["verdict"], result["rule"]) == ("high_risk", 5)


def test_small_loss_needs_a_declared_crop_year_from_before_the_cut_off():
    assert decide(case_for("C-02", crop_since=2022))["verdict"] == "needs_evidence"


def test_caveat_when_the_crop_year_rests_on_the_declaration():
    caveats = decide(case_for("C-02"))["caveats"]
    assert any("no land record was read" in c for c in caveats)
    assert not any("no land record" in c for c in decide(case_for("C-01", read_records=True))["caveats"])


def test_verdict_does_not_change_the_case():
    case = case_for("K-01")
    before = copy.deepcopy(case)
    decide(case)
    assert case == before


def test_missing_loss_follow_ups_block_a_verdict_but_an_unread_record_does_not():
    case = case_for("K-01")
    case["findings"] = [f for f in case["findings"] if f["tool"] != "boundary_sensitivity"]
    assert decide(case)["rule"] == 2
    unread = decide(case_for("C-01"))  # C-01's land record not read: cannot change "no loss"
    assert unread["verdict"] == "clear" and any("land record" in c for c in unread["caveats"])
