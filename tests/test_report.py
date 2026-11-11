"""Tests for the evidence pack. It must be buildable at any stage and must never imply a verdict."""

from __future__ import annotations

import io

from pypdf import PdfReader
from reportlab.pdfgen import canvas

from eudr_agent import report
from eudr_agent.plots import load_plots

LOSS = {"tool": "loss_after_2020", "dataset": "Hansen", "cached": True, "note": "Any canopy loss.", "values": {
    "plot_ha": 9.0, "hansen_loss_ha": 0.22, "hansen_loss_by_year": {"2025": 0.22},
    "tmf_deforestation_ha": 0, "tmf_deforestation_by_year": {}, "tmf_degradation_ha": 0, "tmf_degradation_by_year": {}}}
GEO = {"findings": [LOSS], "summary": "Hansen shows 0.22 ha <loss> & TMF none.", "conflicts": ["Hansen vs TMF"],
       "open_questions": [], "events": [{"node": "geo_tool", "tool": "loss_after_2020", "reason": "Look for loss"}],
       "trace": "run.jsonl"}


def pdf_text(data):
    return " ".join(" ".join(page.extract_text() for page in PdfReader(io.BytesIO(data)).pages).split())


def test_pack_with_nothing_run_says_so():
    text = pdf_text(report.build_evidence_pack(load_plots()["N-01"]))
    assert "No verdict issued" in text and "was not run" in text and "No legal questions" in text


def test_pack_reports_tool_numbers_reasons_and_awkward_text():
    text = pdf_text(report.build_evidence_pack(load_plots()["K-01"], GEO, officer_note="Check on site"))
    assert "0.22 ha (2025: 0.22 ha)" in text and "Look for loss" in text
    assert "<loss> & the EU tropical forest monitor" in text and "Check on site" in text and "run.jsonl" in text


def test_pack_records_a_failed_run():
    text = pdf_text(report.build_evidence_pack(load_plots()["K-01"], {"error": "quota exceeded", "findings": [], "events": []}))
    assert "failed: quota exceeded" in text


def test_pack_includes_documents_and_adds_pdf_pages(tmp_path):
    note, pdf = tmp_path / "note.md", tmp_path / "scan.pdf"
    note.write_text("Recorded since 1998 – “coffee”", encoding="utf-8")
    page = canvas.Canvas(str(pdf))
    page.drawString(72, 720, "Original upload page")
    page.save()
    plot = {**load_plots()["C-02"], "documents": [str(note), str(pdf)]}
    text = pdf_text(report.build_evidence_pack(plot))
    assert "Recorded since 1998" in text and "Original upload page" in text


def test_non_western_text_does_not_break_the_pack():
    plot = load_plots()["C-02"]
    assert report.build_evidence_pack(plot, officer_note="ಕಾಫಿ ≥ 10") .startswith(b"%PDF")


def test_rule_citations_show_how_binding_they_are():
    from eudr_agent.report import _legal

    check = {"question": "Is shade coffee forest?", "answer": "No [EUDR Art. 2(6)].", "citations": ["EUDR Art. 2(6)"],
             "sources": [{"ref": "EUDR Art. 2(6)", "kind": "law", "label": "EU law (binding)"},
                         {"ref": "EC Guidance §11.4(d)", "kind": "guidance",
                          "label": "Commission guidance (not legally binding)"}]}
    text = " ".join(getattr(item, "text", "") for item in _legal([check]))
    assert "EUDR Art. 2(6) (EU law (binding))" in text and "not legally binding" in text


CASE = {"case_id": "K-01-20261110-000000-test", "status": "closed", "draft_verdict": "needs_evidence",
        "final_verdict": "clear", "reruns": 0,
        "verdict_detail": {"rule": 7, "reasons": ["The two satellite sources disagree."], "basis": ["EUDR Art. 2(28)"],
                           "caveats": ["Coffee since 1998 rests on the exporter's declaration."]},
        "officer": {"decision": "override", "verdict": "clear", "reason": "Field visit showed pruning"},
        "guardrail": {"passed": True, "blocked": [], "dropped": [], "warnings": []},
        "rules": [{"question": "Is shade coffee forest?", "answer": "No [EUDR Art. 2(6)].", "point": "A",
                   "status": "answered", "citations": ["EUDR Art. 2(6)"]}],
        "findings": [LOSS], "open_items": []}
EVENTS = [{"node": "input_guardrail", "passed": True},
          {"node": "geo_tool", "tool": "loss_after_2020", "reason": "Look for loss", "finding": LOSS},
          {"node": "supervisor", "decision": {"action": "ask_rules", "reason": "A legal point is open.",
                                              "questions": ["Is shade coffee forest?"]}},
          {"node": "retrieve"}, {"node": "answer"},
          {"node": "verdict", "verdict": "needs_evidence", "detail": {"rule": 7}},
          {"node": "officer", "decision": {"decision": "override", "reason": "Field visit showed pruning"}},
          {"node": "pack"}]


def test_pack_reads_as_a_document_in_order():
    geo = {**GEO, "events": EVENTS}
    text = pdf_text(report.build_evidence_pack(load_plots()["K-01"], geo, CASE["rules"], "Seen on site", case=CASE))
    sections = ["In short", "1. What was submitted", "2. What happened, step by step", "3. Evidence in detail",
                "4. Legal questions and answers", "5. Verdict and why", "6. What is still needed",
                "7. Officer decision", "Appendix C. Audit trail"]
    positions = [text.find(s) for s in sections]
    assert all(p >= 0 for p in positions) and positions == sorted(positions), positions
    assert "Final verdict Clear" in text and "Overrode the draft. Reason: Field visit showed pruning" in text
    assert "1. Input check: passed." in text and "Why: Look for loss Result: Tree loss after 2020" in text
    assert '"Is shade coffee forest?" - answered with citations' in text
    assert "Draft verdict: Needs evidence (the evidence is not conclusive)" in text
    assert "field visit report" in text and "land record (for example an RTC extract)" in text  # what to supply
    for internal in ("rule 7", "supervisor", "geo-analyst", "ask_rules", "loss_after_2020", "UMD/hansen"):
        assert internal not in text, internal


def test_pack_for_a_stopped_plot_says_why_and_nothing_else_ran():
    case = {**CASE, "status": "blocked", "verdict_detail": None, "officer": None, "rules": [], "findings": [],
            "guardrail": {"passed": False, "blocked": ["the plot is outside Karnataka"], "dropped": [], "warnings": []}}
    text = pdf_text(report.build_evidence_pack(load_plots()["W-01"], None, [], case=case))
    assert "Not screened" in text and "Input check: stopped. The plot is outside Karnataka. Nothing else was run" in text


def test_an_overridden_pack_does_not_repeat_the_drafts_reasoning_as_the_verdict():
    text = pdf_text(report.build_evidence_pack(load_plots()["K-01"], {**GEO, "events": EVENTS}, CASE["rules"], case=CASE))
    cover = text[:text.find("In short")]
    assert "Final verdict, set by the officer" in cover and "The draft was Needs evidence" in cover
    assert "a person needs to look" not in cover
    assert "Hansen loss is" not in text and "TMF degradation" not in text  # limits in plain words


def test_a_loss_patch_in_one_year_shows_that_year_once():
    from eudr_agent.report import key_figures

    v = {"pattern": "block", "patch_count": 2, "largest_patch_ha": 1.1, "loss_at_edge_ha": 0.2, "loss_ha": 1.3,
         "loss_in_gfc2020_forest_ha": 1.3, "patches": [
             {"area_ha": 1.1, "first_year": 2021, "last_year": 2024, "distance_to_edge_m": 4},
             {"area_ha": 0.2, "first_year": 2021, "last_year": 2021, "distance_to_edge_m": 0}]}
    rows = dict(key_figures({"tool": "locate_loss", "values": v}))
    assert rows["Patch 1"] == "1.1 ha, 2021-2024, 4 m from edge" and rows["Patch 2"] == "0.2 ha, 2021, 0 m from edge"
