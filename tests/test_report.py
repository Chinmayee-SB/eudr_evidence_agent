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
    assert "No verdict issued" in text and "was not run" in text and "No rule checks" in text


def test_pack_reports_tool_numbers_reasons_and_awkward_text():
    text = pdf_text(report.build_evidence_pack(load_plots()["K-01"], GEO, officer_note="Check on site"))
    assert "0.22 ha (2025: 0.22 ha)" in text and "Look for loss" in text
    assert "<loss> & TMF" in text and "Check on site" in text and "run.jsonl" in text


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
