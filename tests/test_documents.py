"""Tests for mock document uploads and the geo-analyst's access to them."""

from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from eudr_agent import documents
from eudr_agent.geo import agent, tools
from eudr_agent.plots import load_plots


def pdf_with_text(text):
    buffer = io.BytesIO()
    page = canvas.Canvas(buffer)
    page.drawString(72, 720, text)
    page.save()
    return buffer.getvalue()


@pytest.fixture
def uploads(monkeypatch, tmp_path):
    folder = tmp_path / "documents" / "uploads"
    monkeypatch.setattr(documents, "UPLOADS_DIR", folder)
    monkeypatch.setattr(tools, "DOCUMENTS_DIR", tmp_path / "documents")
    return folder


def test_upload_is_stored_for_its_plot_only(uploads):
    documents.save_upload("C-02", "record.md", b"Land use: coffee")
    assert [p.name for p in documents.uploads_for("C-02")] == ["uploaded_record.md"]
    assert documents.uploads_for("K-01") == []


def test_folder_in_the_name_cannot_escape(uploads):
    path = documents.save_upload("C-02", "../../../evil.md", b"text")
    assert path.parent == uploads / "C-02" and path.name == "uploaded_evil.md"


@pytest.mark.parametrize("name, data, message", [
    ("run.sh", b"text", "accepted"),
    ("empty.md", b"  ", "empty"),
    ("binary.txt", b"\xff\xfe\x00", "UTF-8"),
    ("broken.pdf", b"not a pdf", "could not be opened"),
])
def test_bad_files_are_refused_and_not_saved(uploads, name, data, message):
    with pytest.raises(ValueError, match=message):
        documents.save_upload("C-02", name, data)
    assert documents.uploads_for("C-02") == []


def test_pdf_without_text_is_refused(uploads):
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    blank = io.BytesIO()
    writer.write(blank)
    with pytest.raises(ValueError, match="no readable text"):
        documents.save_upload("C-02", "scan.pdf", blank.getvalue())


def test_agent_can_read_an_uploaded_pdf(uploads):
    documents.save_upload("C-02", "record.pdf", pdf_with_text("Land use: coffee since 2010"))
    plot = documents.with_uploads(load_plots()["C-02"])
    finding = agent._run_tool(plot, "read_document", {"document": "uploaded_record.pdf"})
    assert "coffee since 2010" in finding["values"]["text"] and finding["values"]["mocked"] is True


def test_upload_does_not_replace_a_sample_record(uploads):
    documents.save_upload("K-01", "K-01_land_record.md", b"Other text")
    names = [p.rsplit("/", 1)[-1] for p in documents.with_uploads(load_plots()["K-01"])["documents"]]
    assert names == ["K-01_land_record.md", "uploaded_K-01_land_record.md"]


def test_long_document_is_cut_and_marked(uploads):
    path = documents.save_upload("C-02", "long.txt", b"x" * (tools.MAX_DOCUMENT_CHARS + 1))
    values = tools.read_document(str(path))["values"]
    assert len(values["text"]) == tools.MAX_DOCUMENT_CHARS and values["truncated"] is True


def test_remove_touches_only_uploads(uploads):
    documents.save_upload("C-02", "record.md", b"text")
    documents.remove_upload("C-02", "uploaded_record.md")
    documents.remove_upload("K-01", "K-01_land_record.md")
    assert documents.uploads_for("C-02") == []
    assert (tools.PLOTS_DIR / "documents" / "K-01_land_record.md").exists()
