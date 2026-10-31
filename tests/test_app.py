"""Tests for the evidence desk, run without a browser. The agents are never called."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app" / "streamlit_app.py")


def page_markup(at):
    return " ".join(element.proto.body for element in at.get("html"))


def test_every_plot_and_view_opens_without_error():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    for plot_id in ("C-01", "C-02", "K-01", "N-01"):
        at.radio(key="plot_id").set_value(plot_id).run()
        for view in ("Investigate", "Rule checks", "Evidence pack"):
            at.radio(key="view").set_value(view).run()
            assert not at.exception, (plot_id, view)


def test_unbuilt_stages_are_shown_as_not_built():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert page_markup(at).count("Not built yet") == 3


def test_model_and_document_text_cannot_inject_markup():
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["cases"] = {"C-01": {"note": "", "rules": [], "geo": {
        "findings": [], "summary": "<script>alert(1)</script>", "conflicts": ["<img src=x>"], "open_questions": [],
        "events": [{"node": "geo_report", "ms": 1, "summary": "<b>x</b>"}], "trace": "t.jsonl", "documents": []}}}
    at.run()
    markup = page_markup(at)
    assert not at.exception
    assert "&lt;script&gt;" in markup and "<script>" not in markup and "<img src=x>" not in markup


def test_grader_labels_are_not_on_the_screen():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert "High risk" not in page_markup(at)
