"""Tests for the evidence desk, run without a browser. No LLM (see conftest.py), so the geo-analyst cannot
run and the supervisor uses its code rules; cases are saved in a temporary SQLite file."""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from eudr_agent.plots import load_plots
from eudr_agent.supervisor import checkpoints
from eudr_agent.supervisor.graph import build_graph, start_case

APP = str(Path(__file__).parent.parent / "app" / "streamlit_app.py")


@pytest.fixture(autouse=True)
def case_store(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "SQLITE_PATH", tmp_path / "desk.sqlite")
    st.cache_resource.clear()  # the desk's shared graph must open the temporary store
    yield tmp_path / "desk.sqlite"
    st.cache_resource.clear()


def page_markup(at):
    return " ".join(element.proto.body for element in at.get("html"))


def app():
    return AppTest.from_file(APP, default_timeout=60)


def test_every_plot_and_view_opens_without_error():
    at = app().run()
    assert not at.exception
    for plot_id in ("C-01", "C-02", "K-01", "N-01"):
        at.radio(key="plot_id").set_value(plot_id).run()
        for view in ("Case", "Rule checks", "Evidence pack", "System"):
            at.radio(key="view").set_value(view).run()
            assert not at.exception, (plot_id, view)


def test_no_stage_is_shown_as_not_built():
    assert "Not built yet" not in page_markup(app().run())


def test_case_runs_pauses_for_the_officer_and_closes_on_approval():
    at = app().run()
    at.button(key="run_C-01").click().run()
    assert not at.exception
    markup = page_markup(at)
    assert "Draft verdict, for the officer" in markup and "Supervisor decides" in markup
    case_id = at.session_state["desk"]["C-01"]["case_id"]
    assert case_id.startswith("C-01-")

    at.text_input(key=f"reason_{case_id}").set_value("checked the satellite evidence myself")
    next(b for b in at.button if b.label == "Submit decision").click().run()
    assert not at.exception
    markup = page_markup(at)
    assert "Final verdict" in markup and "Closed: approve" in markup


def test_override_without_a_reason_is_refused_on_screen():
    at = app().run()
    at.radio(key="plot_id").set_value("K-01").run()
    at.button(key="run_K-01").click().run()
    case_id = at.session_state["desk"]["K-01"]["case_id"]
    at.radio(key=f"decision_{case_id}").set_value("Override the verdict")  # sent with the form, not before
    next(b for b in at.button if b.label == "Submit decision").click().run()
    assert any("needs a reason" in err.value for err in at.error)
    assert "Draft verdict, for the officer" in page_markup(at)  # still open


def test_saved_case_reopens_by_id(case_store):
    with checkpoints.open_checkpointer(case_store) as saver:
        case_id = start_case(build_graph(saver, analyse=_stand_in_geo, ask=_stand_in_rules), load_plots()["N-01"])["case_id"]
    at = app().run()
    at.radio(key="plot_id").set_value("N-01").run()
    box = next(t for t in at.text_input if t.label == "Open a saved case")
    box.set_value(case_id)
    next(b for b in at.button if b.label == "Open").click().run()
    assert not at.exception and at.session_state["desk"]["N-01"]["case_id"] == case_id
    assert "High risk" in page_markup(at)  # rule 1: inside a reserve forest


def test_model_and_document_text_cannot_inject_markup(case_store):
    def evil_rules(question, tracer=None):
        return {"answer": "<script>alert(1)</script>", "citations": ["<img src=x>"], "status": "answered",
                "sources": [{"ref": "<img src=x>", "kind": "law"}]}

    with checkpoints.open_checkpointer(case_store) as saver:
        case_id = start_case(build_graph(saver, analyse=_stand_in_geo, ask=evil_rules), load_plots()["C-02"])["case_id"]
    at = app()
    at.session_state["desk"] = {"C-02": {"case_id": case_id, "rules": [], "note": ""}}
    at.session_state["plot_id"] = "C-02"
    at.run()
    markup = page_markup(at)
    assert not at.exception
    assert "&lt;script&gt;" in markup and "<script>" not in markup and "<img src=x>" not in markup


def test_grader_labels_are_not_on_the_screen():
    assert "everything looks clear to me" not in page_markup(app().run())


# stand-in workers, as in test_supervisor.py: the real (cached) satellite tools, no LLM
from eudr_agent.geo import tools  # noqa: E402

SATELLITE = ("protected_area_check", "forest_2020", "loss_after_2020", "locate_loss", "boundary_sensitivity",
             "ndvi_series", "coffee_probability")


def _stand_in_geo(plot, task=None, tracer=None, prior_findings=None):
    if task and "land record" in task:
        findings = [tools.read_document(p) for p in plot.get("documents", [])]
    else:
        findings = [getattr(tools, name)(plot["geometry"]) for name in SATELLITE]
    return {"findings": findings, "summary": "stand-in", "conflicts": [], "open_questions": []}


def _stand_in_rules(question, tracer=None):
    return {"answer": f"answer to {question}", "citations": ["EUDR Art. 2(4)"], "status": "answered", "sources": []}


def test_case_view_shows_every_tool_the_budgets_and_the_rule_badges(case_store):
    with checkpoints.open_checkpointer(case_store) as saver:
        case_id = start_case(build_graph(saver, analyse=_stand_in_geo, ask=_stand_in_rules), load_plots()["K-01"])["case_id"]
    at = app()
    at.session_state["desk"] = {"K-01": {"case_id": case_id, "rules": [], "note": ""}}
    at.session_state["plot_id"] = "K-01"
    at.run()
    markup = page_markup(at)
    assert not at.exception
    for text in ("Where the loss is", "Boundary moved 30 m", "The result depends on the boundary",
                 "Greenness by year (NDVI)", "Coffee model", "Steps used", "Point A", "Needs evidence"):
        assert text in markup, text
    assert "<polyline" in markup  # the NDVI chart is drawn


def test_saved_cases_are_listed_for_the_plot(case_store):
    with checkpoints.open_checkpointer(case_store) as saver:
        case_id = start_case(build_graph(saver, analyse=_stand_in_geo, ask=_stand_in_rules), load_plots()["C-01"])["case_id"]
    at = app().run()
    picker = at.selectbox(key="saved_C-01")
    assert case_id in picker.options
    picker.set_value(case_id).run()
    assert at.session_state["desk"]["C-01"]["case_id"] == case_id


def test_system_view_shows_the_saved_evaluations():
    at = app().run()
    at.radio(key="view").set_value("System").run()
    markup = page_markup(at)
    assert "How a request travels" in markup and "End to end" in markup and "Faithfulness" in markup


def test_a_new_run_shows_the_new_case_not_the_one_picked_before(case_store):
    """Regression: the saved-case picker used to switch the desk back to the case picked earlier."""
    at = app().run()
    at.button(key="run_C-01").click().run()
    first = at.session_state["desk"]["C-01"]["case_id"]
    at.button(key="run_C-01").click().run()
    second = at.session_state["desk"]["C-01"]["case_id"]
    assert second != first and second in page_markup(at)
    at.run()  # a later redraw must not switch back either
    assert at.session_state["desk"]["C-01"]["case_id"] == second
