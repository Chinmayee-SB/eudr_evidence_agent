"""Tests for the evidence desk, run without a browser. No LLM (see conftest.py), so the geo-analyst cannot
run and the supervisor uses its code rules; cases are saved in a temporary SQLite file."""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from eudr_agent import submissions
from eudr_agent.plots import load_plots
from eudr_agent.supervisor import checkpoints
from eudr_agent.supervisor.graph import build_graph, start_case

APP = str(Path(__file__).parent.parent / "app" / "streamlit_app.py")


@pytest.fixture(autouse=True)
def case_store(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "SQLITE_PATH", tmp_path / "desk.sqlite")
    monkeypatch.setattr(submissions, "SUBMISSIONS_DIR", tmp_path / "submissions")
    st.cache_resource.clear()  # the desk's shared graph must open the temporary store
    yield tmp_path / "desk.sqlite"
    st.cache_resource.clear()


def page_markup(at):
    return " ".join(element.proto.body for element in at.get("html"))


def app(source="Demo plots"):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["source"] = source  # the desk opens on New submission; most tests use the demo plots
    return at


def test_every_plot_and_view_opens_without_error():
    at = app().run()
    assert not at.exception
    for plot_id in ("C-01", "C-02", "K-01", "N-01", "S-01", "T-01", "W-01"):
        at.selectbox(key="plot_id").set_value(plot_id).run()
        for view in ("Case", "Legal questions", "Evidence pack", "How it works"):
            at.radio(key="view").set_value(view).run()
            assert not at.exception, (plot_id, view)


def test_no_stage_is_shown_as_not_built():
    assert "Not built yet" not in page_markup(app().run())


def test_case_runs_pauses_for_the_officer_and_closes_on_approval():
    at = app().run()
    at.button(key="run_C-01").click().run()
    assert not at.exception
    markup = page_markup(at)
    assert "Draft verdict, for the officer" in markup and "Case review" in markup
    case_id = at.session_state["desk"]["C-01"]["case_id"]
    assert case_id.startswith("C-01-")

    at.text_input(key=f"reason_{case_id}").set_value("checked the satellite evidence myself")
    at.button(key=f"submit_{case_id}").click().run()
    assert not at.exception
    assert at.session_state["flash"]["title"].startswith("Signed off: draft approved")
    markup = page_markup(at)
    assert "Final verdict" in markup and "Closed: Approved the draft" in markup


def test_override_without_a_reason_is_refused_on_screen():
    at = app().run()
    at.selectbox(key="plot_id").set_value("K-01").run()
    at.button(key="run_K-01").click().run()
    case_id = at.session_state["desk"]["K-01"]["case_id"]
    at.radio(key=f"decision_{case_id}").set_value("Override the verdict").run()
    at.button(key=f"submit_{case_id}").click().run()
    assert any("needs a reason" in err.value for err in at.error)
    assert "Draft verdict, for the officer" in page_markup(at)  # still open


def test_saved_case_reopens_by_id(case_store):
    with checkpoints.open_checkpointer(case_store) as saver:
        case_id = start_case(build_graph(saver, analyse=_stand_in_geo, ask=_stand_in_rules), load_plots()["N-01"])["case_id"]
    at = app().run()
    at.selectbox(key="plot_id").set_value("N-01").run()
    box = next(t for t in at.text_input if t.label == "Or open by case id")
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
                 "Greenness by year (NDVI)", "Coffee map", "review steps", "Is coffee under trees forest?",
                 "Needs evidence", "The evidence is not conclusive"):
        assert text in markup, text
    # the NDVI chart is drawn, as an image: st.html strips inline <svg>, so a bare <svg> never reaches the screen
    import base64
    chart = markup.split("data:image/svg+xml;base64,")[1].split('"')[0]
    assert "<polyline" in base64.b64decode(chart).decode() and "<svg" not in markup.split("</style>")[-1]
    visible = markup.split("</style>")[-1]  # the page, not the stylesheet
    import re
    for internal in (r"\brule \d", r"[Gg]eo-analyst", r"[Ss]upervisor", r"\[llm\]", r"\[code\]", r"send_back",
                     r"ask_rules", r"loss_after_2020", r"[Rr]ules agent", r"guardrail", r"Point [A-D]\b"):
        assert not re.search(internal, visible), internal


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
    at.radio(key="view").set_value("How it works").run()
    markup = page_markup(at)
    assert "How a case travels" in markup and "End to end" in markup and "Faithfulness" in markup


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


def test_land_records_are_read_only_beside_the_case():
    at = app()
    at.session_state["plot_id"] = "C-01"
    at.run()
    assert len(at.file_uploader) == 0  # no uploader until the case waits for the officer
    assert "Opened by agent" not in page_markup(at)


def test_recheck_with_new_records_needs_a_record_and_reruns_once():
    from eudr_agent import documents

    added = documents.UPLOADS_DIR / "C-02" / "uploaded_recheck_record.md"
    try:
        at = app()
        at.session_state["plot_id"] = "C-02"
        at.run()
        at.button(key="run_C-02").click().run()
        case_id = at.session_state["desk"]["C-02"]["case_id"]

        at.radio(key=f"decision_{case_id}").set_value("Re-check with new land records")
        at.button(key=f"submit_{case_id}").click().run()
        assert any("at least one land record" in el.value for el in at.error)

        at.radio(key=f"decision_{case_id}").set_value("Re-check with new land records")
        at.file_uploader(key=f"recheck_{case_id}").upload(
            "recheck_record.md", b"Land record (MOCK)\n\nCoffee under shade since 2010.", "text/markdown")
        at.button(key=f"submit_{case_id}").click().run()
        assert not at.exception and not any("Not accepted" in el.value for el in at.error)
        flash = at.session_state["flash"]
        assert flash["title"].startswith("Re-check finished") and "Added: recheck_record.md." in flash["lines"]
        assert "Re-check finished" in page_markup(at)
        from eudr_agent.supervisor.graph import get_case
        from eudr_agent.supervisor.checkpoints import open_checkpointer

        with open_checkpointer() as saver:
            case = get_case(build_graph(saver), case_id)
        assert case["reruns"] == 1 and any(d.endswith("uploaded_recheck_record.md") for d in case["plot"]["documents"])
        assert f"recheck_{case_id}" not in [u.key for u in at.file_uploader]  # one re-check per case
    finally:
        added.unlink(missing_ok=True)


def test_sign_off_shows_only_the_fields_for_the_chosen_option():
    at = app().run()
    at.button(key="run_C-01").click().run()
    case_id = at.session_state["desk"]["C-01"]["case_id"]
    keys = lambda: {w.key for w in [*at.selectbox, *at.text_input, *at.file_uploader]}
    assert f"reason_{case_id}" in keys() and f"override_{case_id}" not in keys() and f"recheck_{case_id}" not in keys()
    at.radio(key=f"decision_{case_id}").set_value("Override the verdict").run()
    assert f"override_{case_id}" in keys() and f"recheck_{case_id}" not in keys()
    assert len(at.selectbox(key=f"override_{case_id}").options) == 2  # the draft verdict itself is not offered
    at.radio(key=f"decision_{case_id}").set_value("Re-check with new land records").run()
    assert f"recheck_{case_id}" in keys() and f"reason_{case_id}" not in keys()
    assert at.button(key=f"submit_{case_id}").label == "Re-check"


def test_a_checked_plot_opens_on_its_case_and_cannot_be_run_again(case_store):
    with checkpoints.open_checkpointer(case_store) as saver:
        case_id = start_case(build_graph(saver, analyse=_stand_in_geo, ask=_stand_in_rules), load_plots()["N-01"])["case_id"]
    at = app()
    at.session_state["plot_id"] = "N-01"
    at.run()
    assert not at.exception and at.session_state["desk"]["N-01"]["case_id"] == case_id  # a new session finds it
    assert "run_N-01" not in [b.key for b in at.button]
    assert "This plot has been checked" in page_markup(at)


def test_only_an_unfinished_or_failed_run_may_run_again():
    spec = __import__("importlib.util").util.spec_from_file_location("desk", APP)
    desk = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(desk)
    done = {"status": "awaiting_officer", "geo_reports": [{"round": 1}], "findings": [{"tool": "forest_2020"}]}
    assert desk.can_run({}) == (True, "")
    assert not desk.can_run(done)[0] and not desk.can_run({**done, "status": "closed"})[0]
    assert not desk.can_run({"status": "blocked"})[0]
    assert desk.can_run({**done, "status": "running"}) == (True, "the last run did not finish")
    failed = {"status": "awaiting_officer", "geo_reports": [{"round": 1, "error": "timeout"}],
              "findings": [{"tool": "forest_2020", "error": "timeout"}]}
    assert desk.can_run(failed) == (True, "the last run failed")


def test_the_landing_page_opens_each_section_and_home_comes_back():
    for key, source in (("home_01", "New submission"), ("home_02", "Submitted plots"), ("home_03", "Demo plots")):
        at = AppTest.from_file(APP, default_timeout=60).run()
        assert not at.exception and "Canopy" in page_markup(at) and "data:image/jpeg;base64," in page_markup(at)
        at.button(key=key).click().run()
        assert not at.exception and at.radio(key="source").value == source
    at.button(key="go_home").click().run()
    assert at.session_state["home"] and not at.radio
