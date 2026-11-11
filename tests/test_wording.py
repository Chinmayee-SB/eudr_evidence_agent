"""The officer's words: no internal names (rule numbers, agent names, tool names) reach the screen or the PDF."""

import importlib.util
from pathlib import Path

from eudr_agent import wording

APP = Path(__file__).parent.parent / "app" / "streamlit_app.py"


def test_free_text_loses_internal_names():
    text = ("finish [llm]: the geo-analyst ran loss_after_2020 and the supervisor asked the rules agent; "
            "needs_evidence (rule 7: The evidence is not conclusive); no land-record tool is available")
    out = wording.plain(text)
    for internal in ("[llm]", "geo-analyst", "loss_after_2020", "supervisor", "rules agent", "needs_evidence",
                     "rule 7", "tool"):
        assert internal not in out, internal
    assert "Tree loss after 2020" in out and "evidence check" in out and "legal check" in out


def test_every_verdict_rule_has_a_plain_label_and_sentence():
    from eudr_agent.supervisor import verdict_rules
    import inspect, re
    rules = {int(n) for n in re.findall(r'_result\("\w+", (\d+),', inspect.getsource(verdict_rules))}
    assert rules <= set(wording.BASIS)
    case = {"draft_verdict": "clear", "verdict_detail": {"rule": 4}}
    assert wording.verdict_line(case) == "Clear: no forest lost since 2020"


def test_a_repeated_check_in_the_trace_does_not_break_the_timeline():
    spec = importlib.util.spec_from_file_location("desk", APP)
    desk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(desk)
    style, title, detail, tag = desk.describe_step(
        {"node": "geo_tool", "tool": "forest_2020", "reason": "check again", "repeat": True, "ms": 2})
    assert title == "Forest in 2020" and tag.startswith("not re-run")


def test_dataset_names_read_as_sentences():
    assert wording.plain("There is a TMF degradation signal. GFC2020 has no legal status") == \
        "There is an EU tropical forest monitor degradation signal. The EU forest map has no legal status"
    assert wording.boundary_change("whether any loss is found") == \
        "Moving the boundary 30 m in or out changes whether any loss is found."
    assert wording.count(1, "step") == "1 step" and wording.count(2, "question") == "2 questions"


def _desk():
    spec = importlib.util.spec_from_file_location("desk", APP)
    desk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(desk)
    return desk


def test_legal_and_verdict_steps_read_plainly():
    desk = _desk()
    answer = desk.describe_step({"node": "answer", "summary": "cites ['EUDR Art. 2(4)', 'EC FAQ Q5.12'] (repair)"})
    assert answer[2] == "Cites EUDR Art. 2(4), EC FAQ Q5.12 (rewritten after the claim check)"
    assert desk.describe_step({"node": "verify", "summary": "failed: 1 problem(s)"})[2] == "1 claim not backed by the sources"
    assert desk.describe_step({"node": "verify", "summary": "ok"})[2] == "Passed"
    verdict = desk.describe_step({"node": "verdict", "verdict": "clear", "detail": {"rule": 4}, "summary": "clear (x)"})
    assert verdict[2] == "Clear: no forest lost since 2020"


def test_an_override_shows_the_officers_verdict_not_the_drafts_reasoning(monkeypatch):
    desk = _desk()
    shown = []
    monkeypatch.setattr(desk, "html", shown.append)
    desk.render_verdict({"status": "closed", "draft_verdict": "needs_evidence", "final_verdict": "clear",
                         "officer": {"decision": "override", "verdict": "clear", "reason": "Field visit: pruning"},
                         "verdict_detail": {"rule": 7, "reasons": ["Hansen and TMF disagree."], "basis": []}})
    box = shown[0]
    assert "Final verdict, set by the officer" in box and "Field visit: pruning" in box
    assert "The draft was Needs evidence: the evidence is not conclusive" in box
    assert "Why the draft said Needs evidence" in box and "Hansen" not in box and "TMF" not in box


def test_officer_steps_read_plainly():
    desk = _desk()
    step = desk.describe_step({"node": "officer", "decision": {"decision": "override", "verdict": "clear",
                                                               "reason": "Pruning seen on site"}})
    assert step[2] == "Overrode the draft to Clear. Pruning seen on site"
    closed = desk.describe_step({"node": "pack", "summary": "case closed: approve, final verdict needs_evidence"})
    assert closed[2] == "Final verdict: Needs evidence"
