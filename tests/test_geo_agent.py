"""Tests for the geo-analyst loop, with the LLM scripted, so they are free and repeatable.
Satellite results come from the saved cache, so Earth Engine is not contacted."""

from __future__ import annotations

import json
from types import SimpleNamespace

from eudr_agent.geo import agent, tools
from eudr_agent.plots import load_plots

REPORT = json.dumps({"summary": "Done.", "conflicts": ["a"], "open_questions": []})


def tool_call(name, call_id="1", **args):
    args.setdefault("reason", "testing")
    function = SimpleNamespace(name=name, arguments=json.dumps(args))
    return SimpleNamespace(id=call_id, function=function, model_dump=lambda: {
        "id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}})


def scripted(monkeypatch, replies):
    """Make the LLM return these replies in turn, and record the tools offered on each call."""
    offered = []

    def complete(messages, tools=None, max_tokens=600):
        offered.append(tools)
        return replies.pop(0) if len(replies) > 1 else replies[0]

    monkeypatch.setattr(agent.llm, "complete", complete)
    return offered


def says(text):
    return SimpleNamespace(content=text, tool_calls=None)


def calls(*tool_calls):
    return SimpleNamespace(content="", tool_calls=list(tool_calls))


def test_loop_runs_chosen_tools_then_reports(monkeypatch):
    scripted(monkeypatch, [calls(tool_call("protected_area_check")), calls(tool_call("loss_after_2020")), says(REPORT)])
    state = agent.analyse(load_plots()["K-01"])
    assert [f["tool"] for f in state["findings"]] == ["protected_area_check", "loss_after_2020"]
    assert state["findings"][1]["values"]["hansen_loss_ha"] == 0.22
    assert state["summary"] == "Done." and state["conflicts"] == ["a"]


def test_tool_receives_the_plots_own_geometry_whatever_the_llm_sends(monkeypatch):
    seen = []
    monkeypatch.setattr(tools, "loss_after_2020", lambda geometry: seen.append(geometry) or {"tool": "x"})
    fake = {"type": "Point", "coordinates": [0, 0]}
    scripted(monkeypatch, [calls(tool_call("loss_after_2020", geometry=fake)), says(REPORT)])
    plot = load_plots()["K-01"]
    agent.analyse(plot)
    assert seen == [plot["geometry"]]


def test_step_limit_withdraws_tools_and_forces_a_report(monkeypatch):
    forever = calls(tool_call("forest_2020"))
    replies = [forever] * agent.MAX_STEPS + [says(REPORT)]
    offered = scripted(monkeypatch, replies)
    state = agent.analyse(load_plots()["C-01"])
    assert len(state["findings"]) == agent.MAX_STEPS
    assert state["hit_limit"] is True
    assert offered[-1] is None and offered[0] is not None


def test_tool_error_goes_back_to_the_llm(monkeypatch):
    monkeypatch.setattr(tools, "forest_2020", lambda g: {"tool": "forest_2020", "error": "quota exceeded"})
    scripted(monkeypatch, [calls(tool_call("forest_2020")), says(REPORT)])
    state = agent.analyse(load_plots()["C-01"])
    tool_message = next(m for m in state["messages"] if m["role"] == "tool")
    assert "quota exceeded" in tool_message["content"]


def test_unknown_tool_and_foreign_document_are_refused(monkeypatch):
    scripted(monkeypatch, [calls(tool_call("delete_everything")),
                           calls(tool_call("read_document", document="../../.env")), says(REPORT)])
    state = agent.analyse(load_plots()["K-01"])
    assert "unknown tool" in state["findings"][0]["error"]
    assert "no such document" in state["findings"][1]["error"]


def test_document_tool_is_offered_only_when_the_plot_has_a_record(monkeypatch):
    def names(plot_id):
        offered = scripted(monkeypatch, [says(REPORT)])
        agent.analyse(load_plots()[plot_id])
        return [t["function"]["name"] for t in offered[0]]

    assert "read_document" in names("K-01")
    assert "read_document" not in names("C-02")


def test_unparseable_report_is_kept_and_flagged(monkeypatch):
    scripted(monkeypatch, [says("not json")])
    state = agent.analyse(load_plots()["C-01"])
    assert state["summary"] == "not json"
    assert "could not be parsed" in state["open_questions"][0]


def test_llm_never_sees_coordinates_or_labels(monkeypatch):
    scripted(monkeypatch, [says(REPORT)])
    state = agent.analyse(load_plots()["N-01"])
    opening = state["messages"][1]["content"]
    assert "coordinates" not in opening and "control" not in opening and "High risk" not in opening
