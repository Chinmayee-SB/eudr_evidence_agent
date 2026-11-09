"""LangSmith tracing, checked with a fake LangSmith client: nothing is sent anywhere, no key is needed."""

from __future__ import annotations

from collections import Counter
from unittest import mock

import pytest
from langsmith import Client, tracing_context

from eudr_agent import observability
from eudr_agent.geo import agent as geo_agent
from eudr_agent.plots import load_plots
from eudr_agent.rag import agent as rag_agent
from eudr_agent.supervisor.checkpoints import open_checkpointer
from eudr_agent.supervisor.graph import build_graph, resume_case, start_case


def _geo(plot, task=None, tracer=None, prior_findings=None):
    findings = [geo_agent._run_tool(plot, name, {}) for name in ("protected_area_check", "forest_2020", "loss_after_2020")]
    return {"findings": findings, "summary": "s", "conflicts": [], "open_questions": []}


@pytest.fixture
def captured():
    runs: list[dict] = []
    client = mock.MagicMock(spec=Client)
    client.create_run.side_effect = lambda **run: runs.append(run)
    with tracing_context(enabled=True, client=client, project_name="test"):
        yield runs


def test_a_case_is_one_thread_of_nested_runs(captured, tmp_path):
    with open_checkpointer(tmp_path / "c.sqlite") as saver:
        graph = build_graph(saver, analyse=_geo, ask=rag_agent.ask)
        case = start_case(graph, load_plots()["C-02"])
        closed = resume_case(graph, case["case_id"], {"decision": "approve", "reason": "ok"})

    roots = [r for r in captured if not r.get("parent_run_id")]
    assert [r["name"] for r in roots] == [f"case {case['case_id']} (start)", f"case {case['case_id']} (approve)"]
    assert {r["extra"]["metadata"]["thread_id"] for r in roots} == {case["case_id"]}
    assert str(roots[0]["id"]) == case["langsmith_runs"][0] and len(closed["langsmith_runs"]) == 2

    names = Counter(r["name"] for r in captured)
    assert {"guardrail", "geo_analyst", "supervisor", "rules_agent", "verdict", "officer", "pack"} <= set(names)
    tools = [r for r in captured if r.get("run_type") == "tool"]
    assert {r["name"] for r in tools} == {"protected_area_check", "forest_2020", "loss_after_2020"}
    search = next(r for r in captured if r.get("run_type") == "retriever")
    assert search["name"] == "search"


def test_retriever_output_lists_the_passages():
    from eudr_agent.rag.index import get_index

    out = observability.retrieved(get_index().search("What is the legal definition of forest?", k=5))
    doc = out["documents"][0]
    assert doc["type"] == "Document" and doc["page_content"] and {"ref", "kind", "score"} <= set(doc["metadata"])


def test_officer_decision_is_feedback_on_the_draft_run(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-not-a-real-key")
    fake = mock.MagicMock()
    monkeypatch.setattr("langsmith.Client", lambda *a, **k: fake)
    observability.log_officer_decision("run-1", {"decision": "override", "verdict": "high_risk", "reason": "seen it"})
    kwargs = fake.create_feedback.call_args.kwargs
    assert fake.create_feedback.call_args.args == ("run-1",)
    assert kwargs["key"] == "officer_decision" and kwargs["score"] == 0 and "high_risk" in kwargs["comment"]


def test_nothing_is_traced_or_wrapped_without_a_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    client = object()
    assert not observability.enabled() and observability.wrap_llm_client(client) is client
    assert observability.run_url("run-1") == ""
    observability.log_officer_decision("run-1", {"decision": "approve"})  # does nothing, raises nothing
