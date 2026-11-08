"""Tests for the supervisor: review, decision, graph, pause and resume. No LLM key or Earth Engine is used:
satellite results come from the saved cache, and workers are stand-ins unless a test says otherwise."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eudr_agent import llm
from eudr_agent.config import Settings
from eudr_agent.geo import agent as geo_agent
from eudr_agent.geo import tools
from eudr_agent.guardrail import check_input
from eudr_agent.plots import load_plots
from eudr_agent.supervisor import decide as decide_mod
from eudr_agent.supervisor.checkpoints import open_checkpointer
from eudr_agent.supervisor.graph import build_graph, get_case, resume_case, start_case
from eudr_agent.supervisor.review import review_evidence

CORE = ("protected_area_check", "forest_2020", "loss_after_2020")


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    monkeypatch.setattr(Settings, "llm_enabled", property(lambda self: False))


def core_findings(plot):
    return [getattr(tools, name)(plot["geometry"]) for name in CORE]


def stand_in_geo(calls=None, fail=False):
    """Runs the real (cached) tools in a fixed order; reads documents when the task asks."""
    def analyse(plot, task=None, tracer=None):
        if calls is not None:
            calls.append(task)
        if fail:
            raise RuntimeError("geo-analyst down")
        if task and ("land record" in task or "New documents" in task):
            findings = [tools.read_document(p) for p in plot.get("documents", [])]
        else:
            findings = core_findings(plot)
        return {"findings": findings, "summary": "stand-in", "conflicts": [], "open_questions": []}
    return analyse


def stand_in_rules(question, tracer=None):
    return {"answer": f"answer to {question}", "citations": ["EUDR Art. 2(4)"], "status": "answered"}


@pytest.fixture
def graph(tmp_path):
    with open_checkpointer(tmp_path / "cases.sqlite") as saver:
        yield build_graph(saver, analyse=stand_in_geo(), ask=stand_in_rules)


def case_for(plot_id, **extra):
    plot = load_plots()[plot_id]
    case = {"guardrail": check_input(plot), "findings": core_findings(plot), "rules": [], "steps": 2, "send_backs": 0}
    case |= extra
    return case, review_evidence(case["guardrail"]["plot"], case["findings"])


# ---------- review ----------

def test_review_finds_conflicts_from_the_numbers():
    kinds = {pid: {c["kind"] for c in case_for(pid)[1]["conflicts"]} for pid in ("C-01", "C-02", "K-01")}
    assert kinds["C-01"] == {"forest_vs_declared_crop"}
    assert kinds["C-02"] == {"forest_vs_declared_crop", "loss_datasets_disagree"}
    assert kinds["K-01"] == {"forest_maps_disagree", "loss_datasets_disagree"}


def test_land_record_is_required_only_with_a_conflict_and_a_record():
    assert [g["need"] for g in case_for("C-01")[1]["gaps"]] == ["land_record"]
    assert case_for("C-02")[1]["gaps"] == []  # conflict, but no record to read


def test_protected_area_in_force_makes_other_checks_unnecessary():
    plot = load_plots()["N-01"]
    review = review_evidence(plot, [tools.protected_area_check(plot["geometry"])])
    assert review["protected_in_force"] and review["gaps"] == [] and review["skipped"]


def test_failed_tool_is_a_gap_and_a_retry_replaces_it():
    plot = load_plots()["K-01"]
    failed = {"tool": "forest_2020", "dataset": "x", "error": "EEException: quota"}
    review = review_evidence(plot, [tools.protected_area_check(plot["geometry"]), failed])
    assert any("quota" in g["text"] for g in review["gaps"])
    retried = review_evidence(plot, [failed, *core_findings(plot)])
    assert "forest_2020" in retried["answered"]


def test_planned_tools_become_required_once_offered():
    plot = load_plots()["K-01"]  # has loss
    now = review_evidence(plot, core_findings(plot))
    later = review_evidence(plot, core_findings(plot),
                            available={*CORE, "read_document", "locate_loss", "boundary_sensitivity"})
    assert len(now["not_checkable"]) == 2
    assert {"loss_location", "boundary_shift"} <= {g["need"] for g in later["gaps"]}


# ---------- decision ----------

def test_code_decision_without_llm():
    assert decide_mod.decide(*case_for("C-01"))["action"] == "send_back"
    d = decide_mod.decide(*case_for("C-02"))
    assert d["action"] == "ask_rules" and d["source"] == "code" and len(d["questions"]) == 2


def test_llm_finish_is_overruled_while_evidence_is_missing():
    d = decide_mod.enforce({"action": "finish", "reason": "fine"}, *case_for("C-01"))
    assert d["action"] == "send_back" and d["source"] == "code" and "missing" in d["overruled"]


def test_second_send_back_is_refused():
    d = decide_mod.enforce({"action": "send_back", "task": "again"}, *case_for("C-01", send_backs=1))
    assert d["action"] != "send_back" and "no send-back left" in d["overruled"]


def test_rule_questions_are_deduplicated_and_capped():
    asked = [{"question": "Q1", "status": "answered"}]
    d = decide_mod.enforce({"action": "ask_rules", "questions": ["q1", "Q2", "Q2", "Q3", "Q4"]},
                           *case_for("K-01", rules=asked))
    assert d["source"] == "llm" and d["questions"] == ["Q2", "Q3"]


def test_unknown_action_and_spent_steps_fall_back_to_code():
    assert decide_mod.enforce({"action": "approve"}, *case_for("C-02"))["source"] == "code"
    d = decide_mod.enforce({"action": "ask_rules", "questions": ["x"]}, *case_for("C-02", steps=decide_mod.MAX_STEPS))
    assert d["action"] == "finish"


def test_llm_error_falls_back_to_code(monkeypatch):
    monkeypatch.setattr(Settings, "llm_enabled", property(lambda self: True))
    monkeypatch.setattr(decide_mod.llm, "chat", lambda *a, **k: "not json")
    d = decide_mod.decide(*case_for("C-02"))
    assert d["source"] == "code" and "LLM unavailable" in d["overruled"]


# ---------- graph ----------

def test_blocked_plot_runs_no_agent(tmp_path):
    calls = []
    with open_checkpointer(tmp_path / "c.sqlite") as saver:
        graph = build_graph(saver, analyse=stand_in_geo(calls), ask=stand_in_rules)
        case = start_case(graph, {**load_plots()["C-01"], "declared_crop": "cocoa"})
    assert case["status"] == "blocked" and calls == []


def test_send_back_happens_at_most_once(tmp_path):
    calls = []
    with open_checkpointer(tmp_path / "c.sqlite") as saver:
        graph = build_graph(saver, analyse=stand_in_geo(calls, fail=True), ask=stand_in_rules)
        case = start_case(graph, load_plots()["K-01"])
    assert len(calls) == 2 and case["status"] == "awaiting_officer"
    assert any("failed" in item for item in case["open_items"])


def test_full_case_pauses_then_closes_on_approval(graph):
    case = start_case(graph, load_plots()["C-01"])
    assert case["status"] == "awaiting_officer"
    assert [d["action"] for d in case["decisions"]] == ["send_back", "ask_rules", "finish"]
    assert any(f["tool"] == "read_document" for f in case["findings"])
    closed = resume_case(graph, case["case_id"], {"decision": "approve", "reason": "checked the land record"})
    assert closed["status"] == "closed" and closed["officer"]["decision"] == "approve"
    with pytest.raises(ValueError, match="not waiting"):
        resume_case(graph, case["case_id"], {"decision": "approve", "reason": "again"})


def test_override_needs_a_verdict_and_a_reason(graph):
    case = start_case(graph, load_plots()["C-02"])
    for bad in ({"decision": "override", "reason": "x"}, {"decision": "override", "verdict": "high_risk"},
                {"decision": "sign"}):
        with pytest.raises(ValueError):
            resume_case(graph, case["case_id"], bad)
    closed = resume_case(graph, case["case_id"], {"decision": "override", "verdict": "high_risk", "reason": "r"})
    assert closed["final_verdict"] == "high_risk"


def test_documents_added_reruns_once(graph):
    case = start_case(graph, load_plots()["C-02"])
    docs = load_plots()["C-01"]["documents"]  # any permitted record will do
    case = resume_case(graph, case["case_id"], {"decision": "documents_added", "documents": docs})
    assert case["status"] == "awaiting_officer" and case["reruns"] == 1
    assert case["geo_reports"][-1]["task"].startswith("New documents")
    with pytest.raises(ValueError, match="already added"):
        resume_case(graph, case["case_id"], {"decision": "documents_added", "documents": docs})


def test_paused_case_resumes_from_a_fresh_connection(tmp_path):
    db = tmp_path / "c.sqlite"
    with open_checkpointer(db) as saver:
        case_id = start_case(build_graph(saver, analyse=stand_in_geo(), ask=stand_in_rules),
                             load_plots()["K-01"])["case_id"]
    with open_checkpointer(db) as saver:  # as if the app had restarted
        graph = build_graph(saver, analyse=stand_in_geo(), ask=stand_in_rules)
        assert get_case(graph, case_id)["status"] == "awaiting_officer"
        assert resume_case(graph, case_id, {"decision": "approve", "reason": "ok"})["status"] == "closed"


def test_case_is_one_trace_file(graph):
    case = start_case(graph, load_plots()["C-01"])
    resume_case(graph, case["case_id"], {"decision": "approve", "reason": "ok"})
    from eudr_agent.config import RUNS_DIR
    lines = [json.loads(line) for line in (RUNS_DIR / "traces" / f"{case['case_id']}.jsonl").open()]
    nodes = [event["node"] for event in lines]
    assert nodes[0] == "input_guardrail" and nodes[-2:] == ["officer", "pack"]
    assert [event["step"] for event in lines] == list(range(1, len(lines) + 1))


# ---------- real workers inside the supervisor ----------

class _Call:
    def __init__(self, name):
        self.id, self.function = f"call_{name}", SimpleNamespace(name=name, arguments=json.dumps({"reason": "r"}))

    def model_dump(self):
        return {"id": self.id, "type": "function", "function": {"name": self.function.name,
                                                                "arguments": self.function.arguments}}


def test_real_worker_graphs_run_inside_the_supervisor(tmp_path, monkeypatch):
    """Worker graphs must not inherit the supervisor's checkpointer: their state holds a tracer."""
    replies = iter([SimpleNamespace(content="", tool_calls=[_Call(name)]) for name in CORE] +
                   [SimpleNamespace(content=json.dumps({"summary": "s", "conflicts": [], "open_questions": []}),
                                    tool_calls=None)])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: next(replies))
    monkeypatch.setattr(Settings, "llm_enabled", property(lambda self: False))
    with open_checkpointer(tmp_path / "c.sqlite") as saver:
        graph = build_graph(saver)  # the real geo-analyst and the real (offline) rules agent
        case = start_case(graph, load_plots()["C-02"])
    assert case["status"] == "awaiting_officer"
    assert [f["tool"] for f in case["findings"]] == list(CORE)
    assert case["rules"] and all(r["status"] == "answered" for r in case["rules"])
