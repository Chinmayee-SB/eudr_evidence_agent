"""Geo-analyst agent: a ReAct loop as a LangGraph state machine.

    think --(tool call)--> act --> think --(no tool call)--> report -> END

The LLM picks the tools and their order; the code runs them. The LLM never sees or writes coordinates:
tools receive the plot's geometry from the state, so a location cannot be mistyped or invented.
Every tool call must state a reason, which is written to the trace, so the planning can be inspected.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from eudr_agent import llm
from eudr_agent.geo import prompts, tools
from eudr_agent.state import Finding, PlotInput
from eudr_agent.tracing import Tracer

MAX_STEPS = 6  # tool rounds; at the limit the tools are withdrawn and the agent must report

REASON = {"type": "string", "description": "One sentence: why this check now, given what you have seen so far."}


def _schema(name: str, description: str, **extra: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {"reason": REASON, **extra}, "required": ["reason", *extra],
    }}}


TOOL_SCHEMAS = [
    _schema("protected_area_check", "Overlap of the plot with protected areas, separating those in force from proposed."),
    _schema("loss_after_2020", "Tree-cover loss on the plot after 2020 from Hansen, and deforestation and degradation "
                               "from JRC TMF, in hectares."),
    _schema("forest_2020", "How much of the plot was forest around 2020, on the JRC GFC2020 map and Hansen tree cover."),
    _schema("read_document", "Read one of the plot's land records.",
            document={"type": "string", "description": "File name from the plot's document list."}),
]


class GeoState(TypedDict, total=False):
    plot: PlotInput
    messages: list[dict[str, Any]]
    steps: int
    findings: list[Finding]
    summary: str
    conflicts: list[str]
    open_questions: list[str]
    hit_limit: bool
    tracer: Tracer


def _tools_for(plot: PlotInput) -> list[dict[str, Any]]:
    """A plot with no land record is not offered the document tool, so no step is wasted on it."""
    if plot.get("documents"):
        return TOOL_SCHEMAS
    return [t for t in TOOL_SCHEMAS if t["function"]["name"] != "read_document"]


def _run_tool(plot: PlotInput, name: str, args: dict[str, Any]) -> Finding:
    if name == "read_document":
        # only the plot's own documents can be opened, whatever name the model asks for
        by_name = {path.rsplit("/", 1)[-1]: path for path in plot.get("documents", [])}
        if args.get("document") not in by_name:
            return Finding(tool=name, dataset="mocked land records", error=f"no such document; available: {list(by_name)}")
        return tools.read_document(by_name[args["document"]])
    if name in ("protected_area_check", "loss_after_2020", "forest_2020"):
        return getattr(tools, name)(plot["geometry"])
    return Finding(tool=name, dataset="", error=f"unknown tool: {name}")


def think(state: GeoState) -> GeoState:
    with state["tracer"].span("geo_think") as rec:
        messages = list(state["messages"])
        hit_limit = state["steps"] >= MAX_STEPS
        if hit_limit:
            messages.append({"role": "user", "content": prompts.GEO_LIMIT_REACHED})
        reply = llm.complete(messages, tools=None if hit_limit else _tools_for(state["plot"]))
        entry: dict[str, Any] = {"role": "assistant", "content": reply.content or ""}
        if reply.tool_calls:
            entry["tool_calls"] = [call.model_dump() for call in reply.tool_calls]
        messages.append(entry)
        calls = [c.function.name for c in reply.tool_calls or []]
        rec["chose"] = calls
        rec["summary"] = f"chose {calls}" if calls else "has enough evidence, writing report"
    return {"messages": messages, "hit_limit": hit_limit}


def act(state: GeoState) -> GeoState:
    messages, findings = list(state["messages"]), list(state.get("findings", []))
    for call in messages[-1]["tool_calls"]:
        name = call["function"]["name"]
        with state["tracer"].span("geo_tool") as rec:
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            finding = _run_tool(state["plot"], name, args)
            findings.append(finding)
            rec.update(tool=name, reason=args.get("reason", ""), finding=finding)
            result = finding.get("error") or ("from cache" if finding.get("cached") else "ok")
            rec["summary"] = f"{name} [{result}]  why: {args.get('reason', '')[:90]}"
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(finding)})
    return {"messages": messages, "findings": findings, "steps": state["steps"] + 1}


def report(state: GeoState) -> GeoState:
    with state["tracer"].span("geo_report") as rec:
        text = state["messages"][-1]["content"].strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(text)
            summary, conflicts, questions = parsed["summary"], parsed["conflicts"], parsed["open_questions"]
        except (json.JSONDecodeError, KeyError, TypeError):  # keep the text rather than lose the report
            summary, conflicts, questions = text, [], ["The geo-analyst report could not be parsed."]
        rec.update(report=summary, conflicts=conflicts, open_questions=questions, hit_limit=state["hit_limit"])
        rec["summary"] = f"{len(conflicts)} conflicts, {len(questions)} open questions"
    return {"summary": summary, "conflicts": conflicts, "open_questions": questions}


def after_think(state: GeoState) -> str:
    return "act" if state["messages"][-1].get("tool_calls") else "report"


def build_graph():
    g = StateGraph(GeoState)
    g.add_node("think", think)
    g.add_node("act", act)
    g.add_node("report", report)
    g.set_entry_point("think")
    g.add_conditional_edges("think", after_think, {"act": "act", "report": "report"})
    g.add_edge("act", "think")
    g.add_edge("report", END)
    return g.compile(checkpointer=False)  # run as a worker inside the supervisor, never saved on its own


_GRAPH = None


def _opening_message(plot: PlotInput, task: str | None) -> str:
    details = {
        "plot_id": plot["plot_id"], "area_ha": plot.get("area_ha"), "declared_crop": plot.get("declared_crop"),
        "crop_since": plot.get("crop_since", "not stated"),
        "documents": [path.rsplit("/", 1)[-1] for path in plot.get("documents", [])],
    }
    return f"Plot: {json.dumps(details)}\nTask: {task or 'Gather the evidence needed to screen this plot.'}"


def analyse(plot: PlotInput, task: str | None = None, tracer: Tracer | None = None, echo: bool = False) -> GeoState:
    """Run the geo-analyst on one plot. `task` lets the supervisor ask for a specific follow-up."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    tracer = tracer or Tracer("geo_analyst", echo=echo)
    messages = [{"role": "system", "content": prompts.GEO_SYSTEM},
                {"role": "user", "content": _opening_message(plot, task)}]
    return _GRAPH.invoke({"plot": plot, "messages": messages, "steps": 0, "findings": [], "tracer": tracer},
                         {"recursion_limit": 3 * MAX_STEPS + 5})
