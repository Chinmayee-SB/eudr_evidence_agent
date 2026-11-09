"""Evidence desk: the officer's screen for one plot's case.

Run with:  streamlit run app/streamlit_app.py

The screen holds no logic of its own. "Run case" starts the same supervisor graph as scripts/run_case.py: input
check, geo-analyst, supervisor review, rules agent, fixed verdict rules, then a pause for the officer. Each
step appears as it finishes. The officer approves, overrides or adds documents here, and the case resumes.
Cases are saved by the checkpointer (SQLite, or Postgres when DATABASE_URL is set), so a case can be reopened
by its id after a restart. Grader labels are never loaded here.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import ExitStack
from html import escape as e
from pathlib import Path
from typing import Any

import streamlit as st

from eudr_agent import documents, guardrail, observability, report
from eudr_agent.config import RUNS_DIR, settings
from eudr_agent.geo import tools as geo_tools
from eudr_agent.plots import load_plots
from eudr_agent.rag import agent as rag_agent
from eudr_agent.rag.chunk import KINDS, KINDS_SHORT
from eudr_agent.state import Finding, PlotInput
from eudr_agent.supervisor import graph as supervisor
from eudr_agent.supervisor import decide as decide_rules
from eudr_agent.supervisor.checkpoints import list_case_ids, open_checkpointer
from eudr_agent.tracing import Tracer

METRES_PER_DEGREE = 111_320
SCALE_BAR_METRES = (50, 100, 200, 500, 1000)
RULE_QUESTIONS = [
    "Who checks my coffee after it is shipped?",
    "What is the cut-off date for deforestation-free products?",
    "What geolocation is required for a plot above 4 hectares?",
]
VERDICT_WORDS = report.VERDICT_WORDS
CARD_ORDER = ["protected_area_check", "forest_2020", "loss_after_2020", "locate_loss", "boundary_sensitivity",
              "ndvi_series", "coffee_probability", "read_document"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root { --black:#060906; --panel:#0E1510; --raised:#16211A; --line:#2A3D30; --leaf:#2F8F4E; --bean:#A9CF7C;
        --ink:#EDF5E8; --muted:#A3B8A5; --alert:#E6B450; }
html, body, [data-testid="stAppViewContainer"], .stMarkdown, button, input, textarea, label
  { font-family:'IBM Plex Sans', sans-serif; }
[data-testid="stAppViewContainer"] { background:
  radial-gradient(1100px 420px at 88% -8%, rgba(47,143,78,.24), transparent 70%), var(--black); }
[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.6rem; max-width:1320px; }

.brand { display:flex; justify-content:space-between; align-items:flex-end; gap:24px;
         border-bottom:1px solid var(--line); padding-bottom:14px; margin-bottom:16px; }
.brand h1 { font-family:'Fraunces', serif; font-weight:600; font-size:2.35rem; line-height:1.05; color:var(--ink);
            margin:0; padding:0; letter-spacing:-.01em; }
.brand h1 span { color:var(--bean); }
.brand p { color:var(--muted); margin:6px 0 0; font-size:.95rem; }
.brand .model { font-family:'IBM Plex Mono', monospace; font-size:.76rem; color:var(--muted); text-align:right; }

.pipe { display:flex; gap:4px; margin:0 0 20px; }
.stage { flex:1; min-width:0; background:var(--panel); border:1px solid var(--line); border-top:3px solid var(--line);
         padding:9px 11px; }
.stage b { display:block; color:var(--ink); font-size:.86rem; font-weight:600; white-space:nowrap; overflow:hidden;
           text-overflow:ellipsis; }
.stage small { color:var(--muted); font-size:.75rem; }
.stage.done { border-top-color:var(--bean); }
.stage.ready { border-top-color:var(--leaf); }
.stage.todo { background:transparent; border-style:dashed; border-top-style:dashed; }
.stage.todo b { color:var(--muted); font-weight:500; }

.eyebrow { font-family:'IBM Plex Mono', monospace; font-size:.72rem; letter-spacing:.09em; text-transform:uppercase;
           color:var(--bean); margin:0 0 7px; }
.facts { display:grid; grid-template-columns:1fr 1fr; gap:8px 16px; margin:10px 0 2px; }
.facts div { color:var(--ink); font-size:.95rem; }
.facts small { display:block; color:var(--muted); font-size:.74rem; }

.card { background:var(--panel); border:1px solid var(--line); padding:15px 16px; margin-bottom:10px; height:100%; }
.card h4 { font-family:'Fraunces', serif; font-weight:500; font-size:1.08rem; color:var(--ink); margin:0 0 9px; padding:0; }
.card.flag { border-left:3px solid var(--alert); }
.card.off { background:transparent; border-style:dashed; }
.off p { color:var(--muted); margin:0; font-size:.88rem; }
.figure { display:flex; flex-direction:column; gap:2px; border-top:1px solid var(--line); padding:7px 0;
          font-size:.8rem; color:var(--muted); }
.figure:first-of-type { border-top:0; }
.figure b { color:var(--ink); font-weight:500; font-family:'IBM Plex Mono', monospace; font-size:.84rem; }
.flagword { color:var(--alert); font-size:.8rem; font-weight:600; }
.source { color:var(--muted); font-size:.74rem; margin:8px 0 0; }

.bar { margin:9px 0 11px; }
.bar label { display:flex; flex-direction:column; gap:2px; color:var(--muted); font-size:.8rem; margin-bottom:5px; }
.bar label b { color:var(--ink); font-family:'IBM Plex Mono', monospace; font-weight:500; }
.track { height:12px; background:var(--raised); border-radius:0 4px 4px 0; }
.fill { height:12px; background:var(--bean); border-radius:0 4px 4px 0; min-width:2px; }

.step { display:grid; grid-template-columns:30px 1fr auto; gap:10px; padding:9px 0; border-top:1px solid var(--line); }
.step:first-child { border-top:0; }
.num { font-family:'IBM Plex Mono', monospace; color:var(--muted); font-size:.8rem; padding-top:2px; }
.step b { color:var(--ink); font-weight:500; font-size:.93rem; }
.step.tool b { color:var(--bean); }
.step.loop { background:rgba(47,143,78,.14); box-shadow:inset 3px 0 0 var(--bean); padding-left:8px; }
.step p { color:var(--muted); margin:2px 0 0; font-size:.85rem; line-height:1.4; }
.tag { font-family:'IBM Plex Mono', monospace; font-size:.71rem; color:var(--muted); white-space:nowrap; }
.chip { display:inline-block; background:var(--raised); border:1px solid var(--line); color:var(--bean);
        font-family:'IBM Plex Mono', monospace; font-size:.76rem; padding:2px 8px; margin:0 5px 5px 0; }
.chip.mock { color:var(--alert); }
.chip.guidance, .chip.recital { border-style:dashed; }  /* explains the law, does not bind */
.chip.note { border-style:dotted; color:var(--muted); }  /* our own dataset note, not a legal source */
.chip small { color:var(--muted); font-size:.68rem; }
.prose { color:var(--ink); font-size:.95rem; line-height:1.55; margin:0 0 8px; }
.prose-list { color:var(--ink); font-size:.92rem; line-height:1.45; margin:4px 0 12px; padding-left:18px; }
.note { color:var(--muted); font-size:.82rem; margin:0 0 10px; }
.notice { border-left:3px solid var(--alert); background:rgba(230,180,80,.08); color:var(--ink);
          padding:8px 12px; font-size:.88rem; margin:0 0 12px; }
.loopflag { color:var(--black); background:var(--bean); font-size:.76rem; font-weight:600; padding:2px 8px; }
.question { font-family:'Fraunces', serif; color:var(--ink); font-size:1.12rem; margin:0 0 8px; }
.packlist { list-style:none; padding:0 !important; margin:0 !important; }
.packlist li { margin:0 !important; display:flex; justify-content:space-between; gap:14px; border-top:1px solid var(--line); padding:8px 0;
               color:var(--ink); font-size:.92rem; }
.packlist li:first-child { border-top:0; }
.packlist span { color:var(--muted); text-align:right; }
.verdict { background:var(--panel); border:1px solid var(--line); border-left:5px solid var(--bean); padding:16px 18px;
           margin:0 0 14px; }
.verdict.needs_evidence { border-left-color:var(--alert); }
.verdict.high_risk { border-left-color:#E0705A; }
.verdict h3 { font-family:'Fraunces', serif; font-weight:600; font-size:1.6rem; color:var(--ink); margin:0 0 4px; padding:0; }
.verdict .rule { font-family:'IBM Plex Mono', monospace; font-size:.78rem; color:var(--muted); margin:0 0 10px; }
.decision { border-top:1px solid var(--line); padding:8px 0; color:var(--ink); font-size:.9rem; }
.decision b { font-family:'IBM Plex Mono', monospace; color:var(--bean); font-weight:500; }
.decision small { display:block; color:var(--alert); font-size:.78rem; }
.who { display:inline-block; font-family:'IBM Plex Mono', monospace; font-size:.66rem; letter-spacing:.06em;
       text-transform:uppercase; color:var(--muted); border:1px solid var(--line); padding:0 5px; margin-right:6px; }
.badge { display:inline-block; font-size:.72rem; font-weight:600; padding:1px 7px; margin:0 5px 6px 0;
         border:1px solid var(--line); color:var(--ink); }
.badge.ok { border-color:var(--bean); color:var(--bean); }
.badge.warn { border-color:var(--alert); color:var(--alert); }
.budget { display:grid; grid-template-columns:repeat(4, 1fr); gap:8px; margin:6px 0 14px; }
.budget div { background:var(--panel); border:1px solid var(--line); padding:8px 10px; color:var(--ink);
              font-family:'IBM Plex Mono', monospace; font-size:.9rem; }
.budget small { display:block; font-family:'IBM Plex Sans', sans-serif; color:var(--muted); font-size:.72rem; }
.metric-table { width:100%; border-collapse:collapse; margin:4px 0 14px; font-size:.88rem; }
.metric-table th, .metric-table td { border-top:1px solid var(--line); padding:6px 8px; text-align:left; color:var(--ink); }
.metric-table th { color:var(--muted); font-weight:500; font-size:.78rem; }
a.ls { color:var(--bean); font-size:.85rem; }

[data-testid="stRadio"] div[role="radiogroup"] { gap:6px; }
[data-testid="stRadioOption"] { background:var(--panel); border:1px solid var(--line); padding:7px 16px; margin:0; }
[data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) { display:none; }
[data-testid="stRadioOption"][data-selected="true"] { background:var(--bean); border-color:var(--bean); }
[data-testid="stRadioOption"][data-selected="true"] p { color:var(--black); font-weight:600; }
.stButton button, .stFormSubmitButton button, .stDownloadButton button { border-radius:0; font-weight:600; height:auto; }
.stButton button p { white-space:normal; overflow:visible; text-overflow:clip; line-height:1.3; }
.stButton button div { overflow:visible; }
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"], .stFormSubmitButton button[kind="primary"]
  { background:var(--bean); border-color:var(--bean); color:var(--black); }
.stButton button[kind="primary"] p, .stDownloadButton button[kind="primary"] p, .stFormSubmitButton button[kind="primary"] p
  { color:var(--black); }
[data-testid="stFileUploaderDropzone"] { background:var(--panel); border:1px dashed var(--leaf); border-radius:0; }
[data-testid="stForm"] { border:0; padding:0; }
</style>
"""



def html(markup: str) -> None:
    st.html(markup)


# ---------- the case: one shared graph, cases kept by the checkpointer ----------

@st.cache_resource
def _shared_case_graph() -> tuple[Any, ExitStack]:
    """One graph and one checkpointer (and so one Postgres connection pool) for the whole app, open for its life.
    The ExitStack is returned with it so the connection stays open as long as the graph is cached."""
    stack = ExitStack()
    saver = stack.enter_context(open_checkpointer())
    return supervisor.build_graph(saver), stack


def case_graph() -> Any:
    return _shared_case_graph()[0]


def desk(plot_id: str) -> dict[str, Any]:
    """What the screen remembers per plot: the case id, ad-hoc rule checks and the officer's note."""
    plots = st.session_state.setdefault("desk", {})
    return plots.setdefault(plot_id, {"case_id": "", "rules": [], "note": ""})


def load_case(case_id: str) -> dict[str, Any]:
    return supervisor.get_case(case_graph(), case_id) if case_id else {}


def case_events(case_id: str) -> list[dict[str, Any]]:
    path = RUNS_DIR / "traces" / f"{case_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def as_geo(case: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The case's geo-analyst rounds in the shape the evidence cards and the pack use."""
    reports = case.get("geo_reports", [])
    if not reports:
        return None
    ok = [r for r in reports if not r.get("error")]
    if not ok:
        return {"error": reports[-1]["error"], "findings": [], "events": events, "trace": f"{case['case_id']}.jsonl"}
    return {"findings": case.get("findings", []),
            "summary": " ".join(f"Round {r['round']} ({r['task']}): {r.get('summary', '')}" for r in ok),
            "conflicts": [c for r in ok for c in r.get("conflicts", [])],
            "open_questions": [q for r in ok for q in r.get("open_questions", [])],
            "hit_limit": any(r.get("hit_limit") for r in ok), "events": events,
            "trace": f"{case['case_id']}.jsonl"}


# ---------- header and case summary ----------

def render_header() -> None:
    model = f"LLM {settings.llm_model}" if settings.llm_enabled else "No LLM key: supervisor uses its code rules"
    storage = "Postgres" if os.getenv("DATABASE_URL") else "SQLite"
    tracing = f"LangSmith tracing on ({observability.project()})" if observability.enabled() else \
        "LangSmith tracing off; local traces only"
    html(f"""<div class="brand"><div><h1>Evidence <span>Desk</span></h1>
    <p>EUDR screening for Karnataka coffee plots. Advisory only: an officer makes every decision.</p></div>
    <div class="model">{e(model)}<br>Satellite data: Earth Engine, saved per plot<br>Cases saved in {storage}
    <br>{e(tracing)}</div></div>""")


def render_pipeline(plot: PlotInput, check: dict[str, Any], case: dict[str, Any]) -> None:
    rounds = case.get("geo_reports", [])
    decisions = case.get("decisions", [])
    status = case.get("status")
    officer = case.get("officer", {})
    stages = [
        ("Input check", "Passed" if check["passed"] else "Blocked", "done" if check["passed"] else "ready"),
        ("Documents", f"{len(plot.get('documents', []))} on case", "done" if plot.get("documents") else "ready"),
        ("Geo-analyst", f"{len(rounds)} round{'s' if len(rounds) != 1 else ''}" if rounds else "Waiting",
         "done" if rounds else "ready"),
        ("Supervisor", f"{len(decisions)} decision{'s' if len(decisions) != 1 else ''}" if decisions else "Waiting",
         "done" if decisions else "ready"),
        ("Rules agent", f"{len(case.get('rules', []))} answer(s)" if case.get("rules") else "Waiting",
         "done" if case.get("rules") else "ready"),
        ("Verdict rules", VERDICT_WORDS.get(case.get("draft_verdict"), "Waiting") if case.get("verdict_note") else "Waiting",
         "done" if case.get("verdict_note") else "ready"),
        ("Officer sign-off", "Waiting for you" if status == "awaiting_officer" else
         officer.get("decision", "Waiting").replace("_", " ").capitalize() if status == "closed" else "Waiting",
         "done" if status == "closed" else "ready"),
        ("Evidence pack", "Final" if status == "closed" else "Draft", "done" if status == "closed" else "ready"),
    ]
    cells = "".join(f'<div class="stage {state}"><b>{e(name)}</b><small>{e(text)}</small></div>'
                    for name, text, state in stages)
    html(f'<div class="pipe">{cells}</div>')


def render_outline(plot: PlotInput) -> None:
    """The plot's true shape, north up, drawn from its coordinates. Needs no map service, so nothing leaves the machine."""
    width, height, margin = 320, 220, 22
    ring = plot["geometry"]["coordinates"][0]
    mean_lat = math.radians(sum(p[1] for p in ring) / len(ring))
    xs = [p[0] * METRES_PER_DEGREE * math.cos(mean_lat) for p in ring]
    ys = [p[1] * METRES_PER_DEGREE for p in ring]
    extent_x, extent_y = max(xs) - min(xs), max(ys) - min(ys)
    scale = min((width - 2 * margin) / extent_x, (height - 2 * margin - 16) / extent_y)
    left = (width - extent_x * scale) / 2
    top = (height - 16 - extent_y * scale) / 2
    points = " ".join(f"{left + (x - min(xs)) * scale:.1f},{top + (max(ys) - y) * scale:.1f}" for x, y in zip(xs, ys))
    bar_m = max((m for m in SCALE_BAR_METRES if m * scale <= width / 3), default=SCALE_BAR_METRES[0])
    grid = "".join(f'<path d="M{x} 0V{height}" stroke="#16211A"/>' for x in range(40, width, 40))
    grid += "".join(f'<path d="M0 {y}H{width}" stroke="#16211A"/>' for y in range(40, height, 40))
    st.image(f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
    <rect width="{width}" height="{height}" fill="#0E1510"/>{grid}
    <polygon points="{points}" fill="#A9CF7C" fill-opacity=".28" stroke="#A9CF7C" stroke-width="2" stroke-linejoin="round"/>
    <path d="M14 {height - 12}h{bar_m * scale:.1f}" stroke="#EDF5E8" stroke-width="2"/>
    <text x="14" y="{height - 18}" fill="#A3B8A5" font-size="10" font-family="IBM Plex Mono, monospace">{bar_m} m</text>
    <text x="{width - 20}" y="20" fill="#A3B8A5" font-size="11" font-family="IBM Plex Mono, monospace">N</text>
    <path d="M{width - 16.5} 40V25m-3.5 5l3.5-5 3.5 5" stroke="#A3B8A5" fill="none"/></svg>""", width="stretch")


def render_facts(plot: PlotInput) -> None:
    check = geo_tools.validate_geometry(plot["geometry"], plot.get("area_ha"))["values"]
    geometry = "Passes Art. 2(28)" if check["valid"] else "; ".join(check["reasons"])
    html(f"""<div class="facts"><div><small>Stated area</small>{e(str(plot.get('area_ha')))} ha</div>
    <div><small>Declared crop</small>{e(str(plot.get('declared_crop', 'unknown')))}</div>
    <div><small>Crop since</small>{e(str(plot.get('crop_since', 'Not stated')))}</div>
    <div><small>Geometry check</small>{e(geometry)}</div></div>""")


def render_input_check(case: dict[str, Any]) -> None:
    """What the input guardrail found. A blocked plot cannot be sent to any agent."""
    check = case["guardrail"]
    if not check["passed"]:
        items = "".join(f"<li>{e(reason)}</li>" for reason in check["blocked"])
        html(f'<div class="notice"><b>Input check: blocked.</b> No agent will run on this plot.<ul>{items}</ul></div>')
    for line in check["dropped"]:
        html(f'<div class="notice"><b>Document held back from the agent.</b> {e(line)}</div>')
    if check["warnings"]:
        items = "".join(f"<li>{e(warning)}</li>" for warning in check["warnings"])
        html(f'<p class="eyebrow" style="margin-top:12px">Input warnings</p><ul class="prose-list">{items}</ul>')



# ---------- documents ----------

def render_documents(plot: PlotInput, case: dict[str, Any]) -> None:
    plot_id = plot["plot_id"]
    html('<p class="eyebrow">Mock documents</p>')
    with st.form(f"upload_{plot_id}", clear_on_submit=True, border=False):
        files = st.file_uploader("Add mock documents (.md, .txt or .pdf)", type=["md", "txt", "pdf"],
                                 accept_multiple_files=True)
        submitted = st.form_submit_button("Add to case", width="stretch")
    if submitted and files:
        refused = st.session_state.setdefault("refused_uploads", [])
        for file in files:
            try:
                documents.save_upload(plot_id, file.name, file.getvalue())
            except ValueError as error:
                refused.append(f"{file.name} was not added: {error}")
        st.rerun()  # the pipeline and pack above need the new document
    for message in st.session_state.pop("refused_uploads", []):
        st.error(message)

    current = [Path(p) for p in plot.get("documents", [])]
    if not current:
        html('<p class="note">No documents on this case yet.</p>')
    uploaded = {p.name for p in documents.uploads_for(plot_id)}
    opened = {f["values"]["file"] for f in case.get("findings", []) if f["tool"] == "read_document" and not f.get("error")}
    for path in current:
        with st.expander(path.name):
            tags = '<span class="chip mock">MOCK</span>'
            tags += '<span class="chip">Opened by agent</span>' if path.name in opened else ""
            html(tags)
            st.text(documents.read_text(path)[:2000])
            if path.name in uploaded and st.button("Remove", key=f"remove_{plot_id}_{path.name}"):
                documents.remove_upload(plot_id, path.name)
                st.rerun()


# ---------- trace timeline ----------

def describe_step(event: dict[str, Any]) -> tuple[str, str, str, str]:
    """A trace event as (row style, title, detail, tag)."""
    node, ms = event["node"], event.get("ms", 0)
    time_tag = f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"
    titles = {"input_guardrail": "Input check", "geo_report": "Geo-analyst report", "grade": "Grade passages",
              "answer": "Write cited answer", "check_citations": "Check citations", "verify": "Check the claims",
              "cache": "Answer reused", "verdict": "Verdict rules", "officer": "Officer decision",
              "pack": "Case closed"}
    if node == "geo_think":
        chose = [report.TOOL_LABELS.get(name, name) for name in event.get("chose", [])]
        return "", "Geo-analyst plans", f"Next: {', '.join(chose)}" if chose else "Has enough evidence and stops", time_tag
    if node == "geo_tool":
        finding = event["finding"]
        source = "error" if finding.get("error") else "saved result" if finding.get("cached") else "live"
        return "tool", report.TOOL_LABELS.get(event["tool"], event["tool"]), event.get("reason", ""), f"{source} · {time_tag}"
    if node == "supervisor":
        return "loop" if "send_back" in event.get("summary", "") else "", "Supervisor decides", event.get("summary", ""), time_tag
    if node == "retrieve":
        return "tool", "Search the regulation", f"Query: {event.get('query', '')}", time_tag
    if node == "rewrite":
        return "loop", "Feedback loop: rewrite the query", event.get("summary", ""), time_tag
    if node == "no_rule_found":
        return "", "No rule found", event.get("summary", "The agent declines to answer"), time_tag
    return "", titles.get(node, node), event.get("summary", ""), time_tag


WHO = {"input_guardrail": "input", "geo_think": "geo", "geo_tool": "geo", "geo_report": "geo",
       "supervisor": "supervisor", "retrieve": "rules", "grade": "rules", "rewrite": "rules", "answer": "rules",
       "check_citations": "rules", "verify": "rules", "cache": "rules", "no_rule_found": "rules",
       "verdict": "verdict", "officer": "officer", "pack": "officer"}


def timeline_html(events: list[dict[str, Any]]) -> str:
    rows = []
    for number, event in enumerate(events, 1):
        style, title, detail, tag = describe_step(event)
        who = WHO.get(event["node"], "")
        badge = f'<span class="who">{e(who)}</span>' if who else ""
        rows.append(f'<div class="step {style}"><div class="num">{number:02d}</div>'
                    f'<div>{badge}<b>{e(title)}</b><p>{e(detail)}</p></div><div class="tag">{e(tag)}</div></div>')
    return "".join(rows)


def live_tracer(run_name: str, slot: Any) -> Tracer:
    """A tracer that redraws the timeline each time a step finishes."""
    seen: list[dict[str, Any]] = []

    def on_event(event: dict[str, Any]) -> None:
        seen.append(event)
        slot.html(timeline_html(seen))

    return Tracer(run_name, on_event=on_event)


# ---------- evidence ----------

def figures_html(finding: Finding) -> str:
    return "".join(f'<div class="figure"><span>{e(label)}</span><b>{e(value)}</b></div>'
                   for label, value in report.key_figures(finding))


def source_html(finding: Finding) -> str:
    source = "Saved result" if finding.get("cached") else "Live query"
    return f'<p class="source">{e(source)}. {e(finding.get("note", ""))}</p>'


def bar_html(label: str, hectares: float, share: float) -> str:
    title = f"{label}: {hectares} ha, {share:.0%} of the plot"
    return (f'<div class="bar" title="{e(title)}"><label><span>{e(label)}</span><b>{hectares} ha · {share:.0%}</b></label>'
            f'<div class="track"><div class="fill" style="width:{min(share, 1) * 100:.0f}%"></div></div></div>')


def protected_card(finding: Finding) -> str:
    inside = not finding.get("error") and finding["values"]["inside_protected_area"]
    flag = '<p class="flagword">Overlaps an area in force</p>' if inside else ""
    return (f'<div class="card {"flag" if inside else ""}"><h4>Protected areas</h4>{flag}{figures_html(finding)}'
            f'{source_html(finding)}</div>')


def forest_card(finding: Finding) -> str:
    if finding.get("error"):
        return f'<div class="card"><h4>Forest in 2020</h4>{figures_html(finding)}</div>'
    v = finding["values"]
    return (f'<div class="card"><h4>Forest in 2020</h4>'
            f'{bar_html("JRC GFC2020 forest", v["gfc2020_forest_ha"], v["gfc2020_forest_share"])}'
            f'{bar_html("Hansen tree cover (proxy)", v["hansen_tree_cover_ha"], v["hansen_tree_cover_share"])}'
            f'{source_html(finding)}</div>')


def loss_card(finding: Finding) -> str:
    return f'<div class="card"><h4>Loss after 2020</h4>{figures_html(finding)}{source_html(finding)}</div>'


def tool_card(finding: Finding) -> str:
    title = report.TOOL_LABELS.get(finding["tool"], finding["tool"])
    return f'<div class="card"><h4>{e(title)}</h4>{figures_html(finding)}{source_html(finding)}</div>'


def locate_card(finding: Finding) -> str:
    if finding.get("error"):
        return tool_card(finding)
    v = finding["values"]
    flag = '<p class="flagword">Loss sits at the plot edge</p>' if v.get("share_at_edge", 0) >= 0.5 else ""
    flag += '<p class="flagword">A block of 0.5 ha or more</p>' if v.get("largest_patch_ha", 0) >= 0.5 else ""
    return f'<div class="card {"flag" if flag else ""}"><h4>Where the loss is</h4>{flag}{figures_html(finding)}{source_html(finding)}</div>'


def boundary_card(finding: Finding) -> str:
    if finding.get("error"):
        return tool_card(finding)
    v = finding["values"]
    most = max((x["loss_ha"] for x in v["variants"].values()), default=0) or 1
    bars = "".join(bar_html(f"{name.replace('_', ' ').capitalize()} ({x['plot_ha']} ha plot): loss",
                            x["loss_ha"], x["loss_ha"] / most) for name, x in v["variants"].items())
    verdict = '<p class="flagword">The result depends on the boundary</p>' if not v["stable"] else \
        '<p class="note">Stable: the result holds 30 m in or out.</p>'
    changes = "".join(f'<p class="note">{e(c)}</p>' for c in v["changes"])
    return (f'<div class="card {"flag" if not v["stable"] else ""}"><h4>Boundary moved 30 m</h4>{verdict}{bars}'
            f'{changes}{source_html(finding)}</div>')


def ndvi_card(finding: Finding) -> str:
    """Dry-season greenness by year: the whole plot, and the pixels where loss was mapped."""
    if finding.get("error"):
        return tool_card(finding)
    v = finding["values"]
    years = sorted(v["plot"])
    series = {"plot": [v["plot"][y]["ndvi"] for y in years]}
    if v["loss_pixels"]:
        series["loss pixels"] = [v["loss_pixels"].get(y) for y in years]
    width, height, pad = 300, 130, 24
    lo, hi = 0.3, 1.0

    def x(i: int) -> float:
        return pad + i * (width - 2 * pad) / max(len(years) - 1, 1)

    def y(value: float) -> float:
        return height - pad - (min(max(value, lo), hi) - lo) / (hi - lo) * (height - 2 * pad)

    colours = {"plot": "#A9CF7C", "loss pixels": "#E6B450"}
    lines = ""
    for name, values in series.items():
        points = " ".join(f"{x(i):.1f},{y(val):.1f}" for i, val in enumerate(values) if val is not None)
        lines += f'<polyline points="{points}" fill="none" stroke="{colours[name]}" stroke-width="2"/>'
    cut = years.index("2020") if "2020" in years else None
    marker = (f'<path d="M{x(cut):.1f} {pad - 6}V{height - pad}" stroke="#A3B8A5" stroke-dasharray="3 3"/>'
              f'<text x="{x(cut) + 3:.1f}" y="{pad - 8}" fill="#A3B8A5" font-size="9">cut-off</text>') if cut is not None else ""
    labels = "".join(f'<text x="{x(i):.1f}" y="{height - 6}" fill="#A3B8A5" font-size="9" text-anchor="middle">'
                     f'{e(str(year)[2:])}</text>' for i, year in enumerate(years))
    legend = " · ".join(f'<span style="color:{colours[n]}">{e(n)}</span>' for n in series)
    svg = (f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="NDVI by year">'
           f'{marker}{lines}{labels}</svg>')
    return (f'<div class="card"><h4>Greenness by year (NDVI)</h4><p class="note">{legend}, {e(v["season"])}. '
            'A dip that recovers looks like pruning; a drop that stays looks like clearing.</p>'
            f'{svg}{source_html(finding)}</div>')


def coffee_card(finding: Finding) -> str:
    return (f'<div class="card"><h4>Coffee model</h4><p class="flagword">Weak for shade coffee in India: '
            'a low score is not evidence against coffee</p>'
            f'{figures_html(finding)}{source_html(finding)}</div>')


def document_card(finding: Finding) -> str:
    if finding.get("error"):
        return tool_card(finding)
    text = finding["values"]["text"]
    return (f'<div class="card"><h4>Land record</h4><span class="chip mock">MOCK</span>{figures_html(finding)}'
            f'<p class="note">{e(text[:300])}{"…" if len(text) > 300 else ""}</p></div>')


CARDS = {"protected_area_check": protected_card, "forest_2020": forest_card, "loss_after_2020": loss_card,
         "locate_loss": locate_card, "boundary_sensitivity": boundary_card, "ndvi_series": ndvi_card,
         "coffee_probability": coffee_card, "read_document": document_card}


def render_evidence(case: dict[str, Any]) -> None:
    ok = [f for f in case.get("findings", []) if not f.get("error")]
    latest = {f["tool"]: f for f in ok}
    latest |= {f["tool"]: f for f in case.get("findings", []) if f.get("error") and f["tool"] not in latest}
    documents_read = [f for f in ok if f["tool"] == "read_document"]
    cards = [latest[t] for t in CARD_ORDER if t in latest and t != "read_document"] + documents_read
    tools_run = {f["tool"] for f in cards}
    html('<p class="eyebrow">Tool findings</p>'
         f'<p class="note">The geo-analyst chose {len(tools_run)} of {len(CARD_ORDER)} checks for this plot; '
         'the others were not needed. Figures come straight from the tools.</p>')
    for start in range(0, len(cards), 3):
        for column, finding in zip(st.columns(3), cards[start:start + 3]):
            with column:
                html(CARDS.get(finding["tool"], tool_card)(finding))


def render_verdict(case: dict[str, Any]) -> None:
    detail = case.get("verdict_detail") or {}
    verdict = case.get("final_verdict") if case.get("status") == "closed" else case.get("draft_verdict")
    label = "Final verdict" if case.get("status") == "closed" else "Draft verdict, for the officer"
    lists = ""
    for title, items in (("Why", detail.get("reasons", [])), ("Caveats", detail.get("caveats", [])),
                         ("Still missing or not checked", case.get("open_items", []))):
        if items:
            lists += f'<p class="eyebrow" style="margin-top:8px">{title}</p><ul class="prose-list">' + \
                     "".join(f"<li>{e(i)}</li>" for i in items) + "</ul>"
    basis = "".join(f'<span class="chip">{e(b)}</span>' for b in detail.get("basis", []))
    html(f'<div class="verdict {e(str(verdict))}"><p class="eyebrow">{label}</p>'
         f'<h3>{e(VERDICT_WORDS.get(verdict, "Not issued"))}</h3>'
         f'<p class="rule">Fixed rules, {e(case.get("verdict_note", ""))}</p>{lists}{basis}</div>')


def render_budgets(case: dict[str, Any]) -> None:
    """How much of each limit the case used: the supervisor cannot loop or ask without end."""
    cells = [("Steps", case.get("steps", 0), decide_rules.MAX_STEPS),
             ("Send-backs", case.get("send_backs", 0), decide_rules.MAX_SEND_BACKS),
             ("Rule questions", len(case.get("rules", [])), decide_rules.MAX_RULE_QUESTIONS),
             ("Document re-runs", case.get("reruns", 0), supervisor.MAX_RERUNS)]
    html('<div class="budget">' + "".join(f"<div><small>{e(name)} used</small>{used} / {limit}</div>"
                                          for name, used, limit in cells) + "</div>")


def rule_badges(rule: dict[str, Any]) -> str:
    """Where a rule answer came from and how it was checked."""
    badges = []
    if rule.get("point"):
        badges.append(f'<span class="badge">Point {e(rule["point"])}: '
                      f'{e(decide_rules.LEGAL_POINTS.get(rule["point"], "")[:60])}</span>')
    verification = rule.get("verification")
    words = {"passed": ("ok", "Claims checked"), "repaired": ("ok", "Claims checked after one rewrite"),
             "trimmed": ("warn", "Unsupported sentences removed"), "not_run": ("warn", "Claims not checked"),
             "failed": ("warn", "Claims failed the check")}
    if verification in words:
        style, text = words[verification]
        badges.append(f'<span class="badge {style}">{e(text)}</span>')
    if rule.get("status") == "no_rule_found":
        badges.append('<span class="badge warn">No rule found</span>')
    if rule.get("cached"):
        badges.append(f'<span class="badge">Reused answer from {e(rule["cached"])}</span>')
    return "".join(badges)


def render_decisions(case: dict[str, Any]) -> None:
    rows = ""
    for d in case.get("decisions", []):
        extra = f'<small>Code overruled the LLM: {e(d["overruled"])}</small>' if d.get("overruled") else ""
        detail = f" Task: {d['task']}" if d.get("task") else ""
        detail += " Questions: " + "; ".join(d["questions"]) if d.get("questions") else ""
        rows += (f'<div class="decision"><b>{e(d["action"])}</b> [{e(d["source"])}] '
                 f'{e(d["reason"])}{e(detail)}{extra}</div>')
    html('<p class="eyebrow" style="margin-top:14px">Supervisor decisions</p>' + (rows or '<p class="note">None yet.</p>'))


def render_officer(case: dict[str, Any], plot: PlotInput) -> None:
    """Approve, override or add documents. The case resumes from where it paused, in the saved checkpoint."""
    case_id = case["case_id"]
    html('<p class="eyebrow" style="margin-top:6px">Officer sign-off</p>')
    if case.get("status") == "closed":
        officer = case.get("officer", {})
        html(f'<p class="prose">Closed: {e(officer.get("decision", ""))}'
             + (f' ({e(officer["reason"])})' if officer.get("reason") else "") + "</p>")
        return
    with st.form(f"officer_{case_id}", border=False):
        choice = st.radio("Decision", ["Approve the draft", "Override the verdict", "Documents were added"],
                          horizontal=True, key=f"decision_{case_id}")
        override = st.selectbox("Verdict, if overriding", list(supervisor.VERDICTS), key=f"override_{case_id}",
                                format_func=lambda v: VERDICT_WORDS[v])
        reason = st.text_input("Reason (required for an override)", key=f"reason_{case_id}")
        submitted = st.form_submit_button("Submit decision", type="primary")
    if not submitted:
        return
    if choice == "Approve the draft":
        officer = {"decision": "approve", "reason": reason}
    elif choice == "Override the verdict":
        officer = {"decision": "override", "verdict": override, "reason": reason}
    else:
        officer = {"decision": "documents_added", "documents": plot.get("documents", [])}
    slot = st.empty()
    seen = case_events(case_id)
    try:
        with st.spinner("Resuming the case"):
            supervisor.resume_case(case_graph(), case_id, officer,
                                   on_event=lambda ev: (seen.append(ev), slot.html(timeline_html(seen))))
    except ValueError as error:
        st.error(f"Not accepted: {error}")
        return
    st.rerun()


def render_case(plot: PlotInput, state: dict[str, Any]) -> None:
    plot_id = plot["plot_id"]
    check = state["guardrail"]
    case = state["case"]
    running = case.get("status") == "running"
    clicked = st.button(f"Run case on {plot_id}" if not case else f"Run a new case on {plot_id}", type="primary",
                        key=f"run_{plot_id}", disabled=not check["passed"] or running)
    if not settings.llm_enabled:
        html('<p class="note">No LLM key: the geo-analyst cannot run, so the case records that and the supervisor '
             'uses its code rules.</p>')
    html('<p class="eyebrow" style="margin-top:14px">What happened, step by step</p>')
    slot = st.empty()
    if clicked:
        seen: list[dict[str, Any]] = []
        with st.spinner("Running the case: input check, geo-analyst, supervisor, rules agent, verdict"):
            new = supervisor.start_case(case_graph(), plot,
                                        on_event=lambda ev: (seen.append(ev), slot.html(timeline_html(seen))))
        desk(plot_id)["case_id"] = new["case_id"]
        st.rerun()
    if not case:
        slot.html('<p class="note">No case yet. Run one: the agents pick their own checks, the supervisor reviews '
                  'them, and the case stops here for your decision.</p>')
        return
    events = case_events(case["case_id"])
    with slot.container():
        with st.expander(f"Trace: {len(events)} steps (runs/traces/{case['case_id']}.jsonl)",
                         expanded=case.get("status") != "closed"):
            html(timeline_html(events))
    html(f'<p class="note">Case {e(case["case_id"])} · status {e(case.get("status", ""))}</p>')
    link = observability.run_url((case.get("langsmith_runs") or [None])[-1])
    if link:
        html(f'<p><a class="ls" href="{e(link)}" target="_blank">Open this case\'s trace in LangSmith</a></p>')
    render_budgets(case)
    if case.get("status") == "blocked":
        st.error("The input check blocked this plot: " + "; ".join(case["guardrail"]["blocked"]))
        return
    if case.get("verdict_note"):
        render_verdict(case)
        render_officer(case, plot)
    render_evidence(case)
    for r in case.get("geo_reports", []):
        if r.get("error"):
            st.error(f"Geo-analyst round {r['round']} failed: {r['error']}")
    render_decisions(case)
    if case.get("rules"):
        html('<p class="eyebrow" style="margin-top:14px">Rules agent answers</p>')
        html('<p class="note">Asked by the supervisor, one question per legal point. Each answer must rest on a '
             'legal source, and its claims are checked against the passages it cites.</p>')
        for index, rule in enumerate(case["rules"]):
            html(rule_badges(rule))
            render_rule_result({**rule, "events": [], "trace": f"{case['case_id']}.jsonl"}, index)


def render_open_case(plot_id: str) -> None:
    html('<p class="eyebrow">Saved cases</p>')
    try:
        saved = list_case_ids(case_graph().checkpointer, f"{plot_id}-", limit=15)
    except Exception:  # listing is a convenience; opening by id still works
        saved = []
    current = desk(plot_id)["case_id"]
    if saved:
        key = f"saved_{plot_id}"
        st.session_state[key] = current if current in saved else "—"  # follow the case on screen, e.g. a new run

        def pick() -> None:  # only a choice the officer makes switches the case
            if st.session_state[key] != "—":
                desk(plot_id)["case_id"] = st.session_state[key]

        st.selectbox("Recent cases for this plot", ["—", *saved], key=key, on_change=pick)
    with st.form(f"open_{plot_id}", clear_on_submit=True, border=False):
        typed = st.text_input("Open a saved case", placeholder=f"Case id, e.g. {plot_id}-20261109-...")
        if st.form_submit_button("Open") and typed.strip():
            found = load_case(typed.strip())
            if not found:
                st.session_state["open_error"] = f"No saved case {typed.strip()}"
            elif found["plot"]["plot_id"] != plot_id:
                st.session_state["open_error"] = f"Case {typed.strip()} is for plot {found['plot']['plot_id']}"
            else:
                desk(plot_id)["case_id"] = typed.strip()
            st.rerun()
    if st.session_state.get("open_error"):
        st.error(st.session_state.pop("open_error"))


# ---------- rule checks (ad hoc, outside the case) ----------

def run_rule_check(state: dict[str, Any], question: str, slot: Any) -> None:
    seen: list[dict[str, Any]] = []
    tracer = Tracer("rules_agent", on_event=lambda ev: (seen.append(ev), slot.html(timeline_html(seen))))
    try:
        final = rag_agent.ask(question, tracer=tracer)
        state["rules"].insert(0, {"question": question, "answer": final["answer"], "citations": final["citations"],
                                  "sources": final.get("sources", []), "status": final["status"],
                                  "events": tracer.events, "trace": tracer.path.name})
    except Exception as error:
        state["rules"].insert(0, {"question": question, "error": f"{type(error).__name__}: {error}"[:300],
                                  "citations": [], "status": "failed", "events": tracer.events,
                                  "trace": tracer.path.name})


def render_rule_result(check: dict[str, Any], index: int) -> None:
    rewrites = sum(1 for event in check.get("events", []) if event["node"] == "rewrite")
    loop = f'<span class="loopflag">Feedback loop ran {rewrites}x</span>' if rewrites else ""
    sources = check.get("sources") or [{"ref": ref, "kind": ""} for ref in check["citations"]]
    chips = "".join(f'<span class="chip {e(s["kind"])}" title="{e(KINDS.get(s["kind"], ""))}">{e(s["ref"])}'
                    + (f' <small>· {e(KINDS_SHORT[s["kind"]])}</small>' if s["kind"] in KINDS_SHORT else "")
                    + '</span>' for s in sources)
    body = check.get("answer") or check.get("error", "")
    html(f'<div class="card"><p class="question">{e(check["question"])}</p><p class="prose">{e(body)}</p>{chips} {loop}</div>')
    if check.get("events"):
        with st.expander("Show this check's trace", expanded=bool(rewrites) and index == 0):
            html(timeline_html(check["events"]))
            html(f'<p class="note">Trace file: runs/traces/{e(check["trace"])}</p>')



def render_rules(plot: PlotInput, state: dict[str, Any]) -> None:
    plot_id = plot["plot_id"]
    html('<p class="note">Ask the rules agent directly, outside the case. Each answer must cite at least one legal '
         'source it retrieved (EU law, Commission guidance or Indian law), or the agent declines. Each citation '
         'shows how binding it is. The case asks its own questions; these are extra.</p>')
    question = None
    for column, preset in zip(st.columns(len(RULE_QUESTIONS)), RULE_QUESTIONS):
        with column:
            if st.button(preset, key=f"preset_{plot_id}_{preset}", width="stretch"):
                question = preset
    with st.form(f"rule_{plot_id}", clear_on_submit=True, border=False):
        typed = st.text_input("Or check another rule", placeholder="Type a question about the regulation")
        if st.form_submit_button("Check rule") and typed.strip():
            question = typed.strip()
    slot = st.empty()
    if question:
        with st.spinner("Searching the regulation"):
            run_rule_check(state, question, slot)
        st.rerun()
    for index, check in enumerate(state["rules"]):
        render_rule_result(check, index)


# ---------- evidence pack ----------

def render_pack(plot: PlotInput, state: dict[str, Any]) -> None:
    plot_id = plot["plot_id"]
    case = state["case"]
    geo = as_geo(case, case_events(case["case_id"])) if case else None
    checks = [{**r, "trace": f"{case['case_id']}.jsonl"} for r in case.get("rules", [])] if case else []
    checks += state["rules"]
    verdict = (f"{VERDICT_WORDS.get(case.get('final_verdict'))} (final)" if case.get("status") == "closed" else
               f"{VERDICT_WORDS.get(case.get('draft_verdict'))} (draft, not signed off)") if case.get("verdict_note") \
        else "Not issued; no case run yet"
    geo_line = ("Failed, and the error is recorded" if geo.get("error") else f"{len(geo['findings'])} findings with reasons") \
        if geo else "Not run, and the pack says so"
    html(f"""<div class="card"><h4>What goes in the {e(plot_id)} pack</h4><ul class="packlist">
    <li>Verdict and officer sign-off<span>{e(verdict)}</span></li>
    <li>Plot details and geometry check<span>Included</span></li>
    <li>Geo-analyst evidence<span>{e(geo_line)}</span></li>
    <li>Supervisor decisions<span>{len(case.get('decisions', []))}</span></li>
    <li>Rule checks with citations<span>{len(checks)}</span></li>
    <li>Mock documents, in full<span>{len(plot.get('documents', []))}</span></li></ul></div>""")
    state["note"] = st.text_area("Officer note (optional)", value=state["note"], key=f"note_{plot_id}",
                                 placeholder="Anything the officer wants on the record")
    st.download_button(f"Download {plot_id} evidence pack (PDF)", type="primary", mime="application/pdf",
                       file_name=f"{plot_id}_evidence_pack.pdf", key=f"pack_{plot_id}",
                       data=report.build_evidence_pack(plot, geo, checks, state["note"], case=case or None))


# ---------- system: what is built and how well it does ----------

def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="metric-table"><tr>{head}</tr>{body}</table>'


def _latest(pattern: str) -> Path | None:
    files = sorted((Path(__file__).parent.parent / "eval").glob(pattern))
    return files[-1] if files else None


def render_system(plot: PlotInput, state: dict[str, Any]) -> None:
    from eudr_agent.rag import index as rag_index, rerank

    html('<p class="eyebrow">How a request travels</p><p class="prose">Input check → geo-analyst (picks its own '
         'satellite and document checks) ⇄ supervisor (reviews the evidence, sends back once at most, asks the '
         'rules agent one question per legal point) → rules agent (searches EU law, Commission guidance and '
         'Karnataka law; cites a legal source; checks its claims) → fixed verdict rules → officer sign-off → '
         'evidence pack. Every step is saved, so a case survives a restart.</p>')
    html('<p class="eyebrow">Running on</p>' + _table(["Part", "Setting"], [
        ["LLM", settings.llm_model if settings.llm_enabled else "none (offline: code rules and keyword search)"],
        ["Search", f"{settings.retrieval} keyword + vector, two shelves, "
                   + (f"reranked by {rerank.MODEL}" if rag_index.RERANK and rerank.enabled() else "no reranker")],
        ["Saved cases", "Postgres" if os.getenv("DATABASE_URL") else "SQLite (runs/checkpoints.sqlite)"],
        ["Traces", f"LangSmith project {observability.project()}" if observability.enabled()
                   else "local JSONL only (runs/traces/)"],
        ["Supervisor limits", f"{decide_rules.MAX_STEPS} steps, {decide_rules.MAX_SEND_BACKS} send-back, "
                              f"{decide_rules.MAX_RULE_QUESTIONS} rule questions per case"],
    ]))

    end_to_end = _latest("end_to_end_*.csv")
    if end_to_end:
        import csv as _csv
        rows = list(_csv.DictReader(end_to_end.open()))
        html(f'<p class="eyebrow">End to end ({e(end_to_end.stem[-10:])}): draft verdicts on the test plots</p>'
             '<p class="note">Compared with a human label and with FAO Whisp. Four plots show the system works; '
             'they are too few to claim accuracy.</p>'
             + _table(["Plot", "Label", "Whisp", "Ours", "Rule", "vs label", "Seconds"],
                      [[r["plot_id"], VERDICT_WORDS.get(r["label"], r["label"]), r["whisp"],
                        VERDICT_WORDS.get(r["ours"], r["ours"]), r["rule"], r["agreement"], r["seconds"]] for r in rows]))
    quality = _latest("rag_quality_2026-*.json")
    if quality:
        data = json.loads(quality.read_text())
        answers = [a for a in data.get("answers", []) if not a.get("out_of_scope") and a["status"] == "answered"]

        def mean(key: str) -> str:
            vals = [a[key] for a in answers if a.get(key) is not None]
            return f"{sum(vals) / len(vals):.2f}" if vals else "-"
        retrieval = [r for r in data.get("retrieval", []) if r["variant"].startswith("rerank: keep")]
        html(f'<p class="eyebrow">Rules agent quality ({e(quality.stem[-10:])}, {data["questions"]} questions)</p>'
             + _table(["Metric", "Score"], [
                 ["Faithfulness (claims backed by the passages)", mean("faithfulness")],
                 ["Citation precision", mean("citation_precision")],
                 ["Context precision", mean("context_precision")],
                 ["Answer relevance", mean("answer_relevance")],
                 ["Abstained when it should", f"{sum(a['abstain_ok'] for a in data.get('answers', []))}/"
                                              f"{len(data.get('answers', []))}"]])
             + (_table(["Question set", "hit@8", "MRR", "nDCG@8"],
                       [[r["set"], f"{r['hit']:.2f}", f"{r['mrr']:.2f}", f"{r['ndcg']:.2f}"] for r in retrieval])
                if retrieval else ""))


# ---------- page ----------

def main() -> None:
    st.set_page_config(page_title="Evidence Desk", page_icon="🌱", layout="wide")
    st.html(CSS)
    render_header()

    plots = load_plots()
    plot_id = st.radio("Plot", list(plots), horizontal=True, key="plot_id", label_visibility="collapsed")
    state = desk(plot_id)
    plot = documents.with_uploads(plots[plot_id])
    state["guardrail"] = guardrail.check_input(plot)  # recomputed each redraw, since uploads change the documents
    state["case"] = load_case(state["case_id"])
    render_pipeline(plot, state["guardrail"], state["case"])

    left, right = st.columns([1, 2.3], gap="large")
    with left:
        html('<p class="eyebrow">Plot</p>')
        render_outline(plot)
        render_facts(plot)
        render_input_check(state)
        st.write("")
        render_documents(plot, state["case"])
        st.write("")
        render_open_case(plot_id)
    with right:
        # a radio rather than st.tabs: tabs jump back to the first one when a finished run redraws the page
        views = {"Case": render_case, "Rule checks": render_rules, "Evidence pack": render_pack,
                 "System": render_system}
        view = st.radio("View", list(views), horizontal=True, key="view", label_visibility="collapsed")
        views[view](plot, state)


if __name__ == "__main__":
    main()
