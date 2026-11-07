"""Evidence desk: the officer's screen for one plot's case.

Run with:  streamlit run app/streamlit_app.py

The screen holds no logic of its own. It calls the same agents as the scripts and shows their trace events
as they happen. Stages that are not built yet are shown as not built. Grader labels are never loaded here.
"""

from __future__ import annotations

import math
from html import escape as e
from pathlib import Path
from typing import Any

import streamlit as st

from eudr_agent import documents, guardrail, report
from eudr_agent.config import settings
from eudr_agent.geo import agent as geo_agent
from eudr_agent.geo import tools as geo_tools
from eudr_agent.plots import load_plots
from eudr_agent.rag import agent as rag_agent
from eudr_agent.state import Finding, PlotInput
from eudr_agent.tracing import Tracer

METRES_PER_DEGREE = 111_320
SCALE_BAR_METRES = (50, 100, 200, 500, 1000)
RULE_QUESTIONS = [
    "Who checks my coffee after it is shipped?",
    "What is the cut-off date for deforestation-free products?",
    "What geolocation is required for a plot above 4 hectares?",
]

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


def get_case(plot_id: str) -> dict[str, Any]:
    cases = st.session_state.setdefault("cases", {})
    return cases.setdefault(plot_id, {"geo": None, "rules": [], "note": ""})


# ---------- header and case summary ----------

def render_header() -> None:
    model = f"LLM {settings.llm_model}" if settings.llm_enabled else "No LLM key: geo-analyst unavailable"
    html(f"""<div class="brand"><div><h1>Evidence <span>Desk</span></h1>
    <p>EUDR screening for Karnataka coffee plots. Advisory only: an officer makes every decision.</p></div>
    <div class="model">{e(model)}<br>Satellite data: Earth Engine, saved per plot</div></div>""")


def render_pipeline(plot: PlotInput, case: dict[str, Any]) -> None:
    geo, rules = case["geo"], case["rules"]
    geo_ok = bool(geo) and not geo.get("error")
    check = case["guardrail"]
    input_status = "Passed" if check["passed"] and not check["warnings"] else \
        f"Passed, {len(check['warnings'])} warning(s)" if check["passed"] else "Blocked"
    stages = [
        ("Input check", input_status, "done" if check["passed"] else "ready"),
        ("Documents", f"{len(plot.get('documents', []))} on case", "done" if plot.get("documents") else "ready"),
        ("Geo-analyst", "Evidence gathered" if geo_ok else "Failed" if geo else "Ready to run", "done" if geo_ok else "ready"),
        ("Supervisor", "Not built yet", "todo"),
        ("Rules agent", f"{len(rules)} check{'s' if len(rules) != 1 else ''}", "done" if rules else "ready"),
        ("Verdict rules", "Not built yet", "todo"),
        ("Officer sign-off", "Not built yet", "todo"),
        ("Evidence pack", "Available", "done"),
    ]
    cells = "".join(f'<div class="stage {state}"><b>{e(name)}</b><small>{e(status)}</small></div>'
                    for name, status, state in stages)
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
    opened = {f["values"]["file"] for f in (case["geo"] or {}).get("findings", [])
              if f["tool"] == "read_document" and not f.get("error")}
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
    if node == "input_guardrail":
        return "", "Input check", event.get("summary", ""), time_tag
    if node == "geo_think":
        chose = [report.TOOL_LABELS.get(name, name) for name in event.get("chose", [])]
        return "", "Planning", f"Next: {', '.join(chose)}" if chose else "Has enough evidence and stops checking", time_tag
    if node == "geo_tool":
        finding = event["finding"]
        source = "error" if finding.get("error") else "saved result" if finding.get("cached") else "live"
        return "tool", report.TOOL_LABELS.get(event["tool"], event["tool"]), event.get("reason", ""), f"{source} · {time_tag}"
    if node == "geo_report":
        return "", "Report written", event.get("summary", ""), time_tag
    if node == "retrieve":
        return "tool", "Search the regulation", f"Query: {event.get('query', '')}", time_tag
    if node == "grade":
        return "", "Grade passages", event.get("summary", ""), time_tag
    if node == "rewrite":
        return "loop", "Feedback loop: rewrite the query", event.get("summary", ""), time_tag
    if node == "answer":
        return "", "Write cited answer", event.get("summary", ""), time_tag
    if node == "check_citations":
        return "", "Check citations", event.get("summary", ""), time_tag
    if node == "no_rule_found":
        return "", "No rule found", "The agent declines to answer", time_tag
    return "", node, event.get("summary", ""), time_tag


def timeline_html(events: list[dict[str, Any]]) -> str:
    rows = []
    for number, event in enumerate(events, 1):
        style, title, detail, tag = describe_step(event)
        rows.append(f'<div class="step {style}"><div class="num">{number:02d}</div>'
                    f'<div><b>{e(title)}</b><p>{e(detail)}</p></div><div class="tag">{e(tag)}</div></div>')
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


def not_checked(title: str) -> str:
    return f'<div class="card off"><h4>{e(title)}</h4><p>The agent did not run this check on this plot.</p></div>'


def render_evidence(geo: dict[str, Any]) -> None:
    latest = {f["tool"]: f for f in geo["findings"]}
    cards = [(("protected_area_check"), protected_card, "Protected areas"), ("forest_2020", forest_card, "Forest in 2020"),
             ("loss_after_2020", loss_card, "Loss after 2020")]
    html('<p class="eyebrow">Tool findings</p>')
    for column, (tool, card, title) in zip(st.columns(3), cards):
        with column:
            html(card(latest[tool]) if tool in latest else not_checked(title))

    html('<p class="eyebrow" style="margin-top:14px">Geo-analyst report</p>'
         '<p class="note">Written by the LLM. The figures above come straight from the tools.</p>'
         f'<p class="prose">{e(geo["summary"])}</p>')
    left, right = st.columns(2)
    for column, title, items, empty in ((left, "Conflicts reported", geo["conflicts"], "None reported."),
                                        (right, "Open questions", geo["open_questions"], "None reported.")):
        rows = "".join(f"<li>{e(item)}</li>" for item in items)
        body = f'<ul class="prose-list">{rows}</ul>' if rows else f'<p class="note">{empty}</p>'
        with column:
            html(f'<p class="eyebrow">{title}</p>{body}')


def run_geo(plot: PlotInput, case: dict[str, Any], slot: Any) -> None:
    tracer = live_tracer("geo_analyst", slot)
    names = [Path(p).name for p in plot.get("documents", [])]
    check = guardrail.check_input(plot, tracer=tracer)  # checked again at run time, so it is in the trace
    if not check["passed"]:
        case["geo"] = {"error": "input check blocked the plot: " + "; ".join(check["blocked"]), "events": tracer.events,
                       "trace": tracer.path.name, "documents": names, "findings": []}
        return
    names = [Path(p).name for p in check["plot"].get("documents", [])]
    try:
        final = geo_agent.analyse(check["plot"], tracer=tracer)
        case["geo"] = {"findings": final["findings"], "summary": final["summary"], "conflicts": final["conflicts"],
                       "open_questions": final["open_questions"], "hit_limit": final.get("hit_limit", False),
                       "events": tracer.events, "trace": tracer.path.name, "documents": names}
    except Exception as error:  # a failed run must reach the screen and the pack, not crash the desk
        case["geo"] = {"error": f"{type(error).__name__}: {error}"[:300], "events": tracer.events,
                       "trace": tracer.path.name, "documents": names, "findings": []}


def render_investigate(plot: PlotInput, case: dict[str, Any]) -> None:
    plot_id = plot["plot_id"]
    clicked = st.button(f"Run geo-analyst on {plot_id}", type="primary", key=f"run_{plot_id}",
                        disabled=not settings.llm_enabled or not case["guardrail"]["passed"])
    html('<p class="eyebrow" style="margin-top:14px">What the agent did, step by step</p>')
    slot = st.empty()
    if clicked:
        with st.spinner("The agent is choosing and running its checks"):
            run_geo(plot, case, slot)
        st.rerun()  # redraws the pipeline and document list with the new result

    geo = case["geo"]
    if not geo:
        slot.html('<p class="note">Not run yet. The agent picks its own checks, one at a time, and each step '
                  'appears here as it finishes.</p>')
        return
    slot.html(timeline_html(geo["events"]))
    html(f'<p class="note">Trace file: runs/traces/{e(geo["trace"])}</p>')
    if geo.get("error"):
        st.error(f"The geo-analyst run failed: {geo['error']}")
        return
    if sorted(geo["documents"]) != sorted(Path(p).name for p in case["guardrail"]["plot"].get("documents", [])):
        html('<div class="notice">The documents changed after this run. Run the agent again so it can see them.</div>')
    if geo.get("hit_limit"):
        html('<div class="notice">The agent reached its step limit and reported with what it had.</div>')
    render_evidence(geo)


# ---------- rule checks ----------

def run_rule_check(case: dict[str, Any], question: str, slot: Any) -> None:
    tracer = live_tracer("rules_agent", slot)
    try:
        final = rag_agent.ask(question, tracer=tracer)
        case["rules"].insert(0, {"question": question, "answer": final["answer"], "citations": final["citations"],
                                 "status": final["status"], "events": tracer.events, "trace": tracer.path.name})
    except Exception as error:
        case["rules"].insert(0, {"question": question, "error": f"{type(error).__name__}: {error}"[:300],
                                 "citations": [], "status": "failed", "events": tracer.events,
                                 "trace": tracer.path.name})


def render_rule_result(check: dict[str, Any], index: int) -> None:
    rewrites = sum(1 for event in check["events"] if event["node"] == "rewrite")
    loop = f'<span class="loopflag">Feedback loop ran {rewrites}x</span>' if rewrites else ""
    chips = "".join(f'<span class="chip">{e(ref)}</span>' for ref in check["citations"])
    body = check.get("answer") or check.get("error", "")
    html(f'<div class="card"><p class="question">{e(check["question"])}</p><p class="prose">{e(body)}</p>{chips} {loop}</div>')
    with st.expander("Show this check's trace", expanded=bool(rewrites) and index == 0):
        html(timeline_html(check["events"]))
        html(f'<p class="note">Trace file: runs/traces/{e(check["trace"])}</p>')


def render_rules(plot: PlotInput, case: dict[str, Any]) -> None:
    plot_id = plot["plot_id"]
    html('<p class="note">Each answer must cite retrieved EUDR text, or the agent declines. Until the supervisor is '
         'built, an officer starts these checks by hand.</p>')
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
            run_rule_check(case, question, slot)
        st.rerun()
    for index, check in enumerate(case["rules"]):
        render_rule_result(check, index)


# ---------- evidence pack ----------

def render_pack(plot: PlotInput, case: dict[str, Any]) -> None:
    plot_id = plot["plot_id"]
    geo, checks = case["geo"], case["rules"]
    documents_count = len(plot.get("documents", []))
    geo_line = ("Failed, and the error is recorded" if geo.get("error") else f"{len(geo['findings'])} findings with reasons") \
        if geo else "Not run, and the pack says so"
    html(f"""<div class="card"><h4>What goes in the {e(plot_id)} pack</h4><ul class="packlist">
    <li>Plot details and geometry check<span>Included</span></li>
    <li>Geo-analyst evidence<span>{e(geo_line)}</span></li>
    <li>Rule checks with citations<span>{len(checks)}</span></li>
    <li>Mock documents, in full<span>{documents_count}</span></li>
    <li>Verdict<span>Not issued; rules not built</span></li></ul></div>""")
    case["note"] = st.text_area("Officer note (optional)", value=case["note"], key=f"note_{plot_id}",
                                placeholder="Anything the officer wants on the record")
    st.download_button(f"Download {plot_id} evidence pack (PDF)", type="primary", mime="application/pdf",
                       file_name=f"{plot_id}_evidence_pack.pdf", key=f"pack_{plot_id}",
                       data=report.build_evidence_pack(plot, geo, checks, case["note"]))


# ---------- page ----------

def main() -> None:
    st.set_page_config(page_title="Evidence Desk", page_icon="🌱", layout="wide")
    st.html(CSS)
    render_header()

    plots = load_plots()
    plot_id = st.radio("Plot", list(plots), horizontal=True, key="plot_id", label_visibility="collapsed")
    case = get_case(plot_id)
    plot = documents.with_uploads(plots[plot_id])
    case["guardrail"] = guardrail.check_input(plot)  # recomputed each redraw, since uploads change the documents
    render_pipeline(plot, case)

    left, right = st.columns([1, 2.3], gap="large")
    with left:
        html('<p class="eyebrow">Plot</p>')
        render_outline(plot)
        render_facts(plot)
        render_input_check(case)
        st.write("")
        render_documents(plot, case)
    with right:
        # a radio rather than st.tabs: tabs jump back to the first one when a finished run redraws the page
        views = {"Investigate": render_investigate, "Rule checks": render_rules, "Evidence pack": render_pack}
        view = st.radio("View", list(views), horizontal=True, key="view", label_visibility="collapsed")
        views[view](plot, case)


main()
