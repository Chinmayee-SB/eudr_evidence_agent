"""Evidence desk: the officer's screen for one plot's case.

Run with:  streamlit run app/streamlit_app.py

The screen holds no logic of its own. "Run case" starts the same supervisor graph as scripts/run_case.py: input
check, geo-analyst, supervisor review, rules agent, fixed verdict rules, then a pause for the officer. Each
step appears as it finishes. The officer approves, overrides or adds documents here, and the case resumes.
Cases are saved by the checkpointer (SQLite, or Postgres when DATABASE_URL is set), so a case can be reopened
by its id after a restart. Grader labels are never loaded here.
"""

from __future__ import annotations

import base64
import json
import math
import re
import os
import time
from contextlib import ExitStack
from html import escape as e
from pathlib import Path
from typing import Any

import streamlit as st

from eudr_agent import documents, guardrail, observability, report, submissions, wording
from eudr_agent.wording import plain
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
RECHECK = "Re-check with new land records"
SOURCES = ["New submission", "Submitted plots", "Demo plots"]  # demo plots are the labelled test set
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
[data-testid="stMain"] { overflow-x:hidden; }  /* the full-width photo must not cause a sideways scroll */
.block-container { padding-top:1.6rem; max-width:1320px; }

.brand { display:flex; justify-content:space-between; align-items:flex-end; gap:24px;
         border-bottom:1px solid var(--line); padding-bottom:14px; margin-bottom:16px; }
.brand h1 { font-family:'Fraunces', serif; font-weight:600; font-size:2.35rem; line-height:1.05; color:var(--ink);
            margin:0; padding:0; letter-spacing:-.01em; }
.brand h1 span { color:var(--bean); }
.brand p { color:var(--muted); margin:6px 0 0; font-size:.95rem; }
.brand .model { font-family:'IBM Plex Mono', monospace; font-size:.76rem; color:var(--muted); text-align:right; }

/* landing page */
.hero { position:relative; margin:-1.6rem calc(50% - 50vw) 0; width:100vw; min-height:86vh; display:flex;
        align-items:flex-end; background:
        linear-gradient(180deg, rgba(6,9,6,.10) 0%, rgba(6,9,6,.45) 50%, rgba(6,9,6,.97) 100%),
        linear-gradient(90deg, rgba(6,9,6,.70) 0%, rgba(6,9,6,.05) 60%),
        var(--hero) center 35% / cover no-repeat; }
.hero-inner { width:100%; max-width:1320px; margin:0 auto; padding:0 5rem 9vh; }  /* lines up with the page */
.hero .kicker { font-family:'IBM Plex Mono', monospace; font-size:.78rem; letter-spacing:.14em; text-transform:uppercase;
                color:var(--bean); margin:0 0 14px; }
.hero h1 { font-family:'Fraunces', serif; font-weight:600; font-size:clamp(3rem, 7vw, 5.6rem); line-height:.98;
           color:#F4F9EF; margin:0; padding:0; letter-spacing:-.02em; }
.hero h1 span { color:var(--bean); }
.hero .lead { color:#DCE7D6; font-size:clamp(1.05rem, 1.6vw, 1.3rem); max-width:620px; margin:18px 0 0;
              line-height:1.5; }
.hero .fine { color:var(--muted); font-size:.85rem; margin:14px 0 0; }
.hero .cue { position:absolute; left:50%; bottom:22px; transform:translateX(-50%); color:var(--muted);
             font-family:'IBM Plex Mono', monospace; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase;
             text-align:center; animation:bob 2.4s ease-in-out infinite; }
@keyframes bob { 0%,100% { transform:translate(-50%,0); } 50% { transform:translate(-50%,6px); } }
.home-head { margin:46px 0 16px; }
.home-head h2 { font-family:'Fraunces', serif; font-weight:500; font-size:1.9rem; color:var(--ink); margin:4px 0 0;
                padding:0; }
[class*="st-key-home_card_"] { background:var(--panel); border:1px solid var(--line); border-top:3px solid var(--leaf);
                               padding:22px 22px 18px; height:100%; transition:border-color .2s, transform .2s; }
[class*="st-key-home_card_"]:hover { border-color:var(--bean); transform:translateY(-3px); }
.home-card .num { font-family:'IBM Plex Mono', monospace; color:var(--bean); font-size:.8rem; }
.home-card h3 { font-family:'Fraunces', serif; font-weight:500; font-size:1.45rem; color:var(--ink); margin:6px 0 8px;
                padding:0; }
.home-card p { color:var(--muted); font-size:.93rem; line-height:1.5; margin:0 0 6px; min-height:4.4em; }
.home-card .stat { font-family:'IBM Plex Mono', monospace; color:var(--ink); font-size:.8rem; margin:4px 0 0; }
.flow { display:grid; grid-template-columns:repeat(5, 1fr); gap:4px; margin:6px 0 30px; }
.flow div { background:var(--panel); border:1px solid var(--line); padding:14px 14px 12px; }
.flow b { display:block; color:var(--ink); font-size:.92rem; font-weight:600; margin-bottom:4px; }
.flow small { color:var(--muted); font-size:.8rem; line-height:1.45; }
.flow i { font-style:normal; font-family:'IBM Plex Mono', monospace; color:var(--bean); font-size:.72rem; }
.home-foot { color:var(--muted); font-size:.78rem; border-top:1px solid var(--line); padding:14px 0 30px; }
@media (max-width: 900px) { .flow { grid-template-columns:1fr; } }

/* inner pages: a slim bar with the name and the way home */
.topbrand { font-family:'Fraunces', serif; font-weight:600; font-size:1.45rem; color:var(--ink); margin:2px 0 0;
            white-space:nowrap; }
.topbrand span { color:var(--bean); }
.st-key-topbar { border-bottom:1px solid var(--line); padding-bottom:8px; margin-bottom:14px; }

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

.flash{border:1px solid var(--line,#3a4a3a);border-left:4px solid #7fb069;padding:12px 16px;margin:6px 0 10px;
  border-radius:4px;background:rgba(127,176,105,.08)}
.flash.info{border-left-color:#6aa9d8;background:rgba(106,169,216,.08)}
.flash.error{border-left-color:#d86a6a;background:rgba(216,106,106,.08)}
.flash b{display:block;margin-bottom:4px}.flash ul{margin:4px 0 0 18px;padding:0}.flash li{margin:2px 0}
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

NAME = ("Canopy", "Check")
HERO = Path(__file__).parent / "static" / "hero_coffee.jpg"
CARDS_HOME = [
    ("New submission", "01", "Submit a plot",
     "Upload the plot boundary and its land records. The check starts as soon as you submit."),
    ("Submitted plots", "02", "Submitted plots",
     "Open a plot that was submitted earlier, see its case, and sign it off."),
    ("Demo plots", "03", "Demo plots",
     "Seven test plots with known answers: coffee, protected forest, and plots outside Karnataka."),
]


@st.cache_data
def hero_url() -> str:
    """The landing photo as a data URL, so it needs no file server."""
    return "data:image/jpeg;base64," + base64.b64encode(HERO.read_bytes()).decode() if HERO.exists() else ""


def open_section(source: str) -> None:
    st.session_state.update(home=False, source=source)


def go_home() -> None:
    st.session_state["home"] = True


def render_home() -> None:
    """The landing page: the photo with the name, then the three ways in, then how a case travels."""
    background = f"url({hero_url()})" if hero_url() else "none"
    html(f"""<div class="hero" style="--hero:{background}"><div class="hero-inner">
      <p class="kicker">EU Deforestation Regulation · Karnataka coffee</p>
      <h1>{NAME[0]} <span>{NAME[1]}</span></h1>
      <p class="lead">Was this coffee plot forest at the end of 2020, and has any of it been cleared since?
      Satellite evidence, land records and the law, gathered into one case for an officer to decide.</p>
      <p class="fine">Advisory only: an officer makes every decision.</p></div>
      <div class="cue">Start below<br>↓</div></div>""")
    html('<div class="home-head"><p class="eyebrow">Start here</p><h2>What would you like to do?</h2></div>')
    counts = {"Submitted plots": wording.count(len(submissions.load_submissions()), "plot") + " on file",
              "Demo plots": wording.count(len(load_plots()), "plot") + " with known answers",
              "New submission": "KML or GeoJSON, plus land records"}
    for column, (source, number, title, text) in zip(st.columns(3, gap="medium"), CARDS_HOME):
        with column, st.container(key=f"home_card_{number}"):
            html(f'<div class="home-card"><span class="num">{number}</span><h3>{e(title)}</h3><p>{e(text)}</p>'
                 f'<p class="stat">{e(counts[source])}</p></div>')
            st.button(f"Open {title.lower()} →" if number != "01" else "Start a submission →",
                      key=f"home_{number}", type="primary" if number == "01" else "secondary",
                      width="stretch", on_click=open_section, args=(source,))
    steps = [("Input check", "Stops plots and records that should not be screened."),
             ("Evidence check", "An AI assistant picks the satellite and land-record checks."),
             ("Case review", "Looks over the evidence and asks the legal questions that matter."),
             ("Draft verdict", "Fixed, published criteria. Never decided by the AI."),
             ("Officer sign-off", "Approve, override with a reason, or re-check with new records.")]
    html('<div class="home-head"><p class="eyebrow">How a case travels</p></div><div class="flow">'
         + "".join(f"<div><i>{n:02d}</i><b>{e(t)}</b><small>{e(d)}</small></div>" for n, (t, d) in enumerate(steps, 1))
         + '</div><p class="home-foot">Every step is saved and can be traced. The evidence pack records what was '
           'checked, what was not, and who decided.</p>')


def render_header() -> None:
    """The slim bar on every inner page: the name, the three sections, and the way back to the landing page."""
    with st.container(key="topbar"):
        brand, sections, home = st.columns([1.3, 4, 0.8], vertical_alignment="center")
        brand.html(f'<p class="topbrand">{NAME[0]} <span>{NAME[1]}</span></p>')
        with sections:
            st.radio("Source", SOURCES, horizontal=True, key="source", label_visibility="collapsed")
        home.button("← Home", key="go_home", on_click=go_home, width="stretch")


def render_pipeline(plot: PlotInput, check: dict[str, Any], case: dict[str, Any]) -> None:
    rounds = case.get("geo_reports", [])
    decisions = case.get("decisions", [])
    status = case.get("status")
    officer = case.get("officer", {})
    stages = [
        ("Input check", "Passed" if check["passed"] else "Blocked", "done" if check["passed"] else "ready"),
        ("Land records", f"{len(plot.get('documents', []))} on file", "done" if plot.get("documents") else "ready"),
        (wording.STAGE["geo"], f"{len(rounds)} round{'s' if len(rounds) != 1 else ''}" if rounds else "Waiting",
         "done" if rounds else "ready"),
        (wording.STAGE["supervisor"], f"{len(decisions)} decision{'s' if len(decisions) != 1 else ''}" if decisions
         else "Waiting", "done" if decisions else "ready"),
        (wording.STAGE["rules"], wording.count(len(case.get('rules', [])), "question") if case.get("rules") else "Waiting",
         "done" if case.get("rules") else "ready"),
        ("Verdict", VERDICT_WORDS.get(case.get("draft_verdict"), "Waiting") if case.get("verdict_note") else "Waiting",
         "done" if case.get("verdict_note") else "ready"),
        ("Officer sign-off", "Waiting for you" if status == "awaiting_officer" else
         wording.DECISION.get(officer.get("decision"), "Waiting") if status == "closed" else "Waiting",
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
    geometry = "Meets the EU location rules" if check["valid"] else "; ".join(check["reasons"])
    html(f"""<div class="facts"><div><small>Stated area</small>{e(str(plot.get('area_ha')))} ha</div>
    <div><small>Declared crop</small>{e(str(plot.get('declared_crop', 'unknown')))}</div>
    <div><small>Crop since</small>{e(str(plot.get('crop_since', 'Not stated')))}</div>
    <div><small>Boundary</small>{e(geometry)}</div></div>""")


def render_input_check(case: dict[str, Any]) -> None:
    """What the input guardrail found. A blocked plot cannot be sent to any agent."""
    check = case["guardrail"]
    if not check["passed"]:
        items = "".join(f"<li>{e(plain(reason)[:1].upper() + plain(reason)[1:])}.</li>" for reason in check["blocked"])
        html(f'<div class="notice"><b>Input check: stopped.</b> Nothing else will run on this plot.<ul>{items}</ul></div>')
    for line in check["dropped"]:
        html(f'<div class="notice"><b>Land record held back from the check.</b> {e(line.removeprefix("uploaded_"))}</div>')
    if check["warnings"]:
        items = "".join(f"<li>{e(warning)}</li>" for warning in check["warnings"])
        html(f'<p class="eyebrow" style="margin-top:12px">Notes from the input check</p><ul class="prose-list">{items}</ul>')



# ---------- documents ----------

def render_documents(plot: PlotInput, case: dict[str, Any]) -> None:
    plot_id = plot["plot_id"]
    html('<p class="eyebrow">Land records</p>')
    for message in st.session_state.pop("refused_uploads", []):
        st.error(message)
    current = [Path(p) for p in plot.get("documents", [])]
    if not current:
        html('<p class="note">No land records on this plot.</p>')
    opened = {f["values"]["file"] for f in case.get("findings", []) if f["tool"] == "read_document" and not f.get("error")}
    for path in current:
        with st.expander(path.name):
            tags = '<span class="chip mock">MOCK</span>' if "MOCK" in documents.read_text(path)[:400].upper() else ""
            tags += '<span class="chip">Read during the check</span>' if path.name in opened else ""
            html(tags)
            st.text(documents.read_text(path)[:2000])
    if case.get("status") == "awaiting_officer" and case.get("reruns", 0) < supervisor.MAX_RERUNS:
        html('<p class="note">To add a record, use "Re-check with new land records" in the officer sign-off. '
             'That is the only point where a new record can change the case.</p>')


# ---------- trace timeline ----------

def describe_step(event: dict[str, Any]) -> tuple[str, str, str, str]:
    """A trace event as (row style, title, detail, tag), in the officer's words (wording.py)."""
    node, ms = event["node"], event.get("ms", 0)
    time_tag = f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"
    titles = {"input_guardrail": "Input check", "geo_report": "Evidence summary", "grade": "Keep the relevant passages",
              "answer": "Write an answer with citations", "check_citations": "Check the answer cites the law",
              "verify": "Check each claim against the sources", "cache": "Reuse an earlier checked answer",
              "verdict": "Draft verdict", "officer": "Officer decision", "pack": "Case closed"}
    if node == "geo_think":
        chose = [wording.CHECK.get(name, name) for name in event.get("chose", [])]
        return "", "Plan the next check", f"Next: {', '.join(chose)}" if chose else "Enough evidence: stop", time_tag
    if node == "geo_tool":
        finding = event.get("finding") or {}  # a repeated call is logged without a new result
        source = ("not re-run" if not finding else "error" if finding.get("error")
                  else "saved result" if finding.get("cached") else "live")
        return "tool", wording.CHECK.get(event["tool"], event["tool"]), plain(event.get("reason", "")), \
            f"{source} · {time_tag}"
    if node == "supervisor":
        decision = event.get("decision") or {}
        action = decision.get("action", "")
        detail = plain(decision.get("reason") or event.get("summary", ""))
        if decision.get("questions"):
            detail += " Question: " + "; ".join(decision["questions"])
        return "loop" if action == "send_back" else "", wording.ACTION.get(action, "Case review"), detail, time_tag
    if node == "verdict":
        detail = event.get("detail") or {}
        label = wording.BASIS.get(detail.get("rule"), ("",))[0] if "rule" in detail else ""
        words = VERDICT_WORDS.get(event.get("verdict"), "")
        return "", "Draft verdict", (f"{words}: {label[:1].lower() + label[1:]}" if words and label
                                     else words or plain(event.get("summary", ""))), time_tag
    if node == "officer":
        decision = event.get("decision") or {}
        words = wording.DECISION.get(decision.get("decision"), "Signed off")
        if decision.get("verdict") and decision.get("decision") == "override":
            words += f" to {VERDICT_WORDS.get(decision['verdict'], decision['verdict'])}"
        return "", titles["officer"], words + (f". {decision['reason']}" if decision.get("reason") else ""), time_tag
    if node == "pack":
        found = re.search(r"final verdict (\w+)", event.get("summary", ""))
        return "", titles["pack"], (f"Final verdict: {VERDICT_WORDS.get(found.group(1), found.group(1))}" if found
                                    else "The case is closed"), time_tag
    if node == "answer":
        cited = re.findall(r"'([^']+)'", event.get("summary", ""))
        repair = " (rewritten after the claim check)" if "(repair)" in event.get("summary", "") else ""
        return "", titles["answer"], (f"Cites {', '.join(cited)}" if cited else "No legal source cited") + repair, \
            time_tag
    if node in ("verify", "check_citations"):
        summary = event.get("summary", "")
        found = re.search(r"(\d+) problem", summary)
        text = ("Passed" if summary == "ok" else "Fixed by one rewrite" if summary == "repaired"
                else f"{wording.count(int(found.group(1)), 'claim')} not backed by the sources" if found
                else plain(summary))
        return "", titles[node], text, time_tag
    if node == "retrieve":
        return "tool", "Search the law and guidance", f"Query: {event.get('query', '')}", time_tag
    if node == "rewrite":
        return "loop", "Search again with different words", plain(event.get("summary", "")), time_tag
    if node == "no_rule_found":
        return "", "No answer in the sources", plain(event.get("summary", "The question is not answered")), time_tag
    return "", titles.get(node, "Step"), plain(event.get("summary", "")), time_tag


WHO = {"input_guardrail": "input", "geo_think": "geo", "geo_tool": "geo", "geo_report": "geo",
       "supervisor": "supervisor", "retrieve": "rules", "grade": "rules", "rewrite": "rules", "answer": "rules",
       "check_citations": "rules", "verify": "rules", "cache": "rules", "no_rule_found": "rules",
       "verdict": "verdict", "officer": "officer", "pack": "officer"}


def timeline_html(events: list[dict[str, Any]]) -> str:
    rows = []
    for number, event in enumerate(events, 1):
        style, title, detail, tag = describe_step(event)
        who = wording.STAGE.get(WHO.get(event["node"], ""), "")
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
    return f'<p class="source">{e(source)}. {e(plain(finding.get("note", "")))}</p>'


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
            f'{bar_html("EU forest map 2020", v["gfc2020_forest_ha"], v["gfc2020_forest_share"])}'
            f'{bar_html("Global Forest Watch tree cover", v["hansen_tree_cover_ha"], v["hansen_tree_cover_share"])}'
            f'{source_html(finding)}</div>')


def loss_card(finding: Finding) -> str:
    return f'<div class="card"><h4>Tree loss after 2020</h4>{figures_html(finding)}{source_html(finding)}</div>'


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
    bars = "".join(bar_html(f"{name.replace('_', ' ').capitalize()} ({x['plot_ha']} ha plot): loss",
                            x["loss_ha"], x["loss_ha"] / x["plot_ha"] if x["plot_ha"] else 0)
                   for name, x in v["variants"].items())
    verdict = '<p class="flagword">The result depends on the boundary</p>' if not v["stable"] else \
        '<p class="note">Stable: the result holds 30 m in or out.</p>'
    changes = "".join(f'<p class="note">{e(wording.boundary_change(c))}</p>' for c in v["changes"])
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
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
           f'height="{height}">{marker}{lines}{labels}</svg>')
    svg = (f'<img alt="Greenness by year" style="width:100%" '
           f'src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"/>')
    return (f'<div class="card"><h4>Greenness by year (NDVI)</h4><p class="note">{legend}, {e(v["season"])}. '
            'A dip that recovers looks like pruning; a drop that stays looks like clearing.</p>'
            f'{svg}{source_html(finding)}</div>')


def coffee_card(finding: Finding) -> str:
    return (f'<div class="card"><h4>Coffee map</h4><p class="flagword">Not reliable for shade-grown coffee in '
            'India: a low score is not evidence against coffee</p>'
            f'{figures_html(finding)}{source_html(finding)}</div>')


def document_card(finding: Finding) -> str:
    if finding.get("error"):
        return tool_card(finding)
    text = finding["values"]["text"]
    mock = '<span class="chip mock">MOCK</span>' if "MOCK" in text[:400].upper() else ""
    return (f'<div class="card"><h4>Land record</h4>{mock}{figures_html(finding)}'
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
    html('<p class="eyebrow" style="margin-top:14px">Evidence</p>'
         f'<p class="note">{wording.count(len({c["tool"] for c in cards}), "check")} '
         f'{"was" if len({c["tool"] for c in cards}) == 1 else "were"} run for this plot; the others were not needed. Figures come '
         'straight from the data sources.</p>')
    for start in range(0, len(cards), 3):
        for column, finding in zip(st.columns(3), cards[start:start + 3]):
            with column:
                html(CARDS.get(finding["tool"], tool_card)(finding))


def render_verdict(case: dict[str, Any]) -> None:
    detail = case.get("verdict_detail") or {}
    verdict = case.get("final_verdict") if case.get("status") == "closed" else case.get("draft_verdict")
    label = "Final verdict" if case.get("status") == "closed" else "Draft verdict, for the officer"
    officer = case.get("officer") or {}
    overridden = case.get("status") == "closed" and officer.get("decision") == "override"
    lists = ""
    why_title = f"Why the draft said {VERDICT_WORDS.get(case.get('draft_verdict'))}" if overridden else "Why"
    for title, items in ((why_title, [plain(i) for i in detail.get("reasons", [])]),
                         ("Points to note", [plain(i) for i in detail.get("caveats", [])]),
                         ("Still missing or not checked", [plain(i) for i in case.get("open_items", [])])):
        if items:
            lists += f'<p class="eyebrow" style="margin-top:8px">{title}</p><ul class="prose-list">' + \
                     "".join(f"<li>{e(i)}</li>" for i in items) + "</ul>"
    basis = "".join(f'<span class="chip">{e(b)}</span>' for b in detail.get("basis", []))
    if basis:
        basis = '<p class="eyebrow" style="margin-top:8px">Legal basis</p>' + basis
    based_on, why = wording.basis(case)
    if overridden:
        label = "Final verdict, set by the officer"
        explain = (f'<p class="prose"><b>Officer\'s reason.</b> {e(officer.get("reason", ""))}</p>'
                   f'<p class="prose">The draft was {e(VERDICT_WORDS.get(case.get("draft_verdict")))}: '
                   f'{e(based_on[:1].lower() + based_on[1:])}. {e(why)}</p>')
    else:
        explain = f'<p class="prose"><b>{e(based_on)}.</b> {e(why)}</p>'
    html(f'<div class="verdict {e(str(verdict))}"><p class="eyebrow">{label}</p>'
         f'<h3>{e(VERDICT_WORDS.get(verdict, "Not issued"))}</h3>{explain}{lists}{basis}</div>')


def render_budgets(case: dict[str, Any]) -> None:
    """How much of each limit the case used, in one line: the case review cannot loop or ask without end."""
    parts = [f"{case.get('steps', 0)} of {decide_rules.MAX_STEPS} review steps",
             f"{case.get('send_backs', 0)} of {decide_rules.MAX_SEND_BACKS} requests for more evidence",
             f"{len(case.get('rules', []))} of {decide_rules.MAX_RULE_QUESTIONS} legal questions",
             f"{case.get('reruns', 0)} of {supervisor.MAX_RERUNS} re-checks with new records"]
    html(f'<p class="note">Used: {e(", ".join(parts))}.</p>')


def rule_badges(rule: dict[str, Any]) -> str:
    """Where a rule answer came from and how it was checked."""
    badges = []
    if rule.get("point"):
        badges.append(f'<span class="badge">{e(wording.TOPIC.get(rule["point"], ""))}</span>')
    verification = rule.get("verification")
    words = {"passed": ("ok", "Claims checked"), "repaired": ("ok", "Claims checked after one rewrite"),
             "trimmed": ("warn", "Unsupported sentences removed"), "not_run": ("warn", "Claims not checked"),
             "failed": ("warn", "Claims failed the check")}
    if verification in words:
        style, text = words[verification]
        badges.append(f'<span class="badge {style}">{e(text)}</span>')
    if rule.get("status") == "no_rule_found":
        badges.append('<span class="badge warn">No answer in the sources</span>')
    if rule.get("cached"):
        badges.append(f'<span class="badge">Reused answer from {e(rule["cached"])}</span>')
    return "".join(badges)


def render_decisions(case: dict[str, Any]) -> None:
    rows = ""
    for d in case.get("decisions", []):
        extra = (f'<small>Adjusted by the system\'s limits: {e(plain(d["overruled"]))}</small>'
                 if d.get("overruled") else "")
        detail = f" Task: {plain(d['task'])}" if d.get("task") else ""
        detail += " Question: " + "; ".join(d["questions"]) if d.get("questions") else ""
        rows += (f'<div class="decision"><b>{e(wording.ACTION.get(d["action"], d["action"]))}.</b> '
                 f'{e(plain(d["reason"]))}{e(detail)}{extra}</div>')
    html('<p class="eyebrow" style="margin-top:14px">Case review decisions</p>'
         + (rows or '<p class="note">None yet.</p>'))


def render_officer(case: dict[str, Any], plot: PlotInput) -> None:
    """Approve, override or add documents. The case resumes from where it paused, in the saved checkpoint."""
    case_id = case["case_id"]
    html('<p class="eyebrow" style="margin-top:6px">Officer sign-off</p>')
    if case.get("status") == "closed":
        officer = case.get("officer", {})
        html(f'<p class="prose">Closed: {e(wording.DECISION.get(officer.get("decision"), ""))}'
             + (f' ({e(officer["reason"])})' if officer.get("reason") else "") + "</p>")
        return
    options = ["Approve the draft", "Override the verdict"]
    if case.get("reruns", 0) < supervisor.MAX_RERUNS:  # one re-check per case
        options.append(RECHECK)
    # outside a form, so only the fields for the chosen option are shown
    choice = st.radio("Decision", options, horizontal=True, key=f"decision_{case_id}")
    override, files = None, []
    if choice == "Approve the draft":
        reason = st.text_input("Note (optional)", key=f"reason_{case_id}")
    elif choice == "Override the verdict":
        override = st.selectbox("New verdict", [v for v in supervisor.VERDICTS if v != case.get("draft_verdict")],
                                key=f"override_{case_id}", format_func=lambda v: VERDICT_WORDS[v])
        reason = st.text_input("Reason (required)", key=f"reason_{case_id}")
    else:
        reason = ""
        files = st.file_uploader("New land records (.pdf, .md or .txt)", type=["pdf", "md", "txt"],
                                 accept_multiple_files=True, key=f"recheck_{case_id}",
                                 help="The case runs again with these records, once.")
    submitted = st.button({"Approve the draft": "Approve", "Override the verdict": "Override",
                           RECHECK: "Re-check"}[choice], type="primary", key=f"submit_{case_id}")
    if not submitted:
        return
    if choice == "Approve the draft":
        officer = {"decision": "approve", "reason": reason}
    elif choice == "Override the verdict":
        officer = {"decision": "override", "verdict": override, "reason": reason}
    else:
        if case.get("reruns", 0) >= supervisor.MAX_RERUNS:
            st.error("Not accepted: this case has already been re-checked once. Approve or override it.")
            return
        if not files:
            st.error("Not accepted: add at least one land record for the re-check.")
            return
        refused = []
        for file in files:
            try:
                documents.save_upload(plot["plot_id"], file.name, file.getvalue())
            except ValueError as error:
                refused.append(f"{file.name} was not added: {error}")
        if len(refused) == len(files):
            st.error("Not accepted: " + "; ".join(refused))
            return
        st.session_state.setdefault("refused_uploads", []).extend(refused)
        officer = {"decision": "documents_added", "documents": documents.with_uploads(plot)["documents"]}
    slot = st.empty()
    seen: list[dict[str, Any]] = []
    doing = {"approve": "Signing off", "override": "Recording the override",
             "documents_added": "Re-checking the case with the new land records: the input check screens them, "
                                "the evidence check reads them, and the verdict is drawn again"}[officer["decision"]]

    def show(event: dict[str, Any]) -> None:
        seen.append(event)
        slot.html(f'<p class="note">{e(doing)} · {len(seen)} steps so far</p>' + timeline_html(seen))

    try:
        with st.spinner(doing):
            after = supervisor.resume_case(case_graph(), case_id, officer, on_event=show)
    except ValueError as error:
        st.error(f"Not accepted: {error}")
        return
    st.session_state["flash"] = {"case_id": case_id, **outcome(case, after, officer)}
    st.session_state["_scroll_case"] = True
    st.rerun()


def _verdict_text(case: dict[str, Any], final: bool = False) -> str:
    return wording.verdict_line(case, final)


def outcome(before: dict[str, Any], after: dict[str, Any], officer: dict[str, Any]) -> dict[str, Any]:
    """What a sign-off or re-check changed, in words, for the message shown at the top of the case."""
    decision = officer["decision"]
    if decision == "approve":
        return {"kind": "success", "title": f"Signed off: draft approved. Final verdict {_verdict_text(after, True)}.",
                "lines": ["The case is closed. The evidence pack now holds your decision."]}
    if decision == "override":
        return {"kind": "success",
                "title": f"Signed off: verdict overridden to {_verdict_text(after, True)}.",
                "lines": [f"Your reason is on the record: {officer.get('reason', '')}",
                          "The case is closed. The evidence pack shows both the draft and your override."]}
    lines = []
    new_files = [Path(d).name for d in officer.get("documents", []) if d not in before["plot"].get("documents", [])]
    names = {n: n.removeprefix("uploaded_") for n in new_files}
    if names:
        lines.append("Added: " + ", ".join(names.values()) + ".")
    for line in after.get("guardrail", {}).get("dropped", []):
        if any(line.startswith(n) for n in names):
            lines.append("Held back by the input check, not shown to the agents: " + line.removeprefix("uploaded_"))
    fresh = after.get("findings", [])[len(before.get("findings", [])):]
    read = [f["values"].get("file", "") for f in fresh if f.get("tool") == "read_document" and not f.get("error")]
    if read:
        lines.append("Read during the evidence check: " + ", ".join(n.removeprefix("uploaded_") for n in read) + ".")
    elif names:
        lines.append("The evidence check did not open the new record this time.")
    old_c, new_c = set((before.get("verdict_detail") or {}).get("caveats", [])), \
        set((after.get("verdict_detail") or {}).get("caveats", []))
    lines += [f"Resolved: {c}" for c in sorted(old_c - new_c)] + [f"New caveat: {c}" for c in sorted(new_c - old_c)]
    was, now = _verdict_text(before), _verdict_text(after)
    title = (f"Re-check finished: the verdict changed from {was} to {now}." if was != now
             else f"Re-check finished: the verdict is still {now}.")
    lines.append("The case waits for your sign-off again. It cannot be re-checked a second time.")
    return {"kind": "info", "title": title, "lines": lines}


def render_flash(case: dict[str, Any]) -> None:
    flash = st.session_state.get("flash")
    if not flash or flash.get("case_id") != case.get("case_id"):
        return
    items = "".join(f"<li>{e(line)}</li>" for line in flash["lines"])
    html(f'<div class="flash {e(flash["kind"])}"><b>{e(flash["title"])}</b><ul>{items}</ul></div>')
    if st.button("Dismiss", key=f"dismiss_{case['case_id']}"):
        st.session_state.pop("flash")
        st.rerun()


def can_run(case: dict[str, Any]) -> tuple[bool, str]:
    """One case per plot. A plot runs again only if its last run did not produce a result (it stopped part way,
    or every evidence round failed), never just to get another answer. New records go through the sign-off's
    re-check (once); a changed boundary or declaration is a new submission."""
    if not case:
        return True, ""
    if case.get("status") == "running":
        return True, "the last run did not finish"
    rounds = case.get("geo_reports", [])
    if case.get("status") != "blocked" and rounds and all(r.get("error") for r in rounds) \
            and not any(not f.get("error") for f in case.get("findings", [])):
        return True, "the last run failed"
    return False, ""


def render_case(plot: PlotInput, state: dict[str, Any]) -> None:
    plot_id = plot["plot_id"]
    check = state["guardrail"]
    case = state["case"]
    html('<div id="case-top"></div>')
    if st.session_state.pop("_scroll_case", False):
        scroll_to("case-top")
    if case:
        render_flash(case)
    allowed, why = can_run(case)
    clicked = False
    if allowed:
        clicked = st.button(f"Run the check on {plot_id}" if not case else f"Run again: {why}", type="primary",
                            key=f"run_{plot_id}", disabled=not check["passed"])
    elif case.get("status") == "closed":
        html('<p class="note">This case is closed. To check a changed boundary or declaration, submit it as a new '
             'plot.</p>')
    elif case.get("status") != "blocked":
        recheck = ('To add land records, use "Re-check with new land records" in the sign-off. '
                   if case.get("reruns", 0) < supervisor.MAX_RERUNS else "")
        html(f'<p class="note">This plot has been checked. {recheck}To check a changed boundary or declaration, '
             'submit it as a new plot.</p>')
    if not settings.llm_enabled:
        html('<p class="note">The AI assistant is off (no key), so the evidence check cannot run and the case '
             'review uses its fixed checks only.</p>')
    slot = st.empty()  # the steps appear here live while a case runs
    if st.session_state.get("_autorun") == plot_id:  # a submission runs as soon as it is submitted
        st.session_state.pop("_autorun")
        clicked = True
    if clicked:
        seen: list[dict[str, Any]] = []
        started = time.monotonic()

        def show(event: dict[str, Any]) -> None:
            seen.append(event)
            took = (event.get("ms") or 0) / 1000
            slow = " (slow step: calls give up after their timeout and are retried)" if took > 60 else ""
            slot.html(f'<p class="note">Running for {time.monotonic() - started:.0f} s · {len(seen)} steps · '
                      f'last step {took:.0f} s{e(slow)}</p>' + timeline_html(seen))

        with st.spinner("Running the case: input check, evidence check, case review, legal check, verdict"):
            new = supervisor.start_case(case_graph(), plot, on_event=show)
        desk(plot_id)["case_id"] = new["case_id"]
        if new.get("status") == "blocked":
            flash = {"kind": "error", "title": "The input check stopped this plot. Nothing else was run.",
                     "lines": [plain(b)[:1].upper() + plain(b)[1:] + "." for b in new["guardrail"]["blocked"]]}
        else:
            flash = {"kind": "info", "title": f"The case ran: draft verdict {_verdict_text(new)}.",
                     "lines": ["It now waits for your sign-off below: approve, override, or re-check with new "
                               "land records."]}
        st.session_state["flash"] = {"case_id": new["case_id"], **flash}
        st.session_state["_scroll_case"] = True
        st.rerun()
    if not case:
        slot.html('<p class="note">No case yet. Run one: the evidence check picks its own checks, the case review '
                  'looks over them, and the case stops here for your decision.</p>')
        return
    status = {"awaiting_officer": "waiting for your sign-off", "closed": "closed", "blocked": "stopped at the input "
              "check", "running": "running"}.get(case.get("status", ""), case.get("status", ""))
    html(f'<p class="note">Case {e(case["case_id"])} · {e(status)}</p>')

    # 1. the verdict and 2. the sign-off: what the officer is here for
    if case.get("status") == "blocked":
        blocked = case["guardrail"]["blocked"]
        extra = (" This system screens coffee plots in Karnataka only, so this plot needs a manual review."
                 if any("Karnataka" in b for b in blocked) else "")
        st.error("The input check stopped this plot: " + "; ".join(plain(b)[:1].lower() + plain(b)[1:] for b in blocked) + "." + extra)
    elif case.get("verdict_note"):
        render_verdict(case)
        render_officer(case, plot)
    # 3. the evidence, one line per check
    if case.get("status") != "blocked":
        render_evidence(case)
        for r in case.get("geo_reports", []):
            if r.get("error"):
                st.error(f"Evidence check round {r['round']} failed: {plain(r['error'])}")
    # 4. the legal answers and how the case ran, folded away
    if case.get("rules"):
        with st.expander(f"Legal questions asked during the case ({len(case['rules'])})"):
            html('<p class="note">One per legal topic. Each answer must rest on a legal source, and its claims are '
                 'checked against the passages it cites.</p>')
            for index, rule in enumerate(case["rules"]):
                html(rule_badges(rule))
                render_rule_result({**rule, "events": [], "trace": f"{case['case_id']}.jsonl"}, index)
    events = case_events(case["case_id"])
    with st.expander(f"How the case ran ({wording.count(len(events), 'step')})"):
        render_budgets(case)
        render_decisions(case)
        html('<p class="eyebrow" style="margin-top:14px">Every step</p>' + timeline_html(events))
        link = observability.run_url((case.get("langsmith_runs") or [None])[-1])
        if link:
            html(f'<p><a class="ls" href="{e(link)}" target="_blank">Open the technical trace (LangSmith)</a></p>')


def render_submission_form() -> None:
    """A new plot: boundary file, declared details and land records. Saved, then run straight away."""
    html('<p class="eyebrow">New submission</p>')
    html('<p class="note">Upload the plot boundary and land records, and enter the declared details. The check '
         'starts as soon as you submit. Your files are used only for this check, and are kept with the case on '
         'this system so it can be reviewed and signed off.</p>')
    with st.form("new_submission", clear_on_submit=True):
        boundary = st.file_uploader("Plot boundary", type=["kml", "geojson", "json"], key="sub_boundary",
                                    help="A KML file from Google Earth, or GeoJSON, with one polygon.")
        reference = st.text_input("Reference (optional)", key="sub_reference", max_chars=120,
                                  help="For your records. It is not used in the check.")
        one, two, three = st.columns(3)
        crop = one.selectbox("Declared crop", submissions.CROPS, key="sub_crop")
        since = two.number_input("Crop since (year)", min_value=1900, max_value=2100, value=None, step=1,
                                 key="sub_since", placeholder="not stated")
        area = three.number_input("Declared area (ha, optional)", min_value=0.0, value=None, step=0.1,
                                  key="sub_area", placeholder="from the polygon")
        records = st.file_uploader("Land records", type=["pdf", "md", "txt"], help="PDF or text files.",
                                   accept_multiple_files=True, key="sub_records")
        sent = st.form_submit_button("Submit and run the check", type="primary")
    if not sent:
        return
    if boundary is None:
        st.error("Add the plot boundary file.")
        return
    try:
        plot_id, refused = submissions.create(
            boundary.name, boundary.getvalue(), reference=reference or "", declared_crop=crop,
            crop_since=int(since) if since is not None else None, declared_area_ha=area or None,
            records=[(f.name, f.getvalue()) for f in records or []])
    except ValueError as error:
        st.error(f"The boundary file was not accepted: {error}")
        return
    st.session_state.setdefault("refused_uploads", []).extend(f"{line} (not added)" for line in refused)
    st.session_state["_goto"] = plot_id
    st.rerun()


def render_open_case(plot_id: str) -> None:
    """Earlier runs of this plot. Every click of Run makes a new case with its own id; this switches between them."""
    with st.expander("Earlier runs of this plot"):
        _earlier_runs(plot_id)
    if st.session_state.get("open_error"):
        st.error(st.session_state.pop("open_error"))


def _earlier_runs(plot_id: str) -> None:
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

        st.selectbox("Pick a run", ["—", *saved], key=key, on_change=pick,
                     help="Case ids are the plot id, the date and time of the run, and a short code.")
    else:
        html('<p class="note">No earlier runs.</p>')
    with st.form(f"open_{plot_id}", clear_on_submit=True, border=False):
        typed = st.text_input("Or open by case id", placeholder=f"{plot_id}-20261109-...")
        if st.form_submit_button("Open") and typed.strip():
            found = load_case(typed.strip())
            if not found:
                st.session_state["open_error"] = f"No saved case has the id \"{typed.strip()}\"."
            elif found["plot"]["plot_id"] != plot_id:
                st.session_state["open_error"] = f"Case {typed.strip()} is for plot {found['plot']['plot_id']}"
            else:
                desk(plot_id)["case_id"] = typed.strip()
            st.rerun()


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
    loop = f'<span class="loopflag">Searched again {rewrites}x</span>' if rewrites else ""
    sources = check.get("sources") or [{"ref": ref, "kind": ""} for ref in check["citations"]]
    chips = "".join(f'<span class="chip {e(s["kind"])}" title="{e(KINDS.get(s["kind"], ""))}">{e(s["ref"])}'
                    + (f' <small>· {e(KINDS_SHORT[s["kind"]])}</small>' if s["kind"] in KINDS_SHORT else "")
                    + '</span>' for s in sources)
    body = check.get("answer") or check.get("error", "")
    html(f'<div class="card"><p class="question">{e(check["question"])}</p><p class="prose">{e(body)}</p>{chips} {loop}</div>')
    if check.get("events"):
        with st.expander("Show how this answer was found", expanded=bool(rewrites) and index == 0):
            html(timeline_html(check["events"]))
            html(f'<p class="note">Record: runs/traces/{e(check["trace"])}</p>')



def render_rules(plot: PlotInput, state: dict[str, Any]) -> None:
    plot_id = plot["plot_id"]
    html('<p class="note">Ask a legal question directly, outside the case. Each answer must cite at least one '
         'legal source (EU law, Commission guidance or Indian law), or no answer is given. Each citation shows how '
         'binding it is. The case asks its own questions; these are extra.</p>')
    question = None
    for column, preset in zip(st.columns(len(RULE_QUESTIONS)), RULE_QUESTIONS):
        with column:
            if st.button(preset, key=f"preset_{plot_id}_{preset}", width="stretch"):
                question = preset
    with st.form(f"rule_{plot_id}", clear_on_submit=True, border=False):
        typed = st.text_input("Or ask your own question", placeholder="Type a question about the regulation")
        if st.form_submit_button("Ask") and typed.strip():
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
    geo_line = ("Failed, and the error is recorded" if geo.get("error") else f"{wording.count(len({f['tool'] for f in geo['findings']}), 'check')}, with reasons") \
        if geo else "Not run, and the pack says so"
    needed = "Listed" if case.get("draft_verdict") == "needs_evidence" else "Nothing further" if case.get(
        "verdict_note") else "Not applicable yet"
    signed = wording.DECISION.get((case.get("officer") or {}).get("decision"), "Not signed off yet")
    html(f"""<div class="card"><h4>The {e(plot_id)} evidence pack</h4>
    <p class="note">A PDF document, in this order. It can be downloaded at any stage and says what was not done.</p>
    <ul class="packlist">
    <li>Cover: verdict and a short summary<span>{e(verdict)}</span></li>
    <li>1. What was submitted, and the input check<span>{e(wording.count(len(plot.get('documents', [])), 'land record'))}</span></li>
    <li>2. What happened, step by step<span>{e(wording.count(len(case_events(case['case_id'])) if case else 0, 'step'))} recorded</span></li>
    <li>3. Evidence in detail<span>{e(geo_line)}</span></li>
    <li>4. Legal questions and answers<span>{e(wording.count(len(checks), 'question') if checks else 'None asked')}</span></li>
    <li>5. Verdict and why<span>{e(VERDICT_WORDS.get(case.get('draft_verdict'), 'Not issued'))}</span></li>
    <li>6. What is still needed<span>{e(needed)}</span></li>
    <li>7. Officer decision and note<span>{e(signed)}</span></li>
    <li>Appendices: land records, data sources, audit trail<span>Included</span></li></ul></div>""")
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

    html('<p class="eyebrow">How a case travels</p><p class="prose">Input check (stops plots and records that '
         'should not be screened) → evidence check (an AI assistant picks the satellite and land-record checks) ⇄ '
         'case review (looks over the evidence, asks for more once at most, asks one legal question per topic) → '
         'legal check (searches EU law, Commission guidance and Karnataka law; cites a legal source; checks its '
         'claims) → draft verdict (fixed, published criteria, never the AI assistant) → officer sign-off → '
         'evidence pack. Every step is saved, so a case survives a restart.</p>')
    html('<p class="eyebrow">Running on</p>' + _table(["Part", "Setting"], [
        ["AI assistant", settings.llm_model if settings.llm_enabled else "off (fixed checks and keyword search only)"],
        ["Satellite data", "Google Earth Engine, saved per plot"],
        ["Search", f"{settings.retrieval} keyword + vector, two shelves, "
                   + (f"reranked by {rerank.MODEL}" if rag_index.RERANK and rerank.enabled() else "no reranker")],
        ["Saved cases", "Postgres" if os.getenv("DATABASE_URL") else "SQLite (runs/checkpoints.sqlite)"],
        ["Traces", f"LangSmith project {observability.project()}" if observability.enabled()
                   else "local JSONL only (runs/traces/)"],
        ["Case review limits", f"{decide_rules.MAX_STEPS} steps, {decide_rules.MAX_SEND_BACKS} request for more "
                               f"evidence, {decide_rules.MAX_RULE_QUESTIONS} legal questions per case"],
    ]))

    end_to_end = _latest("end_to_end_*.csv")
    if end_to_end:
        import csv as _csv
        rows = list(_csv.DictReader(end_to_end.open()))
        html(f'<p class="eyebrow">End to end ({e(end_to_end.stem[-10:])}): draft verdicts on the test plots</p>'
             f'<p class="note">Compared with a human label and with FAO Whisp. {len(rows)} plots show the system works; '
             'they are too few to claim accuracy.</p>'
             + _table(["Plot", "Label", "FAO Whisp", "Ours", "Based on", "vs label", "Seconds"],
                      [[r["plot_id"], VERDICT_WORDS.get(r["label"], r["label"]), wording.WHISP.get(r["whisp"], r["whisp"] or "Not run"),
                        VERDICT_WORDS.get(r["ours"], r["ours"]) if r["ours"] else "Stopped at input check",
                        wording.BASIS.get(int(r["rule"]), ("",))[0] if r["rule"] not in ("", "None") else "",
                        r["agreement"], r["seconds"]] for r in rows]))
    quality = _latest("rag_quality_2026-*.json")
    if quality:
        data = json.loads(quality.read_text())
        answers = [a for a in data.get("answers", []) if not a.get("out_of_scope") and a["status"] == "answered"]

        def mean(key: str) -> str:
            vals = [a[key] for a in answers if a.get(key) is not None]
            return f"{sum(vals) / len(vals):.2f}" if vals else "-"
        retrieval = [r for r in data.get("retrieval", []) if r["variant"].startswith("rerank: keep")]
        html(f'<p class="eyebrow">Legal check quality ({e(quality.stem[-10:])}, {data["questions"]} questions)</p>'
             + _table(["Metric", "Score"], [
                 ["Faithfulness (claims backed by the passages)", mean("faithfulness")],
                 ["Citation precision", mean("citation_precision")],
                 ["Context precision", mean("context_precision")],
                 ["Answer relevance", mean("answer_relevance")],
                 ["Abstained when it should", f"{sum(a['abstain_ok'] for a in data.get('answers', []))}/"
                                              f"{len(data.get('answers', []))}"]])
             + (_table(["Question set", "hit@8", "MRR", "nDCG@8"],
                       [[plain(r["set"]), f"{r['hit']:.2f}", f"{r['mrr']:.2f}", f"{r['ndcg']:.2f}"] for r in retrieval])
                if retrieval else ""))


# ---------- page ----------

def main() -> None:
    st.set_page_config(page_title="Canopy Check", page_icon="🌱", layout="wide")
    st.html(CSS)
    if "home" not in st.session_state:  # a fresh visit opens on the landing page (tests may open a section)
        st.session_state["home"] = "source" not in st.session_state
    if st.session_state["home"]:
        render_home()
        return
    goto = st.session_state.pop("_goto", None)
    if goto:  # just submitted: show it below the form, scroll to it, and run it once
        st.session_state.update(_current_submission=goto, _autorun=goto, _scroll=True, view="Case")
    render_header()
    source = st.session_state["source"]
    if source == "New submission":
        render_submission_form()
        current = st.session_state.get("_current_submission")
        plots = submissions.load_submissions()
        if current in plots:
            html('<div id="submission-case"></div>')
            html(f'<p class="eyebrow" style="margin-top:24px">Submission {e(current)}</p>')
            if st.session_state.pop("_scroll", False):
                scroll_to("submission-case")
            render_workspace(plots, current)
        return
    if source == "Submitted plots":
        plots = submissions.load_submissions()
        if not plots:
            st.info("No submissions yet. Use New submission to upload a plot.")
            return
        plot_id = st.selectbox("Plot", list(plots), key="sub_plot_id",
                               format_func=lambda p: f"{p}  {submissions.details(p).get('reference', '')}".strip())
    else:
        plots = load_plots()
        plot_id = st.selectbox("Plot", sorted(plots), key="plot_id",
                               format_func=lambda p: f"{p}  ({plots[p].get('area_ha', '?')} ha)")
    render_workspace(plots, plot_id)


def scroll_to(anchor: str) -> None:
    """Scroll the page to an element, e.g. the case after a run or sign-off. `anchor` is always a fixed id from
    this file, never user or model text: the frame may run script against the page."""
    st.iframe(f"""<script>
      const go = () => {{ const el = window.parent.document.getElementById("{anchor}");
                         if (el) el.scrollIntoView({{behavior: "smooth", block: "start"}}); else setTimeout(go, 150); }};
      go();</script>""", height=1)


def render_workspace(plots: dict[str, PlotInput], plot_id: str) -> None:
    """One plot: the pipeline strip, the plot and its records on the left, the case and its views on the right."""
    state = desk(plot_id)
    if not state["case_id"]:  # a plot already checked opens on its case, so it is not run again
        try:
            latest = list_case_ids(case_graph().checkpointer, f"{plot_id}-", limit=1)
        except Exception:
            latest = []
        state["case_id"] = latest[0] if latest else ""
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
        views = {"Case": render_case, "Legal questions": render_rules, "Evidence pack": render_pack,
                 "How it works": render_system}
        view = st.radio("View", list(views), horizontal=True, key="view", label_visibility="collapsed")
        views[view](plot, state)


if __name__ == "__main__":
    main()
