"""The words the officer sees: one place, so the desk and the evidence pack say the same thing.

Inside the code, a case has a guardrail, a geo-analyst, a supervisor, a rules agent and seven numbered verdict
rules. None of that means anything to an officer, so every screen and the PDF use the names below instead, and
pass free text (reasons written by the AI assistant, trace summaries) through `plain()` to catch what slips in.
"""

from __future__ import annotations

import re
from typing import Any

VERDICT = {"clear": "Clear", "needs_evidence": "Needs evidence", "high_risk": "High risk", None: "Not issued"}

# the stages of a case, in order
STAGE = {"input": "Input check", "geo": "Evidence check", "supervisor": "Case review", "rules": "Legal check",
         "verdict": "Verdict", "officer": "Officer"}

CHECK = {
    "validate_geometry": "Boundary shape",
    "protected_area_check": "Protected areas",
    "forest_2020": "Forest in 2020",
    "loss_after_2020": "Tree loss after 2020",
    "locate_loss": "Where the loss is",
    "boundary_sensitivity": "Boundary moved 30 m",
    "ndvi_series": "Greenness by year",
    "coffee_probability": "Coffee map",
    "read_document": "Land record",
    "run_whisp": "FAO Whisp",
}

# what decided the verdict: a short label, and one sentence anyone can follow
BASIS = {
    1: ("Inside a protected forest area",
        "The plot overlaps a protected forest area that is in force. Farming there is not allowed, whatever the "
        "satellite images show."),
    2: ("A required check could not be completed",
        "A check this decision depends on did not return a result, so the plot cannot be cleared yet."),
    3: ("Not forest in 2020",
        "Forest maps show the land was not forest at the end of 2020, so it cannot have been deforested since."),
    4: ("No forest lost since 2020",
        "Satellite maps show no forest was cleared on this plot after 2020."),
    5: ("Forest cleared after 2020",
        "Two independent satellite sources show more than half a hectare of forest was cleared after 2020."),
    6: ("Only small tree loss on an older coffee farm",
        "Only small patches of trees were lost, which is typical of pruning or shade-tree work on a coffee farm "
        "that existed before 2020."),
    7: ("The evidence is not conclusive",
        "The satellite sources disagree, or the result changes when the plot boundary is moved slightly, so a "
        "person needs to look."),
    0: ("No standard check settles it",
        "None of the standard checks settles this case, so a person needs to look."),
}

ACTION = {"send_back": "Asked for more evidence", "ask_rules": "Asked a legal question",
          "finish": "Evidence complete"}

TOPIC = {"A": "Is coffee under trees forest?", "B": "What counts as deforestation after 2020?",
         "C": "What a protected area means", "D": "Location and document requirements"}

# data sets by their published names, for the pack's "Data source" lines
DATASET = [
    ("UMD/hansen/global_forest_change", "Global Forest Watch tree cover and loss (Hansen et al., University of Maryland)"),
    ("projects/JRC/TMF", "EU tropical moist forest monitor (JRC TMF)"),
    ("JRC/GFC2020", "EU forest map 2020 (JRC GFC2020)"),
    ("WCMC/WDPA", "World Database on Protected Areas (UNEP-WCMC)"),
    ("COPERNICUS/S2", "Sentinel-2 satellite images (Copernicus)"),
    ("GOOGLE/CLOUD_SCORE_PLUS", ""),
    ("projects/forestdatapartnership", "Forest Data Partnership coffee model"),
]


def dataset_names(ids: str) -> str:
    """'UMD/hansen/...; projects/JRC/TMF/...' as published names, each once, in order."""
    names = []
    for part in str(ids or "").split(";"):
        part = part.strip()
        name = next((n for prefix, n in DATASET if part.startswith(prefix)), part)
        if name and name not in names:
            names.append(name)
    return "; ".join(names)


WHISP = {"low": "Low risk", "high": "High risk", "more_info_needed": "More information needed"}


def count(n: int, word: str, plural: str = "") -> str:
    """'1 question', '3 questions'."""
    return f"{n} {word if n == 1 else plural or word + 's'}"


def boundary_change(change: str) -> str:
    """A boundary-check finding ('whether any loss is found') as a sentence an officer can read."""
    change = plain(change).rstrip(".")
    if change.startswith("whether"):
        return f"Moving the boundary 30 m in or out changes {change}."
    return change[:1].upper() + change[1:] + "."


DECISION = {"approve": "Approved the draft", "override": "Overrode the draft",
            "documents_added": "Re-checked with new land records"}


def basis(case: dict[str, Any]) -> tuple[str, str]:
    """(label, sentence) for what decided this case's verdict, or ("", "") before there is one."""
    detail = case.get("verdict_detail") or {}
    if "rule" not in detail:
        return "", ""
    return BASIS.get(detail["rule"], BASIS[0])


def verdict_line(case: dict[str, Any], final: bool = False) -> str:
    """'Clear: no forest lost since 2020', for messages and headings."""
    word = VERDICT.get(case.get("final_verdict" if final else "draft_verdict"), "Not issued")
    label = basis(case)[0]
    return f"{word}: {label[0].lower() + label[1:]}" if label and not final else word


# free text from the agents and the trace: internal names to plain words
_REPLACE = [
    (r"\brule \d+: ", ""), (r"\s*\(rule \d+(: [^)]*)?\)", ""), (r"\[(llm|code)\]:?", ""),
    (r"\bsend[_ -]backs?\b", "request for more evidence"), (r"\bask_rules\b", "asked a legal question"),
    (r"\bneeds_evidence\b", "needs evidence"), (r"\bhigh_risk\b", "high risk"),
    (r"\bgeo[- ]analyst\b", "evidence check"), (r"\bGeo[- ]analyst\b", "Evidence check"),
    (r"\bgeo task\b", "evidence task"), (r"\bcode review\b", "evidence review"),
    (r"\bsupervisor\b", "case review"), (r"\bSupervisor\b", "Case review"),
    (r"\brules agent\b", "legal check"), (r"\bRules agent\b", "Legal check"),
    (r"\b(input )?guardrail\b", "input check"),
    (r"\bLLM\b", "AI assistant"),
    # data sources by their public names (the figures' own labels already use these)
    (r"\bJRC GFC ?2020\b", "the EU forest map"), (r"\bGFC ?2020\b", "the EU forest map"),
    (r"\bJRC TMF\b", "the EU tropical forest monitor"), (r"\bTMF\b", "the EU tropical forest monitor"),
    (r"\bHansen(’s|'s)?( GFC)?\b", "Global Forest Watch"), (r"\b(WCMC/)?WDPA\b", "the protected-areas database"),
    (r"\b(the|The) (the|The) ", r"\1 "), (r"\b([Aa])n? the EU\b", r"\1n EU"), (r"\b([Aa])n? the (?=\w)", r"\1 "),
    # boundary-check findings saved before boundary_change() existed
    (r"The result depends on where the boundary is drawn: whether", "Moving the boundary 30 m in or out changes whether"),
    (r"The result depends on where the boundary is drawn: most", "Most"),
    (r"\btools\b", "checks"), (r"\btool\b", "check"), (r"\bagents\b", "checks"), (r"\bagent\b", "system"),
]


def plain(text: Any) -> str:
    """Free text with internal names (tools, rule numbers, agent names) replaced by the officer's words."""
    out = str(text or "")
    for name, label in CHECK.items():
        out = re.sub(rf"\b{name}\b", label, out)
    for pattern, repl in _REPLACE:
        out = re.sub(pattern, repl, out)
    out = re.sub(r"the the\b", "the", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    # a sentence that now starts with a lower-case "the" (e.g. "the EU forest map has ...")
    return re.sub(r"(^|[.!?]\s+)the\b", lambda m: m.group(1) + "The", out)
