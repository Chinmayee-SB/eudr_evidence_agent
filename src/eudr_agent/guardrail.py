"""Input guardrail: the first stop for every plot, before any agent or LLM sees it.

Fixed rules, no LLM, so the same input always gets the same answer. Three outcomes:
  block  - the plot cannot be screened; no agent runs and the officer is told why
  drop   - one document is unsafe or unreadable; it is removed from the case, the rest goes on
  warn   - the plot can be screened, but a gap is carried forward for the officer and the reviewer

Why each rule exists:
  plot_id      the ID goes into the LLM prompt and the evidence pack, so it must be an anonymised code
               (e.g. "K-01"): no real estate names, and no room for text that reads as an instruction
  geometry     EUDR Art. 2(28) geolocation rules, reusing the geo tool's check
  location     every corner must lie in Karnataka; the datasets notes and test plots only cover it
  crop         the agent is built and evaluated for coffee only
  crop_since   if stated, it must be a real past year
  documents    only files in the permitted folder, readable as text, and not containing instructions
               aimed at the agent (a prompt-injection screen; it catches obvious attempts only)
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import TypedDict

from eudr_agent import documents
from eudr_agent.config import PLOTS_DIR
from eudr_agent.geo import tools as geo_tools
from eudr_agent.state import PlotInput
from eudr_agent.tracing import Tracer

PLOT_ID = re.compile(r"^[A-Z]{1,3}-\d{2,4}$")
# Karnataka's extent, rounded outwards. A plot outside it is out of scope, not wrong.
KARNATAKA = {"lon": (74.0, 78.6), "lat": (11.5, 18.5)}
IN_SCOPE_CROPS = {"coffee"}
EARLIEST_CROP_YEAR = 1900
AREA_MISMATCH_WARN = 0.10  # stated vs computed area; the test polygons are approximate, so warn, don't block
DOCUMENTS_DIR = PLOTS_DIR / "documents"

# Phrases that address the agent rather than describe land. A land record never needs them.
INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|messages)",
    r"disregard (all |any )?(the )?(previous|prior|above|your) (instructions|rules)",
    r"\byou are (now )?(an?|the) (ai|assistant|agent|model)\b",
    r"\b(system|developer) prompt\b",
    r"\b(mark|classify|label|report) (this|the) plot as\b",
    r"\b(do not|don't) (run|call|use|check)\b.*\b(tool|check|satellite)",
]
INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


class GuardrailResult(TypedDict):
    passed: bool              # False means: do not run any agent
    blocked: list[str]        # why the plot cannot be screened
    dropped: list[str]        # documents removed from the case, with the reason
    warnings: list[str]       # gaps carried forward to the officer and reviewer
    plot: PlotInput           # the plot the agents may see: same input, minus dropped documents


def _check_plot_id(plot: PlotInput) -> list[str]:
    plot_id = plot.get("plot_id")
    if not isinstance(plot_id, str) or not PLOT_ID.match(plot_id):
        return ["plot ID must be an anonymised code such as K-01 (letters, a dash, digits)"]
    return []


def _corners(geometry: dict) -> list[list[float]]:
    coords = geometry.get("coordinates")
    return [coords] if geometry.get("type") == "Point" else coords[0]


def _check_geometry(plot: PlotInput) -> tuple[list[str], list[str]]:
    geometry = plot.get("geometry")
    if not isinstance(geometry, dict):
        return ["the plot has no geometry"], []
    check = geo_tools.validate_geometry(geometry, plot.get("area_ha"))["values"]
    if not check["valid"]:
        return [f"geometry: {reason}" for reason in check["reasons"]], []

    blocked, warnings = [], []
    (lon_min, lon_max), (lat_min, lat_max) = KARNATAKA["lon"], KARNATAKA["lat"]
    if not all(lon_min <= lon <= lon_max and lat_min <= lat <= lat_max for lon, lat, *_ in _corners(geometry)):
        blocked.append("the plot is outside Karnataka, the only region this screening covers")

    computed, stated = check.get("computed_area_ha"), plot.get("area_ha")
    if computed is not None and computed <= 0:
        blocked.append("the polygon has no area")
    elif computed and stated and abs(stated - computed) / computed > AREA_MISMATCH_WARN:
        warnings.append(f"stated area {stated} ha differs from the polygon's {computed} ha by more than "
                        f"{AREA_MISMATCH_WARN:.0%}; the boundary may not match the plot")
    return blocked, warnings


def _check_crop(plot: PlotInput) -> tuple[list[str], list[str]]:
    blocked, warnings = [], []
    crop = str(plot.get("declared_crop", "")).strip().lower()
    if crop not in IN_SCOPE_CROPS:
        blocked.append(f"declared crop is '{crop or 'not stated'}'; this agent only screens coffee")

    since = plot.get("crop_since")
    this_year = datetime.date.today().year
    if since is None:
        warnings.append("no 'crop since' year is stated, so land use before 2020 must come from documents")
    elif isinstance(since, bool) or not isinstance(since, int) or not EARLIEST_CROP_YEAR <= since <= this_year:
        blocked.append(f"'crop since' must be a year between {EARLIEST_CROP_YEAR} and {this_year}, not {since!r}")
    return blocked, warnings


def _check_document(path: str) -> str | None:
    """The reason to drop this document, or None if it may go to the agent."""
    resolved = Path(path).resolve()
    if DOCUMENTS_DIR.resolve() not in resolved.parents:
        return "it is outside the permitted documents folder"
    if not resolved.exists():
        return "the file does not exist"
    try:
        text = documents.read_text(resolved)
    except ValueError as error:
        return str(error)
    if match := INJECTION.search(text):
        return f"it contains instructions to the screening system ('{match.group(0)}'), not land-record content"
    return None


def check_input(plot: PlotInput, tracer: Tracer | None = None) -> GuardrailResult:
    """Check one plot before the pipeline starts. Returns the verdict on the input and the plot agents may see."""
    blocked = _check_plot_id(plot)
    geometry_blocked, warnings = _check_geometry(plot)
    crop_blocked, crop_warnings = _check_crop(plot)
    blocked += geometry_blocked + crop_blocked
    warnings += crop_warnings

    kept, dropped = [], []
    for path in plot.get("documents", []):
        if reason := _check_document(path):
            dropped.append(f"{Path(path).name}: {reason}")
        else:
            kept.append(path)
    if not kept:
        warnings.append("no usable land record is on the case")

    result = GuardrailResult(passed=not blocked, blocked=blocked, dropped=dropped, warnings=warnings,
                             plot=PlotInput(**{**plot, "documents": kept}))
    if tracer:
        with tracer.span("input_guardrail") as rec:
            rec.update(plot_id=plot.get("plot_id"), passed=result["passed"], blocked=blocked,
                       dropped=dropped, warnings=warnings)
            rec["summary"] = ("passed" if result["passed"] else f"blocked: {'; '.join(blocked)}") + \
                (f", {len(dropped)} document(s) dropped" if dropped else "") + \
                (f", {len(warnings)} warning(s)" if warnings else "")
    return result
