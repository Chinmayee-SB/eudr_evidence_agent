"""The evidence pack: one PDF holding everything known about a plot, whatever stage the case reached.

The pack reports tool numbers as returned and marks what the LLM wrote. It issues no verdict, because the
verdict rules are not built yet, and says so. Uploaded PDFs are added unchanged after the report.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from pypdf import PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from eudr_agent import documents
from eudr_agent.geo import tools
from eudr_agent.state import Finding, PlotInput

TOOL_LABELS = {
    "protected_area_check": "Protected areas",
    "loss_after_2020": "Loss after 2020",
    "forest_2020": "Forest in 2020",
    "read_document": "Land record",
    "locate_loss": "Where the loss is",
    "boundary_sensitivity": "Boundary shift",
    "ndvi_series": "Greenness by year",
    "coffee_probability": "Coffee model",
}
NO_VERDICT = ("No verdict issued. The supervisor review, verdict rules and officer sign-off are not built yet, "
              "so this pack contains evidence only.")
ADVISORY = "Advisory screening. A human officer makes every compliance decision."

BLACK = colors.HexColor("#060906")
LEAF = colors.HexColor("#1F6B3A")
PALE = colors.HexColor("#EEF5E8")
LINE = colors.HexColor("#C4D4BC")
MUTED = colors.HexColor("#4C5D50")

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=BLACK)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=11, textColor=MUTED)
HEADING = ParagraphStyle("heading", parent=BODY, fontName="Helvetica-Bold", fontSize=12.5, leading=16,
                         textColor=LEAF, spaceBefore=12, spaceAfter=5)
LABEL = ParagraphStyle("label", parent=BODY, fontName="Helvetica-Bold")


def _ha(value: Any) -> str:
    return f"{value} ha"


def _areas(rows: list[dict[str, Any]]) -> str:
    return "; ".join(f"{r['name']} ({r['designation']}, {r['status']}): {r['overlap_ha']} ha" for r in rows) or "None"


def _by_year(values: dict[str, float]) -> str:
    return ", ".join(f"{year}: {ha} ha" for year, ha in sorted(values.items())) or "none"


def key_figures(finding: Finding) -> list[tuple[str, str]]:
    """A finding's numbers as label and value pairs, shared by the screen and the PDF."""
    if finding.get("error"):
        return [("Error", finding["error"])]
    v = finding.get("values", {})
    tool = finding["tool"]
    if tool == "protected_area_check":
        return [("In force", _areas(v["in_force"])), ("Proposed only", _areas(v["proposed"]))]
    if tool == "forest_2020":
        return [("JRC GFC2020 forest", f"{v['gfc2020_forest_ha']} ha ({v['gfc2020_forest_share']:.0%})"),
                ("Hansen tree cover (proxy)", f"{v['hansen_tree_cover_ha']} ha ({v['hansen_tree_cover_share']:.0%})"),
                ("Plot area", _ha(v["plot_ha"]))]
    if tool == "loss_after_2020":
        return [("Hansen loss", f"{v['hansen_loss_ha']} ha ({_by_year(v['hansen_loss_by_year'])})"),
                ("TMF deforestation", f"{v['tmf_deforestation_ha']} ha ({_by_year(v['tmf_deforestation_by_year'])})"),
                ("TMF degradation", f"{v['tmf_degradation_ha']} ha ({_by_year(v['tmf_degradation_by_year'])})")]
    if tool == "read_document":
        return [("File", v["file"]), ("Status", "Mock document, not a real record")]
    if tool == "locate_loss":
        rows = [("Pattern", f"{v['pattern']}: {v['patch_count']} patch(es), largest {_ha(v['largest_patch_ha'])}"),
                ("Loss at the edge", f"{v['loss_at_edge_ha']} ha of {v['loss_ha']} ha (within 30 m)"),
                ("Loss on 2020 forest", _ha(v["loss_in_gfc2020_forest_ha"]))]
        return rows + [(f"Patch {i}", f"{p['area_ha']} ha, {p['first_year']}-{p['last_year']}, "
                                      f"{p['distance_to_edge_m']} m from edge") for i, p in enumerate(v["patches"], 1)]
    if tool == "boundary_sensitivity":
        rows = [("Result", "Stable" if v["stable"] else "Changes: " + "; ".join(v["changes"]))]
        return rows + [(name.replace("_", " ").capitalize(),
                        f"{x['plot_ha']} ha plot, {x['loss_ha']} ha loss, {x['gfc2020_forest_share']:.0%} forest")
                       for name, x in v["variants"].items()]
    if tool == "ndvi_series":
        rows = [("Plot", ", ".join(f"{y}: {d['ndvi']}" for y, d in v["plot"].items()))]
        if v["loss_pixels"]:
            rows.append(("Loss pixels", ", ".join(f"{y}: {n}" for y, n in v["loss_pixels"].items())))
        return rows + [("Season", v["season"])]
    if tool == "coffee_probability":
        return [(f"{year}", f"mean {d['mean_probability']}, {d['share_above_threshold']:.0%} above {v['threshold']}")
                for year, d in v["by_year"].items()]
    return [(key, str(value)) for key, value in v.items()]


def _clean(text: Any) -> str:
    """Make text safe for the PDF's built-in font, which covers Western characters only."""
    plain = str(text).encode("cp1252", "replace").decode("cp1252")
    return escape(plain)


def _para(text: Any, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(_clean(text).replace("\n", "<br/>"), style)


def _table(rows: list[list[Any]], widths: list[float], header: bool = False) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
             ("LEFTPADDING", (0, 0), (-1, -1), 5)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), PALE))
    table.setStyle(TableStyle(style))
    return table


def _bullets(items: list[str], empty: str) -> list[Any]:
    return [_para(f"- {item}") for item in items] or [_para(empty)]


def _on_page(canvas: Any, doc: Any) -> None:
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(BLACK)
    canvas.rect(0, height - 17 * mm, width, 17 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#A9CF7C"))
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(18 * mm, height - 11 * mm, "EUDR Evidence Pack")
    canvas.setFont("Helvetica", 8.5)
    canvas.drawRightString(width - 18 * mm, height - 11 * mm, f"Plot {doc.plot_id}")
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, ADVISORY)
    canvas.drawRightString(width - 18 * mm, 9 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def _plot_section(plot: PlotInput) -> list[Any]:
    check = tools.validate_geometry(plot["geometry"], plot.get("area_ha"))["values"]
    geometry = "Valid" if check["valid"] else "Invalid: " + "; ".join(check["reasons"])
    rows = [["Declared crop", plot.get("declared_crop", "unknown")], ["Crop since", plot.get("crop_since", "not stated")],
            ["Stated area", _ha(plot.get("area_ha"))], ["Geometry check (Art. 2(28))", geometry]]
    return [_table([[_para(a, LABEL), _para(b)] for a, b in rows], [55 * mm, 119 * mm])]


def _geo_section(geo: dict[str, Any] | None) -> list[Any]:
    if not geo:
        return [Paragraph("Geo-analyst findings", HEADING), _para("The geo-analyst was not run on this plot.")]
    if geo.get("error"):
        return [Paragraph("Geo-analyst findings", HEADING), _para(f"The geo-analyst run failed: {geo['error']}")]

    story: list[Any] = [Paragraph("Tool findings", HEADING)]
    for finding in geo["findings"]:
        source = "error" if finding.get("error") else "saved result" if finding.get("cached") else "live"
        rows = [[_para(TOOL_LABELS.get(finding["tool"], finding["tool"]), LABEL), _para(f"Source: {source}", SMALL)]]
        rows += [[_para(a), _para(b)] for a, b in key_figures(finding)]
        if finding.get("dataset"):
            rows.append([_para("Dataset", SMALL), _para(finding["dataset"], SMALL)])
        if finding.get("note"):
            rows.append([_para("Limits", SMALL), _para(finding["note"], SMALL)])
        story += [_table(rows, [50 * mm, 124 * mm]), Spacer(1, 6)]

    story.append(Paragraph("What the agent chose to check, in order", HEADING))
    plan = [[_para("Step", LABEL), _para("Tool", LABEL), _para("Agent's stated reason", LABEL)]]
    for i, event in enumerate((e for e in geo["events"] if e["node"] == "geo_tool"), 1):
        plan.append([_para(i), _para(TOOL_LABELS.get(event["tool"], event["tool"])), _para(event.get("reason", ""))])
    story.append(_table(plan, [12 * mm, 38 * mm, 124 * mm], header=True))

    story += [Paragraph("Geo-analyst report (written by the LLM; the numbers above come from the tools)", HEADING),
              _para(geo["summary"]), Spacer(1, 5), _para("Conflicts reported", LABEL),
              *_bullets(geo["conflicts"], "None reported."), Spacer(1, 5), _para("Open questions", LABEL),
              *_bullets(geo["open_questions"], "None reported.")]
    if geo.get("hit_limit"):
        story.append(_para("The agent reached its step limit before reporting."))
    return story


def _rules_section(checks: list[dict[str, Any]]) -> list[Any]:
    story: list[Any] = [Paragraph("Rule checks", HEADING)]
    if not checks:
        return [*story, _para("No rule checks were run on this plot.")]
    for check in checks:
        # each citation with its weight: binding law, a recital, non-binding guidance, Indian law or our own note
        cited = [f"{s['ref']} ({s['label']})" for s in check.get("sources", [])] or check.get("citations", [])
        story += [_para(check["question"], LABEL), _para(check.get("answer") or check.get("error", "")),
                  _para(f"Citations: {'; '.join(cited) or 'none'}", SMALL), Spacer(1, 7)]
    return story


def _documents_section(paths: list[Path], geo: dict[str, Any] | None) -> list[Any]:
    story: list[Any] = [Paragraph("Documents on this case (all mock)", HEADING)]
    if not paths:
        return [*story, _para("No documents were provided.")]
    opened = {f["values"]["file"] for f in (geo or {}).get("findings", [])
              if f["tool"] == "read_document" and not f.get("error")}
    rows = [[_para("File", LABEL), _para("Opened by the agent", LABEL), _para("In this pack", LABEL)]]
    for path in paths:
        where = "PDF pages added at the end" if path.suffix.lower() == ".pdf" else "Text in appendix"
        rows.append([_para(path.name), _para("Yes" if path.name in opened else "No"), _para(where)])
    story.append(_table(rows, [84 * mm, 35 * mm, 55 * mm], header=True))
    return story


def _appendix(paths: list[Path]) -> list[Any]:
    story: list[Any] = []
    for path in (p for p in paths if p.suffix.lower() != ".pdf"):
        story += [Paragraph(f"Appendix: {_clean(path.name)}", HEADING), _para(documents.read_text(path))]
    return story


def build_evidence_pack(plot: PlotInput, geo: dict[str, Any] | None = None, checks: list[dict[str, Any]] | None = None,
                        officer_note: str = "") -> bytes:
    """Build the PDF for one plot. Works at any stage, and states what was not done."""
    checks = checks or []
    paths = [Path(p) for p in plot.get("documents", [])]
    trace_files = [item["trace"] for item in [geo or {}, *checks] if item.get("trace")]

    story: list[Any] = [
        Paragraph(f"Plot {_clean(plot['plot_id'])}", HEADING),
        _para(f"Generated {time.strftime('%d %b %Y, %H:%M')}", SMALL), Spacer(1, 6),
        _para(NO_VERDICT, LABEL), _para(ADVISORY), Spacer(1, 6),
        *_plot_section(plot), *_geo_section(geo), *_rules_section(checks), *_documents_section(paths, geo),
        Paragraph("Officer note", HEADING), _para(officer_note.strip() or "None recorded."),
        Paragraph("Trace files", HEADING),
        *_bullets(trace_files, "No agent runs on this plot."),
        *_appendix(paths),
    ]

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title=f"EUDR evidence pack {plot['plot_id']}",
                            leftMargin=18 * mm, rightMargin=18 * mm, topMargin=24 * mm, bottomMargin=16 * mm)
    doc.plot_id = plot["plot_id"]
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)

    pdfs = [p for p in paths if p.suffix.lower() == ".pdf"]
    if not pdfs:
        return buffer.getvalue()
    writer = PdfWriter()
    writer.append(io.BytesIO(buffer.getvalue()))
    for path in pdfs:
        writer.append(str(path))
    merged = io.BytesIO()
    writer.write(merged)
    return merged.getvalue()
