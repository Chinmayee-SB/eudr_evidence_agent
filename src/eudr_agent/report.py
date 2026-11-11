"""The evidence pack: one PDF document holding everything known about a plot, whatever stage the case reached.

Cover (verdict and a summary built from the facts), then 1. what was submitted, 2. what happened step by step,
3. the evidence, 4. the legal questions, 5. the verdict and why, 6. what is still needed, 7. the officer's
decision, and appendices (land records, data sources, audit trail). It uses the officer's words (wording.py),
reports figures as the data sources returned them, and marks what the AI assistant wrote. Before a case has
run it holds evidence only and says so. Uploaded PDFs are added unchanged at the end.
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
from eudr_agent import wording
from eudr_agent.state import Finding, PlotInput

TOOL_LABELS = wording.CHECK  # the officer's names for the checks (wording.py)
NO_VERDICT = "No verdict issued: no case has been run on this plot, so this pack contains evidence only."
VERDICT_WORDS = wording.VERDICT
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
        return [("EU forest map 2020", f"{v['gfc2020_forest_ha']} ha ({v['gfc2020_forest_share']:.0%})"),
                ("Global Forest Watch tree cover", f"{v['hansen_tree_cover_ha']} ha ({v['hansen_tree_cover_share']:.0%})"),
                ("Plot area", _ha(v["plot_ha"]))]
    if tool == "loss_after_2020":
        return [("Global Forest Watch: tree loss", f"{v['hansen_loss_ha']} ha ({_by_year(v['hansen_loss_by_year'])})"),
                ("EU tropical forest monitor: cleared", f"{v['tmf_deforestation_ha']} ha ({_by_year(v['tmf_deforestation_by_year'])})"),
                ("EU tropical forest monitor: thinned", f"{v['tmf_degradation_ha']} ha ({_by_year(v['tmf_degradation_by_year'])})")]
    if tool == "read_document":
        return [("File", v["file"]), ("Status", "Mock document, not a real record")]
    if tool == "locate_loss":
        rows = [("Pattern", f"{v['pattern']}: {v['patch_count']} patch{'es' if v['patch_count'] != 1 else ''}, "
                            f"largest {_ha(v['largest_patch_ha'])}"),
                ("Loss at the edge", f"{v['loss_at_edge_ha']} ha of {v['loss_ha']} ha (within 30 m)"),
                ("Loss on 2020 forest", _ha(v["loss_in_gfc2020_forest_ha"]))]
        def years(p):  # "2021", not "2021-2021"
            return str(p["first_year"]) if p["first_year"] == p["last_year"] else f"{p['first_year']}-{p['last_year']}"
        return rows + [(f"Patch {i}", f"{p['area_ha']} ha, {years(p)}, {p['distance_to_edge_m']} m from edge")
                       for i, p in enumerate(v["patches"], 1)]
    if tool == "boundary_sensitivity":
        rows = [("Result", "Stable" if v["stable"] else "Changes: " + "; ".join(v["changes"]))]
        return rows + [(name.replace("_", " ").capitalize(),
                        f"{x['plot_ha']} ha plot, {x['loss_ha']} ha loss, {x['gfc2020_forest_share']:.0%} forest")
                       for name, x in v["variants"].items()]
    if tool == "ndvi_series":
        rows = [("Plot", ", ".join(f"{y}: {d['ndvi']}" for y, d in v["plot"].items()))]
        if v["loss_pixels"]:
            rows.append(("Where loss was mapped", ", ".join(f"{y}: {n}" for y, n in v["loss_pixels"].items())))
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


# ---------- what each part of the pack says ----------

TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=BLACK,
                       spaceAfter=2)
SUB = ParagraphStyle("sub", parent=BODY, fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=BLACK,
                     spaceBefore=8, spaceAfter=3)
BIG = ParagraphStyle("big", parent=BODY, fontName="Helvetica-Bold", fontSize=17, leading=21)
VERDICT_COLOUR = {"clear": colors.HexColor("#E3F1DA"), "needs_evidence": colors.HexColor("#FBEFD3"),
                  "high_risk": colors.HexColor("#F7DCD8"), None: PALE}

# what the grower or officer can supply, by what decided a "needs evidence" verdict
NEEDED = {
    7: ["Recent high-resolution imagery or a field visit report for the area where tree loss was mapped, showing "
        "whether it is pruning or shade-tree work, or forest clearing.",
        "If the boundary is in doubt, a corrected boundary (submitted as a new plot)."],
    0: ["Recent high-resolution imagery or a field visit report for the area where tree loss was mapped."],
}
LAND_RECORD_NEEDED = "A land record (for example an RTC extract) showing the plot was under coffee before 31 December 2020."


def finding_line(finding: Finding) -> str:
    """One sentence saying what a check found."""
    if finding.get("error"):
        return f"Could not run: {wording.plain(finding['error'])}"
    v, tool = finding.get("values", {}), finding["tool"]
    if tool == "protected_area_check":
        if v.get("inside_protected_area"):
            return "Overlaps " + ", ".join(a.get("name", "?") for a in v.get("in_force", [])) + ", which is in force."
        return "Only proposed areas overlap; none is in force." if v.get("proposed") else "Not in a protected area."
    if tool == "forest_2020":
        return (f"The EU forest map shows {v['gfc2020_forest_share']:.0%} of the plot as forest in 2020; "
                f"Global Forest Watch shows {v['hansen_tree_cover_share']:.0%} tree cover.")
    if tool == "loss_after_2020":
        hansen, tmf = v.get("hansen_loss_ha", 0), v.get("tmf_deforestation_ha", 0)
        if not hansen and not tmf:
            return "No tree loss after 2020 on either satellite source."
        return f"Tree loss after 2020: Global Forest Watch {hansen} ha, EU tropical forest monitor {tmf} ha."
    if tool == "locate_loss":
        n = v.get("patch_count", 0)
        return (f"{n} patch{'es' if n != 1 else ''}, the largest {v.get('largest_patch_ha', 0)} ha; "
                f"{v.get('loss_at_edge_ha', 0)} of {v.get('loss_ha', 0)} ha lies within 30 m of the boundary.")
    if tool == "boundary_sensitivity":
        return ("The result holds when the boundary moves 30 m in or out." if v.get("stable")
                else "The result changes when the boundary moves 30 m: " + "; ".join(v.get("changes", [])) + ".")
    if tool == "ndvi_series":
        years = sorted(v.get("plot", {}))
        return (f"Dry-season greenness {years[0]}: {v['plot'][years[0]]['ndvi']}, {years[-1]}: "
                f"{v['plot'][years[-1]]['ndvi']}." if years else "No clear-sky images.")
    if tool == "coffee_probability":
        return "Not used: the global coffee map is not reliable for shade-grown coffee in India."
    if tool == "read_document":
        return f"Read {v.get('file', '').removeprefix('uploaded_')}."
    return ""


def _kv(rows: list[tuple[str, Any]]) -> Table:
    return _table([[_para(a, LABEL), _para(b)] for a, b in rows], [52 * mm, 122 * mm])


def _stopped(case: dict[str, Any]) -> str:
    """Why the input check stopped the case, as sentences."""
    reasons = [wording.plain(b) for b in case.get("guardrail", {}).get("blocked", [])]
    return " ".join(r[0].upper() + r[1:] + ("" if r.endswith(".") else ".") for r in reasons)


def _summary(plot: PlotInput, case: dict[str, Any], findings: list[Finding]) -> str:
    """The case in one paragraph, built from its facts (not written by the AI assistant)."""
    since = f", declared as {plot.get('declared_crop', 'coffee')} since {plot['crop_since']}" if plot.get("crop_since") \
        else f", declared as {plot.get('declared_crop', 'coffee')} (start year not stated)"
    parts = [f"Plot {plot['plot_id']} ({plot.get('area_ha')} ha{since}) was screened against the EU Deforestation "
             f"Regulation."]
    if case.get("status") == "blocked":
        return " ".join(parts + ["The input check stopped it before any evidence was gathered. " + _stopped(case)])
    checks = {f["tool"] for f in findings if not f.get("error")}
    parts.append(f"{len(checks)} check{'s were' if len(checks) != 1 else ' was'} run on satellite data and land "
                 "records, and "
                 f"{len(case.get('rules', []))} legal question{'s were' if len(case.get('rules', [])) != 1 else ' was'} "
                 "answered from EU law, Commission guidance and Karnataka law.")
    label, why = wording.basis(case)
    if label:
        parts.append(f"The draft verdict is {VERDICT_WORDS.get(case.get('draft_verdict'))} "
                     f"({label[0].lower() + label[1:]}): {why[0].lower() + why[1:]}")
    officer = case.get("officer") or {}
    if case.get("status") == "closed":
        parts.append(f"The officer {wording.DECISION.get(officer.get('decision'), '').lower()}, so the final verdict "
                     f"is {VERDICT_WORDS.get(case.get('final_verdict'))}.")
    elif label:
        parts.append("The case is waiting for the officer's sign-off.")
    return " ".join(parts)


def _cover(plot: PlotInput, case: dict[str, Any] | None, findings: list[Finding]) -> list[Any]:
    story: list[Any] = [Paragraph(f"Evidence pack: plot {_clean(plot['plot_id'])}", TITLE),
                        _para(f"Generated {time.strftime('%d %b %Y, %H:%M')}"
                              + (f" · Case {case['case_id']}" if case else ""), SMALL), Spacer(1, 8)]
    if not case:
        return story + [_para(NO_VERDICT, LABEL), Spacer(1, 4), _para(ADVISORY, SMALL)]
    closed = case.get("status") == "closed"
    verdict = case.get("final_verdict") if closed else case.get("draft_verdict")
    label, why = wording.basis(case)
    officer = case.get("officer") or {}
    if case.get("status") == "blocked":
        rows = [[_para("Not screened", BIG)], [_para("Stopped by the input check. " + _stopped(case))]]
    else:
        overridden = closed and officer.get("decision") == "override"
        heading = ("Final verdict, set by the officer" if overridden else "Final verdict" if closed
                   else "Draft verdict, waiting for the officer's sign-off")
        rows = [[_para(heading, SMALL)], [_para(VERDICT_WORDS.get(verdict, "Not issued"), BIG)]]
        if label and overridden:  # the draft's reasoning would contradict the officer's verdict
            rows.append([_para(f"The draft was {VERDICT_WORDS.get(case.get('draft_verdict'))}: "
                               f"{label[:1].lower() + label[1:]}.", BODY)])
        elif label:
            rows.append([Paragraph(f"<b>{_clean(label)}.</b> {_clean(why)}", BODY)])
        if closed:
            rows.append([_para(f"Officer: {wording.DECISION.get(officer.get('decision'), '')}"
                               + (f". Reason: {officer['reason']}" if officer.get("reason") else "."), BODY)])
    box = Table(rows, colWidths=[174 * mm])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), VERDICT_COLOUR.get(verdict, PALE)),
                             ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return story + [box, Spacer(1, 10), Paragraph("In short", SUB), _para(_summary(plot, case, findings))]


def _submitted(plot: PlotInput, case: dict[str, Any] | None, paths: list[Path], findings: list[Finding]) -> list[Any]:
    check = tools.validate_geometry(plot["geometry"], plot.get("area_ha"))["values"]
    boundary = "Meets the EU location rules (Art. 2(28))" if check["valid"] else "Does not meet the EU location " \
        "rules: " + "; ".join(check["reasons"])
    story: list[Any] = [Paragraph("1. What was submitted", HEADING), _kv([
        ("Declared crop", plot.get("declared_crop", "unknown")), ("Crop since", plot.get("crop_since", "Not stated")),
        ("Area", _ha(plot.get("area_ha"))), ("Boundary", boundary)])]
    read = {f["values"]["file"] for f in findings if f["tool"] == "read_document" and not f.get("error")}
    story.append(Paragraph("Land records", SUB))
    if paths:
        rows = [[_para("File", LABEL), _para("Read during the check", LABEL), _para("In this pack", LABEL)]]
        for path in paths:
            rows.append([_para(path.name.removeprefix("uploaded_")), _para("Yes" if path.name in read else "No"),
                         _para("Pages added at the end" if path.suffix.lower() == ".pdf" else "Appendix A")])
        story.append(_table(rows, [84 * mm, 40 * mm, 50 * mm], header=True))
    else:
        story.append(_para("None provided."))
    guard = (case or {}).get("guardrail") or {}
    if guard.get("dropped"):
        story += [Paragraph("Held back by the input check", SUB),
                  *_bullets([d.removeprefix("uploaded_") for d in guard["dropped"]], "")]
    if guard.get("warnings"):
        story += [Paragraph("Notes from the input check", SUB),
                  *_bullets([w[0].upper() + w[1:] for w in guard["warnings"]], "")]
    return story


def _steps(case: dict[str, Any], events: list[dict[str, Any]]) -> list[str]:
    """The case's record, step by step, in plain words: consecutive checks and legal-search steps are grouped."""
    findings = {}
    for f in case.get("findings", []):
        findings.setdefault(f["tool"], []).append(f)
    steps, round_no, i = [], 0, 0
    answers = {r["question"]: r for r in case.get("rules", [])}
    rag = {"retrieve", "grade", "rewrite", "answer", "check_citations", "verify", "cache", "no_rule_found"}
    while i < len(events):
        ev = events[i]
        node = ev["node"]
        if node == "input_guardrail":
            text = "Input check: passed." if ev.get("passed") else \
                "Input check: stopped. " + "; ".join(ev.get("blocked", [])) + "."
            if ev.get("dropped"):
                text += " Held back: " + "; ".join(d.removeprefix("uploaded_") for d in ev["dropped"]) + "."
            steps.append(text)
        elif node in {"geo_think", "geo_tool"}:
            round_no += 1
            lines = []
            while i < len(events) and events[i]["node"] in {"geo_think", "geo_tool", "geo_report"}:
                e = events[i]
                if e["node"] == "geo_tool":
                    name = wording.CHECK.get(e["tool"], e["tool"])
                    if e.get("repeat"):
                        lines.append(f"{name}: asked again; the earlier result was reused.")
                    else:
                        result = finding_line(e["finding"]) if e.get("finding") else ""
                        lines.append(f"{name}. Why: {wording.plain(e.get('reason', ''))} Result: {result}")
                i += 1
            steps.append(f"Evidence check, round {round_no}: {len(lines)} check{'s' if len(lines) != 1 else ''}.\n"
                         + "\n".join(f"    - {line}" for line in lines))
            continue
        elif node == "supervisor":
            d = ev.get("decision") or {}
            text = f"Case review: {wording.ACTION.get(d.get('action'), 'decision')}. {wording.plain(d.get('reason', ''))}"
            if d.get("task"):
                text += f" Asked for: {wording.plain(d['task'])}"
            steps.append(text)
            for question in d.get("questions") or []:  # the legal check answers these next
                r = answers.get(question, {})
                status = ("answered with citations" if r.get("status") == "answered" else
                          "not answered from the sources" if r else "no answer recorded")
                steps.append(f"Legal check: \"{question}\" - {status}.")
        elif node in rag:
            pass  # the legal check's own steps: summarised above, with the question
        elif node == "verdict":
            label, _ = wording.basis({"verdict_detail": ev.get("detail") or {}})
            steps.append(f"Draft verdict: {VERDICT_WORDS.get(ev.get('verdict'))} ({label.lower()}).")
        elif node == "officer":
            d = ev.get("decision") or {}
            steps.append(f"Officer: {wording.DECISION.get(d.get('decision'), '')}"
                         + (f". Reason: {d['reason']}" if d.get("reason") else "") + ".")
        elif node == "pack":
            steps.append(f"Case closed. Final verdict: {VERDICT_WORDS.get(case.get('final_verdict'))}.")
        i += 1
    return steps


def _happened(case: dict[str, Any] | None, events: list[dict[str, Any]]) -> list[Any]:
    story: list[Any] = [Paragraph("2. What happened, step by step", HEADING)]
    if not events:
        if case and case.get("status") == "blocked":
            return story + [_para("1. Input check: stopped. " + _stopped(case) + " Nothing else was run.")]
        return story + [_para("No case has been run on this plot.")]
    story.append(_para("From the case record. The figures come from the data sources; the reasons for each check "
                       "and each case-review decision were written by the AI assistant.", SMALL))
    for n, step in enumerate(_steps(case or {}, events), 1):
        first, *rest = step.split("\n")
        story.append(_para(f"{n}. {first}"))
        for line in rest:
            story.append(Paragraph(_clean(line.strip()), ParagraphStyle("li", parent=BODY, leftIndent=14)))
        story.append(Spacer(1, 3))
    return story


def _evidence(geo: dict[str, Any] | None) -> list[Any]:
    story: list[Any] = [Paragraph("3. Evidence in detail", HEADING)]
    if not geo:
        return story + [_para("The evidence check was not run on this plot.")]
    if geo.get("error"):
        return story + [_para(f"The evidence check failed: {wording.plain(geo['error'])}")]
    latest: dict[str, Finding] = {}
    for f in geo["findings"]:
        if not f.get("error") or f["tool"] not in latest:
            latest[f["tool"] if f["tool"] != "read_document" else f"doc:{len(latest)}"] = f
    for finding in latest.values():
        source = "error" if finding.get("error") else "saved result" if finding.get("cached") else "live query"
        rows = [[_para(TOOL_LABELS.get(finding["tool"], finding["tool"]), LABEL), _para(finding_line(finding))]]
        rows += [[_para(a, SMALL), _para(b)] for a, b in key_figures(finding)]
        if finding.get("dataset"):
            rows.append([_para("Data source", SMALL),
                         _para(f"{wording.dataset_names(finding['dataset'])} ({source})", SMALL)])
        if finding.get("note"):
            rows.append([_para("Limits", SMALL), _para(wording.plain(finding["note"]), SMALL)])
        story += [_table(rows, [52 * mm, 122 * mm]), Spacer(1, 6)]
    story += [Paragraph("Evidence summary (written by the AI assistant)", SUB), _para(wording.plain(geo["summary"]))]
    if geo.get("conflicts"):
        story += [Paragraph("Where the sources disagree", SUB), *_bullets([wording.plain(c) for c in geo["conflicts"]], "")]
    if geo.get("hit_limit"):
        story.append(_para("The evidence check reached its step limit before reporting."))
    return story


def _legal(checks: list[dict[str, Any]]) -> list[Any]:
    story: list[Any] = [Paragraph("4. Legal questions and answers", HEADING)]
    if not checks:
        return story + [_para("No legal questions were asked on this plot.")]
    in_case = [c for c in checks if c.get("point")]
    extra = [c for c in checks if not c.get("point")]
    for title, group in (("Asked during the case review", in_case), ("Asked by the officer", extra)):
        if not group:
            continue
        if in_case and extra:
            story.append(Paragraph(title, SUB))
        for check in group:
            topic = wording.TOPIC.get(check.get("point"), "")
            # each citation with its weight: binding law, a recital, non-binding guidance, Indian law or our own note
            cited = [f"{s['ref']} ({s['label']})" for s in check.get("sources", [])] or check.get("citations", [])
            story += [_para((f"{topic} " if topic else "") + check["question"], LABEL),
                      _para(check.get("answer") or check.get("error", "")),
                      _para(f"Citations: {'; '.join(cited) or 'none'}", SMALL), Spacer(1, 7)]
    return story


def _verdict(case: dict[str, Any] | None, findings: list[Finding]) -> list[Any]:
    story: list[Any] = [Paragraph("5. Verdict and why", HEADING)]
    if not case or not case.get("verdict_detail"):
        return story + [_para("No verdict has been issued." if not case or case.get("status") != "blocked"
                              else "Not screened: the input check stopped the plot.")]
    detail = case["verdict_detail"]
    label, why = wording.basis(case)
    story += [_kv([("Draft verdict", VERDICT_WORDS.get(case.get("draft_verdict"))), ("Based on", f"{label}. {why}")]),
              Paragraph("Why", SUB), *_bullets([wording.plain(r) for r in detail.get("reasons", [])],
                                               "No reasons recorded."),
              Paragraph("Legal basis", SUB), *_bullets(detail.get("basis", []), "None."),
              Paragraph("Points to note", SUB), *_bullets([wording.plain(c) for c in detail.get("caveats", [])],
                                                          "None.")]
    return story


def _needed(case: dict[str, Any] | None, findings: list[Finding]) -> list[Any]:
    story: list[Any] = [Paragraph("6. What is still needed", HEADING)]
    if not case or not case.get("verdict_detail"):
        return story + [_para("Not applicable: no verdict has been issued.")]
    rule = case["verdict_detail"].get("rule")
    needed = list(NEEDED.get(rule, [])) if case.get("draft_verdict") == "needs_evidence" else []
    if rule == 2:
        needed += [f"Complete this check: {r}" for r in case["verdict_detail"].get("reasons", [])]
    read = any(f["tool"] == "read_document" and not f.get("error") for f in findings)
    if case.get("draft_verdict") == "needs_evidence" and not read:
        needed.append(LAND_RECORD_NEEDED)
    open_items = [wording.plain(i) for i in case.get("open_items", [])]
    if needed:
        story += [_para("For the officer to reach a final verdict:"), *_bullets(needed, "")]
    else:
        story.append(_para("Nothing further is needed for this verdict."))
    if open_items:
        story += [Paragraph("Not checked, or still open", SUB), *_bullets(open_items, "")]
    return story


def _officer(case: dict[str, Any] | None, note: str) -> list[Any]:
    story: list[Any] = [Paragraph("7. Officer decision", HEADING)]
    officer = (case or {}).get("officer") or {}
    if case and case.get("status") == "closed":
        rows = [("Decision", wording.DECISION.get(officer.get("decision"), "")),
                ("Reason", officer.get("reason") or "None given"),
                ("Final verdict", VERDICT_WORDS.get(case.get("final_verdict")))]
        if case.get("reruns"):
            rows.append(("Re-checks with new land records", case["reruns"]))
        story.append(_kv(rows))
    else:
        story.append(_para("Not signed off yet."))
    return story + [Paragraph("Officer note", SUB), _para(note.strip() or "None recorded.")]


def _appendices(paths: list[Path], findings: list[Finding], case: dict[str, Any] | None,
                traces: list[str]) -> list[Any]:
    story: list[Any] = []
    texts = [p for p in paths if p.suffix.lower() != ".pdf"]
    if texts:
        story.append(Paragraph("Appendix A. Land records, in full", HEADING))
        for path in texts:
            story += [Paragraph(_clean(path.name.removeprefix("uploaded_")), SUB), _para(documents.read_text(path))]
    sources = sorted({(TOOL_LABELS.get(f["tool"], f["tool"]), wording.dataset_names(f.get("dataset", "")),
                       wording.plain(f.get("note", "")))
                      for f in findings if f.get("dataset") and f["tool"] != "read_document"})
    if sources:
        story.append(Paragraph("Appendix B. Data sources and their limits", HEADING))
        rows = [[_para("Check", LABEL), _para("Data source", LABEL), _para("Limits", LABEL)]]
        rows += [[_para(a), _para(b, SMALL), _para(c, SMALL)] for a, b, c in sources]
        story.append(_table(rows, [36 * mm, 62 * mm, 76 * mm], header=True))
    story.append(Paragraph("Appendix C. Audit trail", HEADING))
    rows = []
    if case:
        rows += [("Case", case["case_id"]), ("Status", case.get("status", ""))]
        if case.get("langsmith_runs"):
            rows.append(("Technical trace runs", ", ".join(case["langsmith_runs"])))
    rows += [("Step-by-step record", t) for t in traces] or [("Step-by-step record", "None: no checks were run.")]
    story.append(_kv(rows))
    return story


def build_evidence_pack(plot: PlotInput, geo: dict[str, Any] | None = None, checks: list[dict[str, Any]] | None = None,
                        officer_note: str = "", case: dict[str, Any] | None = None) -> bytes:
    """The PDF for one plot, as a document: cover with the verdict and a summary, then what was submitted, what
    happened step by step, the evidence, the legal answers, the verdict, what is still needed, the officer's
    decision, and appendices. Works at any stage and says what was not done."""
    checks = checks or []
    paths = [Path(p) for p in plot.get("documents", [])]
    findings = (geo or {}).get("findings", []) if not (geo or {}).get("error") else []
    events = (geo or {}).get("events", [])
    traces = list(dict.fromkeys(item["trace"] for item in [geo or {}, *checks] if item.get("trace")))

    story: list[Any] = [
        *_cover(plot, case, findings), *_submitted(plot, case, paths, findings), *_happened(case, events),
        *_evidence(geo), *_legal(checks), *_verdict(case, findings), *_needed(case, findings),
        *_officer(case, officer_note), *_appendices(paths, findings, case, traces),
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
