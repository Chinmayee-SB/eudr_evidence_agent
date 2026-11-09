"""End-to-end evaluation: the full case on every plot, next to the human label and FAO Whisp.

Each plot runs through the real supervisor graph (input check, geo-analyst, supervisor, rules agent, fixed
verdict rules) up to the officer pause, exactly as on the desk. The officer step is not simulated: the table
compares the draft verdict, which is what the officer is shown.

Columns
  label        the human labeller's verdict (data/plots/labels.csv), from looking at before/after imagery
  whisp        FAO Whisp's perennial-crop risk (eval/whisp_baseline.csv, from scripts/run_whisp_baseline.py)
  ours         the draft verdict and the rule that decided it
  agreement    ours vs the label. "needs_evidence" against a "Clear" label is counted separately: the system
               asked for evidence rather than calling it wrong
  cost         geo-analyst rounds, supervisor decisions, rule questions, tools run, seconds

Needs LLM_API_KEY (the agents) and the satellite cache or EE_PROJECT. Saves eval/end_to_end_<date>.csv and
the cases themselves in the checkpointer, so each can be opened on the desk by its case id.

Usage:  python eval/run_end_to_end.py
"""

from __future__ import annotations

import csv
import os
import sys
import time

from eudr_agent import documents
from eudr_agent.config import PLOTS_DIR, ROOT, settings
from eudr_agent.plots import load_plots
from eudr_agent.supervisor.checkpoints import open_checkpointer
from eudr_agent.supervisor.graph import build_graph, start_case

LABEL = {"Clear": "clear", "Needs evidence": "needs_evidence", "High risk": "high_risk"}
WHISP = ROOT / "eval" / "whisp_baseline.csv"
WORDS = {"clear": "Clear", "needs_evidence": "Needs evidence", "high_risk": "High risk", None: "-"}


def agreement(ours: str | None, label: str) -> str:
    if ours == label:
        return "agrees"
    if ours == "needs_evidence":
        return "asks for evidence"
    return "disagrees"


def main() -> None:
    if not settings.llm_enabled:
        sys.exit("needs LLM_API_KEY and LLM_MODEL: the agents are the thing being evaluated")
    labels = {r["plot_id"]: r for r in csv.DictReader((PLOTS_DIR / "labels.csv").open())}
    whisp = {r["plot_id"]: r["whisp_risk_pcrop"] for r in csv.DictReader(WHISP.open())} if WHISP.exists() else {}
    plots = load_plots()
    rows = []
    with open_checkpointer() as saver:
        graph = build_graph(saver)
        for plot_id in sorted(plots):
            started = time.time()
            case = start_case(graph, documents.with_uploads(plots[plot_id]))
            seconds = round(time.time() - started)
            label = LABEL.get(labels.get(plot_id, {}).get("label", ""), "")
            ours = case.get("draft_verdict")
            rows.append({
                "plot_id": plot_id, "label": label, "whisp": whisp.get(plot_id, ""), "ours": ours,
                "rule": (case.get("verdict_detail") or {}).get("rule"), "verdict_note": case.get("verdict_note", ""),
                "agreement": agreement(ours, label),
                "geo_rounds": len(case.get("geo_reports", [])), "send_backs": case.get("send_backs", 0),
                "decisions": len(case.get("decisions", [])), "rule_questions": len(case.get("rules", [])),
                "tools_run": sorted({f["tool"] for f in case.get("findings", []) if not f.get("error")}),
                "llm_overruled": sum(bool(d.get("overruled")) for d in case.get("decisions", [])),
                "open_items": len(case.get("open_items", [])), "seconds": seconds, "case_id": case["case_id"],
            })
            print(f"  {plot_id}: {WORDS[ours]} ({case.get('verdict_note', '')}) in {seconds}s", flush=True)

    out = ROOT / "eval" / f"end_to_end_{time.strftime('%Y-%m-%d')}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**r, "tools_run": " ".join(r["tools_run"])} for r in rows)

    print(f"\n{'plot':<6}{'label':<16}{'Whisp':<18}{'ours':<16}{'rule':<6}{'vs label':<19}"
          f"{'rounds':<8}{'rules Qs':<10}{'tools':<7}{'secs'}")
    for r in rows:
        print(f"{r['plot_id']:<6}{WORDS.get(r['label'], r['label']):<16}{r['whisp']:<18}{WORDS[r['ours']]:<16}"
              f"{str(r['rule']):<6}{r['agreement']:<19}{r['geo_rounds']:<8}{r['rule_questions']:<10}"
              f"{len(r['tools_run']):<7}{r['seconds']}")
    agree = sum(r["agreement"] == "agrees" for r in rows)
    asks = sum(r["agreement"] == "asks for evidence" for r in rows)
    wrong = sum(r["agreement"] == "disagrees" for r in rows)
    print(f"\nagrees with the label {agree}/{len(rows)}, asks for evidence {asks}, disagrees {wrong}")
    high = [r for r in rows if r["label"] == "high_risk"]
    print(f"high-risk plots caught: ours {sum(r['ours'] == 'high_risk' for r in high)}/{len(high)}, "
          f"Whisp {sum(r['whisp'] == 'high' for r in high)}/{len(high)}")
    print(f"\nsaved {out.relative_to(ROOT)}  (open any case on the desk by its case_id)")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # onnxruntime can hang in interpreter shutdown on macOS (see scripts/run_case.py)
