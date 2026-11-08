"""Run one plot's case through the supervisor, and resume it after the officer decides.

Usage:
  python scripts/run_case.py K-01                                    start; stops when the officer is needed
  python scripts/run_case.py --show CASE_ID                          print a saved case
  python scripts/run_case.py --resume CASE_ID --approve [--reason "..."]
  python scripts/run_case.py --resume CASE_ID --override high_risk --reason "..."
  python scripts/run_case.py --resume CASE_ID --documents-added      after uploading documents for the plot

Cases are saved in runs/checkpoints.sqlite (or Postgres if DATABASE_URL is set), so a case can be resumed
in a later session. Each case's full trace is in runs/traces/<CASE_ID>.jsonl.
"""

from __future__ import annotations

import argparse
import os
import sys

from eudr_agent import documents
from eudr_agent.plots import load_plots
from eudr_agent.rag.chunk import KINDS_SHORT
from eudr_agent.state import CaseFile
from eudr_agent.supervisor.checkpoints import open_checkpointer
from eudr_agent.supervisor.graph import build_graph, get_case, resume_case, start_case


def print_case(case: CaseFile) -> None:
    if not case:
        print("no such case")
        return
    print(f"\ncase {case['case_id']}   status: {case['status'].upper()}")
    if case["status"] == "blocked":
        for reason in case["guardrail"]["blocked"]:
            print(f"  blocked: {reason}")
        return
    for d in case.get("decisions", []):
        print(f"  supervisor -> {d['action']} [{d['source']}] {d['reason']}")
        if d.get("overruled"):
            print(f"      (overruled: {d['overruled']})")
    for r in case.get("rules", []):
        cited = [f"{s['ref']} [{KINDS_SHORT.get(s['kind'], s['kind'])}]" for s in r.get("sources", [])]
        print(f"  rule ({r.get('point') or '-'}): {r['question']} -> {r['status']}")
        print(f"      cites: {'; '.join(cited or r.get('citations', [])) or 'none'}")
    print(f"  draft verdict: {case.get('draft_verdict') or '-'} ({case.get('verdict_note', '')})")
    for item in case.get("open_items", []):
        print(f"  open: {item}")
    if case.get("officer"):
        print(f"  officer: {case['officer']}   final verdict: {case.get('final_verdict') or '-'}")
    if case["status"] == "awaiting_officer":
        print(f"\nWaiting for the officer. Resume with:\n  python scripts/run_case.py --resume {case['case_id']} --approve")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plot_id", nargs="?")
    parser.add_argument("--show")
    parser.add_argument("--resume")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--approve", action="store_true")
    group.add_argument("--override", choices=["clear", "needs_evidence", "high_risk"])
    group.add_argument("--documents-added", action="store_true")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    with open_checkpointer() as saver:
        graph = build_graph(saver)
        if args.show:
            print_case(get_case(graph, args.show))
        elif args.resume:
            if args.approve:
                officer = {"decision": "approve", "reason": args.reason}
            elif args.override:
                officer = {"decision": "override", "verdict": args.override, "reason": args.reason}
            elif args.documents_added:
                plot_id = get_case(graph, args.resume)["plot"]["plot_id"]
                officer = {"decision": "documents_added",
                           "documents": documents.with_uploads(load_plots()[plot_id])["documents"]}
            else:
                sys.exit("--resume needs --approve, --override or --documents-added")
            try:
                print_case(resume_case(graph, args.resume, officer, echo=True))
            except ValueError as error:
                sys.exit(f"not accepted: {error}")
        elif args.plot_id:
            plots = load_plots()
            if args.plot_id not in plots:
                sys.exit(f"unknown plot {args.plot_id}; choose from {sorted(plots)}")
            print_case(start_case(graph, documents.with_uploads(plots[args.plot_id]), echo=True))
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
    # By here the case is saved and the database connection is closed. onnxruntime (used by the rules agent's
    # vector search) can deadlock in its shutdown code on macOS, leaving the process hanging after its work is
    # done; exiting directly skips that cleanup. A sys.exit() above still reports errors as usual.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
