"""Evaluate the rules agent on eval/rule_questions.jsonl.

Metrics
  retrieval hit@k     expected passage appears among retrieved passages (any attempt)
  citation correct    the answer cites at least one expected passage
  abstention correct  out-of-scope questions end in "no rule found", in-scope ones do not
  cites a legal source / guidance   share of answers resting on law, and drawing on the guidance or Indian law
Questions are in two sets: "original" (general EUDR questions) and "supervisor" (worded as the supervisor
asks them in real cases, one legal point each). A "(part N)" citation counts as its whole section.
  steps               graph steps per question (cost proxy)

Usage:  python eval/run_rag_eval.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import re

from eudr_agent.config import RUNS_DIR, settings
from eudr_agent.rag.agent import ask

QUESTIONS = Path(__file__).with_name("rule_questions.jsonl")
LEGAL_KINDS = {"law", "recital", "guidance", "national_law"}


def section(ref: str) -> str:
    """'EC Guidance §11.4(d) (part 2)' counts as 'EC Guidance §11.4(d)'."""
    return re.sub(r" \(part \d+\)$", "", ref)


def main() -> None:
    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    results = []
    for row in rows:
        state = ask(row["question"], use_cache=False)  # measure the agent, not the cache
        events = state["tracer"].events
        retrieved = {section(ref) for e in events if e["node"] == "retrieve" for ref, _ in e["hits"]}
        cited = {section(ref) for ref in state["citations"]}
        kinds = [s["kind"] for s in state.get("sources", [])]
        expected = set(row["expected"])
        oos = row.get("out_of_scope", False)
        results.append({
            "id": row["id"],
            "set": row.get("set", "original"),
            "question": row["question"],
            "status": state["status"],
            "citations": state["citations"],
            "hit": None if oos else bool(expected & retrieved),
            "cite_ok": None if oos else bool(expected & cited),
            "legal_cited": None if oos else any(k in LEGAL_KINDS for k in kinds),
            "guidance_cited": None if oos else any(k in {"guidance", "national_law"} for k in kinds),
            "abstain_ok": (state["status"] == "no_rule_found") == oos,
            "steps": len(events),
            "rewrites": sum(e["node"] == "rewrite" for e in events),
        })

    in_scope = [r for r in results if r["hit"] is not None]
    pct = lambda xs: f"{100 * sum(xs) / len(xs):.0f}%" if xs else "-"  # noqa: E731
    print(f"mode: {'LLM ' + settings.llm_model if settings.llm_enabled else 'offline'}  |  "
          f"retrieval: {settings.retrieval}  |  {len(rows)} questions\n")
    print(f"{'id':<5}{'status':<15}{'hit':<6}{'cite':<6}{'abst':<6}{'steps':<7}citations")
    for r in results:
        mark = lambda v: "-" if v is None else ("yes" if v else "NO")  # noqa: E731
        print(f"{r['id']:<5}{r['status']:<15}{mark(r['hit']):<6}{mark(r['cite_ok']):<6}"
              f"{mark(r['abstain_ok']):<6}{r['steps']:<7}{', '.join(r['citations'])[:70]}")
    print()
    sets = {"all": results, **{name: [r for r in results if r["set"] == name]
                               for name in dict.fromkeys(r["set"] for r in results)}}
    print(f"{'':<22}" + "".join(f"{name:>12}" for name in sets))
    rows_out = [
        ("retrieval hit@k", lambda rs: pct([r["hit"] for r in rs if r["hit"] is not None])),
        ("citation correct", lambda rs: pct([r["cite_ok"] for r in rs if r["hit"] is not None])),
        ("cites a legal source", lambda rs: pct([r["legal_cited"] for r in rs if r["hit"] is not None])),
        ("cites guidance/Indian law", lambda rs: pct([r["guidance_cited"] for r in rs if r["hit"] is not None])),
        ("abstention correct", lambda rs: pct([r["abstain_ok"] for r in rs])),
        ("avg steps", lambda rs: f"{sum(r['steps'] for r in rs) / len(rs):.1f}"),
        ("questions rewritten", lambda rs: f"{sum(r['rewrites'] > 0 for r in rs)}/{len(rs)}"),
    ]
    for label, fn in rows_out:
        print(f"{label:<26}" + "".join(f"{fn(rs):>12}" for rs in sets.values())[4:])

    out = RUNS_DIR / "eval" / f"rag_eval_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
