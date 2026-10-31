"""Evaluate the rules agent on eval/rule_questions.jsonl.

Metrics
  retrieval hit@k     expected passage appears among retrieved passages (any attempt)
  citation correct    the answer cites at least one expected passage
  abstention correct  out-of-scope questions end in "no rule found", in-scope ones do not
  steps               graph steps per question (cost proxy)

Usage:  python eval/run_rag_eval.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from eudr_agent.config import RUNS_DIR, settings
from eudr_agent.rag.agent import ask

QUESTIONS = Path(__file__).with_name("rule_questions.jsonl")


def main() -> None:
    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    results = []
    for row in rows:
        state = ask(row["question"])
        events = state["tracer"].events
        retrieved = {ref for e in events if e["node"] == "retrieve" for ref, _ in e["hits"]}
        expected = set(row["expected"])
        oos = row.get("out_of_scope", False)
        results.append({
            "id": row["id"],
            "question": row["question"],
            "status": state["status"],
            "citations": state["citations"],
            "hit": None if oos else bool(expected & retrieved),
            "cite_ok": None if oos else bool(expected & set(state["citations"])),
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
    print(f"retrieval hit@k      {pct([r['hit'] for r in in_scope])}")
    print(f"citation correct     {pct([r['cite_ok'] for r in in_scope])}")
    print(f"abstention correct   {pct([r['abstain_ok'] for r in results])}")
    print(f"avg steps            {sum(r['steps'] for r in results) / len(results):.1f}")
    print(f"questions rewritten  {sum(r['rewrites'] > 0 for r in results)}/{len(results)}")

    out = RUNS_DIR / "eval" / f"rag_eval_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
