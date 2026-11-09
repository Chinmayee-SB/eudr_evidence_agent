"""LangSmith tracing: every case as one trace tree, viewable at smith.langchain.com.

Switched on by three lines in .env (and off, with nothing sent anywhere, without them):
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=...
    LANGSMITH_PROJECT=eudr-evidence-agent        (LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com for EU)

What appears, nested under one run per case:
  * every LangGraph node of the supervisor, the geo-analyst and the rules agent (LangGraph traces these itself)
  * every LLM call, with prompt, reply, tokens and latency (our OpenAI client is wrapped, see llm.py)
  * every satellite or document tool the geo-analyst runs, as a tool run named after the tool
  * every search, as a retriever run listing the passages found, with reference, source type and score,
    and the reranker's scoring
Start and resume of a case share thread_id = case id, so LangSmith shows them as one thread. The officer's
decision is logged as feedback on the run that produced the draft verdict.

The local JSONL traces (tracing.py) are kept: they work offline and are what the desk shows. LangSmith is the
shared, searchable record across runs. Sending a trace publishes its content (plot coordinates, document
text, prompts) to LangSmith's servers; the test plots are anonymised.
"""

from __future__ import annotations

import os
from typing import Any

from langsmith import traceable  # noqa: F401  (re-exported: a no-op until tracing is switched on)

PROJECT_DEFAULT = "eudr-evidence-agent"


def enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def project() -> str:
    return os.getenv("LANGSMITH_PROJECT", PROJECT_DEFAULT)


def wrap_llm_client(client: Any) -> Any:
    """Trace every LLM call made through this OpenAI client."""
    if not enabled():
        return client
    from langsmith.wrappers import wrap_openai

    return wrap_openai(client)


def case_config(case_id: str, plot_id: str, stage: str, run_id: str | None = None) -> dict[str, Any]:
    """The run settings LangSmith reads: a readable name, tags to filter by, and the thread the run belongs to."""
    config: dict[str, Any] = {
        "run_name": f"case {case_id} ({stage})",
        "tags": ["eudr", f"plot:{plot_id}", stage],
        "metadata": {"thread_id": case_id, "case_id": case_id, "plot_id": plot_id, "stage": stage},
    }
    if run_id:
        config["run_id"] = run_id
    return config


def retrieved(hits: Any) -> dict[str, Any]:
    """A search result as LangSmith shows a retriever's output: one document per passage."""
    return {"documents": [{"type": "Document", "page_content": h.chunk.text[:2000],
                           "metadata": {"ref": h.chunk.ref, "kind": h.chunk.kind, "title": h.chunk.title,
                                        "score": round(float(h.score), 4), "found_by": h.found_by}}
                          for h in hits]}


def log_officer_decision(run_id: str | None, officer: dict[str, Any]) -> None:
    """The officer's decision as feedback on the run that produced the draft. Never fails the case."""
    if not (enabled() and run_id):
        return
    try:
        from langsmith import Client

        decision = officer.get("decision", "")
        Client().create_feedback(
            run_id, key="officer_decision", value=decision,
            score={"approve": 1, "override": 0}.get(decision),  # documents_added is neither
            comment=(officer.get("reason") or "") + (f" (verdict set to {officer['verdict']})" if officer.get("verdict") else ""))
    except Exception:  # an unreachable LangSmith must not stop the officer
        pass


def run_url(run_id: str | None) -> str:
    """A link to the run on LangSmith, or "" if tracing is off or LangSmith cannot be reached."""
    if not (enabled() and run_id):
        return ""
    try:
        from langsmith import Client

        client = Client()
        return client.get_run_url(run=client.read_run(run_id), project_name=project())
    except Exception:
        return ""
