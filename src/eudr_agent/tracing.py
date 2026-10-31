"""Minimal step tracer: every node writes one JSON line to runs/traces/<run_id>.jsonl.

This is a stand-in for Langfuse so traces exist from day one; the event shape
(run_id, step, node, data, ms) maps directly onto Langfuse spans later.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from eudr_agent.config import RUNS_DIR


class Tracer:
    def __init__(self, run_name: str, echo: bool = False, on_event: Callable[[dict[str, Any]], None] | None = None):
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.run_name = run_name
        self.echo = echo
        self.on_event = on_event  # lets a screen show each step as it finishes
        self.step = 0
        self.events: list[dict[str, Any]] = []
        self.path = RUNS_DIR / "traces" / f"{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def span(self, node: str) -> Iterator[dict[str, Any]]:
        """Use as `with tracer.span("grade") as rec: rec["out"] = ...`."""
        self.step += 1
        rec: dict[str, Any] = {}
        start = time.perf_counter()
        try:
            yield rec
        finally:
            event = {
                "run_id": self.run_id,
                "run": self.run_name,
                "step": self.step,
                "node": node,
                "ms": round((time.perf_counter() - start) * 1000, 1),
                **rec,
            }
            self.events.append(event)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            if self.echo:
                summary = rec.get("summary", "")
                print(f"  [{self.step:02d}] {node:<16} {summary}")
            if self.on_event:
                self.on_event(event)
