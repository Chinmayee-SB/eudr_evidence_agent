"""Watch the geo-analyst investigate plots step by step.

Usage:  python scripts/run_geo_agent.py [plot_id ...]
"""

from __future__ import annotations

import sys

from eudr_agent.geo.agent import analyse
from eudr_agent.guardrail import check_input
from eudr_agent.plots import load_plots
from eudr_agent.tracing import Tracer


def main() -> None:
    plots = load_plots()
    for plot_id in sys.argv[1:] or sorted(plots):
        print(f"\n=== {plot_id}")
        tracer = Tracer("geo_analyst", echo=True)
        check = check_input(plots[plot_id], tracer=tracer)
        for line in check["blocked"] + check["dropped"] + check["warnings"]:
            print(f"  input: {line}")
        if not check["passed"]:
            print("  input check blocked this plot; the agent was not run")
            continue
        state = analyse(check["plot"], tracer=tracer)
        print(f"\nsummary: {state['summary']}")
        print(f"conflicts: {state['conflicts']}")
        print(f"open questions: {state['open_questions']}")
        print(f"tools used: {[f['tool'] for f in state['findings']]}")
        print(f"trace: {state['tracer'].path}")


if __name__ == "__main__":
    main()
