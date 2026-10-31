"""Ask the rules agent one question and watch each step.

Usage:  python scripts/ask.py "Is shade-grown coffee under trees counted as forest?"
"""

from __future__ import annotations

import sys

from eudr_agent.config import settings
from eudr_agent.rag.agent import ask


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Is shade-grown coffee under trees counted as forest?"
    mode = f"LLM ({settings.llm_model})" if settings.llm_enabled else "offline (no LLM key)"
    print(f"Q: {question}\nmode: {mode}\n")
    state = ask(question, echo=True)
    print(f"\nstatus: {state['status']}")
    print(f"citations: {state['citations']}")
    print(f"\n{state['answer']}")
    print(f"\ntrace: {state['tracer'].path}")


if __name__ == "__main__":
    main()
