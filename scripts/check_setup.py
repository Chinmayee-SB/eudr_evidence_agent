"""Check the project is ready to run: package import, rule index, LLM chat and LLM tool calling.

Usage:  python scripts/check_setup.py
"""

from __future__ import annotations

import sys

from eudr_agent import llm
from eudr_agent.config import CHUNKS_PATH, settings
from eudr_agent.rag.chunk import load_chunks

TOOL = {
    "type": "function",
    "function": {
        "name": "get_plot_area",
        "description": "Return the area of a plot in hectares.",
        "parameters": {
            "type": "object",
            "properties": {"plot_id": {"type": "string"}},
            "required": ["plot_id"],
        },
    },
}


def check_index() -> str:
    if not CHUNKS_PATH.exists():
        raise RuntimeError("no index; run scripts/build_index.py")
    return f"{len(load_chunks())} chunks"


def check_chat() -> str:
    return repr(llm.chat("Reply with exactly one word.", "Say ready.", max_tokens=200))


def check_tool_call() -> str:
    message = llm.complete([{"role": "user", "content": "What is the area of plot K-06? Use the tool."}], tools=[TOOL])
    calls = message.tool_calls or []
    if not calls:
        raise RuntimeError("model replied without calling the tool")
    return f"{calls[0].function.name}({calls[0].function.arguments})"


def main() -> None:
    print(f"model: {settings.llm_model or '(none)'}  endpoint: {settings.llm_base_url or '(default)'}\n")
    failed = False
    for name, check in [("rule index", check_index), ("llm chat", check_chat), ("llm tool call", check_tool_call)]:
        try:
            print(f"  ok    {name:<14} {check()}")
        except Exception as e:  # report every failure instead of stopping at the first
            failed = True
            print(f"  FAIL  {name:<14} {type(e).__name__}: {str(e)[:300]}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
