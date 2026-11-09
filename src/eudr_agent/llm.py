"""Thin wrapper around any OpenAI-compatible chat endpoint (OpenAI, Groq, Gemini, Ollama, ...)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from openai import OpenAI

from eudr_agent import observability
from eudr_agent.config import settings


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    # wrapped so every call is traced in LangSmith when it is switched on (observability.py)
    return observability.wrap_llm_client(OpenAI(base_url=settings.llm_base_url or None, api_key=settings.llm_api_key))


def complete(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             max_tokens: int = 600) -> Any:
    """One deterministic completion. Returns the model's message, which may hold tool calls."""
    if not settings.llm_enabled:
        raise RuntimeError("No LLM configured; set LLM_API_KEY and LLM_MODEL in .env")
    args: dict[str, Any] = {}
    if tools:
        args["tools"] = tools
        args["parallel_tool_calls"] = False  # one tool at a time, so each result can shape the next choice
    if settings.llm_reasoning_effort:
        args["reasoning_effort"] = settings.llm_reasoning_effort
    resp = _client().chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        max_completion_tokens=max_tokens,
        messages=messages,
        **args,
    )
    return resp.choices[0].message


def chat(system: str, user: str, max_tokens: int = 600) -> str:
    """One text reply. Raises on an empty reply, which would otherwise read as a "no" from the grader."""
    message = complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                       max_tokens=max_tokens)
    text = (message.content or "").strip()
    if not text:
        raise RuntimeError(f"Empty reply from {settings.llm_model}; max_tokens={max_tokens} may be too low")
    return text
