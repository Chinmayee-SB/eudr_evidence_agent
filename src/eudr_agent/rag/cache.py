"""Reuse the rules agent's answer to a question it has already answered, with the same setup.

The supervisor asks the same few questions in case after case ("Is shade-grown coffee under trees counted as
forest?"). Answering again costs about 4 LLM calls, takes several seconds, and can come out worded differently
for two plots, which an officer comparing cases would rightly question. A cached answer is free, instant and
identical.

The key is the exact question (ignoring case and spacing) plus a fingerprint of everything that shapes the
answer: the indexed texts, the prompts, the model, and the search settings. Rebuilding the index, editing a
prompt or switching model therefore starts a fresh cache by itself. Reworded questions are not matched on
purpose: in law, a slightly different question can need a different answer.

Only verified answers from the LLM are kept. Abstentions are not, so a question that failed once is tried
again next time. Files: data/cache/rules/<key>.json. RAG_CACHE=0 switches the cache off; the evaluation
scripts always bypass it, so they measure the agent and not the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from eudr_agent.config import CHUNKS_PATH, DATA_DIR, settings

CACHE_DIR = DATA_DIR / "cache" / "rules"
KEPT = ("answer", "citations", "sources", "status", "verification", "passages")
SHAPES_THE_ANSWER = ("agent.py", "prompts.py", "verify.py", "index.py", "rerank.py")


def enabled() -> bool:
    return os.getenv("RAG_CACHE", "1") != "0" and settings.llm_enabled


@lru_cache(maxsize=1)
def _code_and_texts() -> str:
    """The indexed texts and the rules agent's code: a change to either gives a new fingerprint."""
    h = hashlib.sha1(CHUNKS_PATH.read_bytes() if CHUNKS_PATH.exists() else b"")
    here = Path(__file__).parent
    for name in SHAPES_THE_ANSWER:
        h.update((here / name).read_bytes())
    return h.hexdigest()


def fingerprint() -> str:
    from eudr_agent.rag import index, rerank

    setup = [_code_and_texts(), settings.llm_model, settings.retrieval, settings.top_k, settings.max_retries,
             index.RERANK, index.RERANK_MODE, rerank.MODEL]
    return hashlib.sha1(json.dumps(setup).encode()).hexdigest()[:16]


def key(question: str) -> str:
    q = re.sub(r"\s+", " ", question.strip().lower())
    return hashlib.sha1(f"{fingerprint()}|{q}".encode()).hexdigest()[:20]


def get(question: str) -> dict[str, Any] | None:
    path = CACHE_DIR / f"{key(question)}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):  # a damaged file is a miss, not an error
        return None


def put(question: str, state: dict[str, Any]) -> None:
    if state.get("status") != "answered" or state.get("verification") not in {"passed", "repaired", "trimmed"}:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    record = {"question": question, "model": settings.llm_model, "fingerprint": fingerprint(),
              "created": time.strftime("%Y-%m-%d %H:%M:%S"), **{k: state.get(k) for k in KEPT}}
    path = CACHE_DIR / f"{key(question)}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # whole file or nothing, even if two cases answer at once
