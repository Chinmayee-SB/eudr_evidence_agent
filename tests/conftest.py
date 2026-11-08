"""Shared test setup: no test calls the real LLM, even when .env has a key, and search runs without the
reranker model (test_rag.py has one test that switches it on).

Tests stay free, fast and repeatable. A test that needs LLM behaviour fakes the reply (see test_supervisor.py
and test_geo_agent.py); the live runs and the eval/ scripts are where the real models are used.
"""

import pytest

from eudr_agent.config import Settings
from eudr_agent.rag import index


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    monkeypatch.setattr(Settings, "llm_enabled", property(lambda self: False))


@pytest.fixture(autouse=True)
def no_reranker(monkeypatch):
    monkeypatch.setattr(index, "RERANK", False)
