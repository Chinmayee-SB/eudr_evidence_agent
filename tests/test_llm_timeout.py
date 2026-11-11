"""A hung LLM call must fail within the configured timeout, not the SDK's 10 minutes."""

from dataclasses import replace

from eudr_agent import llm


def test_client_has_a_short_timeout_and_retries(monkeypatch):
    test_settings = replace(llm.settings, llm_api_key="test-not-a-real-key")
    monkeypatch.setattr(llm, "settings", test_settings)
    llm._client.cache_clear()
    try:
        client = llm._client()
        assert client.timeout == test_settings.llm_timeout_s <= 120
        assert client.max_retries == test_settings.llm_retries
    finally:
        llm._client.cache_clear()
